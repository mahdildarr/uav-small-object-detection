import numpy as np
import pandas as pd
import pytest

from uavdet.boxes import (
    Detections,
    box_ioa,
    box_iou,
    label_path_for,
    load_yolo_labels,
    match_detections,
    outside_ignored,
    size_bin,
)
from uavdet.metrics import CocoCollector, image_record, paired_bootstrap, summarize
from uavdet.slice_dataset import clip_to_tile, tile_grid
from uavdet.visdrone import load_det_annotation, map_classes


def test_iou_and_ioa():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30], [0, 0, 20, 20]]
    np.testing.assert_allclose(box_iou(a, b)[0], [1, 50 / 150, 0, 0.25], atol=1e-6)
    np.testing.assert_allclose(box_ioa(a, b)[0], [1, 0.5, 0, 1], atol=1e-6)
    assert box_iou([], b).shape == (0, 4)


def test_yolo_labels(tmp_path):
    (tmp_path / "images").mkdir()
    img = tmp_path / "images" / "a.jpg"
    assert label_path_for(img) == str(tmp_path / "labels" / "a.txt")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("3 0.5 0.5 0.2 0.1\n")
    cls, boxes = load_yolo_labels(label_path_for(img), 100, 200)
    assert cls.tolist() == [3]
    np.testing.assert_allclose(boxes[0], [40, 90, 60, 110])
    assert load_yolo_labels(tmp_path / "missing.txt", 10, 10)[1].shape == (0, 4)


def test_matching_is_greedy_by_confidence_and_class_aware():
    gt = np.array([[0, 0, 10, 10]], np.float32)
    pred = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], np.float32)
    # the second prediction has the highest score and wins the object; the duplicate is a false positive
    m, tp = match_detections(gt, [0], pred, [0, 0, 1], [0.5, 0.9, 0.99])
    assert m.tolist() == [True] and tp.tolist() == [False, True, False]


def test_size_bins_follow_coco():
    boxes = [[0, 0, 31, 31], [0, 0, 32, 32], [0, 0, 95, 95], [0, 0, 96, 96]]
    assert size_bin(boxes).tolist() == [0, 1, 1, 2]


def test_ignored_regions(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("10,10,20,20,1,4,0,0\n0,0,50,50,0,0,0,0\n100,100,5,5,1,11,0,0\n200,200,8,8,0,1,0,0\n")
    cls, boxes, ignore = load_det_annotation(f)
    assert cls.tolist() == [3] and len(ignore) == 3  # category 0, category 11 and score 0 are ignored
    keep = outside_ignored([[5, 5, 15, 15], [300, 300, 310, 310]], ignore)
    assert keep.tolist() == [False, True]


def test_class_mapping_drops_unmapped():
    g, keep = map_classes([0, 6, 3], {0: 0, 3: 1})
    assert g[keep].tolist() == [0, 1] and keep.tolist() == [True, False, True]


def test_recall_by_size_and_precision():
    gt = np.array([[0, 0, 10, 10], [50, 50, 150, 150]], np.float32)
    det = Detections(np.array([[0, 0, 10, 10], [300, 300, 310, 310]], np.float32), np.array([0, 0]),
                     np.array([0.9, 0.8]))
    s = summarize(pd.DataFrame([image_record(gt, [0, 0], det)]))
    assert s["precision"] == 0.5 and s["recall_small"] == 1 and s["recall_large"] == 0
    assert np.isnan(s["recall_medium"])


def test_paired_bootstrap_contains_point_estimate():
    rng = np.random.default_rng(0)
    rows_a, rows_b = [], []
    for _ in range(40):
        g = int(rng.integers(5, 15))
        rows_a.append({"tp": 5, "fp": 2, "gt_small": g, "hit_small": int(g * 0.3), "gt_medium": 0, "hit_medium": 0,
                       "gt_large": 0, "hit_large": 0})
        rows_b.append({**rows_a[-1], "hit_small": int(g * 0.5)})
    ci = paired_bootstrap(pd.DataFrame(rows_a), pd.DataFrame(rows_b), n=200)
    d, lo, hi = ci["recall_small"]
    assert lo <= d <= hi and lo > 0


def test_coco_ap_perfect_and_ignored_regions():
    gt = np.array([[0, 0, 20, 20], [40, 40, 140, 140]], np.float32)
    c = CocoCollector(2)
    det = Detections(np.vstack([gt, [[300, 300, 320, 320]]]), np.array([0, 1, 0]), np.array([0.9, 0.8, 0.95]))
    c.add(0, 400, 400, gt, [0, 1], det, ignore_boxes=[[290, 290, 330, 330]])
    r = c.evaluate()
    # the highest-scoring box lies on an ignored region and must not count as a false positive
    assert r["AP50"] == pytest.approx(1.0) and r["AP_small"] == pytest.approx(1.0)


def test_tiles_cover_image_and_clip_boxes():
    tiles = tile_grid(1000, 700, 640, 0.2)
    assert tiles[0] == (0, 0, 640, 640) and tiles[-1] == (360, 60, 1000, 700)
    boxes, cls = clip_to_tile([[600, 10, 700, 30], [630, 10, 700, 30], [10, 10, 20, 20]], [1, 2, 3], (0, 0, 640, 640))
    # 40% of the first box is visible -> dropped; the third box is fully inside
    assert cls.tolist() == [3]
    boxes, cls = clip_to_tile([[600, 10, 660, 30]], [1], (0, 0, 640, 640))
    np.testing.assert_allclose(boxes[0], [600, 10, 640, 30])

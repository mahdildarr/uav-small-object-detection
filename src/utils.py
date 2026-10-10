"""Shared helpers: YOLO label parsing, IoU computation and prediction-to-ground-truth matching."""

import os

import numpy as np


def label_path_for(img_path):
    """Derive the label path from an image path using the Ultralytics convention (.../images/x.jpg -> .../labels/x.txt)."""
    p = str(img_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in p:
        p = sb.join(p.rsplit(sa, 1))
    return os.path.splitext(p)[0] + ".txt"


def load_yolo_labels(label_path, width, height):
    """Convert a YOLO label file (class cx cy w h, normalized) to pixel xyxy boxes.

    Returns: (classes [N], boxes [N, 4])
    """
    if not os.path.exists(label_path):
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                rows.append([float(v) for v in parts[:5]])
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    return cls, boxes


def box_iou(a, b):
    """Pairwise IoU between two box sets. a: [N, 4], b: [M, 4] (xyxy) -> [N, M]"""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a_, b_ = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a_[..., 2], b_[..., 2]) - np.maximum(a_[..., 0], b_[..., 0]), 0, None)
    ih = np.clip(np.minimum(a_[..., 3], b_[..., 3]) - np.maximum(a_[..., 1], b_[..., 1]), 0, None)
    inter = iw * ih
    area_a = (a_[..., 2] - a_[..., 0]) * (a_[..., 3] - a_[..., 1])
    area_b = (b_[..., 2] - b_[..., 0]) * (b_[..., 3] - b_[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def match_detections(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, iou_thr=0.5):
    """Greedy, class-aware matching in descending confidence order.

    Returns: (matched ground-truth boxes [N] bool, true-positive predictions [M] bool)
    """
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_matched, pred_tp
    ious = box_iou(pred_boxes, gt_boxes)
    for i in np.argsort(-np.asarray(pred_conf)):
        cand = np.where((gt_cls == pred_cls[i]) & (~gt_matched))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= iou_thr:
            gt_matched[j] = True
            pred_tp[i] = True
    return gt_matched, pred_tp

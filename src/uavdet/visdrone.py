"""VisDrone annotation formats, class lists and ignored regions.

Raw VisDrone-DET line:  x, y, w, h, score, category, truncation, occlusion
Raw VisDrone-MOT line:  frame, target_id, x, y, w, h, score, category, truncation, occlusion

Category 0 marks *ignored regions* and 11 marks *others*; the official evaluation discards detections
inside those areas. ``score`` is 0 for boxes that are not considered in the evaluation. The
Ultralytics conversion to YOLO format drops these regions, so detections inside them are scored as
false positives unless the raw annotations are used (``--annotations``).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

# YOLO class index = VisDrone category - 1
CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
IGNORE_CATEGORIES = {0, 11}
# Categories scored in the VisDrone2019-MOT challenge: pedestrian, car, van, truck, bus.
MOT_EVAL_CATEGORIES = (1, 4, 5, 6, 9)

# Shared groups for the zero-shot baseline: 0 person, 1 vehicle, 2 two-wheeler (-1 = dropped).
GROUPS = ["person", "vehicle", "two-wheeler"]
VISDRONE_TO_GROUP = {0: 0, 1: 0, 2: 2, 3: 1, 4: 1, 5: 1, 6: -1, 7: -1, 8: 1, 9: 2}
COCO_TO_GROUP = {0: 0, 1: 2, 2: 1, 3: 2, 5: 1, 7: 1}


def map_classes(cls, mapping: dict) -> tuple[np.ndarray, np.ndarray]:
    """Apply a class mapping; returns (mapped classes, keep mask). Classes mapped to -1 are dropped."""
    g = np.array([mapping.get(int(c), -1) for c in cls], dtype=int)
    return g, g >= 0


def _read_rows(path, n: int) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            v = line.strip().rstrip(",").split(",")
            if len(v) >= n:
                rows.append([float(x) for x in v[:n]])
    return np.array(rows, dtype=np.float64).reshape(-1, n)


def load_det_annotation(path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw VisDrone-DET file -> (YOLO class ids [N], xyxy boxes [N, 4], ignored-region boxes [K, 4])."""
    a = _read_rows(path, 6)
    xyxy = np.c_[a[:, 0], a[:, 1], a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]].astype(np.float32)
    cat, score = a[:, 5].astype(int), a[:, 4].astype(int)
    ignore = np.isin(cat, list(IGNORE_CATEGORIES)) | (score == 0)
    return cat[~ignore] - 1, xyxy[~ignore], xyxy[ignore]


def annotation_path_for(img_path, annotations_dir) -> Path:
    return Path(annotations_dir) / (Path(img_path).stem + ".txt")


class MotSequence:
    """Ground truth of one VisDrone-MOT sequence, indexed by frame (1-based)."""

    def __init__(self, path, categories=MOT_EVAL_CATEGORIES):
        a = _read_rows(path, 8)
        self.objects: dict[int, list[tuple[int, int, list[float]]]] = defaultdict(list)
        # Ignored regions, plus boxes that are not scored (other categories or score 0): detections on
        # them are neither required nor penalised.
        self.ignore: dict[int, list[list[float]]] = defaultdict(list)
        self.unscored: dict[int, list[list[float]]] = defaultdict(list)
        for frame, tid, x, y, w, h, score, cat in a:
            f, c, box = int(frame), int(cat), [x, y, x + w, y + h]
            if c in IGNORE_CATEGORIES:
                self.ignore[f].append(box)
            elif int(score) == 1 and (categories is None or c in categories):
                self.objects[f].append((int(tid), c, box))
            else:
                self.unscored[f].append(box)
        self.n_frames = int(a[:, 0].max()) if len(a) else 0


def load_mot_results(path) -> dict[int, list[tuple[int, int, list[float], float]]]:
    """Tracker output in MOT format (frame, id, x, y, w, h, conf, category, -1, -1) -> frame -> objects."""
    out = defaultdict(list)
    for frame, tid, x, y, w, h, conf, cat in _read_rows(path, 8):
        out[int(frame)].append((int(tid), int(cat), [x, y, x + w, y + h], float(conf)))
    return out

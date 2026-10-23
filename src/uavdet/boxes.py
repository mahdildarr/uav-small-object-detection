"""Box utilities: YOLO label parsing, IoU / IoA, size buckets and prediction-to-ground-truth matching."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# COCO object-size buckets (pixel area of the box).
SIZE_BINS = [("small", 0.0, 32.0**2), ("medium", 32.0**2, 96.0**2), ("large", 96.0**2, float("inf"))]


@dataclass
class Detections:
    """Boxes in pixel xyxy, integer class ids and confidences (all numpy arrays of the same length)."""

    boxes: np.ndarray
    cls: np.ndarray
    conf: np.ndarray

    @classmethod
    def empty(cls) -> Detections:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32))

    def __len__(self) -> int:
        return len(self.boxes)

    def subset(self, keep: np.ndarray) -> Detections:
        return Detections(self.boxes[keep], self.cls[keep], self.conf[keep])


def label_path_for(img_path) -> str:
    """Ultralytics convention: .../images/x.jpg -> .../labels/x.txt"""
    p = str(img_path)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa in p:
        p = sb.join(p.rsplit(sa, 1))
    return os.path.splitext(p)[0] + ".txt"


def load_yolo_labels(label_path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO label file (class cx cy w h, normalised) -> (classes [N], pixel xyxy boxes [N, 4])."""
    rows = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            rows = [[float(v) for v in line.split()[:5]] for line in f if len(line.split()) >= 5]
    if not rows:
        return np.zeros(0, dtype=int), np.zeros((0, 4), dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    cx, cy, bw, bh = arr[:, 1] * width, arr[:, 2] * height, arr[:, 3] * width, arr[:, 4] * height
    return arr[:, 0].astype(int), np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def box_area(b) -> np.ndarray:
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_, b_ = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a_[..., 2], b_[..., 2]) - np.maximum(a_[..., 0], b_[..., 0]), 0, None)
    ih = np.clip(np.minimum(a_[..., 3], b_[..., 3]) - np.maximum(a_[..., 1], b_[..., 1]), 0, None)
    return iw * ih


def box_iou(a, b) -> np.ndarray:
    """Pairwise IoU between two xyxy box sets: [N, 4] x [M, 4] -> [N, M]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter = _intersection(a, b)
    return inter / np.maximum(box_area(a)[:, None] + box_area(b)[None, :] - inter, 1e-9)


def box_ioa(a, b) -> np.ndarray:
    """Intersection over the area of each box in ``a``: [N, 4] x [M, 4] -> [N, M]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return _intersection(a, b) / np.maximum(box_area(a)[:, None], 1e-9)


def outside_ignored(boxes, ignore_boxes, thr: float = 0.5) -> np.ndarray:
    """Mask of boxes that do *not* lie mostly (IoA >= thr) inside an ignored region.

    VisDrone marks crowded or unclear areas as ignored regions; detections there are neither true
    nor false positives and must be removed before scoring.
    """
    ioa = box_ioa(boxes, ignore_boxes)
    if ioa.shape[1] == 0:
        return np.ones(len(ioa), dtype=bool)
    return ioa.max(axis=1) < thr


def size_bin(boxes) -> np.ndarray:
    """Index into SIZE_BINS for each box."""
    area = box_area(boxes)
    return np.searchsorted([hi for _, _, hi in SIZE_BINS[:-1]], area, side="right")


def match_detections(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf, iou_thr: float = 0.5):
    """Greedy, class-aware matching in descending confidence order (as in the COCO / VOC protocols).

    Returns: (matched ground-truth mask [N], true-positive prediction mask [M])
    """
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    pred_tp = np.zeros(len(pred_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_matched, pred_tp
    gt_cls, pred_cls = np.asarray(gt_cls), np.asarray(pred_cls)
    ious = box_iou(pred_boxes, gt_boxes)
    for i in np.argsort(-np.asarray(pred_conf), kind="stable"):
        cand = np.where((gt_cls == pred_cls[i]) & (~gt_matched))[0]
        if len(cand) == 0:
            continue
        j = cand[np.argmax(ious[i, cand])]
        if ious[i, j] >= iou_thr:
            gt_matched[j] = True
            pred_tp[i] = True
    return gt_matched, pred_tp

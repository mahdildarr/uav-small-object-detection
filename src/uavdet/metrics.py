"""Detection metrics: operating-point precision / recall by object size, paired bootstrap confidence
intervals, and COCO-style AP (pycocotools) with VisDrone ignored regions treated as crowd boxes."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from .boxes import SIZE_BINS, Detections, match_detections, size_bin


def image_record(gt_boxes, gt_cls, det: Detections, iou_thr: float = 0.5) -> dict:
    """Counts for one image at a fixed confidence threshold."""
    gt_matched, pred_tp = match_detections(gt_boxes, gt_cls, det.boxes, det.cls, det.conf, iou_thr)
    rec = {"tp": int(pred_tp.sum()), "fp": int((~pred_tp).sum())}
    bins = size_bin(gt_boxes)
    for k, (name, _, _) in enumerate(SIZE_BINS):
        rec[f"gt_{name}"] = int((bins == k).sum())
        rec[f"hit_{name}"] = int(((bins == k) & gt_matched).sum())
    return rec


def summarize(df: pd.DataFrame) -> dict:
    """Pool per-image records (one row per image) into precision, recall per size and overall recall."""
    s = df.sum(numeric_only=True)
    out = {"precision": s["tp"] / max(s["tp"] + s["fp"], 1)}
    hits = gts = 0
    for name, _, _ in SIZE_BINS:
        out[f"recall_{name}"] = s[f"hit_{name}"] / s[f"gt_{name}"] if s[f"gt_{name}"] else float("nan")
        hits, gts = hits + s[f"hit_{name}"], gts + s[f"gt_{name}"]
    out["recall"] = hits / max(gts, 1)
    if "time" in df:
        out["time_per_image"] = float(df["time"].mean())
    return out


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """95% CI of summarize(b) - summarize(a), resampling *images* (both methods see the same resample).

    Objects in one image are correlated, so resampling images rather than objects gives honest intervals.
    """
    assert len(a) == len(b), "both methods must be evaluated on the same images"
    rng = np.random.default_rng(seed)
    point = {k: summarize(b)[k] - summarize(a)[k] for k in summarize(a) if k != "time_per_image"}
    va, vb = a.reset_index(drop=True), b.reset_index(drop=True)
    draws = {k: [] for k in point}
    for _ in range(n):
        idx = rng.integers(0, len(a), len(a))
        sa, sb = summarize(va.iloc[idx]), summarize(vb.iloc[idx])
        for k in point:
            draws[k].append(sb[k] - sa[k])
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def pct(v, q):
        v = np.asarray(v, dtype=float)
        return float(np.percentile(v[np.isfinite(v)], q)) if np.isfinite(v).any() else float("nan")

    return {k: (point[k], pct(draws[k], lo), pct(draws[k], hi)) for k in point}


class CocoCollector:
    """Accumulates ground truth and detections and runs the official COCO evaluation.

    Ignored regions are added as ``iscrowd=1`` boxes: COCOeval then neither rewards nor penalises
    detections on them, which matches the VisDrone protocol.
    """

    def __init__(self, n_classes: int):
        self.cats = [{"id": i, "name": str(i)} for i in range(n_classes)]
        self.images, self.gt, self.dt = [], [], []

    def add(self, image_id: int, width: int, height: int, gt_boxes, gt_cls, det: Detections, ignore_boxes=()):
        self.images.append({"id": image_id, "width": width, "height": height})

        def xywh(b):
            return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]

        for b, c in zip(gt_boxes, gt_cls):
            w = xywh(b)
            self.gt.append({"id": len(self.gt) + 1, "image_id": image_id, "category_id": int(c), "bbox": w,
                            "area": w[2] * w[3], "iscrowd": 0})
        for b in ignore_boxes:  # a crowd box per class so that it applies to every category
            w = xywh(b)
            for cat in self.cats:
                self.gt.append({"id": len(self.gt) + 1, "image_id": image_id, "category_id": cat["id"], "bbox": w,
                                "area": w[2] * w[3], "iscrowd": 1})
        for b, c, s in zip(det.boxes, det.cls, det.conf):
            self.dt.append({"image_id": image_id, "category_id": int(c), "bbox": xywh(b), "score": float(s)})

    def evaluate(self, max_dets: int = 500) -> dict:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        gt = COCO()
        gt.dataset = {"images": self.images, "annotations": self.gt, "categories": self.cats}
        with contextlib.redirect_stdout(io.StringIO()):
            gt.createIndex()
            dt = gt.loadRes(self.dt) if self.dt else COCO()
            if not self.dt:
                return {k: 0.0 for k in ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")}
            ev = COCOeval(gt, dt, "bbox")
            ev.params.maxDets = [1, 100, max_dets]  # aerial images hold far more than 100 objects
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        s = [float(v) if v >= 0 else float("nan") for v in ev.stats]  # -1: no objects of that size
        return {"AP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AP_large": s[5],
                f"AR@{max_dets}": s[8]}

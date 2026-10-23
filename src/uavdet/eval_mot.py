"""Score MOT-format tracking results against VisDrone-MOT ground truth.

Metrics come from py-motmetrics (CLEAR-MOT and identity metrics): MOTA, MOTP, IDF1, ID switches,
fragmentations, mostly tracked / mostly lost. Following the VisDrone-MOT protocol:

* only the evaluated categories are required (default: pedestrian, car, van, truck, bus),
* predictions inside ignored regions, or on objects that are not scored (other categories,
  score 0), are removed before matching instead of being counted as false positives,
* matching is class-agnostic with IoU >= 0.5.

Tracking is decoupled from scoring, so any tracker that writes MOT text files can be compared.

Example::

    python -m uavdet.eval_mot --results runs/track/bytetrack --annotations VisDrone2019-MOT-val/annotations
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .boxes import box_iou, outside_ignored
from .visdrone import MOT_EVAL_CATEGORIES, MotSequence, load_mot_results

METRICS = ["mota", "motp", "idf1", "idp", "idr", "num_switches", "num_fragmentations", "mostly_tracked",
           "mostly_lost", "num_false_positives", "num_misses", "num_objects", "num_unique_objects"]


def scorable(gt: MotSequence, frame: int, g: list, p: list, iou_thr: float) -> np.ndarray:
    """Mask of predictions that take part in scoring.

    A prediction that cannot match any scored object (IoU < thr) and lies on an ignored region or on
    an unscored object is removed; one that could match a scored object is always kept.
    """
    p_boxes = np.array([x[2] for x in p], dtype=np.float32).reshape(-1, 4)
    if len(p_boxes) == 0:
        return np.zeros(0, dtype=bool)
    could_match = box_iou(p_boxes, [x[2] for x in g]).max(axis=1, initial=0) >= iou_thr
    on_ignored = ~outside_ignored(p_boxes, gt.ignore.get(frame, []))
    on_unscored = box_iou(p_boxes, gt.unscored.get(frame, [])).max(axis=1, initial=0) >= iou_thr
    return could_match | ~(on_ignored | on_unscored)


def accumulate(gt: MotSequence, pred: dict, iou_thr: float = 0.5, n_frames: int | None = None):
    """Build a motmetrics accumulator for one sequence."""
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    n_frames = n_frames or max(gt.n_frames, max(pred, default=0))
    for f in range(1, n_frames + 1):
        g = gt.objects.get(f, [])
        p = pred.get(f, [])
        p = [x for x, k in zip(p, scorable(gt, f, g, p, iou_thr)) if k]
        iou = box_iou([x[2] for x in g], [x[2] for x in p])
        dist = np.where(iou >= iou_thr, 1 - iou, np.nan)
        acc.update([x[0] for x in g], [x[0] for x in p], dist, frameid=f)
    return acc


def evaluate(results_dir, annotations_dir, categories=MOT_EVAL_CATEGORIES, iou_thr: float = 0.5):
    """Return a DataFrame with one row per sequence plus an OVERALL row."""
    import motmetrics as mm

    accs, names = [], []
    for res in sorted(Path(results_dir).glob("*.txt")):
        ann = Path(annotations_dir) / res.name
        if not ann.exists():
            print(f"skip {res.name}: no ground truth")
            continue
        accs.append(accumulate(MotSequence(ann, categories), load_mot_results(res), iou_thr))
        names.append(res.stem)
    if not accs:
        raise SystemExit(f"No result files with matching ground truth in {results_dir}")
    mh = mm.metrics.create()
    return mh.compute_many(accs, names=names, metrics=METRICS, generate_overall=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="VisDrone-MOT evaluation")
    p.add_argument("--results", required=True, help="Folder of MOT-format result files (<sequence>.txt)")
    p.add_argument("--annotations", required=True, help="VisDrone-MOT annotation folder")
    p.add_argument("--all-categories", action="store_true", help="Score categories 1-10 instead of the official five")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", default=None, help="CSV output (default: <results>/metrics.csv)")
    a = p.parse_args(argv)

    df = evaluate(a.results, a.annotations, None if a.all_categories else MOT_EVAL_CATEGORIES, a.iou)
    out = Path(a.out or Path(a.results) / "metrics.csv")
    df.to_csv(out)
    import motmetrics as mm

    print(mm.io.render_summary(df, formatters=mm.metrics.create().formatters,
                               namemap=mm.io.motchallenge_metric_names))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

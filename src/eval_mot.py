"""Measure tracking performance (MOTA and IDF1) on a VisDrone-MOT sequence.

Tracking is evaluated class-agnostically. Only ground-truth objects marked for evaluation (score = 1)
with categories 1–10 are used.

Note: this is a compact re-implementation of the CLEAR-MOT and IDF1 definitions. For official numbers,
cross-check with TrackEval.

Example:
    python src/eval_mot.py --weights best.pt \
        --seq-dir VisDrone2019-MOT-val/sequences/uav0000086_00000_v \
        --gt VisDrone2019-MOT-val/annotations/uav0000086_00000_v.txt
"""

import argparse
import os
from collections import defaultdict
from glob import glob

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from utils import box_iou


def parse_args():
    p = argparse.ArgumentParser(description="MOTA / IDF1 evaluation")
    p.add_argument("--weights", required=True)
    p.add_argument("--seq-dir", required=True, help="Sequence frame folder")
    p.add_argument("--gt", required=True, help="VisDrone-MOT annotation .txt file")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    return p.parse_args()


def load_gt(path):
    """frame -> [(id, [x1, y1, x2, y2]), ...]"""
    gt = defaultdict(list)
    for line in open(path):
        v = line.strip().split(",")
        if len(v) < 8:
            continue
        frame, tid, x, y, w, h, score, cat = map(float, v[:8])
        if int(score) == 1 and 1 <= int(cat) <= 10:
            gt[int(frame)].append((int(tid), [x, y, x + w, y + h]))
    return gt


def run_tracker(weights, frames, tracker, imgsz, conf):
    model = YOLO(weights)
    preds = defaultdict(list)
    for i, f in enumerate(frames, start=1):
        r = model.track(f, persist=True, tracker=tracker, imgsz=imgsz, conf=conf, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue
        for tid, box in zip(r.boxes.id.int().cpu().tolist(), r.boxes.xyxy.cpu().numpy().tolist()):
            preds[i].append((tid, box))
        if i % 100 == 0:
            print(f"  {i}/{len(frames)} frames")
    return preds


def evaluate(gt, pr, n_frames, thr):
    tp = fp = fn = idsw = n_gt = n_pr = 0
    last = {}  # ground-truth ID -> last matched predicted ID
    pair_counts = defaultdict(int)
    for f in range(1, n_frames + 1):
        g, p = gt.get(f, []), pr.get(f, [])
        gids, pids = [x[0] for x in g], [x[0] for x in p]
        iou = box_iou([x[1] for x in g], [x[1] for x in p])
        n_gt += len(g)
        n_pr += len(p)

        for a_, b_ in zip(*np.where(iou >= thr)):
            pair_counts[(gids[a_], pids[b_])] += 1

        matches, used_g, used_p = [], set(), set()
        for a_, gid in enumerate(gids):  # keep previous matches when still valid (CLEAR-MOT)
            if gid in last and last[gid] in pids:
                b_ = pids.index(last[gid])
                if b_ not in used_p and iou[a_, b_] >= thr:
                    matches.append((a_, b_))
                    used_g.add(a_)
                    used_p.add(b_)
        rem_g = [i for i in range(len(gids)) if i not in used_g]
        rem_p = [j for j in range(len(pids)) if j not in used_p]
        if rem_g and rem_p:
            cost = 1 - iou[np.ix_(rem_g, rem_p)]
            for r_, c_ in zip(*linear_sum_assignment(cost)):
                if cost[r_, c_] <= 1 - thr:
                    a_, b_ = rem_g[r_], rem_p[c_]
                    if gids[a_] in last and last[gids[a_]] != pids[b_]:
                        idsw += 1
                    matches.append((a_, b_))
        for a_, b_ in matches:
            last[gids[a_]] = pids[b_]
        tp += len(matches)
        fn += len(gids) - len(matches)
        fp += len(pids) - len(matches)

    mota = 1 - (fn + fp + idsw) / max(n_gt, 1)

    idtp = 0
    if pair_counts:
        g_ids = sorted({k[0] for k in pair_counts})
        p_ids = sorted({k[1] for k in pair_counts})
        gi, pi = {g: i for i, g in enumerate(g_ids)}, {p: i for i, p in enumerate(p_ids)}
        m = np.zeros((len(g_ids), len(p_ids)))
        for (g_, p_), c in pair_counts.items():
            m[gi[g_], pi[p_]] = c
        r_, c_ = linear_sum_assignment(-m)
        idtp = m[r_, c_].sum()
    idf1 = 2 * idtp / max(2 * idtp + (n_pr - idtp) + (n_gt - idtp), 1)

    return {
        "MOTA": mota,
        "IDF1": idf1,
        "ID switches (IDSW)": idsw,
        "Precision": tp / max(tp + fp, 1),
        "Recall": tp / max(n_gt, 1),
        "Ground-truth boxes": n_gt,
    }


def main():
    a = parse_args()
    frames = sorted(glob(os.path.join(a.seq_dir, "*.jpg")))
    if not frames:
        raise SystemExit(f"No frames found: {a.seq_dir}")
    print(f"Tracking {len(frames)} frames...")
    gt = load_gt(a.gt)
    pr = run_tracker(a.weights, frames, a.tracker, a.imgsz, a.conf)
    res = evaluate(gt, pr, len(frames), a.iou)
    print("\n=== Tracking results ===")
    for k, v in res.items():
        print(f"{k:<20}: {v:.4f}" if isinstance(v, float) else f"{k:<20}: {v}")


if __name__ == "__main__":
    main()

"""Standard vs sliced (SAHI) inference, broken down by object size.

For each method the script reports, on the same images:

* precision and recall per size bucket at a fixed confidence threshold (the operating point),
* COCO AP / AP50 / AP_small / AP_medium / AP_large from pycocotools (``--ap``),
* time per image,
* a paired bootstrap 95% confidence interval for every difference (sliced - standard).

Size buckets follow COCO: small < 32², medium 32²–96², large > 96² pixels.

With ``--annotations`` the raw VisDrone files are read instead of the YOLO labels, so that ignored
regions are honoured (detections there are not counted as false positives).

Example::

    python -m uavdet.evaluate --weights runs/visdrone/yolo11s_1024/weights/best.pt --limit 0 --ap
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .boxes import SIZE_BINS, Detections, box_iou, label_path_for, load_yolo_labels, match_detections, outside_ignored
from .metrics import CocoCollector, image_record, paired_bootstrap, summarize
from .visdrone import (
    CLASSES,
    COCO_TO_GROUP,
    GROUPS,
    VISDRONE_TO_GROUP,
    annotation_path_for,
    load_det_annotation,
    map_classes,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Size-aware comparison of standard and sliced inference")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", default=None, help="Validation image folder (default: VisDrone.yaml val split)")
    p.add_argument("--annotations", default=None, help="Raw VisDrone-DET annotation folder (enables ignored regions)")
    p.add_argument("--limit", type=int, default=200, help="Number of images (0 = all)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25, help="Operating-point confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    p.add_argument("--slice", type=int, default=640, help="SAHI tile size")
    p.add_argument("--overlap", type=float, default=0.2, help="SAHI tile overlap ratio")
    p.add_argument("--no-sahi", action="store_true", help="Evaluate standard inference only")
    p.add_argument("--ap", action="store_true",
                   help="Also compute COCO AP; inference then runs at --ap-conf and the operating point is "
                        "obtained by filtering, so timings include the extra low-score boxes")
    p.add_argument("--ap-conf", type=float, default=0.01)
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples (0 = off)")
    p.add_argument("--coco-zero-shot", action="store_true",
                   help="Evaluate a COCO-pretrained model: COCO and VisDrone classes are mapped to shared groups "
                        "(person, vehicle, two-wheeler); tricycles have no COCO counterpart and are dropped")
    p.add_argument("--save-vis", type=int, default=0, help="Number of side-by-side visualisations")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/eval")
    return p.parse_args(argv)


def resolve_images(images):
    if images:
        return images
    from ultralytics.data.utils import check_det_dataset

    val = check_det_dataset("VisDrone.yaml")["val"]
    return val[0] if isinstance(val, list) else val


def load_ground_truth(img_path, w, h, annotations=None):
    if annotations:
        return load_det_annotation(annotation_path_for(img_path, annotations))
    cls, boxes = load_yolo_labels(label_path_for(img_path), w, h)
    return cls, boxes, np.zeros((0, 4), np.float32)


def draw(img, gt_boxes, gt_matched, det: Detections, title):
    vis = img.copy()
    for b in det.boxes.astype(int):
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 160, 0), 1)
    for b, m in zip(gt_boxes.astype(int), gt_matched):
        if not m:  # missed objects in red
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis


def plot_recall_by_size(summary: dict, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n, _, _ in SIZE_BINS]
    x = np.arange(len(names))
    w = 0.8 / len(summary)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, (method, s) in enumerate(summary.items()):
        ax.bar(x + i * w - 0.4 + w / 2, [s[f"recall_{n}"] for n in names], w, label=method)
    ax.set_xticks(x, [n.capitalize() for n in names])
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_report(summary, ap, ci, counts, args, out: Path) -> str:
    names = [n for n, _, _ in SIZE_BINS]
    lines = ["| Method | Precision | " + " | ".join(f"Recall {n}" for n in names) + " | Recall | Time / image |",
             "|---|---|" + "---|" * len(names) + "---|---|"]
    for m, s in summary.items():
        lines.append(f"| {m} | {s['precision']:.3f} | " + " | ".join(f"{s[f'recall_{n}']:.3f}" for n in names)
                     + f" | {s['recall']:.3f} | {s['time_per_image']:.2f} s |")
    lines += ["", "Ground-truth objects: " + ", ".join(f"{n} {counts[n]:,}" for n in names),
              f"Operating point: confidence {args.conf}, IoU {args.iou}. "
              + (f"Inference at confidence {args.ap_conf} (for AP); sliced predictions merged with NMS."
                 if args.ap else "Sliced predictions merged with greedy NMM (SAHI default).")]
    if ap:
        keys = list(next(iter(ap.values())))
        lines += ["", "| Method | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
        lines += [f"| {m} | " + " | ".join(f"{v[k]:.3f}" for k in keys) + " |" for m, v in ap.items()]
    if ci:
        lines += ["", f"Sliced − standard, paired bootstrap over images ({args.bootstrap} resamples, 95% CI):", "",
                  "| Metric | Difference | 95% CI |", "|---|---|---|"]
        lines += [f"| {k} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |" for k, (d, lo, hi) in ci.items()]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def evaluate(args, detectors: dict | None = None) -> dict:
    """Run the comparison. ``detectors`` (name -> callable) can be injected, e.g. in tests."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    img_dir = Path(resolve_images(args.images))
    images = sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png")])
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found in {img_dir}")

    if detectors is None:
        from .detectors import SlicedDetector, StandardDetector

        conf = args.ap_conf if args.ap else args.conf
        detectors = {"Standard": StandardDetector(args.weights, args.imgsz, conf, args.device)}
        if not args.no_sahi:
            detectors["Sliced (SAHI)"] = SlicedDetector(args.weights, args.slice, args.overlap, conf,
                                                        postprocess="NMS" if args.ap else "GREEDYNMM",
                                                        device=args.device)
    n_classes = len(GROUPS) if args.coco_zero_shot else len(CLASSES)
    records = {m: [] for m in detectors}
    coco = {m: CocoCollector(n_classes) for m in detectors} if args.ap else {}
    print(f"Evaluating {len(images)} images from {img_dir} with {', '.join(detectors)}")

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt_cls, gt_boxes, ignore = load_ground_truth(img_path, w, h, args.annotations)
        if args.coco_zero_shot:
            gt_cls, keep = map_classes(gt_cls, VISDRONE_TO_GROUP)
            gt_cls, gt_boxes = gt_cls[keep], gt_boxes[keep]
        panels = []
        for m, detect in detectors.items():
            t0 = time.perf_counter()
            det = detect(img_path)
            elapsed = time.perf_counter() - t0
            if args.coco_zero_shot:
                cls, keep = map_classes(det.cls, COCO_TO_GROUP)
                det = Detections(det.boxes[keep], cls[keep], det.conf[keep])
            if args.ap:  # COCOeval handles ignored regions itself (crowd boxes)
                coco[m].add(idx, w, h, gt_boxes, gt_cls, det, ignore)
            # operating point: drop detections on ignored regions unless they could match an object
            could_match = box_iou(det.boxes, gt_boxes).max(axis=1, initial=0) >= args.iou
            op = det.subset((det.conf >= args.conf) & (could_match | outside_ignored(det.boxes, ignore)))
            records[m].append({"image": img_path.name, **image_record(gt_boxes, gt_cls, op, args.iou), "time": elapsed})
            if idx < args.save_vis:
                matched, _ = match_detections(gt_boxes, gt_cls, op.boxes, op.cls, op.conf, args.iou)
                panels.append(draw(img, gt_boxes, matched, op, m))
        if panels:
            (out / "vis").mkdir(exist_ok=True)
            cv2.imwrite(str(out / "vis" / img_path.name), np.hstack(panels))
        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(images)}")

    frames = {m: pd.DataFrame(r) for m, r in records.items()}
    pd.concat([f.assign(method=m) for m, f in frames.items()]).to_csv(out / "per_image.csv", index=False)
    summary = {m: summarize(f) for m, f in frames.items()}
    ap = {m: c.evaluate() for m, c in coco.items()}
    ci = {}
    if args.bootstrap and len(frames) == 2:
        a, b = frames.values()
        ci = paired_bootstrap(a, b, n=args.bootstrap)
    counts = {n: int(next(iter(frames.values()))[f"gt_{n}"].sum()) for n, _, _ in SIZE_BINS}
    plot_recall_by_size(summary, out / "recall_by_size.png")
    text = write_report(summary, ap, ci, counts, args, out)
    result = {"summary": summary, "ap": ap, "bootstrap_ci": ci, "gt_counts": counts, "n_images": len(images),
              "args": vars(args)}
    (out / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("\n" + text + f"\nSaved to {out}/")
    return result


def main(argv=None):
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()

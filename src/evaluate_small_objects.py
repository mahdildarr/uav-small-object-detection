"""Compare standard inference with sliced (SAHI) inference, broken down by object size.

For each method the script reports precision, recall per object-size bucket and time per image.
Size buckets follow the COCO definition: small < 32², medium 32²–96², large > 96² pixel area.

Example:
    python src/evaluate_small_objects.py --weights runs/visdrone/yolo11s_1024/weights/best.pt --limit 200 --save-vis 6
"""

import argparse
import os
import time
from glob import glob

import cv2
import numpy as np
from ultralytics import YOLO

from utils import label_path_for, load_yolo_labels, match_detections

# Shared groups for the zero-shot baseline: 0 person, 1 vehicle, 2 two-wheeler (-1 = dropped).
VISDRONE_TO_GROUP = {0: 0, 1: 0, 2: 2, 3: 1, 4: 1, 5: 1, 6: -1, 7: -1, 8: 1, 9: 2}
COCO_TO_GROUP = {0: 0, 1: 2, 2: 1, 3: 2, 5: 1, 7: 1}


def to_groups(cls, boxes, mapping, conf=None):
    g = np.array([mapping.get(int(c), -1) for c in cls], dtype=int)
    keep = g >= 0
    out = (g[keep], boxes[keep]) if conf is None else (boxes[keep], g[keep], conf[keep])
    return out


SIZE_BINS = [("Small (<32²)", 0, 32**2), ("Medium (32²–96²)", 32**2, 96**2), ("Large (>96²)", 96**2, float("inf"))]


def parse_args():
    p = argparse.ArgumentParser(description="Small-object-focused comparative evaluation")
    p.add_argument("--weights", required=True)
    p.add_argument("--images", default=None, help="Validation image folder (defaults to VisDrone.yaml val split)")
    p.add_argument("--limit", type=int, default=200, help="Number of images to evaluate (0 = all)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5, help="IoU threshold for matching")
    p.add_argument("--slice", type=int, default=640, help="SAHI slice size")
    p.add_argument("--overlap", type=float, default=0.2, help="SAHI slice overlap ratio")
    p.add_argument("--no-sahi", action="store_true", help="Evaluate standard inference only")
    p.add_argument("--save-vis", type=int, default=0, help="Number of side-by-side visualizations to save")
    p.add_argument("--coco-zero-shot", action="store_true",
                   help="Evaluate a COCO-pretrained model before fine-tuning: COCO and VisDrone classes are mapped to "
                        "shared groups (person, vehicle, two-wheeler); tricycles have no COCO counterpart and are dropped")
    p.add_argument("--out", default="runs/eval")
    return p.parse_args()


def resolve_images(images):
    if images:
        return images
    from ultralytics.data.utils import check_det_dataset

    val = check_det_dataset("VisDrone.yaml")["val"]
    return val[0] if isinstance(val, list) else val


def standard_predict(model, img_path, imgsz, conf):
    r = model.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)[0]
    b = r.boxes
    return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


def load_sahi_model(weights, conf, slice_size):
    import torch
    from sahi import AutoDetectionModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    last_err = None
    for model_type in ("ultralytics", "yolov8"):  # model type name differs between SAHI versions
        try:
            return AutoDetectionModel.from_pretrained(
                model_type=model_type, model_path=weights, confidence_threshold=conf, device=device, image_size=slice_size
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


def sliced_predict(sahi_model, img_path, slice_size, overlap):
    from sahi.predict import get_sliced_prediction

    res = get_sliced_prediction(
        str(img_path),
        sahi_model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        verbose=0,
    )
    ops = res.object_prediction_list
    if not ops:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0)
    boxes = np.array([op.bbox.to_xyxy() for op in ops], dtype=np.float32)
    cls = np.array([op.category.id for op in ops], dtype=int)
    conf = np.array([op.score.value for op in ops], dtype=np.float32)
    return boxes, cls, conf


def new_stats():
    return {"tp": 0, "fp": 0, "gt": [0] * len(SIZE_BINS), "hit": [0] * len(SIZE_BINS), "time": 0.0, "n": 0}


def update(stats, gt_boxes, gt_cls, pred, iou_thr, elapsed):
    boxes, cls, conf = pred
    gt_matched, pred_tp = match_detections(gt_boxes, gt_cls, boxes, cls, conf, iou_thr)
    stats["tp"] += int(pred_tp.sum())
    stats["fp"] += int((~pred_tp).sum())
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) if len(gt_boxes) else np.zeros(0)
    for k, (_, lo, hi) in enumerate(SIZE_BINS):
        in_bin = (areas >= lo) & (areas < hi)
        stats["gt"][k] += int(in_bin.sum())
        stats["hit"][k] += int((in_bin & gt_matched).sum())
    stats["time"] += elapsed
    stats["n"] += 1
    return gt_matched


def draw(img, gt_boxes, gt_matched, pred_boxes, title):
    vis = img.copy()
    for b in pred_boxes.astype(int):
        cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (255, 160, 0), 1)
    for b, m in zip(gt_boxes.astype(int), gt_matched):
        if not m:  # missed objects in red
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis


def report(results, out_dir):
    lines = ["| Method | Precision | " + " | ".join(f"Recall {n}" for n, _, _ in SIZE_BINS) + " | Time / image |",
             "|---|---|" + "---|" * len(SIZE_BINS) + "---|"]
    for name, s in results.items():
        prec = s["tp"] / max(s["tp"] + s["fp"], 1)
        recs = [f"{h / g:.3f}" if g else "-" for h, g in zip(s["hit"], s["gt"])]
        lines.append(f"| {name} | {prec:.3f} | " + " | ".join(recs) + f" | {s['time'] / max(s['n'], 1):.3f} s |")
    counts = next(iter(results.values()))["gt"]
    lines.append("")
    lines.append("Ground-truth object counts: " + ", ".join(f"{n}: {c}" for (n, _, _), c in zip(SIZE_BINS, counts)))
    text = "\n".join(lines)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "small_object_report.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nReport saved: {os.path.join(out_dir, 'small_object_report.md')}")


def main():
    a = parse_args()
    img_dir = resolve_images(a.images)
    images = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
    if a.limit:
        images = images[: a.limit]
    print(f"Evaluating {len(images)} images from {img_dir}")

    model = YOLO(a.weights)
    sahi_model = None if a.no_sahi else load_sahi_model(a.weights, a.conf, a.slice)
    results = {"Standard": new_stats()}
    if sahi_model is not None:
        results["Sliced (SAHI)"] = new_stats()
    vis_dir = os.path.join(a.out, "vis")

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        gt_cls, gt_boxes = load_yolo_labels(label_path_for(img_path), w, h)
        if a.coco_zero_shot:
            gt_cls, gt_boxes = to_groups(gt_cls, gt_boxes, VISDRONE_TO_GROUP)

        t0 = time.perf_counter()
        std = standard_predict(model, img_path, a.imgsz, a.conf)
        if a.coco_zero_shot:
            std = to_groups(std[1], std[0], COCO_TO_GROUP, std[2])
        m_std = update(results["Standard"], gt_boxes, gt_cls, std, a.iou, time.perf_counter() - t0)

        if sahi_model is not None:
            t0 = time.perf_counter()
            sl = sliced_predict(sahi_model, img_path, a.slice, a.overlap)
            if a.coco_zero_shot:
                sl = to_groups(sl[1], sl[0], COCO_TO_GROUP, sl[2])
            m_sl = update(results["Sliced (SAHI)"], gt_boxes, gt_cls, sl, a.iou, time.perf_counter() - t0)

        if idx < a.save_vis:
            os.makedirs(vis_dir, exist_ok=True)
            panels = [draw(img, gt_boxes, m_std, std[0], "Standard")]
            if sahi_model is not None:
                panels.append(draw(img, gt_boxes, m_sl, sl[0], "Sliced (SAHI)"))
            cv2.imwrite(os.path.join(vis_dir, os.path.basename(img_path)), np.hstack(panels))

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(images)}")

    report(results, a.out)


if __name__ == "__main__":
    main()

"""Fine-tune a YOLO detector on the VisDrone dataset.

Ultralytics downloads VisDrone and converts it to YOLO format on the first run via VisDrone.yaml.

Example:
    python src/train.py --model yolo11s.pt --epochs 50 --imgsz 1024 --batch 8 --name yolo11s_1024
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO on VisDrone")
    p.add_argument("--model", default="yolo11s.pt", help="Initial weights (COCO-pretrained)")
    p.add_argument("--data", default="VisDrone.yaml", help="Dataset config")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1024, help="High input resolution for small objects")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--name", default="yolo11s_1024", help="Experiment name (runs/visdrone/<name>)")
    p.add_argument("--device", default=None, help="e.g. 0 or cpu; auto-selected if omitted")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    a = parse_args()
    model = YOLO(a.model)
    model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project="runs/visdrone",
        name=a.name,
        exist_ok=True,
        seed=a.seed,
        patience=20,
        close_mosaic=10,  # disable mosaic for the last epochs to adapt to real object scales
        plots=True,
    )
    metrics = model.val(data=a.data, imgsz=a.imgsz, split="val", project="runs/visdrone", name=f"{a.name}_val", exist_ok=True)
    print("\n=== Validation results ===")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Weights      : runs/visdrone/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()

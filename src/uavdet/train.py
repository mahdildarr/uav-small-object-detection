"""Fine-tune a YOLO detector on VisDrone from a YAML experiment config.

Every key under ``train:`` is passed to ``ultralytics.YOLO.train``; command-line ``--set key=value``
pairs override it. The resolved config and the validation metrics are saved next to the weights.

Example::

    python -m uavdet.train --config configs/visdrone_yolo11s_1024.yaml
    python -m uavdet.train --config configs/visdrone_yolo11s_1024.yaml --set epochs=10 imgsz=640 name=quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def parse_value(v: str):
    """YAML scalar, with "1e-4"-style floats (strings under YAML 1.1) converted to float."""
    x = yaml.safe_load(v)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            pass
    return x


def load_config(path, overrides=()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg["train"][k] = parse_value(v)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune YOLO on VisDrone")
    p.add_argument("--config", default="configs/visdrone_yolo11s_1024.yaml")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="Override train arguments")
    a = p.parse_args(argv)

    from ultralytics import YOLO

    cfg = load_config(a.config, a.set)
    t = dict(cfg["train"])
    model = YOLO(t.pop("model"))
    model.train(**t)
    run_dir = Path(model.trainer.save_dir)
    m = model.val(data=t["data"], imgsz=t.get("imgsz", 640), split="val", project=str(run_dir), name="val",
                  exist_ok=True)
    metrics = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
               "per_class_AP50-95": {m.names[i]: float(ap) for i, ap in zip(m.box.ap_class_index, m.box.maps)}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nmAP@0.5 {metrics['mAP50']:.4f} | mAP@0.5:0.95 {metrics['mAP50-95']:.4f}")
    print(f"Weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

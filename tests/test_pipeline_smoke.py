"""End-to-end run on synthetic data with injected detectors (no weights or GPU needed)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from uavdet.boxes import Detections, label_path_for, load_yolo_labels
from uavdet.evaluate import evaluate, parse_args

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_synthetic_visdrone.py"), "--out", str(out)], check=True)
    return out


def oracle(min_side=0):
    """Detector that returns the ground truth, except objects smaller than ``min_side`` pixels."""

    def detect(img_path):
        cls, boxes = load_yolo_labels(label_path_for(img_path), 640, 480)
        keep = (boxes[:, 2] - boxes[:, 0]) >= min_side
        return Detections(boxes[keep], cls[keep], np.full(int(keep.sum()), 0.9, np.float32))

    return detect


def test_evaluate_end_to_end(data, tmp_path):
    args = parse_args(["--weights", "unused", "--images", str(data / "images" / "val"), "--limit", "0", "--ap",
                       "--bootstrap", "100", "--save-vis", "2", "--annotations", str(data / "annotations"),
                       "--out", str(tmp_path)])
    res = evaluate(args, detectors={"Standard": oracle(min_side=32), "Sliced (SAHI)": oracle()})
    s = res["summary"]
    assert s["Sliced (SAHI)"]["recall"] == pytest.approx(1.0)
    assert s["Standard"]["recall_small"] == 0 and s["Standard"]["recall_large"] == 1
    assert res["ap"]["Sliced (SAHI)"]["AP50"] > res["ap"]["Standard"]["AP50"]
    d, lo, hi = res["bootstrap_ci"]["recall_small"]
    assert d == pytest.approx(1.0) and lo == pytest.approx(1.0)
    for f in ("report.md", "metrics.json", "per_image.csv", "recall_by_size.png"):
        assert (tmp_path / f).exists()
    assert len(list((tmp_path / "vis").glob("*.jpg"))) == 2
    json.loads((tmp_path / "metrics.json").read_text())


def test_eval_mot_cli(data, tmp_path):
    ann = data / "mot" / "annotations"
    res = tmp_path / "res"
    res.mkdir()
    rows = [line.split(",") for line in (ann / "uav_synthetic_v.txt").read_text().splitlines()]
    res.joinpath("uav_synthetic_v.txt").write_text(
        "\n".join(",".join(r[:6] + ["0.9", r[7], "-1", "-1"]) for r in rows if r[7] != "0"))
    subprocess.run([sys.executable, "-m", "uavdet.eval_mot", "--results", str(res), "--annotations", str(ann)],
                   check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert (res / "metrics.csv").exists()

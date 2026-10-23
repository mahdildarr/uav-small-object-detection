"""Create a tiny synthetic dataset in VisDrone layout to exercise the pipeline without real data.

Writes ``images/val`` + YOLO ``labels/val``, raw VisDrone-DET ``annotations`` (with one ignored region
per image) and one VisDrone-MOT sequence of moving boxes with ground truth.

    python tools/make_synthetic_visdrone.py --out data/synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def objects_for(rng, w, h, n):
    """Random boxes of mixed sizes (mostly small, like VisDrone)."""
    sizes = rng.choice([12, 24, 48, 120], size=n, p=[0.5, 0.3, 0.15, 0.05])
    out = []
    for s in sizes:
        x, y = rng.integers(0, w - s), rng.integers(0, h - s)
        out.append((int(rng.integers(0, 10)), int(x), int(y), int(s), int(s)))
    return out


def draw(img, objs):
    for c, x, y, bw, bh in objs:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (40 + 20 * c, 220 - 15 * c, 120), -1)


def make_det(out: Path, n_images: int, w: int, h: int, rng):
    for d in ("images/val", "labels/val", "annotations"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        img = np.full((h, w, 3), 90, np.uint8)
        objs = objects_for(rng, w, h, 25)
        draw(img, objs)
        name = f"img{i:04d}"
        cv2.imwrite(str(out / "images/val" / f"{name}.jpg"), img)
        yolo = [f"{c} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}"
                for c, x, y, bw, bh in objs]
        (out / "labels/val" / f"{name}.txt").write_text("\n".join(yolo))
        raw = [f"{x},{y},{bw},{bh},1,{c + 1},0,0" for c, x, y, bw, bh in objs]
        raw.append(f"0,0,{w // 8},{h // 8},0,0,0,0")  # ignored region in the top-left corner
        (out / "annotations" / f"{name}.txt").write_text("\n".join(raw))


def make_mot(out: Path, n_frames: int, w: int, h: int, rng):
    seq = "uav_synthetic_v"
    frames = out / "mot" / "sequences" / seq
    frames.mkdir(parents=True, exist_ok=True)
    (out / "mot" / "annotations").mkdir(parents=True, exist_ok=True)
    tracks = [(tid, int(rng.integers(20, w - 100)), int(rng.integers(20, h - 100)),
               int(rng.integers(-3, 4)), int(rng.integers(-3, 4))) for tid in range(1, 6)]
    lines = []
    for f in range(1, n_frames + 1):
        img = np.full((h, w, 3), 90, np.uint8)
        for tid, x0, y0, vx, vy in tracks:
            x, y = x0 + vx * f, y0 + vy * f
            cv2.rectangle(img, (x, y), (x + 40, y + 40), (200, 60, 60), -1)
            lines.append(f"{f},{tid},{x},{y},40,40,1,4,0,0")
        lines.append(f"{f},-1,0,0,60,60,0,0,0,0")  # ignored region
        cv2.imwrite(str(frames / f"{f:07d}.jpg"), img)
    (out / "mot" / "annotations" / f"{seq}.txt").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--images", type=int, default=6)
    p.add_argument("--frames", type=int, default=20)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    make_det(Path(a.out), a.images, a.width, a.height, rng)
    make_mot(Path(a.out), a.frames, a.width, a.height, rng)
    print(f"Synthetic VisDrone-style data -> {a.out}")


if __name__ == "__main__":
    main()

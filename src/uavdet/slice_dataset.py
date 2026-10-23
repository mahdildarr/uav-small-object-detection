"""Cut a YOLO-format dataset into overlapping tiles for slicing-aided fine-tuning.

Boxes are clipped to each tile; a box is kept if at least ``--min-visible`` of its area falls inside
the tile (cut-off fragments would teach the detector to fire on object parts). Tiles without objects
are kept with probability ``--keep-empty`` so the model still sees background. Validation images are
copied unsliced, so validation keeps measuring full-image performance.

Example::

    python -m uavdet.slice_dataset --data VisDrone.yaml --out datasets/VisDrone-sliced --slice 640 --overlap 0.2
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from .boxes import box_area, label_path_for, load_yolo_labels


def tile_grid(width: int, height: int, size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Tiles (x1, y1, x2, y2) covering the image; the last row / column is aligned to the border."""
    step = max(int(size * (1 - overlap)), 1)

    def starts(n):
        if n <= size:
            return [0]
        s = list(range(0, n - size, step))
        return s + [n - size]

    return [(x, y, min(x + size, width), min(y + size, height)) for y in starts(height) for x in starts(width)]


def clip_to_tile(boxes, cls, tile, min_visible: float = 0.5):
    """Boxes clipped to a tile, in tile coordinates, with their classes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int)
    x1, y1, x2, y2 = tile
    c = np.stack([np.clip(boxes[:, 0], x1, x2), np.clip(boxes[:, 1], y1, y2),
                  np.clip(boxes[:, 2], x1, x2), np.clip(boxes[:, 3], y1, y2)], axis=1)
    keep = box_area(c) >= min_visible * np.maximum(box_area(boxes), 1e-9)
    keep &= box_area(c) > 0
    return c[keep] - np.array([x1, y1, x1, y1], np.float32), np.asarray(cls)[keep]


def to_yolo_lines(boxes, cls, w: int, h: int) -> list[str]:
    out = []
    for (x1, y1, x2, y2), c in zip(boxes, cls):
        out.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return out


def slice_split(img_dir: Path, out: Path, split: str, size: int, overlap: float, min_visible: float,
                keep_empty: float, rng: random.Random) -> int:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png")]):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        cls, boxes = load_yolo_labels(label_path_for(img_path), w, h)
        for tile in tile_grid(w, h, size, overlap):
            tb, tc = clip_to_tile(boxes, cls, tile, min_visible)
            if len(tb) == 0 and rng.random() >= keep_empty:
                continue
            x1, y1, x2, y2 = tile
            name = f"{img_path.stem}_{x1}_{y1}"
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img[y1:y2, x1:x2])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(to_yolo_lines(tb, tc, x2 - x1, y2 - y1)))
            n += 1
    return n


def copy_split(img_dir: Path, out: Path, split: str) -> int:
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted([*img_dir.glob("*.jpg"), *img_dir.glob("*.png")]):
        shutil.copy2(img_path, out / "images" / split / img_path.name)
        lab = Path(label_path_for(img_path))
        if lab.exists():
            shutil.copy2(lab, out / "labels" / split / lab.name)
        n += 1
    return n


def main(argv=None):
    p = argparse.ArgumentParser(description="Tile a YOLO dataset for slicing-aided fine-tuning")
    p.add_argument("--data", default="VisDrone.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--slice", type=int, default=640)
    p.add_argument("--overlap", type=float, default=0.2)
    p.add_argument("--min-visible", type=float, default=0.5)
    p.add_argument("--keep-empty", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from ultralytics.data.utils import check_det_dataset

    d = check_det_dataset(a.data)
    out = Path(a.out).resolve()
    rng = random.Random(a.seed)
    first = lambda v: Path(v[0] if isinstance(v, list) else v)  # noqa: E731
    n_train = slice_split(first(d["train"]), out, "train", a.slice, a.overlap, a.min_visible, a.keep_empty, rng)
    n_val = copy_split(first(d["val"]), out, "val")
    names = d["names"] if isinstance(d["names"], dict) else dict(enumerate(d["names"]))
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val",
                                                   "names": names}, sort_keys=False))
    print(f"{n_train} training tiles, {n_val} validation images -> {out / 'data.yaml'}")


if __name__ == "__main__":
    main()

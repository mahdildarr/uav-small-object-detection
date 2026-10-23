"""Convert a folder of ordered frames (e.g. a VisDrone-MOT sequence) into an MP4 video.

Example::

    python -m uavdet.video --frames VisDrone2019-MOT-val/sequences/uav0000086_00000_v --out sequence.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def frames_to_video(frames_dir, out, fps: float = 25) -> int:
    frames = sorted([*Path(frames_dir).glob("*.jpg"), *Path(frames_dir).glob("*.png")])
    if not frames:
        raise SystemExit(f"No frames found: {frames_dir}")
    h, w = cv2.imread(str(frames[0])).shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(str(f)))
    writer.release()
    return len(frames)


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert a frame folder to video")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=25)
    a = p.parse_args(argv)
    n = frames_to_video(a.frames, a.out, a.fps)
    print(f"{n} frames -> {a.out} ({a.fps} fps)")


if __name__ == "__main__":
    main()

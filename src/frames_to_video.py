"""Convert a folder of ordered frames (e.g. a VisDrone-MOT sequence) into an MP4 video.

Example:
    python src/frames_to_video.py --frames VisDrone2019-MOT-val/sequences/uav0000086_00000_v --out sequence.mp4
"""

import argparse
import os
from glob import glob

import cv2


def main():
    p = argparse.ArgumentParser(description="Convert a frame folder to video")
    p.add_argument("--frames", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=25)
    a = p.parse_args()

    frames = sorted(glob(os.path.join(a.frames, "*.jpg")) + glob(os.path.join(a.frames, "*.png")))
    if not frames:
        raise SystemExit(f"No frames found: {a.frames}")
    h, w = cv2.imread(frames[0]).shape[:2]
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    for f in frames:
        writer.write(cv2.imread(f))
    writer.release()
    print(f"{len(frames)} frames -> {a.out} ({w}x{h}, {a.fps} fps)")


if __name__ == "__main__":
    main()

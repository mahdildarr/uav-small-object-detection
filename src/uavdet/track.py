"""Multi-object tracking with a trained detector and ByteTrack (or BoT-SORT).

The source can be a video file, a folder of frames (one VisDrone-MOT sequence) or, with
``--sequences``, a folder that contains several sequence folders. For every sequence the tracks are
written in MOT format (frame, id, x, y, w, h, conf, category, -1, -1) so that they can be scored with
``uavdet.eval_mot``; the tracker state is reset between sequences.

Example::

    python -m uavdet.track --weights best.pt --sequences VisDrone2019-MOT-val/sequences --out runs/track/bytetrack
    python -m uavdet.track --weights best.pt --source clip.mp4 --save-video
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

IMAGE_EXT = (".jpg", ".jpeg", ".png")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Multi-object tracking")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="Video file or a folder of frames")
    src.add_argument("--sequences", help="Folder containing one sub-folder of frames per sequence")
    p.add_argument("--weights", required=True)
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml (or a custom YAML)")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.1,
                   help="Detector threshold; ByteTrack uses low-score boxes in its second association step")
    p.add_argument("--device", default=None)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--out", default="runs/track/results")
    return p.parse_args(argv)


def frames_of(source: Path):
    if source.is_dir():
        return sorted(str(f) for f in source.iterdir() if f.suffix.lower() in IMAGE_EXT)
    return str(source)


def track_sequence(weights, source: Path, out_txt: Path, args) -> dict:
    """Track one sequence and write MOT-format results. Returns simple statistics."""
    from ultralytics import YOLO

    model = YOLO(weights)  # a fresh model per sequence also gives a fresh tracker state
    frames = frames_of(source)
    stream = model.track(source=frames, tracker=args.tracker, imgsz=args.imgsz, conf=args.conf, device=args.device,
                         persist=True, stream=True, save=args.save_video, project=str(out_txt.parent),
                         name=source.stem, exist_ok=True, verbose=False)
    lines, ids, n, t0 = [], set(), 0, time.perf_counter()
    for n, r in enumerate(stream, start=1):
        b = r.boxes
        if b is None or b.id is None:
            continue
        for tid, (x1, y1, x2, y2), conf, cls in zip(b.id.int().tolist(), b.xyxy.tolist(), b.conf.tolist(),
                                                    b.cls.int().tolist()):
            # category: YOLO index + 1 = VisDrone category
            lines.append(f"{n},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},{conf:.4f},{cls + 1},-1,-1")
            ids.add(tid)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
    dt = time.perf_counter() - t0
    return {"frames": n, "tracks": len(ids), "boxes": len(lines), "fps": n / max(dt, 1e-9)}


def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    if a.sequences:
        sources = sorted(d for d in Path(a.sequences).iterdir() if d.is_dir())
    else:
        sources = [Path(a.source)]
    for s in sources:
        st = track_sequence(a.weights, s, out / f"{s.stem}.txt", a)
        print(f"{s.stem}: {st['frames']} frames, {st['tracks']} tracks, {st['boxes'] / max(st['frames'], 1):.1f} "
              f"boxes/frame, {st['fps']:.1f} FPS")
    print(f"MOT-format results: {out}/")


if __name__ == "__main__":
    main()

"""Multi-object tracking on video with a trained detector + ByteTrack.

The source can be a video file. For frame folders such as VisDrone-MOT sequences, create a video first
with frames_to_video.py.

Example:
    python src/track.py --weights runs/visdrone/yolo11s_1024/weights/best.pt --source sample.mp4
"""

import argparse

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Multi-object tracking with ByteTrack")
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="Video file")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--name", default="tracking")
    return p.parse_args()


def main():
    a = parse_args()
    model = YOLO(a.weights)
    stream = model.track(
        source=a.source,
        tracker=a.tracker,
        imgsz=a.imgsz,
        conf=a.conf,
        persist=True,
        stream=True,
        save=True,
        project="runs/track",
        name=a.name,
        exist_ok=True,
        verbose=False,
    )
    track_ids, frames, boxes = set(), 0, 0
    for r in stream:
        frames += 1
        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            track_ids.update(ids)
            boxes += len(ids)
    print(f"Frames processed     : {frames}")
    print(f"Unique track IDs     : {len(track_ids)}")
    print(f"Mean boxes per frame : {boxes / max(frames, 1):.1f}")
    print(f"Output video         : runs/track/{a.name}/")


if __name__ == "__main__":
    main()

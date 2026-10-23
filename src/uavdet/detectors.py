"""Standard (full-image) and sliced (SAHI) inference behind one interface: ``detector(img_path) -> Detections``."""
from __future__ import annotations

import numpy as np

from .boxes import Detections


class StandardDetector:
    name = "Standard"

    def __init__(self, weights: str, imgsz: int = 1024, conf: float = 0.25, device=None):
        from ultralytics import YOLO

        self.model, self.imgsz, self.conf, self.device = YOLO(weights), imgsz, conf, device

    def __call__(self, img_path) -> Detections:
        r = self.model.predict(str(img_path), imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        b = r[0].boxes
        return Detections(b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())


class SlicedDetector:
    """SAHI sliced inference with an additional full-image pass (``full_image=True``), as recommended
    by Akyon et al. so that large objects cut by tile borders are still found."""

    name = "Sliced (SAHI)"

    def __init__(self, weights: str, slice_size: int = 640, overlap: float = 0.2, conf: float = 0.25,
                 full_image: bool = True, postprocess: str = "GREEDYNMM", device=None):
        import torch
        from sahi import AutoDetectionModel

        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        last_err = None
        for model_type in ("ultralytics", "yolov8"):  # the model type name differs between SAHI versions
            try:
                self.model = AutoDetectionModel.from_pretrained(
                    model_type=model_type, model_path=weights, confidence_threshold=conf, device=device,
                    image_size=slice_size)
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
        else:
            raise last_err
        self.slice_size, self.overlap, self.full_image = slice_size, overlap, full_image
        # SAHI's default merges overlapping tile predictions (greedy NMM, IoS 0.5); with low score
        # thresholds (AP evaluation) plain NMS is used, as SAHI itself recommends.
        self.postprocess = postprocess
        self.match_metric = "IOU" if postprocess == "NMS" else "IOS"

    def __call__(self, img_path) -> Detections:
        from sahi.predict import get_sliced_prediction

        res = get_sliced_prediction(
            str(img_path), self.model, slice_height=self.slice_size, slice_width=self.slice_size,
            overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
            perform_standard_pred=self.full_image, postprocess_type=self.postprocess,
            postprocess_match_metric=self.match_metric, force_postprocess_type=True, verbose=0)
        ops = res.object_prediction_list
        if not ops:
            return Detections.empty()
        return Detections(np.array([op.bbox.to_xyxy() for op in ops], dtype=np.float32),
                          np.array([op.category.id for op in ops], dtype=int),
                          np.array([op.score.value for op in ops], dtype=np.float32))

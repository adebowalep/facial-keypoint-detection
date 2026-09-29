"""Haar cascade face detection."""

from typing import List, Tuple

import cv2
import numpy as np

from .config import HAAR_CASCADE_PATH

BoundingBox = Tuple[int, int, int, int]  # (x, y, w, h)

_face_cascade: cv2.CascadeClassifier | None = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(str(HAAR_CASCADE_PATH))
    return _face_cascade


def detect_faces(
    image_rgb: np.ndarray,
    scale_factor: float = 1.2,
    min_neighbors: int = 2,
) -> List[BoundingBox]:
    """Detect faces in an RGB image, returning ``(x, y, w, h)`` boxes."""
    cascade = _get_cascade()
    detections = cascade.detectMultiScale(image_rgb, scale_factor, min_neighbors)
    return [tuple(int(v) for v in box) for box in detections]

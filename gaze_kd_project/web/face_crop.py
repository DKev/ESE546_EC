"""
Face detection + crop using OpenCV's bundled Haar cascade (no extra model download).

Used before gaze inference so the network sees a face-centered crop similar to training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

_Cascade: Optional[cv2.CascadeClassifier] = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _Cascade
    if _Cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _Cascade = cv2.CascadeClassifier(path)
        if _Cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return _Cascade


def crop_largest_face(
    pil_img: Image.Image,
    expand: float = 1.35,
    min_size: int = 48,
) -> tuple[Image.Image, bool, Optional[Tuple[int, int, int, int]]]:
    """
    Detect the largest frontal face, expand the box, crop to RGB image.

    Returns:
        cropped_rgb: PIL image fed to the gaze model (full frame if no face).
        found: whether any face was detected.
        bbox_xywh: ``(x, y, w, h)`` of the **clamped expanded** crop in the
            original image (for drawing overlays), or ``None`` if no face.
    """
    rgb = np.array(pil_img.convert("RGB"))
    h0, w0 = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    cascade = _get_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(min_size, min_size),
    )
    if faces is None or len(faces) == 0:
        return pil_img.convert("RGB"), False, None

    x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
    cx = x + w / 2.0
    cy = y + h / 2.0
    w2 = int(w * expand)
    h2 = int(h * expand)
    x1 = int(round(cx - w2 / 2.0))
    y1 = int(round(cy - h2 / 2.0))
    x2 = x1 + w2
    y2 = y1 + h2

    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w0, x2)
    y2c = min(h0, y2)

    if x2c <= x1c or y2c <= y1c:
        return pil_img.convert("RGB"), False, None

    crop = rgb[y1c:y2c, x1c:x2c]
    bbox = (x1c, y1c, x2c - x1c, y2c - y1c)
    return Image.fromarray(crop), True, bbox

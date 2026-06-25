"""Detect and crop the largest face from a new image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class FaceDetection:
    """The selected face crop and its bounding box in the original image."""

    crop: Image.Image
    box: tuple[int, int, int, int]


class FaceDetector:
    """Use OpenCV's Haar cascade for a small educational deployment."""

    def __init__(self, cascade_path: str | Path | None = None) -> None:
        if cascade_path is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.cascade_path = Path(cascade_path)
        self.detector = cv2.CascadeClassifier(str(self.cascade_path))
        if self.detector.empty():
            raise RuntimeError(f"No se pudo cargar el detector facial: {self.cascade_path}")

    def detect_largest(self, image: Image.Image) -> FaceDetection | None:
        """Return the largest frontal face, or None when no face is found."""

        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )
        if len(faces) == 0:
            return None

        x, y, width, height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
        box = (int(x), int(y), int(x + width), int(y + height))
        return FaceDetection(crop=rgb_image.crop(box), box=box)

    @staticmethod
    def draw_box(image: Image.Image, detection: FaceDetection) -> Image.Image:
        """Create a copy of the image with the selected face highlighted."""

        annotated = image.convert("RGB").copy()
        drawer = ImageDraw.Draw(annotated)
        drawer.rectangle(detection.box, outline="red", width=3)
        return annotated

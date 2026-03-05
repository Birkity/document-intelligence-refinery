"""OCR backend abstraction — PaddleOCR primary, Tesseract fallback.

Each backend exposes a single function:

    run_ocr(image_bytes: bytes, dpi: int = 300) -> list[OcrBox]

where OcrBox carries the recognized text and its bounding box as
(x1, y1, x2, y2) in pixel coordinates.

The ``get_ocr_backend()`` factory probes available libraries and
returns the best one (PaddleOCR → Tesseract → None).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from io import BytesIO
from typing import Protocol

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data class
# ---------------------------------------------------------------------------


@dataclass
class OcrBox:
    """Single OCR detection box."""

    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class OcrBackend(Protocol):
    """Minimal protocol each OCR backend must satisfy."""

    name: str

    def run_ocr(self, image_bytes: bytes, dpi: int = 300) -> list[OcrBox]:
        ...


# ---------------------------------------------------------------------------
# PaddleOCR backend
# ---------------------------------------------------------------------------


class PaddleOcrBackend:
    """PaddleOCR English-mode backend.

    Install:  pip install paddleocr paddlepaddle
    """

    name = "paddleocr"

    def __init__(self) -> None:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]

        self._reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def run_ocr(self, image_bytes: bytes, dpi: int = 300) -> list[OcrBox]:
        import numpy as np
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        arr = np.array(img)
        result = self._reader.ocr(arr, cls=True)
        boxes: list[OcrBox] = []
        if not result or not result[0]:
            return boxes
        for line in result[0]:
            pts, (text, conf) = line
            x_coords = [p[0] for p in pts]
            y_coords = [p[1] for p in pts]
            boxes.append(
                OcrBox(
                    text=text,
                    x1=min(x_coords),
                    y1=min(y_coords),
                    x2=max(x_coords),
                    y2=max(y_coords),
                    confidence=float(conf),
                )
            )
        return boxes


# ---------------------------------------------------------------------------
# Tesseract backend
# ---------------------------------------------------------------------------


class TesseractBackend:
    """pytesseract backend.

    Install:  pip install pytesseract
    Also requires Tesseract-OCR binary on PATH.
    """

    name = "tesseract"

    def __init__(self) -> None:
        import pytesseract  # type: ignore[import-untyped]
        self._tess = pytesseract

    def run_ocr(self, image_bytes: bytes, dpi: int = 300) -> list[OcrBox]:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        data = self._tess.image_to_data(img, output_type=self._tess.Output.DICT)
        boxes: list[OcrBox] = []
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i])
            if txt and conf > 0:
                x = data["left"][i]
                y = data["top"][i]
                w = data["width"][i]
                h = data["height"][i]
                boxes.append(
                    OcrBox(
                        text=txt,
                        x1=float(x),
                        y1=float(y),
                        x2=float(x + w),
                        y2=float(y + h),
                        confidence=conf / 100.0,
                    )
                )
        return boxes


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_ocr_backend() -> OcrBackend | None:
    """Return the best available OCR backend, or *None* if nothing is installed."""
    for cls in (PaddleOcrBackend, TesseractBackend):
        try:
            backend = cls()
            log.info("Using OCR backend: %s", backend.name)
            return backend
        except ImportError:
            continue
    log.warning("No OCR backend available (install paddleocr or pytesseract)")
    return None

"""OCR backend abstraction.

Provides ``OcrBox`` dataclass for OCR bounding-box results and
``get_ocr_backend()`` factory.  The real OCR logic lives in
``src.strategies.ocr.OCRExtractor``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__: list[str] = ["get_ocr_backend", "OcrBox"]


@dataclass
class OcrBox:
    """Single OCR detection result with text and bounding box."""

    text: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    page_number: int = 1


def get_ocr_backend() -> None:
    """Return *None*: use ``OCRExtractor`` from ``src.strategies.ocr`` instead."""
    return None


"""Extraction strategies for the Document Intelligence Refinery.

All strategies conform to the BaseExtractor interface and emit
ExtractedDocument instances so downstream stages are strategy-agnostic.

Budget ladder:
  A (Cheap)   → FastTextExtractor (pdfplumber)
  B (Medium)  → LayoutExtractor (Docling)
  C (Hard)    → OCRExtractor (RapidOCR)
  D (Extreme) → VisionExtractor (OpenRouter Vision LLM)
"""

from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.ocr import OCRExtractor
from src.strategies.vision import VisionExtractor

__all__ = [
    "BaseExtractor",
    "FastTextExtractor",
    "LayoutExtractor",
    "OCRExtractor",
    "VisionExtractor",
]

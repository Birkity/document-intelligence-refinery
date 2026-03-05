"""VisionExtractor — Strategy C (high cost).

Renders PDF pages to images via PyMuPDF and runs local OCR
(PaddleOCR → Tesseract → pdfplumber fallback) to extract content
from scanned/image-heavy documents.

Install OCR deps:
    pip install "document-intelligence-refinery[ocr]"        # PaddleOCR
    pip install "document-intelligence-refinery[tesseract]"  # Tesseract
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from src.models.schemas import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
    TextBlock,
)
from src.strategies.base import BaseExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


def _render_page_to_png(pdf_path: str, page_idx: int, dpi: int = 300) -> bytes:
    """Render a single PDF page to a PNG byte buffer using PyMuPDF."""
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


class VisionExtractor(BaseExtractor):
    """Strategy C — vision/OCR extraction.

    Pipeline:
      1. Render each page to a 300-dpi PNG (PyMuPDF).
      2. Run OCR (PaddleOCR first, Tesseract second).
      3. Normalise OCR boxes → TextBlock with bounding boxes.
      4. Fall back to pdfplumber if OCR is unavailable.
    """

    strategy_name: str = "vision_augmented"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)
        esc = rules.get("escalation", {})
        self._max_depth: int = esc.get("max_escalation_depth", 3)

        # Try to get an OCR backend
        self._ocr_backend: Any = None
        try:
            from src.vision.ocr_backends import get_ocr_backend
            self._ocr_backend = get_ocr_backend()
        except Exception:
            pass

        # Probe for PyMuPDF
        self._has_fitz = False
        try:
            import fitz  # noqa: F401
            self._has_fitz = True
        except ImportError:
            log.info("PyMuPDF (fitz) not installed — OCR not available")

    def extract(
        self,
        pdf_path: str,
        document_id: str,
        *,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract content from *pdf_path*.

        Parameters
        ----------
        page_numbers : list[int] | None
            1-indexed page numbers to extract.  ``None`` → all pages.
        """
        if self._has_fitz and self._ocr_backend is not None:
            return self._extract_ocr(pdf_path, document_id, page_numbers)
        return self._extract_fallback(pdf_path, document_id, page_numbers)

    # ------------------------------------------------------------------
    # Real OCR path
    # ------------------------------------------------------------------

    def _extract_ocr(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Render pages → OCR → normalise to ExtractedDocument."""
        import fitz

        doc = fitz.open(pdf_path)
        n_pages = doc.page_count
        doc.close()

        pages_out: list[ExtractedPage] = []
        total_conf = 0.0
        conf_count = 0
        total_chars = 0

        for page_idx in range(n_pages):
            pnum = page_idx + 1  # 1-indexed
            if page_numbers and pnum not in page_numbers:
                continue

            png_bytes = _render_page_to_png(pdf_path, page_idx, dpi=300)
            ocr_boxes = self._ocr_backend.run_ocr(png_bytes, dpi=300)

            # Get page dimensions for coordinate normalisation
            _doc = fitz.open(pdf_path)
            _page = _doc[page_idx]
            pw, ph = float(_page.rect.width), float(_page.rect.height)
            _doc.close()

            dpi = 300
            scale = 72.0 / dpi  # pixel → PDF points

            text_blocks: list[TextBlock] = []
            page_text = ""
            for box in ocr_boxes:
                page_text += box.text + " "
                total_conf += box.confidence
                conf_count += 1
                text_blocks.append(
                    TextBlock(
                        content=box.text,
                        bbox=BoundingBox(
                            x1=box.x1 * scale,
                            y1=box.y1 * scale,
                            x2=box.x2 * scale,
                            y2=box.y2 * scale,
                            page_number=pnum,
                        ),
                    )
                )
            total_chars += len(page_text)

            # Group into a single page-level block as well for chunking
            if page_text.strip() and not text_blocks:
                text_blocks.append(TextBlock(
                    content=page_text.strip(),
                    bbox=BoundingBox(x1=0, y1=0, x2=pw, y2=ph, page_number=pnum),
                ))

            pages_out.append(ExtractedPage(
                page_number=pnum,
                text_blocks=text_blocks,
            ))

        # Confidence: average OCR box confidence × text density factor
        avg_conf = (total_conf / conf_count) if conf_count else 0.0
        n_extracted = len(pages_out) or 1
        avg_chars = total_chars / n_extracted
        text_density_factor = min(avg_chars / 300, 1.0)
        self.confidence_score = round(
            min(avg_conf * 0.7 + text_density_factor * 0.3, 1.0), 4
        )

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

    # ------------------------------------------------------------------
    # pdfplumber fallback (always available)
    # ------------------------------------------------------------------

    def _extract_fallback(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Best-effort extraction from scanned PDFs using pdfplumber."""
        pages_out: list[ExtractedPage] = []
        total_chars = 0

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                if page_numbers and idx not in page_numbers:
                    continue

                pw, ph = float(page.width), float(page.height)
                text = page.extract_text() or ""
                total_chars += len(text)

                text_blocks: list[TextBlock] = []
                if text.strip():
                    text_blocks.append(
                        TextBlock(
                            content=text,
                            bbox=BoundingBox(
                                x1=0, y1=0, x2=pw, y2=ph, page_number=idx
                            ),
                        )
                    )

                pages_out.append(
                    ExtractedPage(
                        page_number=idx,
                        text_blocks=text_blocks,
                    )
                )

        # Conservative confidence — this is a fallback, not true OCR
        n_pages = len(pages_out) or 1
        avg_chars = total_chars / n_pages
        if avg_chars > 200:
            self.confidence_score = 0.55
        elif avg_chars > 50:
            self.confidence_score = 0.35
        else:
            self.confidence_score = 0.15

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

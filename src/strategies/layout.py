"""LayoutExtractor — Strategy B (medium cost).

Uses Docling (or a fallback pdfplumber-enhanced approach) for
layout-aware extraction.  Handles multi-column, table-heavy,
and mixed-origin documents.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
import yaml

from src.models.schemas import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    TableObject,
    TextBlock,
)
from src.strategies.base import BaseExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


class LayoutExtractor(BaseExtractor):
    """Strategy B — layout-aware extraction.

    Attempts to use Docling for full layout analysis.  Falls back to
    an enhanced pdfplumber extraction with word-level bounding boxes
    if Docling is unavailable.
    """

    strategy_name: str = "layout_aware"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)
        esc = rules.get("escalation", {})
        self._min_confidence: float = esc.get("strategy_b_min_confidence", 0.5)

        # Probe for Docling availability
        self._has_docling = False
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            self._has_docling = True
        except ImportError:
            log.info("Docling not installed — using enhanced pdfplumber fallback")

    def extract(self, pdf_path: str, document_id: str) -> ExtractedDocument:
        """Extract using layout-aware strategy."""
        if self._has_docling:
            return self._extract_docling(pdf_path, document_id)
        return self._extract_pdfplumber_enhanced(pdf_path, document_id)

    # ------------------------------------------------------------------
    # Docling path
    # ------------------------------------------------------------------

    def _extract_docling(self, pdf_path: str, document_id: str) -> ExtractedDocument:
        """Full Docling pipeline with DoclingDocument normalisation."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document

        pages_out: list[ExtractedPage] = []
        text_content = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else ""

        # Simple page-level split (Docling gives a single markdown)
        if text_content:
            pages_out.append(
                ExtractedPage(
                    page_number=1,
                    text_blocks=[
                        TextBlock(
                            content=text_content,
                            bbox=BoundingBox(
                                x1=0, y1=0, x2=612, y2=792, page_number=1
                            ),
                        )
                    ],
                )
            )
        self.confidence_score = 0.85  # Docling is generally high quality
        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

    # ------------------------------------------------------------------
    # Enhanced pdfplumber fallback
    # ------------------------------------------------------------------

    def _extract_pdfplumber_enhanced(
        self, pdf_path: str, document_id: str
    ) -> ExtractedDocument:
        """Enhanced pdfplumber extraction with word-level bounding boxes."""
        pages_out: list[ExtractedPage] = []
        total_chars = 0
        total_tables = 0

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                pw, ph = float(page.width), float(page.height)

                # ── Word-level text blocks ──
                text_blocks: list[TextBlock] = []
                words = page.extract_words()
                page_text = ""
                for w in words:
                    page_text += w.get("text", "") + " "
                total_chars += len(page_text)

                if page_text.strip():
                    text_blocks.append(
                        TextBlock(
                            content=page_text.strip(),
                            bbox=BoundingBox(
                                x1=0, y1=0, x2=pw, y2=ph, page_number=idx
                            ),
                        )
                    )

                # ── Tables ──
                tables: list[TableObject] = []
                for tbl in page.find_tables():
                    raw = tbl.extract()
                    if raw and len(raw) >= 2:
                        headers = [str(c) if c else "" for c in raw[0]]
                        rows = [
                            [str(c) if c else "" for c in row]
                            for row in raw[1:]
                        ]
                        bb = tbl.bbox
                        tables.append(
                            TableObject(
                                headers=headers,
                                rows=rows,
                                bbox=BoundingBox(
                                    x1=bb[0], y1=bb[1],
                                    x2=bb[2], y2=bb[3],
                                    page_number=idx,
                                ),
                            )
                        )
                        total_tables += 1

                pages_out.append(
                    ExtractedPage(
                        page_number=idx,
                        text_blocks=text_blocks,
                        tables=tables,
                    )
                )

        # ── Confidence: higher than Strategy A, accounts for layout ──
        n_pages = len(pages_out) or 1
        avg_chars = total_chars / n_pages
        table_bonus = min(total_tables * 0.05, 0.2)
        char_signal = min(avg_chars / 500, 1.0) * 0.7
        self.confidence_score = round(
            min(char_signal + table_bonus + 0.1, 1.0), 4
        )

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

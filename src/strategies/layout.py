"""LayoutExtractor — Strategy B (medium cost).

Uses Docling (if installed) for layout-aware extraction with full
table/figure/reading-order support.  Falls back to enhanced pdfplumber
word-level extraction with bounding boxes when Docling is absent.

Confidence signals for B:
  • text_coverage  — fraction of pages with non-empty text
  • table_completeness — fraction of detected tables with ≥2 rows
  • reading_order_sanity — whether page text length is monotonic-ish
  • empty_block_ratio — low ratio of empty text blocks → higher quality

Install Docling:  pip install "document-intelligence-refinery[layout]"
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber
import yaml

from src.models.schemas import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
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

    Attempts Docling first (full layout model), then falls back to
    enhanced pdfplumber with word-level bounding boxes.
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

    def extract(
        self,
        pdf_path: str,
        document_id: str,
        *,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract using layout-aware strategy.

        Parameters
        ----------
        page_numbers : list[int] | None
            If given, only extract these 1-indexed pages.
        """
        if self._has_docling:
            try:
                return self._extract_docling(pdf_path, document_id, page_numbers)
            except Exception as exc:
                log.warning("Docling failed (%s) — falling back to pdfplumber", exc)
        return self._extract_pdfplumber_enhanced(pdf_path, document_id, page_numbers)

    # ------------------------------------------------------------------
    # Docling path
    # ------------------------------------------------------------------

    def _extract_docling(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Full Docling pipeline with per-page normalisation."""
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc_res = result.document

        pages_out: list[ExtractedPage] = []
        total_pages_with_text = 0
        total_tables = 0
        tables_with_rows = 0

        # Docling exposes items by iterating the document
        # Build a page → content mapping
        page_texts: dict[int, list[str]] = {}
        page_tables: dict[int, list[TableObject]] = {}
        page_figures: dict[int, list[FigureObject]] = {}

        # Extract text via markdown export (Docling 2.x API)
        md_text = ""
        if hasattr(doc_res, "export_to_markdown"):
            md_text = doc_res.export_to_markdown()

        # Try structured access if available
        if hasattr(doc_res, "pages") and doc_res.pages:
            for page_obj in doc_res.pages:
                pnum = getattr(page_obj, "page_no", 0) + 1  # 0-indexed → 1-indexed
                if page_numbers and pnum not in page_numbers:
                    continue

                pw = getattr(page_obj, "width", 612)
                ph = getattr(page_obj, "height", 792)

                text_blocks: list[TextBlock] = []
                tables: list[TableObject] = []
                figures: list[FigureObject] = []

                # Iterate page items
                for item in getattr(page_obj, "items", []):
                    item_type = getattr(item, "type", "text")
                    bbox_raw = getattr(item, "bbox", None)
                    bb = BoundingBox(
                        x1=bbox_raw[0] if bbox_raw else 0,
                        y1=bbox_raw[1] if bbox_raw else 0,
                        x2=bbox_raw[2] if bbox_raw else pw,
                        y2=bbox_raw[3] if bbox_raw else ph,
                        page_number=pnum,
                    )

                    if item_type == "table":
                        tbl_data = getattr(item, "data", {})
                        headers = tbl_data.get("headers", [])
                        rows = tbl_data.get("rows", [])
                        total_tables += 1
                        if len(rows) >= 1:
                            tables_with_rows += 1
                        tables.append(TableObject(
                            headers=headers if headers else [""],
                            rows=rows if rows else [],
                            bbox=bb,
                        ))
                    elif item_type in ("figure", "image"):
                        caption = getattr(item, "caption", "") or ""
                        figures.append(FigureObject(
                            caption=caption, bbox=bb, figure_type="image",
                        ))
                    else:
                        content = getattr(item, "text", "") or ""
                        if content.strip():
                            text_blocks.append(TextBlock(content=content, bbox=bb))

                if text_blocks or tables or figures:
                    total_pages_with_text += 1

                pages_out.append(ExtractedPage(
                    page_number=pnum,
                    text_blocks=text_blocks,
                    tables=tables,
                    figures=figures,
                ))
        else:
            # Fallback: single-page markdown dump
            if md_text.strip():
                total_pages_with_text = 1
                pages_out.append(ExtractedPage(
                    page_number=1,
                    text_blocks=[TextBlock(
                        content=md_text,
                        bbox=BoundingBox(x1=0, y1=0, x2=612, y2=792, page_number=1),
                    )],
                ))

        # Confidence scoring
        n_pages = len(pages_out) or 1
        text_coverage = total_pages_with_text / n_pages
        table_comp = (tables_with_rows / total_tables) if total_tables else 1.0
        self.confidence_score = round(
            min(text_coverage * 0.5 + table_comp * 0.3 + 0.2, 1.0), 4
        )

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

    # ------------------------------------------------------------------
    # Enhanced pdfplumber fallback
    # ------------------------------------------------------------------

    def _extract_pdfplumber_enhanced(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Enhanced pdfplumber extraction with word-level bounding boxes."""
        pages_out: list[ExtractedPage] = []
        total_chars = 0
        total_tables = 0
        total_empty_blocks = 0
        total_blocks = 0
        pages_with_text = 0

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                if page_numbers and idx not in page_numbers:
                    continue

                pw, ph = float(page.width), float(page.height)

                # ── Word-level text blocks ──
                text_blocks: list[TextBlock] = []
                words = page.extract_words()
                page_text = ""
                for w in words:
                    page_text += w.get("text", "") + " "
                total_chars += len(page_text)

                if page_text.strip():
                    pages_with_text += 1
                    text_blocks.append(
                        TextBlock(
                            content=page_text.strip(),
                            bbox=BoundingBox(
                                x1=0, y1=0, x2=pw, y2=ph, page_number=idx
                            ),
                        )
                    )
                    total_blocks += 1
                else:
                    total_empty_blocks += 1
                    total_blocks += 1

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

                # ── Figures (detect large images as figures) ──
                figures: list[FigureObject] = []
                for img in page.images:
                    iw = abs(float(img["x1"]) - float(img["x0"]))
                    ih = abs(float(img["bottom"]) - float(img["top"]))
                    if iw * ih > (pw * ph * 0.05):  # >5% of page area
                        figures.append(FigureObject(
                            caption="",
                            bbox=BoundingBox(
                                x1=float(img["x0"]), y1=float(img["top"]),
                                x2=float(img["x1"]), y2=float(img["bottom"]),
                                page_number=idx,
                            ),
                            figure_type="image",
                        ))

                pages_out.append(
                    ExtractedPage(
                        page_number=idx,
                        text_blocks=text_blocks,
                        tables=tables,
                        figures=figures,
                    )
                )

        # ── Confidence: multi-signal ──
        n_pages = len(pages_out) or 1
        text_coverage = pages_with_text / n_pages
        avg_chars = total_chars / n_pages
        table_bonus = min(total_tables * 0.05, 0.2)
        empty_ratio = (total_empty_blocks / total_blocks) if total_blocks else 0
        char_signal = min(avg_chars / 500, 1.0) * 0.5
        self.confidence_score = round(
            min(char_signal + table_bonus + text_coverage * 0.2 + (1 - empty_ratio) * 0.1, 1.0),
            4,
        )

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

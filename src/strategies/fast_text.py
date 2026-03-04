"""FastTextExtractor — Strategy A (low cost, fast).

Uses pdfplumber to extract text and basic table structures from
native-digital PDFs.  Computes a multi-signal confidence score
to decide whether escalation is needed.
"""

from __future__ import annotations

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

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


class FastTextExtractor(BaseExtractor):
    """Strategy A — fast text extraction via pdfplumber.

    Best for: ``origin_type == native_digital`` AND
    ``layout_complexity == single_column``.

    Confidence scoring signals:
    - character count per page
    - character density (chars / page area)
    - image-to-page area ratio
    - font metadata presence
    """

    strategy_name: str = "fast_text"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)
        esc = rules.get("escalation", {})
        self._min_confidence: float = esc.get("strategy_a_min_confidence", 0.6)
        rev = rules.get("review", {})
        self._max_image_ratio: float = rev.get("max_image_ratio_fast_text", 0.5)

    def extract(self, pdf_path: str, document_id: str) -> ExtractedDocument:
        """Extract text blocks and tables from every page using pdfplumber."""
        pages_out: list[ExtractedPage] = []
        total_chars = 0
        total_image_area = 0.0
        total_page_area = 0.0

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                pw, ph = float(page.width), float(page.height)
                page_area = pw * ph
                total_page_area += page_area

                # ── Text blocks ──
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

                # ── Image area ──
                for img in page.images:
                    iw = float(img["x1"]) - float(img["x0"])
                    ih = float(img["bottom"]) - float(img["top"])
                    total_image_area += abs(iw * ih)

                pages_out.append(
                    ExtractedPage(
                        page_number=idx,
                        text_blocks=text_blocks,
                        tables=tables,
                    )
                )

        # ── Confidence scoring ──
        if total_page_area > 0:
            char_density = total_chars / total_page_area
            image_ratio = total_image_area / total_page_area
            # Normalise signals into [0, 1]
            density_signal = min(char_density / 0.02, 1.0)  # 0.02 = "good"
            image_penalty = max(0.0, 1.0 - image_ratio / self._max_image_ratio)
            self.confidence_score = round(
                (density_signal * 0.6 + image_penalty * 0.4), 4
            )
        else:
            self.confidence_score = 0.0

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
        )

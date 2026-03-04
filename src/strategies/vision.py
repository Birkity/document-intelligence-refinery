"""VisionExtractor — Strategy C (high cost).

Uses a Hugging Face vision model (or VLM API) to extract content
from scanned / image-based PDFs.  Includes budget guard logic.
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
    TextBlock,
)
from src.strategies.base import BaseExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


class VisionExtractor(BaseExtractor):
    """Strategy C — vision-augmented extraction.

    Targets scanned-image PDFs or documents where Strategy A / B
    confidence fell below threshold.  Uses page-image rendering
    and a vision model for OCR + structure recognition.

    The current implementation provides an OCR-simulation fallback
    using pdfplumber (suitable for mixed PDFs that still have an
    extractable text layer).  A full VLM integration can be swapped
    in via the ``_extract_with_vlm`` hook.
    """

    strategy_name: str = "vision_augmented"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)
        esc = rules.get("escalation", {})
        self._max_depth: int = esc.get("max_escalation_depth", 3)

        # Probe for optional vision dependencies
        self._has_vlm = False
        try:
            import torch          # noqa: F401
            from transformers import pipeline  # noqa: F401
            self._has_vlm = True
        except ImportError:
            log.info(
                "transformers/torch not installed — "
                "VisionExtractor will use pdfplumber OCR-sim fallback"
            )

    def extract(self, pdf_path: str, document_id: str) -> ExtractedDocument:
        """Extract content from *pdf_path* using vision strategy."""
        if self._has_vlm:
            return self._extract_with_vlm(pdf_path, document_id)
        return self._extract_fallback(pdf_path, document_id)

    # ------------------------------------------------------------------
    # VLM path (stubbed — ready for HF model integration)
    # ------------------------------------------------------------------

    def _extract_with_vlm(
        self, pdf_path: str, document_id: str
    ) -> ExtractedDocument:
        """Placeholder for full VLM extraction pipeline.

        In a production build this would:
        1. Render each page to an image.
        2. Pass images through a vision model (e.g. Pix2Struct,
           Nougat, or an API like GPT-4o-mini via OpenRouter).
        3. Parse structured output into ExtractedDocument.
        4. Apply budget guard per-document.

        Falls back to pdfplumber for the interim submission.
        """
        log.warning("VLM extraction not fully wired — using fallback")
        return self._extract_fallback(pdf_path, document_id)

    # ------------------------------------------------------------------
    # pdfplumber fallback (always available)
    # ------------------------------------------------------------------

    def _extract_fallback(
        self, pdf_path: str, document_id: str
    ) -> ExtractedDocument:
        """Best-effort extraction from scanned PDFs using pdfplumber.

        pdfplumber can still pull embedded text layers even from
        mixed-origin PDFs.  Confidence is set conservatively.
        """
        pages_out: list[ExtractedPage] = []
        total_chars = 0

        with pdfplumber.open(pdf_path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
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

        # Conservative confidence — this is a fallback, not true VLM
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

"""VisionExtractor — Strategy D (extreme cost, budget-guarded).

Uses OpenRouter Vision LLM (gemma-3-27b-it) to extract text from
pages that failed OCR or have handwriting/complex layouts.

This is the most expensive path and is budget-guarded:
- Only triggered when OCR confidence is very low
- Respects per-document vision call budget
- Feature-flagged via ENABLE_VISION_EXTRACTION

Budget ladder position:
  Cheap → pdfplumber (FastText)
  Medium → Docling (Layout)
  Hard → RapidOCR (OCR)
  Extreme → Vision LLM (this)
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

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


def _render_page_to_png(pdf_path: str, page_idx: int, dpi: int = 200) -> bytes:
    """Render a single PDF page to a PNG byte buffer using PyMuPDF."""
    import fitz

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
    """Strategy D — Vision LLM extraction via OpenRouter.

    Uses the vision model only when absolutely necessary (scanned pages
    with poor OCR, handwriting, repeated extraction failure).
    Budget-guarded: respects max_vision_calls_per_document.
    """

    strategy_name: str = "vision_augmented"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as fh:
                rules = yaml.safe_load(fh)
        else:
            rules = {}
        esc = rules.get("escalation", {})
        self._max_depth: int = esc.get("max_escalation_depth", 3)

        # Load config lazily to avoid import cycle
        self._vision_calls_made = 0

    def _get_config(self) -> Any:
        """Lazy-load config to avoid import cycle."""
        from src.config import get_settings
        return get_settings()

    def _call_vision_llm(self, image_bytes: bytes, prompt: str) -> str:
        """Call OpenRouter vision model with base64 image."""
        cfg = self._get_config()

        if not cfg.openrouter_api_key or cfg.openrouter_api_key.startswith("your-"):
            log.warning("OpenRouter API key not configured — returning empty")
            return ""

        if not cfg.enable_vision_extraction:
            log.info("Vision extraction disabled via feature flag")
            return ""

        if self._vision_calls_made >= cfg.budget_max_vision_calls:
            log.warning("Vision call budget exhausted (%d calls)", self._vision_calls_made)
            return ""

        try:
            import httpx

            img_b64 = base64.b64encode(image_bytes).decode("utf-8")

            response = httpx.post(
                f"{cfg.openrouter_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {cfg.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": cfg.vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{img_b64}"
                                    },
                                },
                            ],
                        }
                    ],
                    "max_tokens": 4096,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            self._vision_calls_made += 1

            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            log.error("Vision LLM call failed: %s", exc)
            return ""

    def extract(
        self,
        pdf_path: str,
        document_id: str,
        *,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract content using Vision LLM or fallback."""
        cfg = self._get_config()

        # If vision is enabled and API key is set, try vision path
        if cfg.enable_vision_extraction and cfg.openrouter_api_key and not cfg.openrouter_api_key.startswith("your-"):
            try:
                return self._extract_vision(pdf_path, document_id, page_numbers)
            except Exception as exc:
                log.warning("Vision extraction failed: %s — falling back", exc)

        # Fallback: PyMuPDF text + pdfplumber
        try:
            import fitz  # noqa: F401
            return self._extract_fitz_text(pdf_path, document_id, page_numbers)
        except ImportError:
            return self._extract_fallback(pdf_path, document_id, page_numbers)

    def _extract_vision(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract using Vision LLM — render pages and send to model."""
        import fitz

        pages_out: list[ExtractedPage] = []
        total_chars = 0

        prompt = (
            "Extract all text from this document page. "
            "Preserve the reading order, headings, paragraphs, and any table data. "
            "Return the text content only, no commentary."
        )

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                pnum = page_idx + 1
                if page_numbers and pnum not in page_numbers:
                    continue

                page = doc[page_idx]
                pw, ph = float(page.rect.width), float(page.rect.height)

                try:
                    png_bytes = _render_page_to_png(pdf_path, page_idx)
                    text = self._call_vision_llm(png_bytes, prompt)
                except Exception as exc:
                    log.warning("Vision extraction failed for page %d: %s", pnum, exc)
                    text = ""

                text_blocks: list[TextBlock] = []
                if text.strip():
                    total_chars += len(text)
                    text_blocks.append(
                        TextBlock(
                            content=text.strip(),
                            bbox=BoundingBox(
                                x1=0, y1=0, x2=pw, y2=ph, page_number=pnum
                            ),
                        )
                    )

                pages_out.append(
                    ExtractedPage(
                        page_number=pnum,
                        text_blocks=text_blocks,
                        extraction_strategy="vision_augmented",
                    )
                )
        finally:
            doc.close()

        n_pages = len(pages_out) or 1
        avg_chars = total_chars / n_pages
        if avg_chars > 200:
            self.confidence_score = 0.75
        elif avg_chars > 50:
            self.confidence_score = 0.50
        else:
            self.confidence_score = 0.20

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
            strategies_used=["vision_augmented"],
        )

    # ------------------------------------------------------------------
    # PyMuPDF text extraction (fallback when vision not available)
    # ------------------------------------------------------------------

    def _extract_fitz_text(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract embedded text via PyMuPDF."""
        import fitz

        pages_out: list[ExtractedPage] = []
        total_chars = 0

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                pnum = page_idx + 1
                if page_numbers and pnum not in page_numbers:
                    continue

                page = doc[page_idx]
                blocks = page.get_text("blocks")
                text_blocks: list[TextBlock] = []
                page_chars = 0

                for blk in blocks:
                    blk_text = blk[4].strip() if len(blk) > 4 else ""
                    if not blk_text:
                        continue
                    page_chars += len(blk_text)
                    text_blocks.append(
                        TextBlock(
                            content=blk_text,
                            bbox=BoundingBox(
                                x1=float(blk[0]),
                                y1=float(blk[1]),
                                x2=float(blk[2]),
                                y2=float(blk[3]),
                                page_number=pnum,
                            ),
                        )
                    )

                if not text_blocks:
                    log.warning(
                        "Page %d of '%s' has no embedded text.",
                        pnum, Path(pdf_path).name,
                    )

                total_chars += page_chars
                pages_out.append(
                    ExtractedPage(page_number=pnum, text_blocks=text_blocks)
                )
        finally:
            doc.close()

        n_pages = len(pages_out) or 1
        avg_chars = total_chars / n_pages
        if avg_chars > 200:
            self.confidence_score = 0.60
        elif avg_chars > 50:
            self.confidence_score = 0.40
        else:
            self.confidence_score = 0.15

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
            strategies_used=["vision_augmented_fallback"],
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
                    ExtractedPage(page_number=idx, text_blocks=text_blocks)
                )

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
            strategies_used=["vision_augmented_pdfplumber"],
        )

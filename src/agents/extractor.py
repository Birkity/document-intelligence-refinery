"""ExtractionRouter — confidence-gated strategy selection with escalation.

Implements the 4-tier extraction budget ladder:
  A (Cheap)   → FastTextExtractor (pdfplumber)
  B (Medium)  → LayoutExtractor (Docling)
  C (Hard)    → OCRExtractor (RapidOCR)
  D (Extreme) → VisionExtractor (OpenRouter Vision LLM)

Features:
- Confidence-gated escalation with configurable thresholds
- Page-level escalation (re-extract only failing pages at next tier)
- Budget guard (max cost per document)
- Exception-gated fallback
- Full ledger logging to .refinery/extraction_ledger.jsonl
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.config import get_settings
from src.models.schemas import (
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    LedgerEntry,
)
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.ocr import OCRExtractor
from src.strategies.vision import VisionExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)
_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / ".refinery" / "extraction_ledger.jsonl"
)

# Cost estimates per strategy (relative units)
_COST_MAP = {
    "fast_text": ("low", 0.0),
    "layout_aware": ("medium", 0.01),
    "ocr_heavy": ("high", 0.05),
    "vision_augmented": ("extreme", 0.10),
}


class ExtractionRouter:
    """Selects and executes the best extraction strategy for a document.

    Implements the A → B → C → D escalation chain with:
    - Confidence gating per strategy
    - Page-level escalation (only re-extract failing pages)
    - Budget guard (configurable max cost per document)
    - Exception-gated fallback
    - Full ledger logging

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.
    ledger_path : str | Path | None
        Path to the JSONL ledger file.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        ledger_path: str | Path | None = None,
    ) -> None:
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)

        cfg = get_settings()

        esc = rules.get("escalation", {})
        self._a_min: float = cfg.confidence_fast_text_min
        self._b_min: float = cfg.confidence_layout_min
        self._c_min: float = esc.get("strategy_c_min_confidence", 0.4)
        self._max_depth: int = esc.get("max_escalation_depth", 4)

        rev = rules.get("review", {})
        self._review_threshold: float = cfg.confidence_review_flag

        self._ledger_path = Path(ledger_path) if ledger_path else _LEDGER_PATH
        self._rules_path = rp

        # Budget configuration
        self._budget_max = cfg.budget_max_per_document
        self._enable_page_escalation = cfg.enable_page_level_escalation
        self._enable_vision = cfg.enable_vision_extraction

        # Instantiate strategies
        self._strategies: dict[str, BaseExtractor] = {
            "fast_text": FastTextExtractor(rules_path=rp),
            "layout_aware": LayoutExtractor(rules_path=rp),
            "ocr_heavy": OCRExtractor(rules_path=rp),
            "vision_augmented": VisionExtractor(rules_path=rp),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_and_extract(
        self, profile: DocumentProfile, pdf_path: str,
        *,
        page_numbers: list[int] | None = None,
    ) -> tuple[ExtractedDocument, list[dict]]:
        """Select strategy based on *profile*, extract, and escalate if needed.

        Supports page-level escalation: if a strategy produces low
        confidence on specific pages, only those pages are re-extracted
        at the next tier.

        Returns
        -------
        tuple[ExtractedDocument, list[dict]]
            Final extraction result and the full list of ledger entries.
        """
        chain = self._build_chain(profile)
        ledger_entries: list[dict] = []
        result: ExtractedDocument | None = None
        accumulated_cost = 0.0
        remaining_pages = page_numbers  # None = all pages

        for depth, strategy_name in enumerate(chain):
            if depth >= self._max_depth:
                break

            # Budget guard
            _, cost_per_page = _COST_MAP.get(strategy_name, ("unknown", 0.0))
            if accumulated_cost >= self._budget_max and strategy_name in ("ocr_heavy", "vision_augmented"):
                log.warning(
                    "Budget exhausted (%.2f >= %.2f) — skipping %s",
                    accumulated_cost, self._budget_max, strategy_name,
                )
                entry = self._make_ledger_entry(
                    profile, strategy_name, 0.0, 0.0,
                    failure_reason=f"budget_exhausted ({accumulated_cost:.2f})",
                    page_scope=remaining_pages,
                )
                ledger_entries.append(entry)
                self._append_ledger(entry)
                continue

            # Vision guard
            if strategy_name == "vision_augmented" and not self._enable_vision:
                log.info("Vision extraction disabled — skipping")
                continue

            extractor = self._strategies[strategy_name]
            t0 = time.perf_counter()

            try:
                new_result = extractor.extract(
                    pdf_path, profile.document_id,
                    page_numbers=remaining_pages,
                )
            except TypeError:
                new_result = extractor.extract(pdf_path, profile.document_id)
            except Exception as exc:
                elapsed = round(time.perf_counter() - t0, 4)
                log.error("Strategy %s failed: %s", strategy_name, exc)
                entry = self._make_ledger_entry(
                    profile, strategy_name, 0.0, elapsed,
                    failure_reason=str(exc),
                    page_scope=remaining_pages,
                    fallback_trigger="exception",
                )
                ledger_entries.append(entry)
                self._append_ledger(entry)
                continue

            elapsed = round(time.perf_counter() - t0, 4)
            confidence = extractor.confidence_score

            # Update cost estimate
            n_pages = len(new_result.pages) if new_result.pages else 1
            accumulated_cost += cost_per_page * n_pages

            # Merge results (page-level escalation)
            if result is None:
                result = new_result
            else:
                result = self._merge_results(result, new_result)

            # Determine if we need to escalate
            needs_escalation = False
            escalation_reason = None
            min_conf = self._get_min_confidence(strategy_name)

            if confidence < min_conf:
                needs_escalation = True
                escalation_reason = f"confidence {confidence:.4f} < {min_conf}"

            # Page-level escalation: find low-confidence pages
            low_conf_pages: list[int] | None = None
            if needs_escalation and self._enable_page_escalation and result:
                low_conf_pages = self._find_low_confidence_pages(result, min_conf)
                if low_conf_pages:
                    remaining_pages = low_conf_pages
                    escalation_reason = (
                        f"page-level escalation: {len(low_conf_pages)} pages "
                        f"below {min_conf} confidence"
                    )

            entry = self._make_ledger_entry(
                profile, strategy_name, confidence, elapsed,
                escalation_triggered=needs_escalation,
                escalation_reason=escalation_reason,
                page_scope=remaining_pages,
            )
            ledger_entries.append(entry)
            self._append_ledger(entry)

            if not needs_escalation:
                break

        if result is None:
            # Last resort: return empty document
            result = ExtractedDocument(
                document_id=profile.document_id,
                source_filename=profile.source_filename,
            )

        return result, ledger_entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_chain(self, profile: DocumentProfile) -> list[str]:
        """Determine starting strategy and full escalation chain."""
        cost = profile.estimated_extraction_cost
        origin = profile.origin_type
        layout = profile.layout_complexity

        # Scanned documents go straight to OCR
        if origin == "scanned_image":
            return ["ocr_heavy", "vision_augmented"]

        # Mixed documents start with layout
        if origin == "mixed":
            return ["layout_aware", "ocr_heavy", "vision_augmented"]

        # Layout-complex or table-heavy: use Docling
        if cost == "needs_layout_model" or layout in (
            "multi_column", "table_heavy", "figure_heavy", "mixed"
        ):
            return ["layout_aware", "ocr_heavy", "vision_augmented"]

        if cost == "needs_vision_model":
            return ["ocr_heavy", "vision_augmented"]

        # Default: cheap path first
        return ["fast_text", "layout_aware", "ocr_heavy", "vision_augmented"]

    def _get_min_confidence(self, strategy_name: str) -> float:
        """Get minimum confidence threshold for a strategy."""
        return {
            "fast_text": self._a_min,
            "layout_aware": self._b_min,
            "ocr_heavy": self._c_min,
            "vision_augmented": 0.0,  # no escalation from vision
        }.get(strategy_name, 0.5)

    def _find_low_confidence_pages(
        self, doc: ExtractedDocument, min_conf: float
    ) -> list[int]:
        """Find pages with low extraction quality."""
        low_pages: list[int] = []
        for page in doc.pages:
            # Heuristic: pages with very little text are likely low quality
            total_chars = sum(len(tb.content) for tb in page.text_blocks)
            if total_chars < 50:  # very sparse page
                low_pages.append(page.page_number)
            elif page.extraction_confidence > 0 and page.extraction_confidence < min_conf:
                low_pages.append(page.page_number)
        return low_pages

    def _merge_results(
        self, existing: ExtractedDocument, new: ExtractedDocument
    ) -> ExtractedDocument:
        """Merge new extraction results into existing, replacing pages."""
        existing_pages = {p.page_number: p for p in existing.pages}
        for new_page in new.pages:
            # Replace if new page has more content
            old_page = existing_pages.get(new_page.page_number)
            if old_page is None:
                existing_pages[new_page.page_number] = new_page
            else:
                old_chars = sum(len(tb.content) for tb in old_page.text_blocks)
                new_chars = sum(len(tb.content) for tb in new_page.text_blocks)
                if new_chars > old_chars:
                    existing_pages[new_page.page_number] = new_page

        merged_pages = [existing_pages[k] for k in sorted(existing_pages)]
        strategies = list(set(
            (existing.strategies_used or []) + (new.strategies_used or [])
        ))

        return ExtractedDocument(
            document_id=existing.document_id,
            source_filename=existing.source_filename,
            pages=merged_pages,
            strategies_used=strategies,
        )

    def _make_ledger_entry(
        self,
        profile: DocumentProfile,
        strategy_name: str,
        confidence: float,
        elapsed: float,
        *,
        failure_reason: str | None = None,
        escalation_triggered: bool = False,
        escalation_reason: str | None = None,
        page_scope: list[int] | None = None,
        fallback_trigger: str | None = None,
    ) -> dict:
        """Create a ledger entry dict."""
        cost_label, _ = _COST_MAP.get(strategy_name, ("unknown", 0.0))
        return {
            "document_id": profile.document_id,
            "source_filename": profile.source_filename,
            "strategy_used": strategy_name,
            "confidence_score": confidence,
            "cost_estimate": cost_label,
            "processing_time_s": elapsed,
            "fallback_trigger": fallback_trigger,
            "page_scope": page_scope,
            "failure_reason": failure_reason,
            "escalation_triggered": escalation_triggered,
            "escalation_reason": escalation_reason,
            "flagged_for_review": confidence < self._review_threshold,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _cost_label(strategy_name: str) -> str:
        label, _ = _COST_MAP.get(strategy_name, ("unknown", 0.0))
        return label

    def _append_ledger(self, entry: dict) -> None:
        """Append one JSON line to the extraction ledger."""
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

"""ExtractionRouter — confidence-gated strategy selection with escalation.

Reads a DocumentProfile and delegates extraction to the appropriate
strategy (A → B → C), escalating automatically when confidence falls
below the configured threshold.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Literal

import yaml

from src.models.schemas import DocumentProfile, ExtractedDocument
from src.strategies.base import BaseExtractor
from src.strategies.fast_text import FastTextExtractor
from src.strategies.layout import LayoutExtractor
from src.strategies.vision import VisionExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)
_LEDGER_PATH = (
    Path(__file__).resolve().parents[2] / ".refinery" / "extraction_ledger.jsonl"
)


class ExtractionRouter:
    """Selects and executes the best extraction strategy for a document.

    Implements the A → B → C escalation chain with confidence gating.
    Every extraction attempt is logged to ``.refinery/extraction_ledger.jsonl``.

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

        esc = rules.get("escalation", {})
        self._a_min: float = esc.get("strategy_a_min_confidence", 0.6)
        self._b_min: float = esc.get("strategy_b_min_confidence", 0.5)
        self._max_depth: int = esc.get("max_escalation_depth", 3)

        rev = rules.get("review", {})
        self._review_threshold: float = rev.get("flag_confidence_below", 0.4)

        self._ledger_path = Path(ledger_path) if ledger_path else _LEDGER_PATH
        self._rules_path = rp

        # Instantiate strategies
        self._strategies: dict[str, BaseExtractor] = {
            "fast_text": FastTextExtractor(rules_path=rp),
            "layout_aware": LayoutExtractor(rules_path=rp),
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

        Parameters
        ----------
        page_numbers : list[int] | None
            If given, only extract these 1-indexed pages (sample mode).

        Returns
        -------
        tuple[ExtractedDocument, list[dict]]
            Final extraction result and the full list of ledger entries
            (one per attempt, including escalation steps).
        """
        chain = self._build_chain(profile)
        ledger_entries: list[dict] = []
        result: ExtractedDocument | None = None

        for depth, strategy_name in enumerate(chain):
            if depth >= self._max_depth:
                break

            extractor = self._strategies[strategy_name]
            t0 = time.perf_counter()
            # Pass page_numbers if the strategy supports it
            try:
                result = extractor.extract(
                    pdf_path, profile.document_id,
                    page_numbers=page_numbers,
                )
            except TypeError:
                # Fallback for strategies that don't accept page_numbers
                result = extractor.extract(pdf_path, profile.document_id)
            elapsed = round(time.perf_counter() - t0, 4)

            confidence = extractor.confidence_score

            # Determine if we need to escalate
            needs_escalation = False
            if strategy_name == "fast_text" and confidence < self._a_min:
                needs_escalation = True
            elif strategy_name == "layout_aware" and confidence < self._b_min:
                needs_escalation = True

            entry = {
                "document_id": profile.document_id,
                "source_filename": profile.source_filename,
                "strategy_used": strategy_name,
                "confidence_score": confidence,
                "escalation_triggered": needs_escalation,
                "escalation_reason": (
                    f"confidence {confidence:.4f} < threshold"
                    if needs_escalation
                    else None
                ),
                "estimated_cost": self._cost_label(strategy_name),
                "processing_time_s": elapsed,
                "flagged_for_review": confidence < self._review_threshold,
            }
            ledger_entries.append(entry)
            self._append_ledger(entry)

            if not needs_escalation:
                break

        assert result is not None
        return result, ledger_entries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_chain(self, profile: DocumentProfile) -> list[str]:
        """Determine starting strategy and full escalation chain."""
        cost = profile.estimated_extraction_cost

        if cost == "fast_text_sufficient":
            return ["fast_text", "layout_aware", "vision_augmented"]
        elif cost == "needs_layout_model":
            return ["layout_aware", "vision_augmented"]
        else:  # needs_vision_model
            return ["vision_augmented"]

    @staticmethod
    def _cost_label(strategy_name: str) -> str:
        return {
            "fast_text": "low",
            "layout_aware": "medium",
            "vision_augmented": "high",
        }.get(strategy_name, "unknown")

    def _append_ledger(self, entry: dict) -> None:
        """Append one JSON line to the extraction ledger."""
        self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._ledger_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

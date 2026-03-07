"""Unit tests for ExtractionRouter escalation logic and confidence scoring.

Mocks extractors to control confidence values and verify correct
escalation chain behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.extractor import ExtractionRouter
from src.models.schemas import DocumentProfile, ExtractedDocument, ExtractedPage

_RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


def _make_profile(**overrides) -> DocumentProfile:
    """Return a valid DocumentProfile with sensible defaults."""
    defaults = dict(
        document_id="test-001",
        source_filename="test_document.pdf",
        origin_type="native_digital",
        layout_complexity="single_column",
        language="en",
        language_confidence=0.9,
        domain_hint="general",
        estimated_extraction_cost="fast_text_sufficient",
    )
    defaults.update(overrides)
    return DocumentProfile(**defaults)


def _stub_extracted_doc(document_id: str = "test-001") -> ExtractedDocument:
    """Minimal valid ExtractedDocument."""
    return ExtractedDocument(
        document_id=document_id,
        source_filename="test_document.pdf",
        pages=[ExtractedPage(page_number=1)],
    )


class TestExtractionRouterChainSelection:
    """Verify the router picks the correct starting strategy."""

    def test_fast_text_sufficient_starts_with_a(self) -> None:
        router = ExtractionRouter(rules_path=_RULES_PATH)
        profile = _make_profile(estimated_extraction_cost="fast_text_sufficient")
        chain = router._build_chain(profile)
        assert chain[0] == "fast_text"
        assert len(chain) == 4  # 4-tier: fast_text → layout → ocr → vision

    def test_needs_layout_starts_with_b(self) -> None:
        router = ExtractionRouter(rules_path=_RULES_PATH)
        profile = _make_profile(estimated_extraction_cost="needs_layout_model")
        chain = router._build_chain(profile)
        assert chain[0] == "layout_aware"
        assert len(chain) == 3  # 3-tier: layout → ocr → vision

    def test_needs_vision_starts_with_c(self) -> None:
        router = ExtractionRouter(rules_path=_RULES_PATH)
        profile = _make_profile(estimated_extraction_cost="needs_vision_model")
        chain = router._build_chain(profile)
        assert chain[0] == "ocr_heavy"
        assert len(chain) == 2  # 2-tier: ocr → vision


class TestExtractionRouterEscalation:
    """Verify confidence-gated escalation from A → B and B → C."""

    def test_escalation_a_to_b_on_low_confidence(self, tmp_path: Path) -> None:
        """When Strategy A returns low confidence, router should escalate to B."""
        ledger_file = tmp_path / "ledger.jsonl"
        router = ExtractionRouter(
            rules_path=_RULES_PATH, ledger_path=ledger_file
        )

        # Mock fast_text to return low confidence
        mock_a = MagicMock()
        mock_a.extract.return_value = _stub_extracted_doc()
        mock_a.confidence_score = 0.3  # below 0.6 threshold

        # Mock layout_aware to return good confidence
        mock_b = MagicMock()
        mock_b.extract.return_value = _stub_extracted_doc()
        mock_b.confidence_score = 0.85

        router._strategies["fast_text"] = mock_a
        router._strategies["layout_aware"] = mock_b

        profile = _make_profile(estimated_extraction_cost="fast_text_sufficient")
        doc, entries = router.route_and_extract(profile, "dummy.pdf")

        # Two attempts: A then B
        assert len(entries) == 2
        assert entries[0]["strategy_used"] == "fast_text"
        assert entries[0]["escalation_triggered"] is True
        assert entries[1]["strategy_used"] == "layout_aware"
        assert entries[1]["escalation_triggered"] is False

    def test_escalation_b_to_c_on_low_confidence(self, tmp_path: Path) -> None:
        """When B returns low confidence, escalate to C (ocr_heavy)."""
        ledger_file = tmp_path / "ledger.jsonl"
        router = ExtractionRouter(
            rules_path=_RULES_PATH, ledger_path=ledger_file
        )

        mock_b = MagicMock()
        mock_b.extract.return_value = _stub_extracted_doc()
        mock_b.confidence_score = 0.2  # below 0.5

        mock_c = MagicMock()
        mock_c.extract.return_value = _stub_extracted_doc()
        mock_c.confidence_score = 0.7

        router._strategies["layout_aware"] = mock_b
        router._strategies["ocr_heavy"] = mock_c

        profile = _make_profile(estimated_extraction_cost="needs_layout_model")
        doc, entries = router.route_and_extract(profile, "dummy.pdf")

        assert len(entries) == 2
        assert entries[0]["strategy_used"] == "layout_aware"
        assert entries[0]["escalation_triggered"] is True
        assert entries[1]["strategy_used"] == "ocr_heavy"

    def test_no_escalation_when_a_is_confident(self, tmp_path: Path) -> None:
        """When A is confident, no escalation should occur."""
        ledger_file = tmp_path / "ledger.jsonl"
        router = ExtractionRouter(
            rules_path=_RULES_PATH, ledger_path=ledger_file
        )

        mock_a = MagicMock()
        mock_a.extract.return_value = _stub_extracted_doc()
        mock_a.confidence_score = 0.95

        router._strategies["fast_text"] = mock_a

        profile = _make_profile(estimated_extraction_cost="fast_text_sufficient")
        doc, entries = router.route_and_extract(profile, "dummy.pdf")

        assert len(entries) == 1
        assert entries[0]["escalation_triggered"] is False

    def test_review_flag_on_very_low_confidence(self, tmp_path: Path) -> None:
        """Documents below review threshold should be flagged."""
        ledger_file = tmp_path / "ledger.jsonl"
        router = ExtractionRouter(
            rules_path=_RULES_PATH, ledger_path=ledger_file
        )

        # OCR + vision both return low confidence
        mock_ocr = MagicMock()
        mock_ocr.extract.return_value = _stub_extracted_doc()
        mock_ocr.confidence_score = 0.1

        mock_vision = MagicMock()
        mock_vision.extract.return_value = _stub_extracted_doc()
        mock_vision.confidence_score = 0.1

        router._strategies["ocr_heavy"] = mock_ocr
        router._strategies["vision_augmented"] = mock_vision

        profile = _make_profile(estimated_extraction_cost="needs_vision_model")
        doc, entries = router.route_and_extract(profile, "dummy.pdf")

        assert entries[-1]["flagged_for_review"] is True


class TestLedgerWriting:
    """Verify extraction ledger entries written to file."""

    def test_ledger_entries_persisted(self, tmp_path: Path) -> None:
        ledger_file = tmp_path / "ledger.jsonl"
        router = ExtractionRouter(
            rules_path=_RULES_PATH, ledger_path=ledger_file
        )

        mock_a = MagicMock()
        mock_a.extract.return_value = _stub_extracted_doc()
        mock_a.confidence_score = 0.9
        router._strategies["fast_text"] = mock_a

        profile = _make_profile(estimated_extraction_cost="fast_text_sufficient")
        router.route_and_extract(profile, "dummy.pdf")

        assert ledger_file.exists()
        lines = ledger_file.read_text().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert entry["document_id"] == "test-001"
        assert entry["strategy_used"] == "fast_text"
        assert "confidence_score" in entry
        assert "cost_estimate" in entry

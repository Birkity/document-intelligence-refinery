"""Unit tests for TriageAgent.detect_origin_type().

All tests mock pdfplumber so no real PDFs are required.
Each test simulates a specific document archetype by controlling
character counts, page dimensions, and image bounding boxes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agents.triage import TriageAgent

# Path to the real extraction_rules.yaml (repo root / rubric / ...)
_RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_mock_page(
    width: float = 612.0,
    height: float = 792.0,
    text: str = "",
    images: list[dict] | None = None,
) -> MagicMock:
    """Build a mock pdfplumber Page with controlled attributes."""
    page = MagicMock()
    page.width = width
    page.height = height
    page.extract_text.return_value = text
    page.images = images or []
    return page


def _open_mock_pdf(pages: list[MagicMock]):
    """Return a context-manager-compatible mock for pdfplumber.open()."""
    pdf = MagicMock()
    pdf.pages = pages
    pdf.__enter__ = lambda self: self
    pdf.__exit__ = MagicMock(return_value=False)
    return pdf


# ── Tests ────────────────────────────────────────────────────────────────

class TestDetectOriginType:
    """Tests for TriageAgent.detect_origin_type()."""

    @patch("src.agents.triage.pdfplumber")
    def test_native_digital_pdf(self, mock_pdfplumber: MagicMock) -> None:
        """A PDF with plenty of text and no images → native_digital."""
        # 612×792 page ≈ 484 704 pt².  With lots of text and no images,
        # char_density will be well above the 0.01 threshold.
        long_text = "A" * 10_000
        pages = [_make_mock_page(text=long_text)]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, confidence = agent.detect_origin_type("dummy.pdf")

        assert origin == "native_digital"
        assert 0.5 <= confidence <= 1.0

    @patch("src.agents.triage.pdfplumber")
    def test_scanned_image_pdf(self, mock_pdfplumber: MagicMock) -> None:
        """A PDF with no text and a full-page image → scanned_image."""
        # Full-page image: covers entire 612×792 area
        full_page_image = {
            "x0": 0, "x1": 612,
            "top": 0, "bottom": 792,
        }
        pages = [_make_mock_page(text="", images=[full_page_image])]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, confidence = agent.detect_origin_type("dummy.pdf")

        assert origin == "scanned_image"
        assert 0.5 <= confidence <= 1.0

    @patch("src.agents.triage.pdfplumber")
    def test_mixed_document(self, mock_pdfplumber: MagicMock) -> None:
        """A PDF with some text but also large images → mixed."""
        # Moderate text — enough to pass char_density threshold
        moderate_text = "B" * 10_000
        # Image covering ~70 % of the page → exceeds scanned_image_ratio
        large_image = {
            "x0": 0, "x1": 612,
            "top": 0, "bottom": 554,  # 612 * 554 ≈ 339 048 / 484 704 ≈ 0.70
        }
        pages = [_make_mock_page(text=moderate_text, images=[large_image])]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, confidence = agent.detect_origin_type("dummy.pdf")

        # Has enough chars (not low) but image ratio is high → mixed
        assert origin == "mixed"

    @patch("src.agents.triage.pdfplumber")
    def test_confidence_is_bounded(self, mock_pdfplumber: MagicMock) -> None:
        """Confidence must always be in [0, 1] regardless of input extremes."""
        # Extreme case: massive text, zero images
        huge_text = "C" * 500_000
        pages = [_make_mock_page(text=huge_text)]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        _, confidence = agent.detect_origin_type("dummy.pdf")

        assert 0.0 <= confidence <= 1.0

    @patch("src.agents.triage.pdfplumber")
    def test_multi_page_sampling(self, mock_pdfplumber: MagicMock) -> None:
        """Agent should analyse up to max_pages_to_sample pages."""
        text_page = _make_mock_page(text="D" * 5_000)
        # 6 pages, but default config samples only 5
        pages = [text_page] * 6
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, _ = agent.detect_origin_type("dummy.pdf")

        assert origin == "native_digital"

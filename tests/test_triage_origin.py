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
    def test_native_digital_with_embedded_images(self, mock_pdfplumber: MagicMock) -> None:
        """A PDF with a full text stream AND large images is still native_digital.

        This covers real-world annual reports (e.g. CBE) whose first pages
        contain corporate photos / charts alongside substantial text.
        With the corrected threshold (min_char_density=0.001) the presence
        of a meaningful character stream is sufficient to classify native_digital,
        regardless of image area ratio.
        """
        # Moderate text — char_density well above 0.001 threshold
        moderate_text = "B" * 10_000
        # Large image covering ~70 % of the page
        large_image = {
            "x0": 0, "x1": 612,
            "top": 0, "bottom": 554,  # 612 * 554 / 484_704 ≈ 0.70
        }
        pages = [_make_mock_page(text=moderate_text, images=[large_image])]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, confidence = agent.detect_origin_type("dummy.pdf")

        # Text stream is present → native_digital (images don't override this)
        assert origin == "native_digital"

    @patch("src.agents.triage.pdfplumber")
    def test_mixed_document(self, mock_pdfplumber: MagicMock) -> None:
        """A PDF with near-zero text extraction but not image-dominated → mixed.

        This models partially-OCR'd or image-slide PDFs: few extractable chars
        but not a full-page scanned image (image_ratio stays below 0.7).
        """
        # Very little extractable text → char_density < 0.001
        sparse_text = "X" * 20   # 20 chars on 484_704 pt² page → ~0.000041
        # Moderate image (covers ~30 % of page) — not enough to be scanned
        moderate_image = {
            "x0": 0, "x1": 612,
            "top": 0, "bottom": 238,  # 612 * 238 / 484_704 ≈ 0.30
        }
        pages = [_make_mock_page(text=sparse_text, images=[moderate_image])]
        mock_pdfplumber.open.return_value = _open_mock_pdf(pages)

        agent = TriageAgent(rules_path=_RULES_PATH)
        origin, confidence = agent.detect_origin_type("dummy.pdf")

        # Low chars but not image-dominated → mixed
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

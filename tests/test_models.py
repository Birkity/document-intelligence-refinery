"""Unit tests for core Pydantic v2 data models.

Validates that every model can be instantiated with valid data and that
Pydantic rejects invalid literal values where appropriate.
"""

import pytest
from pydantic import ValidationError

from src.models import (
    BoundingBox,
    DocumentProfile,
    ExtractedDocument,
    ExtractedPage,
    LDU,
    TableObject,
    TextBlock,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_bbox() -> BoundingBox:
    """Return a minimal valid BoundingBox."""
    return BoundingBox(x1=0.0, y1=0.0, x2=100.0, y2=50.0, page_number=1)


@pytest.fixture
def sample_text_block(sample_bbox: BoundingBox) -> TextBlock:
    """Return a minimal valid TextBlock."""
    return TextBlock(content="Hello world", bbox=sample_bbox)


@pytest.fixture
def sample_table(sample_bbox: BoundingBox) -> TableObject:
    """Return a minimal valid TableObject."""
    return TableObject(
        headers=["Year", "Revenue"],
        rows=[["2023", "$4.2B"], ["2024", "$5.1B"]],
        bbox=sample_bbox,
    )


# ── BoundingBox ──────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_valid_instantiation(self, sample_bbox: BoundingBox) -> None:
        assert sample_bbox.x1 == 0.0
        assert sample_bbox.x2 == 100.0
        assert sample_bbox.page_number == 1

    def test_rejects_page_zero(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x1=0, y1=0, x2=1, y2=1, page_number=0)


# ── TextBlock ────────────────────────────────────────────────────────────

class TestTextBlock:
    def test_valid_instantiation(self, sample_text_block: TextBlock) -> None:
        assert sample_text_block.content == "Hello world"
        assert sample_text_block.bbox.page_number == 1


# ── TableObject ──────────────────────────────────────────────────────────

class TestTableObject:
    def test_valid_instantiation(self, sample_table: TableObject) -> None:
        assert sample_table.headers == ["Year", "Revenue"]
        assert len(sample_table.rows) == 2


# ── ExtractedPage ────────────────────────────────────────────────────────

class TestExtractedPage:
    def test_valid_instantiation(
        self,
        sample_text_block: TextBlock,
        sample_table: TableObject,
    ) -> None:
        page = ExtractedPage(
            page_number=1,
            text_blocks=[sample_text_block],
            tables=[sample_table],
        )
        assert page.page_number == 1
        assert len(page.text_blocks) == 1
        assert len(page.tables) == 1

    def test_defaults_to_empty_lists(self) -> None:
        page = ExtractedPage(page_number=1)
        assert page.text_blocks == []
        assert page.tables == []


# ── ExtractedDocument ────────────────────────────────────────────────────

class TestExtractedDocument:
    def test_valid_instantiation(self) -> None:
        doc = ExtractedDocument(
            document_id="doc-001",
            source_filename="document.pdf",
            pages=[ExtractedPage(page_number=1)],
        )
        assert doc.document_id == "doc-001"
        assert len(doc.pages) == 1

    def test_defaults_to_empty_pages(self) -> None:
        doc = ExtractedDocument(
            document_id="doc-002",
            source_filename="sample.pdf",
        )
        assert doc.pages == []


# ── DocumentProfile ──────────────────────────────────────────────────────

class TestDocumentProfile:
    def test_valid_instantiation(self) -> None:
        profile = DocumentProfile(
            document_id="doc-001",
            source_filename="report.pdf",
            origin_type="native_digital",
            layout_complexity="single_column",
            language="en",
            language_confidence=0.95,
            domain_hint="financial",
            estimated_extraction_cost="fast_text_sufficient",
        )
        assert profile.origin_type == "native_digital"
        assert profile.language_confidence == 0.95

    def test_rejects_invalid_origin_type(self) -> None:
        with pytest.raises(ValidationError):
            DocumentProfile(
                document_id="doc-bad",
                source_filename="bad.pdf",
                origin_type="unknown_type",  # type: ignore[arg-type]
                layout_complexity="single_column",
                language="en",
                language_confidence=0.5,
                domain_hint="general",
                estimated_extraction_cost="fast_text_sufficient",
            )

    def test_rejects_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            DocumentProfile(
                document_id="doc-bad",
                source_filename="bad.pdf",
                origin_type="native_digital",
                layout_complexity="single_column",
                language="en",
                language_confidence=1.5,  # > 1.0
                domain_hint="general",
                estimated_extraction_cost="fast_text_sufficient",
            )


# ── LDU ──────────────────────────────────────────────────────────────────

class TestLDU:
    def test_valid_instantiation(self, sample_bbox: BoundingBox) -> None:
        ldu = LDU(
            content="Some paragraph text.",
            chunk_type="paragraph",
            page_refs=[1, 2],
            bbox=sample_bbox,
            parent_section="Introduction",
            token_count=5,
            content_hash="abc123def456",
        )
        assert ldu.chunk_type == "paragraph"
        assert ldu.parent_section == "Introduction"
        assert ldu.token_count == 5

    def test_optional_fields_default_to_none(self) -> None:
        ldu = LDU(
            content="Minimal chunk.",
            chunk_type="section",
            page_refs=[1],
            token_count=3,
            content_hash="deadbeef",
        )
        assert ldu.bbox is None
        assert ldu.parent_section is None

    def test_rejects_invalid_chunk_type(self) -> None:
        with pytest.raises(ValidationError):
            LDU(
                content="Bad chunk.",
                chunk_type="unknown",  # type: ignore[arg-type]
                page_refs=[1],
                token_count=2,
                content_hash="bad",
            )

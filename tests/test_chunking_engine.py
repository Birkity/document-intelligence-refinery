"""TDD tests for the Semantic Chunking Engine (Phase 3).

Tests are written BEFORE implementation.  They validate:
- Paragraph chunking with token limits
- Table preservation (rule #1: cell never split from header)
- Figure-caption binding (rule #2: caption as figure-chunk metadata)
- List preservation (rule #3: numbered list as single LDU unless > max_tokens)
- Section-header propagation (rule #4: parent metadata on child chunks)
- Cross-reference resolution (rule #5: "see Table 3" stored as relationships)
- Content-hash generation (SHA-256 per LDU)
- ChunkValidator enforcement
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.models.schemas import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
    LDU,
    TableObject,
    TextBlock,
)
from src.utils.hash_utils import generate_content_hash

# Will be imported once implementation exists:
from src.agents.chunker import ChunkingEngine, ChunkValidator

_RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


# =====================================================================
# Helpers
# =====================================================================


def _bb(page: int = 1) -> BoundingBox:
    """Quick bounding box factory."""
    return BoundingBox(x1=0, y1=0, x2=612, y2=792, page_number=page)


def _make_doc(
    pages: list[ExtractedPage] | None = None,
    doc_id: str = "test-doc-001",
) -> ExtractedDocument:
    """Build a minimal ExtractedDocument."""
    return ExtractedDocument(
        document_id=doc_id,
        source_filename="test.pdf",
        pages=pages or [],
    )


def _text_block(text: str, page: int = 1) -> TextBlock:
    return TextBlock(content=text, bbox=_bb(page))


def _table(headers: list[str], rows: list[list[str]], page: int = 1) -> TableObject:
    return TableObject(headers=headers, rows=rows, bbox=_bb(page))


def _figure(caption: str = "", page: int = 1) -> FigureObject:
    return FigureObject(caption=caption, bbox=_bb(page))


# =====================================================================
# Content-hash utility tests
# =====================================================================


class TestContentHash:
    """Verify SHA-256 content-hash generation."""

    def test_deterministic(self) -> None:
        h1 = generate_content_hash("hello world")
        h2 = generate_content_hash("hello world")
        assert h1 == h2

    def test_different_for_different_content(self) -> None:
        h1 = generate_content_hash("alpha")
        h2 = generate_content_hash("beta")
        assert h1 != h2

    def test_strip_normalisation(self) -> None:
        """Leading/trailing whitespace does not change the hash."""
        h1 = generate_content_hash("  hello world  ")
        h2 = generate_content_hash("hello world")
        assert h1 == h2

    def test_sha256_format(self) -> None:
        h = generate_content_hash("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_matches_stdlib(self) -> None:
        text = "some content"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert generate_content_hash(text) == expected


# =====================================================================
# ChunkingEngine — basic paragraph chunking
# =====================================================================


class TestParagraphChunking:
    """Paragraphs should become individual LDUs within token limits."""

    def test_single_paragraph_produces_one_ldu(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("This is a simple paragraph.")],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        assert len(ldus) >= 1
        assert ldus[0].chunk_type == "paragraph"
        assert ldus[0].content.strip() == "This is a simple paragraph."
        assert 1 in ldus[0].page_refs

    def test_ldu_has_content_hash(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("Hash me.")],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        assert ldus[0].content_hash == generate_content_hash("Hash me.")

    def test_ldu_has_token_count(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("One two three four five.")],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        assert ldus[0].token_count > 0

    def test_long_paragraph_is_split(self) -> None:
        """Paragraph exceeding max_tokens must be split into ≥ 2 chunks."""
        long_text = " ".join(["word"] * 1000)  # ~1000 tokens
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block(long_text)],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        assert len(ldus) >= 2
        for ldu in ldus:
            assert ldu.token_count <= 512  # max_tokens_per_chunk from rules

    def test_empty_document_yields_no_ldus(self) -> None:
        doc = _make_doc(pages=[])
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)
        assert ldus == []

    def test_multi_page_paragraphs(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("Page one text.", page=1)],
                ),
                ExtractedPage(
                    page_number=2,
                    text_blocks=[_text_block("Page two text.", page=2)],
                ),
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        assert len(ldus) >= 2
        page_refs_flat = [ref for ldu in ldus for ref in ldu.page_refs]
        assert 1 in page_refs_flat
        assert 2 in page_refs_flat


# =====================================================================
# Rule #1: Table cell never split from header row
# =====================================================================


class TestTablePreservation:
    """A table must stay as a single LDU (rule #1)."""

    def test_table_becomes_single_ldu(self) -> None:
        tbl = _table(
            headers=["Year", "Revenue", "Profit"],
            rows=[
                ["2021", "100M", "10M"],
                ["2022", "120M", "15M"],
                ["2023", "130M", "18M"],
            ],
        )
        doc = _make_doc(
            pages=[ExtractedPage(page_number=1, tables=[tbl])]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        table_ldus = [l for l in ldus if l.chunk_type == "table"]
        assert len(table_ldus) == 1
        # Content should contain headers and data
        assert "Year" in table_ldus[0].content
        assert "Revenue" in table_ldus[0].content
        assert "100M" in table_ldus[0].content

    def test_table_ldu_has_page_ref(self) -> None:
        tbl = _table(headers=["A"], rows=[["1"]], page=3)
        doc = _make_doc(
            pages=[ExtractedPage(page_number=3, tables=[tbl])]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        table_ldus = [l for l in ldus if l.chunk_type == "table"]
        assert 3 in table_ldus[0].page_refs


# =====================================================================
# Rule #2: Figure caption as metadata of parent figure chunk
# =====================================================================


class TestFigureCaptionBinding:
    """Figure caption must be stored as part of the figure LDU."""

    def test_figure_with_caption_becomes_figure_ldu(self) -> None:
        fig = _figure(caption="Figure 1: Revenue trend 2020–2024", page=2)
        doc = _make_doc(
            pages=[ExtractedPage(page_number=2, figures=[fig])]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        fig_ldus = [l for l in ldus if l.chunk_type == "figure"]
        assert len(fig_ldus) == 1
        assert "Figure 1: Revenue trend 2020–2024" in fig_ldus[0].content

    def test_figure_without_caption(self) -> None:
        fig = _figure(caption="", page=1)
        doc = _make_doc(
            pages=[ExtractedPage(page_number=1, figures=[fig])]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        fig_ldus = [l for l in ldus if l.chunk_type == "figure"]
        assert len(fig_ldus) == 1
        assert fig_ldus[0].content  # Should still have some content


# =====================================================================
# Rule #3: Numbered list kept as single LDU unless > max_tokens
# =====================================================================


class TestListPreservation:
    """Numbered/bullet lists should stay as single LDU (rule #3)."""

    def test_numbered_list_stays_together(self) -> None:
        list_text = (
            "1. First item in the list\n"
            "2. Second item in the list\n"
            "3. Third item in the list\n"
            "4. Fourth item in the list"
        )
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block(list_text)],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        # The list should be a single LDU (not split per item)
        list_ldus = [l for l in ldus if l.chunk_type == "list"]
        assert len(list_ldus) == 1
        assert "1." in list_ldus[0].content
        assert "4." in list_ldus[0].content

    def test_bullet_list_stays_together(self) -> None:
        list_text = (
            "• Revenue increased by 20%\n"
            "• Operating costs reduced by 5%\n"
            "• New branches opened in 3 regions"
        )
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block(list_text)],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        list_ldus = [l for l in ldus if l.chunk_type == "list"]
        assert len(list_ldus) == 1

    def test_very_long_list_is_split(self) -> None:
        """A list exceeding max_tokens should be split at list-item boundaries."""
        items = [f"{i}. {'word ' * 60}" for i in range(1, 30)]
        list_text = "\n".join(items)  # ~1800 tokens
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block(list_text)],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        list_ldus = [l for l in ldus if l.chunk_type == "list"]
        assert len(list_ldus) >= 2
        for ldu in list_ldus:
            assert ldu.token_count <= 512


# =====================================================================
# Rule #4: Section headers as parent metadata on child chunks
# =====================================================================


class TestSectionHeaderPropagation:
    """Section headers must propagate as parent_section on children."""

    def test_section_header_propagated(self) -> None:
        """When a text block looks like a section header, subsequent
        paragraphs within the same section carry parent_section."""
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[
                        _text_block("Financial Overview"),
                        _text_block(
                            "The bank reported record revenue in 2024, "
                            "driven by growth in digital banking services."
                        ),
                    ],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        # The paragraph LDU should carry the section header
        para_ldus = [l for l in ldus if l.chunk_type == "paragraph"]
        assert any(
            l.parent_section == "Financial Overview" for l in para_ldus
        )


# =====================================================================
# Rule #5: Cross-references resolved as chunk relationships
# =====================================================================


class TestCrossReferenceResolution:
    """Cross-references like 'see Table 3' should be detected."""

    def test_cross_ref_detected_in_content(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[
                        _text_block(
                            "Revenue increased by 20% as shown in Table 3. "
                            "See also Figure 2 for the trend line."
                        ),
                    ],
                    tables=[
                        _table(headers=["Year", "Rev"], rows=[["2023", "4.2B"]]),
                    ],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        # Cross-references should appear somewhere in the output
        para_ldus = [l for l in ldus if l.chunk_type == "paragraph"]
        assert len(para_ldus) >= 1
        # The paragraph content should still contain the cross-ref text
        assert any("Table 3" in l.content for l in para_ldus)


# =====================================================================
# ChunkValidator
# =====================================================================


class TestChunkValidator:
    """ChunkValidator must reject LDUs that violate chunking rules."""

    def test_valid_ldu_passes(self) -> None:
        ldu = LDU(
            content="A valid chunk.",
            chunk_type="paragraph",
            page_refs=[1],
            token_count=4,
            content_hash=generate_content_hash("A valid chunk."),
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        errors = validator.validate(ldu)
        assert errors == []

    def test_empty_content_fails(self) -> None:
        ldu = LDU(
            content="",
            chunk_type="paragraph",
            page_refs=[1],
            token_count=0,
            content_hash=generate_content_hash(""),
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        errors = validator.validate(ldu)
        assert len(errors) > 0

    def test_token_count_exceeds_max(self) -> None:
        ldu = LDU(
            content="word " * 600,
            chunk_type="paragraph",
            page_refs=[1],
            token_count=600,
            content_hash=generate_content_hash("word " * 600),
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        errors = validator.validate(ldu)
        assert any("token" in e.lower() for e in errors)

    def test_mismatched_hash_fails(self) -> None:
        ldu = LDU(
            content="Actual content",
            chunk_type="paragraph",
            page_refs=[1],
            token_count=2,
            content_hash="badhash0000000000000000000000000000000000000000000000000000000000",
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        errors = validator.validate(ldu)
        assert any("hash" in e.lower() for e in errors)

    def test_missing_page_refs_fails(self) -> None:
        ldu = LDU(
            content="No page refs.",
            chunk_type="paragraph",
            page_refs=[],
            token_count=3,
            content_hash=generate_content_hash("No page refs."),
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        errors = validator.validate(ldu)
        assert any("page" in e.lower() for e in errors)

    def test_validate_batch(self) -> None:
        """validate_batch should check every LDU and return all issues."""
        good = LDU(
            content="OK chunk.",
            chunk_type="paragraph",
            page_refs=[1],
            token_count=2,
            content_hash=generate_content_hash("OK chunk."),
        )
        bad = LDU(
            content="",
            chunk_type="paragraph",
            page_refs=[1],
            token_count=0,
            content_hash=generate_content_hash(""),
        )
        validator = ChunkValidator(rules_path=_RULES_PATH)
        all_errors = validator.validate_batch([good, bad])
        assert len(all_errors) >= 1  # at least the bad one has errors


# =====================================================================
# End-to-end: mixed document with text + tables + figures
# =====================================================================


class TestEndToEndMixedDocument:
    """Full document with paragraphs, tables, and figures."""

    def test_mixed_doc_produces_all_chunk_types(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[
                        _text_block("Executive Summary"),
                        _text_block(
                            "The company achieved strong financial results "
                            "in fiscal year 2024."
                        ),
                    ],
                    tables=[
                        _table(
                            headers=["Metric", "2023", "2024"],
                            rows=[
                                ["Revenue", "3.8B", "4.2B"],
                                ["Net Income", "0.9B", "1.1B"],
                            ],
                        )
                    ],
                    figures=[
                        _figure(
                            caption="Figure 1: Revenue growth 2020-2024",
                            page=1,
                        )
                    ],
                ),
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        types_found = {l.chunk_type for l in ldus}
        # Should have at least paragraph, table, and figure
        assert "table" in types_found
        assert "figure" in types_found
        # Should have paragraph or section
        assert "paragraph" in types_found or "section" in types_found

    def test_all_ldus_have_valid_hashes(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("Hello world.")],
                    tables=[_table(headers=["X"], rows=[["1"]])],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        validator = ChunkValidator(rules_path=_RULES_PATH)
        for ldu in ldus:
            errors = validator.validate(ldu)
            assert errors == [], f"LDU failed validation: {errors}"

    def test_all_ldus_within_token_limits(self) -> None:
        doc = _make_doc(
            pages=[
                ExtractedPage(
                    page_number=1,
                    text_blocks=[_text_block("Short paragraph.")],
                    tables=[
                        _table(
                            headers=["Col"],
                            rows=[[f"row{i}"] for i in range(50)],
                        )
                    ],
                )
            ]
        )
        engine = ChunkingEngine(rules_path=_RULES_PATH)
        ldus = engine.chunk_document(doc)

        for ldu in ldus:
            # Tables may exceed max_tokens but text chunks must not
            if ldu.chunk_type != "table":
                assert ldu.token_count <= 512

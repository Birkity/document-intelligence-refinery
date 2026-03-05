"""TDD tests for the PageIndex Builder (Stage 4).

Covers:
- Single-section document
- Multi-section document
- Section headers spanning multiple pages
- Table-heavy / figure-heavy signal detection
- Deterministic summary generation
- data_types_present population
- JSON artifact persistence
- DB persistence
- PageIndex query (top-N section retrieval)
- Edge cases: empty LDU list, all same section, no headers
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from src.models.schemas import (
    BoundingBox,
    LDU,
    PageIndex,
    PageIndexNode,
)
from src.utils.hash_utils import generate_content_hash

# Will be imported once implementation exists
from src.agents.pageindex import PageIndexBuilder

_RULES_PATH = Path(__file__).resolve().parents[1] / "rubric" / "extraction_rules.yaml"


# =====================================================================
# Helpers
# =====================================================================


def _ldu(
    content: str,
    chunk_type: str = "paragraph",
    page_refs: list[int] | None = None,
    parent_section: str | None = None,
) -> LDU:
    """Quick LDU factory."""
    return LDU(
        content=content,
        chunk_type=chunk_type,
        page_refs=page_refs or [1],
        parent_section=parent_section,
        token_count=len(content.split()),
        content_hash=generate_content_hash(content),
    )


# =====================================================================
# Basic construction
# =====================================================================


class TestSingleSectionDocument:
    """A document with only one section should produce one root node."""

    def test_single_section_produces_one_root(self) -> None:
        ldus = [
            _ldu("Introduction", chunk_type="section", parent_section="Introduction"),
            _ldu(
                "This report covers the fiscal year 2024. "
                "The bank achieved record revenue.",
                parent_section="Introduction",
                page_refs=[1],
            ),
            _ldu(
                "Net income grew by 15% compared to the previous year.",
                parent_section="Introduction",
                page_refs=[1, 2],
            ),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-001")

        assert pi.document_id == "doc-001"
        assert len(pi.root_nodes) == 1
        assert pi.root_nodes[0].title == "Introduction"
        assert pi.root_nodes[0].page_start == 1

    def test_auto_generated_document_id(self) -> None:
        ldus = [_ldu("Hello world.", parent_section="Intro")]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="report.pdf")

        assert pi.document_id  # should be auto-generated, not empty


class TestMultiSectionDocument:
    """Documents with multiple sections → multiple root nodes."""

    def test_two_sections(self) -> None:
        ldus = [
            _ldu("Overview", chunk_type="section", parent_section="Overview"),
            _ldu("The company was founded in 1960.", parent_section="Overview", page_refs=[1]),
            _ldu("Financial Results", chunk_type="section", parent_section="Financial Results"),
            _ldu("Revenue reached 4.2 billion.", parent_section="Financial Results", page_refs=[3]),
            _ldu("Net income was 1.1 billion.", parent_section="Financial Results", page_refs=[4]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-002")

        assert len(pi.root_nodes) == 2
        assert pi.root_nodes[0].title == "Overview"
        assert pi.root_nodes[1].title == "Financial Results"

    def test_section_page_ranges(self) -> None:
        """page_start = min(page_refs), page_end = max(page_refs) for each section."""
        ldus = [
            _ldu("Section A", chunk_type="section", parent_section="Section A"),
            _ldu("Content A1.", parent_section="Section A", page_refs=[1, 2]),
            _ldu("Content A2.", parent_section="Section A", page_refs=[3]),
            _ldu("Section B", chunk_type="section", parent_section="Section B"),
            _ldu("Content B1.", parent_section="Section B", page_refs=[5]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-003")

        assert pi.root_nodes[0].page_start == 1
        assert pi.root_nodes[0].page_end == 3
        assert pi.root_nodes[1].page_start == 5
        assert pi.root_nodes[1].page_end == 5

    def test_sections_ordered_by_first_appearance(self) -> None:
        ldus = [
            _ldu("Zebra Section", chunk_type="section", parent_section="Zebra Section"),
            _ldu("Zebra content.", parent_section="Zebra Section", page_refs=[1]),
            _ldu("Alpha Section", chunk_type="section", parent_section="Alpha Section"),
            _ldu("Alpha content.", parent_section="Alpha Section", page_refs=[2]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-004")

        # Should preserve document order, not alphabetical
        assert pi.root_nodes[0].title == "Zebra Section"
        assert pi.root_nodes[1].title == "Alpha Section"


# =====================================================================
# Data-type signal detection
# =====================================================================


class TestDataTypesDetection:
    """Nodes should report which data types are present."""

    def test_tables_detected(self) -> None:
        ldus = [
            _ldu("Data Tables", chunk_type="section", parent_section="Data Tables"),
            _ldu("Year | Revenue\n2023 | 4.2B", chunk_type="table", parent_section="Data Tables"),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-t1")

        assert "tables" in pi.root_nodes[0].data_types_present

    def test_figures_detected(self) -> None:
        ldus = [
            _ldu("Charts", chunk_type="section", parent_section="Charts"),
            _ldu("[IMAGE] Revenue trend chart", chunk_type="figure", parent_section="Charts"),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-t2")

        assert "figures" in pi.root_nodes[0].data_types_present

    def test_lists_detected(self) -> None:
        ldus = [
            _ldu("Key Points", chunk_type="section", parent_section="Key Points"),
            _ldu("1. First\n2. Second\n3. Third", chunk_type="list", parent_section="Key Points"),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-t3")

        assert "lists" in pi.root_nodes[0].data_types_present

    def test_numeric_density_detected(self) -> None:
        ldus = [
            _ldu("Financials", chunk_type="section", parent_section="Financials"),
            _ldu(
                "Revenue 4,200,000 ETB. Net income 1,100,000. "
                "Assets 50,000,000. Liabilities 30,000,000. "
                "Equity 20,000,000. ROE 0.15. ROA 0.08.",
                parent_section="Financials",
            ),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-t4")

        assert "numeric_dense" in pi.root_nodes[0].data_types_present

    def test_mixed_data_types(self) -> None:
        ldus = [
            _ldu("Analysis", chunk_type="section", parent_section="Analysis"),
            _ldu("Year | Revenue\n2023 | 4.2B", chunk_type="table", parent_section="Analysis"),
            _ldu("[IMAGE] Revenue trend chart", chunk_type="figure", parent_section="Analysis"),
            _ldu("General narrative text.", parent_section="Analysis"),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-t5")

        dtypes = pi.root_nodes[0].data_types_present
        assert "tables" in dtypes
        assert "figures" in dtypes


# =====================================================================
# Summary generation
# =====================================================================


class TestDeterministicSummary:
    """Without LLM, summary should be first N sentences of the section."""

    def test_summary_from_first_paragraph(self) -> None:
        ldus = [
            _ldu("Executive Summary", chunk_type="section", parent_section="Executive Summary"),
            _ldu(
                "The bank reported strong results. "
                "Net income increased by 15%. "
                "Total assets reached 1 trillion ETB. "
                "Digital banking expanded to rural areas.",
                parent_section="Executive Summary",
            ),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-s1")

        summary = pi.root_nodes[0].summary
        assert len(summary) > 0
        # Should contain at most ~3 sentences
        assert summary.count(".") <= 4  # relaxed — could end with partial

    def test_summary_excludes_tables(self) -> None:
        """Summary should come from paragraph text, not table data."""
        ldus = [
            _ldu("Revenue Table", chunk_type="section", parent_section="Revenue Table"),
            _ldu("A | B\n1 | 2", chunk_type="table", parent_section="Revenue Table"),
            _ldu(
                "Revenue grew significantly in 2024. The growth was driven by loan expansion.",
                parent_section="Revenue Table",
            ),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-s2")

        summary = pi.root_nodes[0].summary
        assert "Revenue grew" in summary


# =====================================================================
# Headers spanning multiple pages
# =====================================================================


class TestCrossPageSections:
    """Sections that span multiple pages should be merged."""

    def test_section_spans_pages(self) -> None:
        ldus = [
            _ldu("Chapter 1", chunk_type="section", parent_section="Chapter 1"),
            _ldu("Content on page 1.", parent_section="Chapter 1", page_refs=[1]),
            _ldu("Content on page 2.", parent_section="Chapter 1", page_refs=[2]),
            _ldu("Content on page 3.", parent_section="Chapter 1", page_refs=[3]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-cp1")

        assert len(pi.root_nodes) == 1
        assert pi.root_nodes[0].page_start == 1
        assert pi.root_nodes[0].page_end == 3


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
    """Edge cases: empty input, no sections, standalone headers."""

    def test_empty_ldus_produces_empty_index(self) -> None:
        builder = PageIndexBuilder()
        pi = builder.build([], source_filename="empty.pdf", document_id="doc-e1")

        assert pi.document_id == "doc-e1"
        assert pi.root_nodes == []

    def test_no_section_headers_creates_default_section(self) -> None:
        """LDUs without parent_section should be grouped under a default node."""
        ldus = [
            _ldu("Some paragraph.", parent_section=None, page_refs=[1]),
            _ldu("Another paragraph.", parent_section=None, page_refs=[2]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-e2")

        assert len(pi.root_nodes) >= 1  # at least one default node

    def test_all_ldus_same_section(self) -> None:
        ldus = [
            _ldu("Only Section", chunk_type="section", parent_section="Only Section"),
            _ldu("Para 1.", parent_section="Only Section", page_refs=[1]),
            _ldu("Para 2.", parent_section="Only Section", page_refs=[1]),
            _ldu("Para 3.", parent_section="Only Section", page_refs=[2]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-e3")

        assert len(pi.root_nodes) == 1
        assert pi.root_nodes[0].title == "Only Section"


# =====================================================================
# JSON artifact persistence
# =====================================================================


class TestJsonPersistence:
    """PageIndex should save to .refinery/pageindex/{doc_id}.json."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        ldus = [
            _ldu("Intro", chunk_type="section", parent_section="Intro"),
            _ldu("Content here.", parent_section="Intro", page_refs=[1]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-json1")

        out_dir = tmp_path / "pageindex"
        builder.save_json(pi, output_dir=out_dir)

        json_file = out_dir / "doc-json1.json"
        assert json_file.exists()

        loaded = json.loads(json_file.read_text(encoding="utf-8"))
        assert loaded["document_id"] == "doc-json1"
        assert len(loaded["root_nodes"]) == 1
        assert loaded["root_nodes"][0]["title"] == "Intro"


# =====================================================================
# DB persistence
# =====================================================================


class TestDbPersistence:
    """PageIndex should be stored in refinery.db page_indexes table."""

    def _init_db(self, db_path: Path) -> None:
        """Create just the page_indexes table for testing."""
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS page_indexes (
                document_id TEXT PRIMARY KEY,
                source_filename TEXT NOT NULL,
                tree_json TEXT NOT NULL,
                node_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def test_persist_to_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._init_db(db_path)

        ldus = [
            _ldu("Section A", chunk_type="section", parent_section="Section A"),
            _ldu("Some content.", parent_section="Section A", page_refs=[1]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="report.pdf", document_id="doc-db1")

        builder.persist_to_db(pi, source_filename="report.pdf", db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT document_id, source_filename, tree_json, node_count FROM page_indexes WHERE document_id = ?",
            ("doc-db1",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "doc-db1"
        assert row[1] == "report.pdf"
        assert row[3] >= 1  # node_count
        tree = json.loads(row[2])
        assert len(tree) >= 1

    def test_upsert_replaces_existing(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        self._init_db(db_path)

        ldus1 = [_ldu("V1", chunk_type="section", parent_section="V1")]
        ldus2 = [
            _ldu("V2-A", chunk_type="section", parent_section="V2-A"),
            _ldu("V2-B", chunk_type="section", parent_section="V2-B"),
        ]
        builder = PageIndexBuilder()

        pi1 = builder.build(ldus1, source_filename="r.pdf", document_id="doc-up")
        builder.persist_to_db(pi1, source_filename="r.pdf", db_path=db_path)

        pi2 = builder.build(ldus2, source_filename="r.pdf", document_id="doc-up")
        builder.persist_to_db(pi2, source_filename="r.pdf", db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT node_count FROM page_indexes WHERE document_id = ?",
            ("doc-up",),
        ).fetchone()
        conn.close()

        assert row[0] == 2  # second build had 2 nodes


# =====================================================================
# PageIndex query (top-N section retrieval)
# =====================================================================


class TestPageIndexQuery:
    """Given a topic, find the top-N most relevant sections."""

    def test_query_returns_matching_section(self) -> None:
        ldus = [
            _ldu("Revenue Analysis", chunk_type="section", parent_section="Revenue Analysis"),
            _ldu("Revenue grew by 20%.", parent_section="Revenue Analysis", page_refs=[5]),
            _ldu("Risk Management", chunk_type="section", parent_section="Risk Management"),
            _ldu("The bank manages credit risk.", parent_section="Risk Management", page_refs=[10]),
            _ldu("Corporate Governance", chunk_type="section", parent_section="Corporate Governance"),
            _ldu("The board meets quarterly.", parent_section="Corporate Governance", page_refs=[15]),
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-q1")

        results = builder.query(pi, topic="revenue growth", top_n=2)

        assert len(results) <= 2
        # "Revenue Analysis" should be in the results
        titles = [r.title for r in results]
        assert "Revenue Analysis" in titles

    def test_query_returns_up_to_top_n(self) -> None:
        ldus = [
            _ldu(f"Section {i}", chunk_type="section", parent_section=f"Section {i}")
            for i in range(10)
        ]
        builder = PageIndexBuilder()
        pi = builder.build(ldus, source_filename="test.pdf", document_id="doc-q2")

        results = builder.query(pi, topic="anything", top_n=3)
        assert len(results) <= 3

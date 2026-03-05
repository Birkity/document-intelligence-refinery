"""Unit tests for database initialisation (SQLite) and new schema models."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.db.init_db import initialize_database
from src.models.schemas import (
    PageIndex,
    PageIndexNode,
    ProvenanceChain,
    ProvenanceCitation,
    BoundingBox,
)


# ── SQLite Init ──────────────────────────────────────────────────────────

class TestInitializeDatabase:
    """Tests for idempotent SQLite database creation."""

    def test_creates_database_file(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        result = initialize_database(db_path)
        assert result == db_path
        assert db_path.exists()

    def test_creates_expected_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = sorted(
            row[0] for row in cursor.fetchall()
            if not row[0].startswith("sqlite_")
        )
        conn.close()

        expected = [
            "chunks",
            "documents",
            "fact_tables",
            "page_indexes",
            "provenance_ledger",
            "query_logs",
            "structured_tables",
        ]
        assert tables == expected

    def test_idempotent(self, tmp_path: Path) -> None:
        """Calling init twice should not raise or corrupt data."""
        db_path = tmp_path / "test.db"
        initialize_database(db_path)
        # Second call should be a no-op (CREATE IF NOT EXISTS)
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [
            row[0] for row in cursor.fetchall()
            if not row[0].startswith("sqlite_")
        ]
        conn.close()
        assert len(tables) == 7

    def test_can_insert_and_read_document(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO documents (document_id, source_filename, origin_type, "
            "layout_complexity, domain_hint, estimated_cost, processing_timestamp, "
            "page_count, total_chunks) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("doc-001", "test.pdf", "native_digital", "single_column",
             "financial", "fast_text_sufficient", "2026-03-04T12:00:00", 10, 50),
        )
        conn.commit()
        row = conn.execute(
            "SELECT source_filename FROM documents WHERE document_id = ?",
            ("doc-001",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "test.pdf"


# ── PageIndex Schema ─────────────────────────────────────────────────────

class TestPageIndexNode:
    def test_valid_instantiation(self) -> None:
        node = PageIndexNode(
            title="Introduction",
            page_start=1,
            page_end=5,
            key_entities=["CBE", "Ethiopia"],
            summary="Covers the overview of operations.",
            data_types_present=["tables", "figures"],
        )
        assert node.title == "Introduction"
        assert node.children == []

    def test_nested_children(self) -> None:
        child = PageIndexNode(title="1.1 Background", page_start=2, page_end=3)
        parent = PageIndexNode(
            title="Chapter 1", page_start=1, page_end=5, children=[child]
        )
        assert len(parent.children) == 1
        assert parent.children[0].title == "1.1 Background"


class TestPageIndex:
    def test_valid_instantiation(self) -> None:
        idx = PageIndex(
            document_id="doc-001",
            root_nodes=[
                PageIndexNode(title="Root", page_start=1, page_end=10),
            ],
        )
        assert idx.document_id == "doc-001"
        assert len(idx.root_nodes) == 1

    def test_serialises_to_json(self) -> None:
        idx = PageIndex(
            document_id="doc-002",
            root_nodes=[
                PageIndexNode(
                    title="Section A",
                    page_start=1,
                    page_end=3,
                    children=[
                        PageIndexNode(title="A.1", page_start=1, page_end=2),
                    ],
                )
            ],
        )
        data = idx.model_dump()
        assert data["root_nodes"][0]["children"][0]["title"] == "A.1"


# ── Provenance Schema ────────────────────────────────────────────────────

class TestProvenanceCitation:
    def test_valid_instantiation(self) -> None:
        cite = ProvenanceCitation(
            document_id="doc-001",
            document_name="report.pdf",
            page_number=5,
            bbox=BoundingBox(x1=10, y1=20, x2=300, y2=400, page_number=5),
            content_hash="abc123",
        )
        assert cite.page_number == 5

    def test_optional_bbox(self) -> None:
        cite = ProvenanceCitation(
            document_id="doc-001",
            document_name="report.pdf",
            page_number=1,
        )
        assert cite.bbox is None


class TestProvenanceChain:
    def test_valid_instantiation(self) -> None:
        chain = ProvenanceChain(
            query="What was the revenue?",
            citations=[
                ProvenanceCitation(
                    document_id="doc-001",
                    document_name="report.pdf",
                    page_number=12,
                    content_hash="deadbeef",
                )
            ],
            verified=True,
        )
        assert chain.verified is True
        assert len(chain.citations) == 1

    def test_serialises_to_json(self) -> None:
        chain = ProvenanceChain(
            query="test query",
            citations=[],
        )
        data = chain.model_dump_json()
        assert "test query" in data

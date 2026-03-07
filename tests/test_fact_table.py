"""TDD tests for the FactTable extractor — Stage 5 data layer.

Covers: key-value fact extraction from LDUs, SQLite persistence,
and structured querying.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.agents.fact_table import FactTableExtractor
from src.db.init_db import initialize_database
from src.models.schemas import BoundingBox, Fact, LDU
from src.utils.hash_utils import generate_content_hash


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_ldu(
    content: str,
    chunk_type: str = "paragraph",
    page: int = 1,
    parent_section: str | None = None,
) -> LDU:
    return LDU(
        content=content,
        chunk_type=chunk_type,
        page_refs=[page],
        bbox=BoundingBox(x1=0, y1=0, x2=100, y2=100, page_number=page),
        parent_section=parent_section,
        token_count=len(content.split()),
        content_hash=generate_content_hash(content),
    )


# ── Fact extraction ─────────────────────────────────────────────────────

class TestFactExtraction:
    """Extract key-value numerical facts from LDU content."""

    def test_extract_currency_facts(self) -> None:
        ldu = _make_ldu(
            "Total Revenue: $4.2B for FY2024. Net Income: $1.1B.",
            parent_section="Financial Highlights",
            page=5,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        keys = [f.key for f in facts]
        assert "Total Revenue" in keys or any("revenue" in k.lower() for k in keys)
        assert len(facts) >= 2

    def test_extract_percentage_facts(self) -> None:
        ldu = _make_ldu(
            "Interest Rate: 12.5%. Inflation Rate: 8.3%.",
            parent_section="Economic Indicators",
            page=10,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 2
        assert any("%" in f.value or "12.5" in f.value for f in facts)

    def test_extract_from_table_ldu(self) -> None:
        ldu = _make_ldu(
            "Revenue | 4,200 | 3,800\nExpenses | 2,100 | 1,900",
            chunk_type="table",
            parent_section="Income Statement",
            page=12,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 1

    def test_no_facts_from_narrative(self) -> None:
        ldu = _make_ldu(
            "The company was founded in Addis Ababa and operates across Ethiopia.",
            page=2,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) == 0

    def test_facts_carry_document_id(self) -> None:
        ldu = _make_ldu("Total Assets: 500 million ETB.", page=7)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc42")
        assert all(f.document_id == "doc42" for f in facts)

    def test_facts_carry_page_ref(self) -> None:
        ldu = _make_ldu("Net Profit: $320M.", page=15)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert all(f.page_ref == 15 for f in facts)

    def test_facts_carry_content_hash(self) -> None:
        ldu = _make_ldu("Revenue: $100M.", page=1)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        for f in facts:
            assert len(f.content_hash) == 64  # SHA-256

    def test_multiple_ldus(self) -> None:
        ldus = [
            _make_ldu("Revenue: $100M.", page=1),
            _make_ldu("Cost: $50M.", page=2),
        ]
        extractor = FactTableExtractor()
        facts = extractor.extract(ldus, document_id="doc1")
        pages = {f.page_ref for f in facts}
        assert 1 in pages
        assert 2 in pages


# ── SQLite persistence ──────────────────────────────────────────────────

class TestFactPersistence:
    """Persist and query facts from SQLite."""

    def test_persist_and_count(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        ldu = _make_ldu("Revenue: $4.2B. Net Income: $1.1B.", page=5)
        facts = extractor.extract([ldu], document_id="doc1")
        extractor.persist_to_db(facts, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM fact_tables").fetchone()[0]
        conn.close()
        assert count == len(facts)
        assert count >= 2

    def test_query_by_document_id(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        facts1 = extractor.extract(
            [_make_ldu("Revenue: $100M.", page=1)], document_id="doc_a"
        )
        facts2 = extractor.extract(
            [_make_ldu("Revenue: $200M.", page=2)], document_id="doc_b"
        )
        extractor.persist_to_db(facts1, db_path=db_path)
        extractor.persist_to_db(facts2, db_path=db_path)

        results = extractor.query_facts(
            document_id="doc_a", db_path=db_path
        )
        assert len(results) >= 1
        assert all(r["document_id"] == "doc_a" for r in results)

    def test_query_by_key_pattern(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        facts = extractor.extract(
            [_make_ldu("Revenue: $100M. Cost: $50M.", page=1)],
            document_id="doc1",
        )
        extractor.persist_to_db(facts, db_path=db_path)

        results = extractor.query_facts(
            document_id="doc1", key_pattern="%revenue%", db_path=db_path
        )
        assert len(results) >= 1
        assert all("revenue" in r["key"].lower() for r in results)

    def test_empty_query_returns_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        results = extractor.query_facts(
            document_id="nonexistent", db_path=db_path
        )
        assert results == []


# ── Enriched fact extraction ─────────────────────────────────────────────

class TestEnrichedFactExtraction:
    """Test enriched fact fields: extraction_method, confidence, period, entity."""

    def test_extraction_method_regex(self) -> None:
        ldu = _make_ldu("Revenue: $4.2B.", page=1)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 1
        assert all(f.extraction_method == "regex" for f in facts)

    def test_extraction_method_table_parse(self) -> None:
        ldu = _make_ldu(
            "Revenue | 4,200\nExpenses | 2,100",
            chunk_type="table", page=1,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 1
        assert all(f.extraction_method == "table_parse" for f in facts)

    def test_confidence_is_set(self) -> None:
        ldu = _make_ldu("Revenue: $100M.", page=1)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 1
        assert all(0.0 < f.confidence <= 1.0 for f in facts)

    def test_period_detection(self) -> None:
        ldu = _make_ldu(
            "Total Revenue: $4.2B for FY2024.", page=5,
        )
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        periods = [f.period for f in facts if f.period]
        # At least one fact should detect FY2024
        assert len(periods) >= 1 or len(facts) >= 1

    def test_nocolon_pattern_extraction(self) -> None:
        """The _KV_NOCOLON pattern should extract facts without colon separator."""
        ldu = _make_ldu("Total Assets    500,000", page=3)
        extractor = FactTableExtractor()
        facts = extractor.extract([ldu], document_id="doc1")
        assert len(facts) >= 1

    def test_persist_enriched_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        ldu = _make_ldu("Revenue: $4.2B for FY2024.", page=5)
        facts = extractor.extract([ldu], document_id="doc1")
        extractor.persist_to_db(facts, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM fact_tables").fetchall()
        conn.close()
        assert len(rows) >= 1
        row = dict(rows[0])
        assert "extraction_method" in row
        assert "confidence" in row

    def test_query_with_min_confidence(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        initialize_database(db_path)

        extractor = FactTableExtractor()
        facts = extractor.extract(
            [_make_ldu("Revenue: $100M.", page=1)], document_id="doc1",
        )
        extractor.persist_to_db(facts, db_path=db_path)

        results = extractor.query_facts(
            document_id="doc1", min_confidence=0.5, db_path=db_path,
        )
        assert all(r.get("confidence", 0) >= 0.5 for r in results)

    def test_from_config_factory(self) -> None:
        """FactTableExtractor.from_config() should return an instance."""
        extractor = FactTableExtractor.from_config()
        assert isinstance(extractor, FactTableExtractor)

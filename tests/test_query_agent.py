"""TDD tests for the Query Agent — Stage 5 of the Document Intelligence Refinery.

Covers:
- Three tool functions: pageindex_navigate, semantic_search, structured_query
- ProvenanceChain construction on every answer
- Audit Mode (claim verification)
- LangGraph agent graph construction
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agents.fact_table import FactTableExtractor
from src.agents.pageindex import PageIndexBuilder
from src.agents.query_agent import QueryAgent
from src.db.init_db import initialize_database
from src.models.schemas import (
    BoundingBox,
    LDU,
    PageIndex,
    PageIndexNode,
    ProvenanceChain,
    ProvenanceCitation,
    QueryResult,
)
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


def _build_test_corpus(tmp_path: Path) -> tuple[QueryAgent, str]:
    """Build a small corpus with pageindex, vector store, and fact table.

    Returns the QueryAgent and the document_id.
    """
    doc_id = "test_doc_001"
    source_filename = "test_report.pdf"

    ldus = [
        _make_ldu(
            "The company reported Total Revenue: $4.2B for FY2024.",
            page=5,
            parent_section="Financial Highlights",
        ),
        _make_ldu(
            "Net Income: $1.1B, a 12% increase over the prior year.",
            page=5,
            parent_section="Financial Highlights",
        ),
        _make_ldu(
            "The Board of Directors approved a dividend of $0.50 per share.",
            page=8,
            parent_section="Governance",
        ),
        _make_ldu(
            "Capital expenditure projections for Q3 are estimated at $800M.",
            page=12,
            parent_section="Capital Expenditure",
        ),
        _make_ldu(
            "Revenue | 4,200 | 3,800\nExpenses | 2,100 | 1,900",
            chunk_type="table",
            page=15,
            parent_section="Income Statement",
        ),
    ]

    # Build PageIndex
    builder = PageIndexBuilder()
    pi = builder.build(ldus, source_filename=source_filename, document_id=doc_id)

    # Build fact table
    db_path = tmp_path / "test.db"
    initialize_database(db_path)
    fact_extractor = FactTableExtractor()
    facts = fact_extractor.extract(ldus, document_id=doc_id)
    fact_extractor.persist_to_db(facts, db_path=db_path)

    # Build vector store (use tmp dir)
    chroma_dir = tmp_path / "chroma_store"

    agent = QueryAgent(
        db_path=db_path,
        chroma_dir=chroma_dir,
    )
    # Ingest LDUs into vector store
    agent.ingest_ldus(ldus, document_id=doc_id, source_filename=source_filename)

    # Register the PageIndex
    agent.register_page_index(pi)

    return agent, doc_id


# ── Tool: pageindex_navigate ────────────────────────────────────────────

class TestPageIndexNavigate:
    """The pageindex_navigate tool traverses the PageIndex tree."""

    def test_finds_relevant_section(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        nodes = agent.pageindex_navigate(
            topic="capital expenditure", document_id=doc_id
        )
        assert len(nodes) >= 1
        titles = [n.title for n in nodes]
        assert any("capital" in t.lower() or "expenditure" in t.lower() for t in titles)

    def test_returns_empty_for_unknown_doc(self, tmp_path: Path) -> None:
        agent, _ = _build_test_corpus(tmp_path)
        nodes = agent.pageindex_navigate(
            topic="revenue", document_id="nonexistent"
        )
        assert nodes == []

    def test_respects_top_n(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        nodes = agent.pageindex_navigate(
            topic="financial", document_id=doc_id, top_n=2
        )
        assert len(nodes) <= 2


# ── Tool: semantic_search ───────────────────────────────────────────────

class TestSemanticSearch:
    """The semantic_search tool retrieves relevant chunks from ChromaDB."""

    def test_finds_relevant_chunks(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        results = agent.semantic_search(
            query="capital expenditure projections", n_results=3
        )
        assert len(results) >= 1
        # Results should contain document text
        assert any("capital" in r["content"].lower() or "expenditure" in r["content"].lower() for r in results)

    def test_results_have_metadata(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        results = agent.semantic_search(query="revenue", n_results=2)
        assert len(results) >= 1
        for r in results:
            assert "document_id" in r
            assert "page_number" in r
            assert "content" in r

    def test_empty_corpus_returns_empty(self, tmp_path: Path) -> None:
        db_path = tmp_path / "empty.db"
        initialize_database(db_path)
        agent = QueryAgent(db_path=db_path, chroma_dir=tmp_path / "empty_chroma")
        results = agent.semantic_search(query="anything", n_results=3)
        assert results == []


# ── Tool: structured_query ──────────────────────────────────────────────

class TestStructuredQuery:
    """The structured_query tool queries the SQLite fact table."""

    def test_finds_facts_by_key(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        results = agent.structured_query(
            document_id=doc_id, key_pattern="%Revenue%"
        )
        assert len(results) >= 1
        assert any("revenue" in r["key"].lower() for r in results)

    def test_returns_all_facts_for_document(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        results = agent.structured_query(document_id=doc_id)
        assert len(results) >= 2  # We have multiple facts

    def test_empty_for_unknown_doc(self, tmp_path: Path) -> None:
        agent, _ = _build_test_corpus(tmp_path)
        results = agent.structured_query(document_id="nonexistent")
        assert results == []


# ── Answer with Provenance ──────────────────────────────────────────────

class TestAnswerWithProvenance:
    """Every answer must include a ProvenanceChain."""

    def test_answer_returns_query_result(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        result = agent.answer(
            question="What is the total revenue?",
            document_id=doc_id,
        )
        assert isinstance(result, QueryResult)
        assert len(result.answer) > 0

    def test_provenance_has_citations(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        result = agent.answer(
            question="What is the capital expenditure for Q3?",
            document_id=doc_id,
        )
        assert isinstance(result.provenance, ProvenanceChain)
        assert len(result.provenance.citations) >= 1

    def test_citations_have_page_numbers(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        result = agent.answer(
            question="What is the total revenue?",
            document_id=doc_id,
        )
        for cit in result.provenance.citations:
            assert isinstance(cit, ProvenanceCitation)
            assert cit.page_number >= 1

    def test_tools_used_populated(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        result = agent.answer(
            question="revenue figures",
            document_id=doc_id,
        )
        assert len(result.tools_used) >= 1

    def test_provenance_query_matches_input(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        q = "What is the dividend per share?"
        result = agent.answer(question=q, document_id=doc_id)
        assert result.provenance.query == q


# ── Audit Mode ──────────────────────────────────────────────────────────

class TestAuditMode:
    """Audit Mode verifies claims against source or flags as unverifiable."""

    def test_verifiable_claim(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        chain = agent.audit(
            claim="The company reported Total Revenue of $4.2B for FY2024.",
            document_id=doc_id,
        )
        assert isinstance(chain, ProvenanceChain)
        assert chain.verified is True
        assert len(chain.citations) >= 1

    def test_unverifiable_claim(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        chain = agent.audit(
            claim="The company opened a new branch on Mars in 2025.",
            document_id=doc_id,
        )
        assert isinstance(chain, ProvenanceChain)
        assert chain.verified is False

    def test_audit_chain_contains_claim(self, tmp_path: Path) -> None:
        agent, doc_id = _build_test_corpus(tmp_path)
        claim = "Net Income was $1.1B."
        chain = agent.audit(claim=claim, document_id=doc_id)
        assert chain.query == claim


# ── LangGraph Agent Graph ───────────────────────────────────────────────

class TestAgentGraph:
    """The QueryAgent exposes a LangGraph-compatible graph."""

    def test_graph_is_constructable(self, tmp_path: Path) -> None:
        agent, _ = _build_test_corpus(tmp_path)
        graph = agent.build_graph()
        assert graph is not None

    def test_graph_has_tool_nodes(self, tmp_path: Path) -> None:
        agent, _ = _build_test_corpus(tmp_path)
        graph = agent.build_graph()
        # The graph should have nodes for the tools
        node_names = list(graph.nodes.keys())
        assert "tools" in node_names or "tool_node" in node_names or len(node_names) >= 2

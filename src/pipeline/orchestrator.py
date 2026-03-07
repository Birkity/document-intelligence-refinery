"""PipelineOrchestrator — run the full refinery on a single PDF.

Stages executed in order:
  1  Triage       → DocumentProfile
  2  Extract      → ExtractedDocument  (strategy A/B/C/D with escalation)
  3  Chunk        → list[LDU]          (semantic chunking + cross-refs)
  4  PageIndex    → PageIndex tree     (+ JSON artefact)
  5a FactTable    → list[Fact]         (hybrid regex / table_parse / LLM)
  5b EntityLink   → DocumentKnowledgeGraph (entity extraction + KG)
  5c Index        → SQLite + ChromaDB  (persistence & vector embeddings)

The ``sample_pages`` parameter enables a lightweight demo mode that
processes only N pages (head / mid / tail) while preserving real page
numbers for provenance.

All artefacts are written to ``.refinery/runs/{run_id}/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from src.agents.chunker import ChunkingEngine
from src.agents.entity_linker import EntityLinker
from src.agents.extractor import ExtractionRouter
from src.agents.fact_table import FactTableExtractor
from src.agents.pageindex import PageIndexBuilder
from src.agents.triage import TriageAgent
from src.db.repo import RefineryRepo
from src.db.vector_store import VectorStore
from src.models.schemas import (
    DocumentKnowledgeGraph,
    DocumentProfile,
    ExtractedDocument,
    Fact,
    LDU,
    PageIndex,
)

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RUNS_DIR = _PROJECT_ROOT / ".refinery" / "runs"


# ---------------------------------------------------------------------------
# Sample-page selection
# ---------------------------------------------------------------------------


def select_sample_pages(
    total_pages: int,
    n: int = 3,
    strategy: str = "head_mid_tail",
) -> list[int]:
    """Return up to *n* 1-indexed page numbers.

    Strategies
    ----------
    head_mid_tail
        Pages distributed across the head, middle, and tail thirds.
    head
        First *n* pages.
    uniform
        Evenly spaced pages across the document.
    random
        *n* pages chosen at random (without replacement).
    """
    import random as _random

    if total_pages <= n:
        return list(range(1, total_pages + 1))

    if strategy == "head":
        return list(range(1, min(n, total_pages) + 1))

    if strategy == "uniform":
        # linspace-style: spread n points across [1, total_pages]
        if n == 1:
            return [1]
        step = (total_pages - 1) / (n - 1)
        return sorted({round(1 + i * step) for i in range(n)})[:n]

    if strategy == "random":
        return sorted(_random.sample(range(1, total_pages + 1), n))

    # Default: head_mid_tail — divide doc into thirds and sample evenly within each
    third = total_pages // 3
    head_pages = list(range(1, third + 1))
    mid_pages = list(range(third + 1, 2 * third + 1))
    tail_pages = list(range(2 * third + 1, total_pages + 1))

    n_head = max(1, n // 3)
    n_tail = max(1, n // 3)
    n_mid = n - n_head - n_tail

    def _sample(pool: list[int], k: int) -> list[int]:
        if not pool:
            return []
        k = min(k, len(pool))
        step = max(1, len(pool) // k)
        return [pool[min(i * step, len(pool) - 1)] for i in range(k)]

    pages = _sample(head_pages, n_head) + _sample(mid_pages, n_mid) + _sample(tail_pages, n_tail)
    return sorted(set(pages))[:n]


# ---------------------------------------------------------------------------
# Pipeline result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated output from a single pipeline run."""

    run_id: str
    document_id: str
    profile: DocumentProfile
    extracted_doc: ExtractedDocument
    ldus: list[LDU]
    page_index: PageIndex
    facts: list[Fact]
    knowledge_graph: DocumentKnowledgeGraph | None
    ledger_entries: list[dict]
    sample_pages: list[int] | None
    elapsed_seconds: float = 0.0
    artefact_dir: Path | None = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class PipelineOrchestrator:
    """Run the full refinery pipeline end-to-end.

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.
    db_path : str | Path | None
        Path to SQLite database.
    chroma_dir : str | Path | None
        Path to ChromaDB persistence directory.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        db_path: str | Path | None = None,
        chroma_dir: str | Path | None = None,
    ) -> None:
        rp = Path(rules_path) if rules_path else None
        self._triage = TriageAgent(rules_path=rp)
        self._router = ExtractionRouter(rules_path=rp)
        self._chunker = ChunkingEngine(rules_path=rp)
        self._indexer = PageIndexBuilder()
        self._fact_extractor = FactTableExtractor.from_config()
        self._entity_linker = EntityLinker()
        self._repo = RefineryRepo(db_path=db_path)
        self._vector_store = VectorStore(persist_dir=chroma_dir)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        pdf_path: str | Path,
        *,
        sample_pages: int | None = None,
        page_sample_strategy: str = "head_mid_tail",
        explicit_pages: list[int] | None = None,
    ) -> PipelineResult:
        """Execute the full pipeline on *pdf_path*.

        Parameters
        ----------
        sample_pages : int | None
            If set, pick N pages using *page_sample_strategy*.
        page_sample_strategy : str
            One of ``head_mid_tail``, ``head``, ``uniform``, ``random``.
        explicit_pages : list[int] | None
            Exact 1-indexed page numbers to process.  Takes precedence over
            *sample_pages* when provided.

        Returns
        -------
        PipelineResult
        """
        t0 = time.perf_counter()
        run_id = uuid.uuid4().hex[:12]
        pdf = Path(pdf_path)

        log.info("Pipeline run %s — %s", run_id, pdf.name)

        # ── Stage 1: Triage ──────────────────────────────────────────
        profile: DocumentProfile = self._triage.generate_document_profile(str(pdf))
        doc_id = profile.document_id
        log.info("[%s] Triage → %s / %s / cost=%s",
                 doc_id[:8], profile.origin_type,
                 profile.layout_complexity,
                 profile.estimated_extraction_cost)

        # ── Determine pages to extract ───────────────────────────────
        page_nums: list[int] | None = None
        with pdfplumber.open(str(pdf)) as _p:
            total_pages = len(_p.pages)

        if explicit_pages is not None:
            # Clamp to valid range
            page_nums = sorted({p for p in explicit_pages if 1 <= p <= total_pages})
            log.info("[%s] Explicit pages → %s of %d", doc_id[:8], page_nums, total_pages)
        elif sample_pages is not None:
            page_nums = select_sample_pages(
                total_pages, n=sample_pages, strategy=page_sample_strategy
            )
            log.info("[%s] Sample mode → pages %s of %d", doc_id[:8], page_nums, total_pages)

        # ── Stage 2: Extract ─────────────────────────────────────────
        extracted_doc, ledger = self._router.route_and_extract(
            profile, str(pdf), page_numbers=page_nums,
        )
        log.info("[%s] Extract → %d pages, %d ledger entries",
                 doc_id[:8], len(extracted_doc.pages), len(ledger))

        # ── Stage 3: Chunk ───────────────────────────────────────────
        ldus = self._chunker.chunk_document(extracted_doc)
        log.info("[%s] Chunk → %d LDUs", doc_id[:8], len(ldus))

        # ── Stage 4: PageIndex ───────────────────────────────────────
        page_index = self._indexer.build(
            ldus,
            source_filename=pdf.name,
            document_id=doc_id,
        )
        # Persist PageIndex JSON to .refinery/pageindex/
        self._indexer.save_json(page_index)
        log.info("[%s] PageIndex → %d root nodes",
                 doc_id[:8], len(page_index.root_nodes))

        # ── Stage 5a: FactTable ──────────────────────────────────────
        facts = self._fact_extractor.extract(
            ldus, document_id=doc_id, origin=profile.origin_type
        )
        log.info("[%s] FactTable → %d facts", doc_id[:8], len(facts))

        # ── Stage 5b: Entity Linking & Knowledge Graph ───────────────
        # Collect cross-references from LDUs
        cross_refs = []
        for ldu in ldus:
            if hasattr(ldu, "cross_references") and ldu.cross_references:
                cross_refs.extend(ldu.cross_references)

        knowledge_graph = self._entity_linker.build_knowledge_graph(
            ldus=ldus,
            facts=facts,
            cross_references=cross_refs,
            document_id=doc_id,
        )
        log.info("[%s] KG → %d entities, %d edges",
                 doc_id[:8], len(knowledge_graph.entities),
                 len(knowledge_graph.edges))

        # ── Stage 5c: Persist to SQLite + ChromaDB ───────────────────
        self._persist(profile, extracted_doc, ldus, page_index, facts, ledger, total_pages)
        log.info("[%s] Persisted to SQLite + ChromaDB", doc_id[:8])

        # ── Write artefacts ──────────────────────────────────────────
        art_dir = self._write_artefacts(
            run_id, doc_id, profile, extracted_doc, ldus, page_index,
            facts, knowledge_graph, ledger,
        )

        elapsed = round(time.perf_counter() - t0, 3)
        log.info("[%s] Pipeline complete in %.2fs", doc_id[:8], elapsed)

        return PipelineResult(
            run_id=run_id,
            document_id=doc_id,
            profile=profile,
            extracted_doc=extracted_doc,
            ldus=ldus,
            page_index=page_index,
            facts=facts,
            knowledge_graph=knowledge_graph,
            ledger_entries=ledger,
            sample_pages=page_nums,
            elapsed_seconds=elapsed,
            artefact_dir=art_dir,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist(
        self,
        profile: DocumentProfile,
        doc: ExtractedDocument,
        ldus: list[LDU],
        index: PageIndex,
        facts: list[Fact],
        ledger: list[dict],
        page_count: int = 0,
    ) -> None:
        """Write everything to SQLite + ChromaDB."""
        did = profile.document_id

        # documents table
        self._repo.upsert_document(
            document_id=did,
            source_filename=profile.source_filename,
            origin_type=profile.origin_type,
            layout_complexity=profile.layout_complexity,
            domain_hint=profile.domain_hint,
            estimated_cost=profile.estimated_extraction_cost,
            page_count=page_count,
            total_chunks=len(ldus),
        )

        # chunks table + ChromaDB
        chunk_rows: list[dict] = []
        chroma_ids: list[str] = []
        chroma_docs: list[str] = []
        chroma_metas: list[dict[str, Any]] = []

        for i, ldu in enumerate(ldus):
            cid = f"{did}_{i:04d}"
            page_num = ldu.page_refs[0] if ldu.page_refs else 0
            chunk_rows.append({
                "chunk_id": cid,
                "document_id": did,
                "page_number": page_num,
                "chunk_type": ldu.chunk_type,
                "content": ldu.content,
                "content_hash": ldu.content_hash,
                "section_path": ldu.parent_section or "",
            })
            chroma_ids.append(cid)
            chroma_docs.append(ldu.content)
            chroma_metas.append({
                "document_id": did,
                "page_number": page_num,
                "chunk_type": ldu.chunk_type,
                "section_path": ldu.parent_section or "",
                "content_hash": ldu.content_hash,
            })

        if chunk_rows:
            self._repo.upsert_chunks_batch(chunk_rows)
            self._vector_store.add_chunks(chroma_ids, chroma_docs, chroma_metas)

        # page_indexes table
        tree_json = index.model_dump_json()
        self._repo.upsert_page_index(
            document_id=did,
            source_filename=doc.source_filename,
            tree_json=tree_json,
            node_count=len(index.root_nodes),
        )

        # fact_tables (enriched — use FactTableExtractor.persist_to_db for new cols)
        if facts:
            self._fact_extractor.persist_to_db(facts)

        # provenance ledger
        for entry in ledger:
            self._repo.append_provenance(
                document_id=did,
                action=f"extraction:{entry.get('strategy_used', 'unknown')}",
                metadata=entry,
            )

    # ------------------------------------------------------------------
    # Artefact helpers
    # ------------------------------------------------------------------

    def _write_artefacts(
        self,
        run_id: str,
        doc_id: str,
        profile: DocumentProfile,
        doc: ExtractedDocument,
        ldus: list[LDU],
        index: PageIndex,
        facts: list[Fact],
        knowledge_graph: DocumentKnowledgeGraph | None,
        ledger: list[dict],
    ) -> Path:
        """Write JSON artefacts to ``.refinery/runs/{run_id}/``."""
        art_dir = _RUNS_DIR / run_id
        art_dir.mkdir(parents=True, exist_ok=True)

        # Profile
        (art_dir / "profile.json").write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

        # Ledger
        (art_dir / "ledger.json").write_text(
            json.dumps(ledger, indent=2, default=str), encoding="utf-8"
        )

        # LDU summary
        ldu_summary = [
            {
                "chunk_type": l.chunk_type,
                "page_refs": l.page_refs,
                "parent_section": l.parent_section,
                "content_preview": l.content[:120],
                "content_hash": l.content_hash,
            }
            for l in ldus
        ]
        (art_dir / "ldus.json").write_text(
            json.dumps(ldu_summary, indent=2), encoding="utf-8"
        )

        # PageIndex
        (art_dir / "pageindex.json").write_text(
            index.model_dump_json(indent=2), encoding="utf-8"
        )

        # Facts
        fact_dicts = [f.model_dump() for f in facts]
        (art_dir / "facts.json").write_text(
            json.dumps(fact_dicts, indent=2, default=str), encoding="utf-8"
        )

        # Knowledge Graph
        if knowledge_graph:
            (art_dir / "knowledge_graph.json").write_text(
                knowledge_graph.model_dump_json(indent=2), encoding="utf-8"
            )

        log.info("Artefacts written to %s", art_dir)
        return art_dir

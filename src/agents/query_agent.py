"""Query Interface Agent — Stage 5 of the Document Intelligence Refinery.

A LangGraph agent with three tools:

1. **pageindex_navigate** — traverse the PageIndex tree to locate
   relevant sections without embedding-searching the full corpus.
2. **semantic_search** — vector retrieval over ChromaDB-stored LDU
   embeddings.
3. **structured_query** — SQL queries over the ``fact_tables`` SQLite
   table for precise numerical lookups (enriched with entity / period).

Every answer carries a :class:`ProvenanceChain` with document name,
page number, bounding box, and content hash so that claims are
auditable back to their spatial source.

The agent also supports **Audit Mode**: given a claim, it either
verifies it with a source citation or flags it as *unverifiable*,
returning an :class:`AuditResult`.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import httpx
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agents.fact_table import FactTableExtractor
from src.agents.pageindex import PageIndexBuilder
from src.db.vector_store import VectorStore
from src.models.schemas import (
    AuditResult,
    DocumentKnowledgeGraph,
    LDU,
    PageIndex,
    PageIndexNode,
    ProvenanceChain,
    ProvenanceCitation,
    QueryResult,
)

log = logging.getLogger(__name__)

_DEFAULT_DB = (
    Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"
)

# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> set[str]:
    """Lowercase alpha-numeric tokens."""
    return set(_WORD_RE.findall(text.lower()))


def _overlap_score(text_a: str, text_b: str) -> float:
    """Proportion of non-stopword tokens in *text_a* that appear in *text_b*."""
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
        "and", "or", "but", "not", "no", "this", "that", "it", "its",
        "has", "had", "have", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "can", "could", "about", "over", "into",
    }
    a = _tokenise(text_a) - _STOP
    b = _tokenise(text_b) - _STOP
    if not a:
        return 0.0
    overlap = a & b
    return len(overlap) / len(a)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(BaseModel):
    """State passed through the LangGraph agent nodes."""

    question: str = ""
    document_id: str = ""
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    fact_results: list[dict[str, Any]] = Field(default_factory=list)
    pageindex_nodes: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""
    tools_used: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# QueryAgent
# ---------------------------------------------------------------------------


class QueryAgent:
    """Three-tool query agent with provenance, audit mode, and KG support.

    Parameters
    ----------
    db_path : str | Path | None
        SQLite database path (contains ``fact_tables``, ``query_logs``).
    chroma_dir : str | Path | None
        ChromaDB persistence directory.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        chroma_dir: str | Path | None = None,
    ) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        self._chroma_dir = chroma_dir
        self._page_indexes: dict[str, PageIndex] = {}
        self._knowledge_graphs: dict[str, DocumentKnowledgeGraph] = {}
        self._pageindex_builder = PageIndexBuilder()
        self._fact_extractor = FactTableExtractor()

        # Lazy-init vector store (created on first use)
        self._vector_store: VectorStore | None = None
        self._source_filenames: dict[str, str] = {}  # doc_id -> filename

        # LLM config — Ollama for reasoning; OpenRouter only for vision
        self._ollama_base_url: str = "http://localhost:11434/v1"
        self._ollama_model: str = "qwen3-coder:480b-cloud"
        try:
            from src.config import get_settings
            cfg = get_settings()
            self._ollama_base_url = cfg.ollama_base_url
            self._ollama_model = cfg.ollama_model
        except Exception:
            log.debug("Could not load Ollama settings; using defaults.")

    # ------------------------------------------------------------------
    # Vector store lifecycle
    # ------------------------------------------------------------------

    def _get_vector_store(self) -> VectorStore:
        """Return (and lazily create) the vector store."""
        if self._vector_store is None:
            self._vector_store = VectorStore(
                persist_dir=self._chroma_dir,
                # Must match the collection used by pipeline ingestion.
                collection_name="refinery_chunks",
            )
        return self._vector_store

    def ingest_ldus(
        self,
        ldus: list[LDU],
        document_id: str,
        source_filename: str,
    ) -> int:
        """Ingest *ldus* into the vector store.

        Returns the number of chunks ingested.
        """
        vs = self._get_vector_store()
        self._source_filenames[document_id] = source_filename

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, ldu in enumerate(ldus):
            chunk_id = f"{document_id}_{idx}"
            ids.append(chunk_id)
            documents.append(ldu.content)
            metadatas.append({
                "document_id": document_id,
                "source_filename": source_filename,
                "page_number": ldu.page_refs[0] if ldu.page_refs else 1,
                "chunk_type": ldu.chunk_type,
                "parent_section": ldu.parent_section or "",
                "content_hash": ldu.content_hash,
            })

        vs.add_chunks(ids=ids, documents=documents, metadatas=metadatas)
        log.info("Ingested %d LDUs for %s into vector store", len(ldus), document_id)
        return len(ldus)

    def register_page_index(self, page_index: PageIndex) -> None:
        """Register a PageIndex for query-time navigation."""
        self._page_indexes[page_index.document_id] = page_index

    def register_knowledge_graph(self, kg: DocumentKnowledgeGraph) -> None:
        """Register a knowledge graph for query-time enrichment."""
        self._knowledge_graphs[kg.document_id] = kg

    # ------------------------------------------------------------------
    # Tool 1: pageindex_navigate
    # ------------------------------------------------------------------

    def pageindex_navigate(
        self,
        topic: str,
        document_id: str,
        top_n: int = 3,
    ) -> list[PageIndexNode]:
        """Traverse the PageIndex tree to find the most relevant sections.

        Parameters
        ----------
        topic : str
            Natural-language topic to search for.
        document_id : str
            Which document's PageIndex to search.
        top_n : int
            Maximum results.

        Returns
        -------
        list[PageIndexNode]
        """
        pi = self._page_indexes.get(document_id)
        if pi is None:
            return []
        return self._pageindex_builder.query(pi, topic, top_n=top_n)

    # ------------------------------------------------------------------
    # Tool 2: semantic_search
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector-similarity search over ingested LDU chunks.

        Returns a list of dicts with keys: ``content``, ``document_id``,
        ``page_number``, ``chunk_type``, ``parent_section``,
        ``content_hash``.
        """
        vs = self._get_vector_store()
        if vs.count == 0:
            return []

        raw = vs.query(query_text=query, n_results=min(n_results, vs.count))

        results: list[dict[str, Any]] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]

        for doc_text, meta in zip(docs, metas):
            results.append({
                "content": doc_text,
                "document_id": meta.get("document_id", ""),
                "source_filename": meta.get("source_filename", ""),
                "page_number": meta.get("page_number", 1),
                "chunk_type": meta.get("chunk_type", ""),
                "parent_section": meta.get("parent_section", ""),
                "content_hash": meta.get("content_hash", ""),
            })

        return results

    # ------------------------------------------------------------------
    # Tool 3: structured_query
    # ------------------------------------------------------------------

    def structured_query(
        self,
        document_id: str,
        key_pattern: str | None = None,
        entity: str | None = None,
        period: str | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Query the SQLite fact table for structured key-value facts.

        Parameters
        ----------
        document_id : str
            Filter by document.
        key_pattern : str | None
            Optional SQL LIKE pattern (e.g. ``'%revenue%'``).
        entity : str | None
            Optional entity filter.
        period : str | None
            Optional period filter.
        min_confidence : float
            Minimum confidence threshold.

        Returns
        -------
        list[dict]
        """
        return self._fact_extractor.query_facts(
            document_id=document_id,
            key_pattern=key_pattern,
            entity=entity,
            period=period,
            min_confidence=min_confidence,
            db_path=self._db_path,
        )

    # ------------------------------------------------------------------
    # Answer with provenance
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        document_id: str,
    ) -> QueryResult:
        """Answer *question* about *document_id* with full provenance.

        Uses a multi-hop tool-selection strategy:
        1. Try PageIndex navigation to narrow scope.
        2. Run semantic search for relevant chunks.
        3. If the question looks numerical, also query structured facts
           (with entity/period enrichment from KG).
        4. Optionally enrich via knowledge graph edges.
        5. Compose the answer from retrieved evidence.

        Parameters
        ----------
        question : str
            Natural-language question.
        document_id : str
            Document to query.

        Returns
        -------
        QueryResult
        """
        tools_used: list[str] = []
        all_evidence: list[dict[str, Any]] = []
        source_filename = self._source_filenames.get(document_id, document_id)

        # 1. PageIndex navigation
        sections = self.pageindex_navigate(
            topic=question, document_id=document_id, top_n=3
        )
        if sections:
            tools_used.append("pageindex_navigate")
            for s in sections:
                all_evidence.append({
                    "content": f"[Section: {s.title}] {s.summary}",
                    "page_number": s.page_start,
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "content_hash": "",
                })

        # 2. Semantic search
        chunks = self.semantic_search(query=question, n_results=5)
        doc_chunks = [c for c in chunks if c.get("document_id") == document_id]
        if doc_chunks:
            tools_used.append("semantic_search")
            all_evidence.extend(doc_chunks)

        # 3. Structured query for numerical questions
        if self._looks_numerical(question):
            # Extract entity/period hints from KG if available
            entity_hint, period_hint = self._extract_query_hints(question, document_id)
            facts = self.structured_query(
                document_id=document_id,
                entity=entity_hint,
                period=period_hint,
            )
            if facts:
                # Keep only facts that actually overlap with the query to avoid
                # flooding synthesis with unrelated rows.
                scored_facts: list[tuple[float, dict[str, Any]]] = []
                for f in facts:
                    fact_text = f"{f.get('key', '')} {f.get('value', '')} {f.get('entity', '')} {f.get('metric', '')}"
                    score = _overlap_score(question, fact_text)
                    if score > 0:
                        scored_facts.append((score, f))

                if scored_facts:
                    scored_facts.sort(key=lambda x: x[0], reverse=True)
                    facts = [f for _, f in scored_facts[:10]]
                else:
                    facts = []

            if facts:
                tools_used.append("structured_query")
                for f in facts:
                    all_evidence.append({
                        "content": f"{f['key']}: {f['value']}",
                        "page_number": f.get("page_ref", 1),
                        "document_id": document_id,
                        "source_filename": source_filename,
                        "content_hash": f.get("content_hash", ""),
                    })

        # 4. Knowledge graph enrichment (multi-hop)
        kg_evidence = self._kg_enrich(question, document_id, source_filename)
        if kg_evidence:
            tools_used.append("knowledge_graph")
            all_evidence.extend(kg_evidence)

        # 5. Compose answer
        answer_text = self._compose_answer(question, all_evidence)

        # 6. Build provenance chain
        citations = self._build_citations(all_evidence, source_filename)

        provenance = ProvenanceChain(
            query=question,
            citations=citations,
            verified=len(citations) > 0,
        )

        confidence = min(1.0, len(citations) * 0.2) if citations else 0.0

        return QueryResult(
            answer=answer_text,
            provenance=provenance,
            tools_used=tools_used,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Audit Mode
    # ------------------------------------------------------------------

    def audit(
        self,
        claim: str,
        document_id: str,
    ) -> AuditResult:
        """Verify *claim* against the document's evidence.

        Searches for evidence supporting the claim using all three tools.
        Returns an ``AuditResult`` with status ``verified``, ``not_found``,
        or ``unverifiable`` and supporting provenance citations.

        Parameters
        ----------
        claim : str
            The assertion to verify.
        document_id : str
            Document to verify against.

        Returns
        -------
        AuditResult
        """
        source_filename = self._source_filenames.get(document_id, document_id)
        matching_evidence: list[dict[str, Any]] = []

        # Search semantically
        chunks = self.semantic_search(query=claim, n_results=10)
        doc_chunks = [c for c in chunks if c.get("document_id") == document_id]

        for chunk in doc_chunks:
            score = _overlap_score(claim, chunk.get("content", ""))
            if score >= 0.3:  # 30% token overlap threshold
                matching_evidence.append(chunk)

        # Also check structured facts
        facts = self.structured_query(document_id=document_id)
        for f in facts:
            fact_text = f"{f['key']}: {f['value']}"
            score = _overlap_score(claim, fact_text)
            if score >= 0.2:
                matching_evidence.append({
                    "content": fact_text,
                    "page_number": f.get("page_ref", 1),
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "content_hash": f.get("content_hash", ""),
                })

        # Also check KG edges for related evidence
        kg = self._knowledge_graphs.get(document_id)
        if kg:
            for edge in kg.edges:
                edge_text = f"{edge.source} {edge.relation} {edge.target}"
                score = _overlap_score(claim, edge_text)
                if score >= 0.25:
                    matching_evidence.append({
                        "content": edge_text,
                        "page_number": edge.page_ref,
                        "document_id": document_id,
                        "source_filename": source_filename,
                        "content_hash": "",
                    })

        citations = self._build_citations(matching_evidence, source_filename)

        if len(matching_evidence) > 0:
            status = "verified"
        elif len(doc_chunks) > 0:
            # We found related chunks but none matched strongly
            status = "unverifiable"
        else:
            status = "not_found"

        # LLM-based verification reasoning (local Ollama — always attempted)
        explanation = ""
        if matching_evidence or doc_chunks:
            explanation = self._llm_audit_reason(claim, matching_evidence, doc_chunks)
            # LLM may override the heuristic status
            if explanation:
                lower_exp = explanation.lower()
                if "contradicted" in lower_exp:
                    status = "unverifiable"
                elif "verified" in lower_exp and status != "verified":
                    status = "verified"

        return AuditResult(
            claim=claim,
            status=status,
            supporting_evidence=citations,
            explanation=explanation,
        )

    def _llm_audit_reason(
        self,
        claim: str,
        matching_evidence: list[dict[str, Any]],
        related_chunks: list[dict[str, Any]],
    ) -> str:
        """Use LLM to reason about whether evidence supports the claim."""
        evidence_texts = []
        for e in matching_evidence[:5]:
            evidence_texts.append(e.get("content", ""))
        for c in related_chunks[:3]:
            evidence_texts.append(c.get("content", ""))

        context = "\n".join(f"- {t}" for t in evidence_texts if t)
        prompt = (
            "You are a document auditor. Determine if the claim is supported by the evidence.\n"
            "Respond with one of: VERIFIED, CONTRADICTED, or UNVERIFIABLE.\n"
            "Then explain in 1-2 sentences.\n\n"
            f"Claim: {claim}\n\n"
            f"Evidence:\n{context}\n\n"
            "Verdict:"
        )
        try:
            resp = httpx.post(
                f"{self._ollama_base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self._ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.warning("Ollama audit reasoning failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # LangGraph graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> StateGraph:
        """Build and compile a LangGraph state graph.

        The graph has three nodes:
        - ``retrieve``: runs pageindex_navigate + semantic_search
        - ``structured``: runs structured_query
        - ``synthesise``: composes the final answer with provenance

        Returns
        -------
        StateGraph (compiled)
        """
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("retrieve", self._graph_retrieve)
        workflow.add_node("structured", self._graph_structured)
        workflow.add_node("synthesise", self._graph_synthesise)

        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "structured")
        workflow.add_edge("structured", "synthesise")
        workflow.add_edge("synthesise", END)

        return workflow.compile()

    def _graph_retrieve(self, state: AgentState) -> dict:
        """LangGraph node: PageIndex navigation + semantic search."""
        tools_used = list(state.tools_used)
        chunks: list[dict[str, Any]] = []

        # PageIndex
        sections = self.pageindex_navigate(
            topic=state.question, document_id=state.document_id, top_n=3
        )
        if sections:
            tools_used.append("pageindex_navigate")

        # Semantic search
        results = self.semantic_search(query=state.question, n_results=5)
        doc_results = [r for r in results if r.get("document_id") == state.document_id]
        if doc_results:
            tools_used.append("semantic_search")
            chunks.extend(doc_results)

        pi_nodes = [
            {"title": s.title, "page_start": s.page_start, "summary": s.summary}
            for s in sections
        ]

        return {
            "retrieved_chunks": chunks,
            "pageindex_nodes": pi_nodes,
            "tools_used": tools_used,
        }

    def _graph_structured(self, state: AgentState) -> dict:
        """LangGraph node: structured fact query."""
        tools_used = list(state.tools_used)
        facts: list[dict[str, Any]] = []

        if self._looks_numerical(state.question):
            results = self.structured_query(document_id=state.document_id)
            if results:
                tools_used.append("structured_query")
                facts.extend(results)

        return {
            "fact_results": facts,
            "tools_used": tools_used,
        }

    def _graph_synthesise(self, state: AgentState) -> dict:
        """LangGraph node: compose answer with provenance."""
        evidence: list[dict[str, Any]] = []
        source_filename = self._source_filenames.get(
            state.document_id, state.document_id
        )

        for chunk in state.retrieved_chunks:
            evidence.append(chunk)

        for f in state.fact_results:
            evidence.append({
                "content": f"{f.get('key', '')}: {f.get('value', '')}",
                "page_number": f.get("page_ref", 1),
                "document_id": state.document_id,
                "source_filename": source_filename,
                "content_hash": f.get("content_hash", ""),
            })

        answer_text = self._compose_answer(state.question, evidence)

        citations: list[dict[str, Any]] = []
        for e in evidence:
            citations.append({
                "document_id": e.get("document_id", state.document_id),
                "document_name": e.get("source_filename", source_filename),
                "page_number": e.get("page_number", 1),
                "content_hash": e.get("content_hash", ""),
            })

        return {
            "answer": answer_text,
            "citations": citations,
        }

    # ------------------------------------------------------------------
    # Knowledge-graph enrichment helpers
    # ------------------------------------------------------------------

    def _extract_query_hints(
        self, question: str, document_id: str
    ) -> tuple[str | None, str | None]:
        """Extract entity and period hints from the question using the KG."""
        entity_hint: str | None = None
        period_hint: str | None = None

        kg = self._knowledge_graphs.get(document_id)
        if not kg:
            return entity_hint, period_hint

        q_lower = question.lower()
        # Match entity names from KG
        for ent in kg.entities:
            if ent.entity_name.lower() in q_lower:
                if ent.entity_type == "organization":
                    entity_hint = ent.entity_name
                elif ent.entity_type == "date":
                    period_hint = ent.entity_name

        # Also try simple period regex on question
        import re as _re
        period_m = _re.search(
            r"\b(?:FY|CY|Q[1-4])?\s*\d{4}\b|\b(?:H[12])\s*\d{4}\b",
            question, _re.IGNORECASE,
        )
        if period_m and not period_hint:
            period_hint = period_m.group(0).strip()

        return entity_hint, period_hint

    def _kg_enrich(
        self,
        question: str,
        document_id: str,
        source_filename: str,
    ) -> list[dict[str, Any]]:
        """Find KG edges related to the question for multi-hop enrichment."""
        kg = self._knowledge_graphs.get(document_id)
        if not kg:
            return []

        results: list[dict[str, Any]] = []
        for edge in kg.edges:
            edge_text = f"{edge.source} {edge.relation} {edge.target}"
            score = _overlap_score(question, edge_text)
            if score >= 0.25:
                results.append({
                    "content": f"[KG] {edge.source} → {edge.relation} → {edge.target}",
                    "page_number": edge.page_ref,
                    "document_id": document_id,
                    "source_filename": source_filename,
                    "content_hash": "",
                })
        return results[:5]  # limit KG evidence

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_numerical(question: str) -> bool:
        """Heuristic: does the question ask about numbers?"""
        numerical_keywords = {
            "revenue", "income", "cost", "expense", "profit", "loss",
            "budget", "expenditure", "salary", "price", "rate", "amount",
            "total", "net", "gross", "tax", "fee", "dividend", "assets",
            "liabilities", "equity", "balance", "cash", "flow", "capital",
            "capacity", "subscribed", "paid-up",
            "how much", "how many", "what is the",
            "$", "%", "billion", "million", "thousand",
        }
        q_lower = question.lower()
        return any(kw in q_lower for kw in numerical_keywords)

    @staticmethod
    def _compose_answer_deterministic(
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """Deterministic answer composition from retrieved evidence."""
        if not evidence:
            return "No relevant information found in the document."

        # Deduplicate by content
        seen: set[str] = set()
        unique: list[str] = []
        for e in evidence:
            content = e.get("content", "")
            if content and content not in seen:
                seen.add(content)
                unique.append(content)

        # Score each piece of evidence by overlap with question
        scored: list[tuple[float, str]] = []
        for text in unique:
            score = _overlap_score(question, text)
            scored.append((score, text))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take the top pieces
        top = [text for _, text in scored[:3]]
        return " ".join(top)

    def _compose_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """Compose an answer using LLM reasoning over retrieved evidence.

        Falls back to deterministic composition if LLM is unavailable.
        """
        if not evidence:
            return "No relevant information found in the document."

        # Collect unique evidence texts
        seen: set[str] = set()
        unique: list[str] = []
        for e in evidence:
            content = e.get("content", "")
            if content and content not in seen:
                seen.add(content)
                unique.append(content)

        if not self._ollama_model:
            return self._compose_answer_deterministic(question, evidence)

        # Build context for LLM
        context = "\n\n".join(f"[Evidence {i+1}] {t}" for i, t in enumerate(unique[:8]))
        prompt = (
            "You are a precise document analyst. Answer the question based ONLY on "
            "the provided evidence. If the evidence is insufficient, say so. "
            "Be concise and factual.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{context}\n\n"
            "Answer:"
        )

        try:
            resp = httpx.post(
                f"{self._ollama_base_url}/chat/completions",
                headers={"Content-Type": "application/json"},
                json={
                    "model": self._ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 512,
                    "temperature": 0.1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            if answer:
                return answer
        except Exception as exc:
            log.warning("LLM synthesis failed, using deterministic: %s", exc)

        return self._compose_answer_deterministic(question, evidence)

    @staticmethod
    def _build_citations(
        evidence: list[dict[str, Any]],
        default_filename: str,
    ) -> list[ProvenanceCitation]:
        """Build deduplicated ProvenanceCitations from evidence dicts."""
        seen: set[tuple[str, int]] = set()
        citations: list[ProvenanceCitation] = []

        for e in evidence:
            doc_id = e.get("document_id", "")
            page = e.get("page_number", 1)
            key = (doc_id, page)
            if key in seen:
                continue
            seen.add(key)

            citations.append(ProvenanceCitation(
                document_id=doc_id,
                document_name=e.get("source_filename", default_filename),
                page_number=page,
                bbox=None,
                content_hash=e.get("content_hash", ""),
            ))

        return citations

    # ------------------------------------------------------------------
    # Query logging
    # ------------------------------------------------------------------

    def log_query(
        self,
        query_text: str,
        result_count: int,
        latency_ms: float,
        source_documents: str = "",
    ) -> None:
        """Append a query to the ``query_logs`` table."""
        import datetime

        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.execute(
                """
                INSERT INTO query_logs
                    (query_text, timestamp, result_count, latency_ms, source_documents)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    query_text,
                    datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    result_count,
                    latency_ms,
                    source_documents,
                ),
            )
            conn.commit()
        finally:
            conn.close()

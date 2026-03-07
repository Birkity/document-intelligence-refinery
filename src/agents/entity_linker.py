"""Entity Linking & Knowledge Graph Builder.

Extracts named entities (organisations, persons, metrics, dates, amounts,
locations, regulations) from LDUs and facts, resolves cross-references,
and builds a lightweight ``DocumentKnowledgeGraph`` for retrieval enrichment.

Uses regex/heuristic NER (no heavy ML models) plus optional LLM assist.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Sequence

from src.models.schemas import (
    CrossReference,
    DocumentKnowledgeGraph,
    EntityMention,
    Fact,
    KnowledgeGraphEdge,
    LDU,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Entity extraction patterns
# ---------------------------------------------------------------------------

# Organisation-like patterns (capitalised multi-word or known suffixes)
_ORG_PATTERN = re.compile(
    r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)"   # Multi-word proper noun
    r"(?:\s+(?:Inc|Corp|Ltd|LLC|PLC|Group|Bank|Authority|Commission|Bureau|Fund|"
    r"Association|Institute|Board|Agency|Ministry|Department))?"
    r"\b"
)

# Date/period patterns
_DATE_PATTERN = re.compile(
    r"\b(?:"
    r"(?:FY|CY|Q[1-4])?\s*\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|(?:H[12])\s*\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)

# Currency amounts
_AMOUNT_PATTERN = re.compile(
    r"[\$€£₤]\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|thousand|[BMKTbmkt])?"
    r"|\d[\d,]*(?:\.\d+)?\s*(?:million|billion|thousand)\s*(?:USD|EUR|GBP|ETB|[A-Z]{3})",
    re.IGNORECASE,
)

# Metric keywords
_METRIC_KEYWORDS = {
    "revenue", "income", "profit", "loss", "ebitda", "margin", "ratio",
    "growth", "return", "yield", "cost", "expense", "asset", "liability",
    "equity", "dividend", "earnings", "cash flow", "capital", "debt",
    "turnover", "depreciation", "amortization", "interest", "tax",
    "net worth", "book value", "market cap", "share price", "eps",
    "roe", "roa", "roc", "npat", "pbt", "pat",
}

# Regulation patterns
_REGULATION_PATTERN = re.compile(
    r"\b(?:IFRS|GAAP|Basel\s+[IV]+|IAS|ISA|SOX|Dodd-Frank|MiFID|"
    r"Regulation\s+[A-Z]|Directive\s+\d+/\d+|"
    r"Proclamation\s+No\.?\s*\d+|Circular\s+No\.?\s*\d+)\b",
    re.IGNORECASE,
)

# Person names — simplified: Title + Name pattern
_PERSON_PATTERN = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|CEO|CFO|CTO|Chairman|President|Director)"
    r"\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
)

# Location (simplified: Country / City heuristic)
_LOCATION_KEYWORDS = {
    "ethiopia", "addis ababa", "kenya", "nairobi", "nigeria", "lagos",
    "south africa", "johannesburg", "london", "new york", "washington",
    "beijing", "shanghai", "tokyo", "berlin", "paris", "zurich",
    "united states", "united kingdom", "european union", "africa",
}


# ═══════════════════════════════════════════════════════════════════════════
# Entity Linker
# ═══════════════════════════════════════════════════════════════════════════

class EntityLinker:
    """Extract entities and build a document knowledge graph.

    Usage
    -----
    >>> linker = EntityLinker()
    >>> kg = linker.build_knowledge_graph(
    ...     ldus=ldus,
    ...     facts=facts,
    ...     cross_references=cross_refs,
    ...     document_id="doc-1",
    ... )
    """

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(
        self,
        ldus: list[LDU],
        facts: list[Fact] | None = None,
    ) -> list[EntityMention]:
        """Extract entities from LDUs and (optionally) enriched facts.

        Returns deduplicated ``EntityMention`` objects with mention counts.
        """
        # Accumulate raw mentions: {(canonical, type): [mention_dicts]}
        raw: dict[tuple[str, str], list[dict]] = defaultdict(list)

        for ldu in ldus:
            page = ldu.page_refs[0] if ldu.page_refs else 1
            section = ""
            chash = ldu.content_hash or ""
            text = ldu.content

            mention_ctx = {"page": page, "section": section, "content_hash": chash}

            # Organisations
            for m in _ORG_PATTERN.finditer(text):
                name = m.group(0).strip()
                if len(name) < 4 or name.lower() in _METRIC_KEYWORDS:
                    continue
                raw[(_canonical(name), "organization")].append(mention_ctx)

            # Dates/periods
            for m in _DATE_PATTERN.finditer(text):
                raw[(_canonical(m.group(0)), "date")].append(mention_ctx)

            # Amounts
            for m in _AMOUNT_PATTERN.finditer(text):
                raw[(_canonical(m.group(0)), "amount")].append(mention_ctx)

            # Metrics (keyword-based in context)
            text_lower = text.lower()
            for kw in _METRIC_KEYWORDS:
                if kw in text_lower:
                    raw[(kw.title(), "metric")].append(mention_ctx)

            # Regulations
            for m in _REGULATION_PATTERN.finditer(text):
                raw[(_canonical(m.group(0)), "regulation")].append(mention_ctx)

            # Persons
            for m in _PERSON_PATTERN.finditer(text):
                raw[(_canonical(m.group(0)), "person")].append(mention_ctx)

            # Locations
            for loc in _LOCATION_KEYWORDS:
                if loc in text_lower:
                    raw[(loc.title(), "location")].append(mention_ctx)

        # Add entities from facts
        if facts:
            for f in facts:
                if f.entity:
                    raw[(_canonical(f.entity), "organization")].append(
                        {"page": f.page_ref, "section": f.section, "content_hash": f.content_hash}
                    )
                if f.metric:
                    raw[(_canonical(f.metric), "metric")].append(
                        {"page": f.page_ref, "section": f.section, "content_hash": f.content_hash}
                    )
                if f.period:
                    raw[(_canonical(f.period), "date")].append(
                        {"page": f.page_ref, "section": f.section, "content_hash": f.content_hash}
                    )

        # Build EntityMention objects
        entities: list[EntityMention] = []
        for (name, etype), mentions in raw.items():
            if not name.strip():
                continue
            entities.append(EntityMention(
                entity_name=name,
                entity_type=etype,  # type: ignore[arg-type]
                mentions=mentions,
                aliases=[],
            ))

        log.info("Extracted %d unique entities from %d LDUs", len(entities), len(ldus))
        return entities

    # ------------------------------------------------------------------
    # Knowledge graph construction
    # ------------------------------------------------------------------

    def build_knowledge_graph(
        self,
        ldus: list[LDU],
        facts: list[Fact] | None = None,
        cross_references: list[CrossReference] | None = None,
        document_id: str = "",
    ) -> DocumentKnowledgeGraph:
        """Build a lightweight knowledge graph for a document.

        Nodes are entities; edges represent co-occurrence, fact
        relationships, and cross-references.
        """
        entities = self.extract_entities(ldus, facts)
        edges: list[KnowledgeGraphEdge] = []

        # ── Fact-based edges ─────────────────────────────────────────
        if facts:
            for f in facts:
                if f.entity and f.metric:
                    edges.append(KnowledgeGraphEdge(
                        source=_canonical(f.entity),
                        target=_canonical(f.metric),
                        relation="has_metric",
                        page_ref=f.page_ref,
                        confidence=f.confidence,
                    ))
                if f.metric and f.period:
                    edges.append(KnowledgeGraphEdge(
                        source=_canonical(f.metric),
                        target=_canonical(f.period),
                        relation="reported_for_period",
                        page_ref=f.page_ref,
                        confidence=f.confidence,
                    ))

        # ── Co-occurrence edges (entities on same page) ──────────────
        page_entities: dict[int, list[str]] = defaultdict(list)
        for e in entities:
            for mention in e.mentions:
                pg = mention.get("page", 0)
                if pg:
                    page_entities[pg].append(e.entity_name)

        for pg, ent_names in page_entities.items():
            unique = list(set(ent_names))
            for i in range(len(unique)):
                for j in range(i + 1, min(len(unique), i + 6)):  # limit pairs
                    edges.append(KnowledgeGraphEdge(
                        source=unique[i],
                        target=unique[j],
                        relation="co_occurs_on_page",
                        page_ref=pg,
                        confidence=0.5,
                    ))

        # ── Cross-reference edges ────────────────────────────────────
        cr_list = cross_references or []
        for cr in cr_list:
            edges.append(KnowledgeGraphEdge(
                source=f"page:{cr.source_page}",
                target=cr.target_label,
                relation=f"references_{cr.target_type}",
                page_ref=cr.source_page,
                confidence=0.9 if cr.resolved else 0.4,
            ))

        # Deduplicate edges
        seen: set[tuple[str, str, str]] = set()
        deduped_edges: list[KnowledgeGraphEdge] = []
        for e in edges:
            key = (e.source, e.target, e.relation)
            if key not in seen:
                seen.add(key)
                deduped_edges.append(e)

        kg = DocumentKnowledgeGraph(
            document_id=document_id,
            entities=entities,
            edges=deduped_edges,
            cross_references=cr_list,
        )

        log.info(
            "Built knowledge graph: %d entities, %d edges, %d cross-refs (doc=%s)",
            len(entities), len(deduped_edges), len(cr_list), document_id,
        )
        return kg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical(text: str) -> str:
    """Normalise entity name to a canonical form."""
    return " ".join(text.strip().split()).title()

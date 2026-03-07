"""Pydantic v2 schemas for the Document Intelligence Refinery.

All core data models used across the five-stage pipeline are defined here.
Models are ordered by dependency — leaf models first, composite models last.
No implementation logic — pure data contracts only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Spatial coordinates of a region on a specific page."""

    x1: float = Field(..., description="Left edge x-coordinate")
    y1: float = Field(..., description="Top edge y-coordinate")
    x2: float = Field(..., description="Right edge x-coordinate")
    y2: float = Field(..., description="Bottom edge y-coordinate")
    page_number: int = Field(..., ge=1, description="1-indexed page number")


# ---------------------------------------------------------------------------
# Extraction primitives
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """A contiguous block of text with its spatial location."""

    content: str = Field(..., description="Raw text content of the block")
    bbox: BoundingBox = Field(..., description="Bounding box on the source page")


class TableObject(BaseModel):
    """A structured table extracted from a document page."""

    headers: list[str] = Field(
        ..., description="Column header labels in reading order"
    )
    rows: list[list[str]] = Field(
        ..., description="Row data; each inner list aligns with headers"
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the full table")
    table_id: str = Field(
        default="", description="Table identifier (e.g. 'Table 1')"
    )
    caption: str = Field(
        default="", description="Table caption if detected"
    )
    parent_table_id: str = Field(
        default="",
        description="If this is a sub-group of a larger table, link to parent",
    )
    subgroup_label: str = Field(
        default="",
        description="Label of the subgroup (e.g. year, region, category)",
    )


class FigureObject(BaseModel):
    """A figure (image/chart/diagram) extracted from a document page."""

    caption: str = Field(
        default="",
        description="Figure caption text (may be empty if not detected)",
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the figure")
    figure_type: str = Field(
        default="image",
        description="Type hint: image, chart, diagram, photograph, etc.",
    )
    figure_id: str = Field(
        default="", description="Figure identifier (e.g. 'Figure 1')"
    )


# ---------------------------------------------------------------------------
# Page / Document containers
# ---------------------------------------------------------------------------

class ExtractedPage(BaseModel):
    """Extraction results for a single page of a document."""

    page_number: int = Field(..., ge=1, description="1-indexed page number")
    text_blocks: list[TextBlock] = Field(
        default_factory=list,
        description="Text blocks extracted from this page",
    )
    tables: list[TableObject] = Field(
        default_factory=list,
        description="Tables extracted from this page",
    )
    figures: list[FigureObject] = Field(
        default_factory=list,
        description="Figures/charts extracted from this page",
    )
    reading_order: list[int] = Field(
        default_factory=list,
        description=(
            "Indices into a combined list of text_blocks + tables + figures "
            "giving the intended reading sequence on this page"
        ),
    )
    extraction_strategy: str = Field(
        default="", description="Strategy used for this page"
    )
    extraction_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence of extraction for this page",
    )


class ExtractedDocument(BaseModel):
    """Normalized extraction output for an entire document."""

    document_id: str = Field(
        ..., description="Unique identifier for the source document"
    )
    source_filename: str = Field(
        ..., description="Original PDF filename (e.g. 'report.pdf')"
    )
    pages: list[ExtractedPage] = Field(
        default_factory=list,
        description="Per-page extraction results in page order",
    )
    strategies_used: list[str] = Field(
        default_factory=list,
        description="List of extraction strategies applied",
    )


# ---------------------------------------------------------------------------
# Triage / classification
# ---------------------------------------------------------------------------

class DocumentProfile(BaseModel):
    """Classification output produced by the Triage Agent (Stage 1)."""

    document_id: str = Field(
        ..., description="Unique identifier for the source document"
    )
    source_filename: str = Field(
        ..., description="Original PDF filename (e.g. 'report.pdf')"
    )
    origin_type: Literal[
        "native_digital", "scanned_image", "mixed", "form_fillable"
    ] = Field(
        ...,
        description="How the PDF was created — determines base extraction approach",
    )
    layout_complexity: Literal[
        "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
    ] = Field(..., description="Dominant layout pattern of the document")
    language: str = Field(
        ..., description="ISO 639-1 language code (e.g. 'en', 'am')"
    )
    language_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence of language detection [0, 1]"
    )
    domain_hint: Literal[
        "financial", "legal", "technical", "medical", "general"
    ] = Field(
        ...,
        description="Subject-matter domain — selects extraction prompt strategy",
    )
    estimated_extraction_cost: Literal[
        "fast_text_sufficient", "needs_layout_model", "needs_vision_model"
    ] = Field(
        ...,
        description="Estimated computational tier needed for acceptable extraction",
    )
    page_count: int = Field(
        default=0, ge=0, description="Total page count of the document"
    )
    has_ocr_pages: bool = Field(
        default=False,
        description="Whether any pages appear to need OCR",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall triage confidence",
    )


# ---------------------------------------------------------------------------
# Cross-reference
# ---------------------------------------------------------------------------

class CrossReference(BaseModel):
    """A detected cross-reference within a document."""

    source_page: int = Field(..., ge=1, description="Page where the reference appears")
    source_text: str = Field(..., description="The reference text (e.g. 'see Table 3')")
    target_type: Literal[
        "table", "figure", "section", "appendix", "note", "chart", "unknown"
    ] = Field(..., description="Type of referenced element")
    target_label: str = Field(
        ..., description="Label of target (e.g. 'Table 3', 'Section 2.1')"
    )
    target_page: int | None = Field(
        default=None, description="Resolved target page (None if unresolved)"
    )
    resolved: bool = Field(
        default=False, description="Whether the target has been resolved"
    )


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class LDU(BaseModel):
    """Logical Document Unit — a RAG-ready, semantically coherent chunk."""

    content: str = Field(..., description="Chunk text content")
    chunk_type: Literal[
        "paragraph", "table", "figure", "list", "section"
    ] = Field(..., description="Semantic type of this chunk")
    page_refs: list[int] = Field(
        ..., description="1-indexed page numbers this chunk spans"
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Bounding box (None if chunk spans multiple regions)",
    )
    parent_section: str | None = Field(
        default=None,
        description="Title of the parent section for hierarchical context",
    )
    token_count: int = Field(
        default=0, ge=0, description="Approximate token count of content"
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 hex digest of content for provenance verification",
    )
    cross_references: list[CrossReference] = Field(
        default_factory=list,
        description="Cross-references detected in this chunk",
    )
    table_id: str = Field(
        default="", description="Table ID if this is a table chunk"
    )
    figure_id: str = Field(
        default="", description="Figure ID if this is a figure chunk"
    )
    subgroup_label: str = Field(
        default="",
        description="Subgroup label for table fragments",
    )
    parent_table_id: str = Field(
        default="",
        description="Parent table ID for sub-grouped table chunks",
    )


# ---------------------------------------------------------------------------
# PageIndex — hierarchical navigation
# ---------------------------------------------------------------------------

class PageIndexNode(BaseModel):
    """A node in the hierarchical PageIndex tree."""

    title: str = Field(..., description="Section title")
    page_start: int = Field(..., ge=1, description="First page of this section")
    page_end: int = Field(..., ge=1, description="Last page of this section")
    children: list["PageIndexNode"] = Field(
        default_factory=list,
        description="Child sections (recursive)",
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Named entities found in this section",
    )
    summary: str = Field(
        default="",
        description="2–3 sentence summary of section content",
    )
    data_types_present: list[str] = Field(
        default_factory=list,
        description="Types of data in section: tables, figures, equations, etc.",
    )
    content_hashes: list[str] = Field(
        default_factory=list,
        description="Content hashes of LDUs in this section",
    )


PageIndexNode.model_rebuild()


class PageIndex(BaseModel):
    """Hierarchical navigation structure for a document."""

    document_id: str = Field(
        ..., description="Unique identifier of the source document"
    )
    root_nodes: list[PageIndexNode] = Field(
        default_factory=list,
        description="Top-level sections in document order",
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class ProvenanceCitation(BaseModel):
    """A single source citation linking an extracted fact to its origin."""

    document_id: str = Field(..., description="Source document identifier")
    document_name: str = Field(..., description="Human-readable file name")
    page_number: int = Field(..., ge=1, description="1-indexed page number")
    bbox: BoundingBox | None = Field(
        default=None,
        description="Bounding box of the cited region on the page",
    )
    content_hash: str = Field(
        default="",
        description="SHA-256 hash of the cited content for verification",
    )


class ProvenanceChain(BaseModel):
    """Ordered chain of provenance citations for an answer or extracted fact."""

    query: str = Field(..., description="The question or claim being sourced")
    citations: list[ProvenanceCitation] = Field(
        default_factory=list,
        description="Source citations in relevance order",
    )
    verified: bool = Field(
        default=False,
        description="Whether the chain has been verified against source",
    )


# ---------------------------------------------------------------------------
# Fact table — structured key-value extraction (enriched)
# ---------------------------------------------------------------------------

class Fact(BaseModel):
    """A structured fact extracted from a document (enriched schema)."""

    key: str = Field(..., description="Fact label (e.g. 'Revenue', 'Net Income')")
    value: str = Field(..., description="Fact value (e.g. '$4.2B', '12.5%')")
    unit: str = Field(default="", description="Unit or currency (e.g. 'USD', '%', 'ETB')")
    page_ref: int = Field(..., ge=1, description="Page where fact was found")
    document_id: str = Field(..., description="Source document identifier")
    content_hash: str = Field(default="", description="SHA-256 hash of source content")

    # Enriched fields
    entity: str = Field(default="", description="Entity the fact is about")
    metric: str = Field(default="", description="Metric name (e.g. 'Total Revenue')")
    period: str = Field(default="", description="Time period (e.g. '2024', 'Q3 2023')")
    section: str = Field(default="", description="Section where fact was extracted")
    bbox: BoundingBox | None = Field(default=None, description="Bounding box on page")
    extraction_method: Literal[
        "regex", "table_parse", "llm_assisted", "hybrid"
    ] = Field(default="regex", description="How the fact was extracted")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the extraction [0, 1]",
    )


# ---------------------------------------------------------------------------
# Entity linking & Knowledge Graph
# ---------------------------------------------------------------------------

class EntityMention(BaseModel):
    """An entity mentioned in the document."""

    entity_name: str = Field(..., description="Canonical entity name")
    entity_type: Literal[
        "organization", "person", "metric", "date", "amount",
        "location", "regulation", "product", "other"
    ] = Field(..., description="Entity type")
    mentions: list[dict] = Field(
        default_factory=list,
        description="List of mentions with page, section, content_hash",
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="Known aliases for this entity",
    )


class KnowledgeGraphEdge(BaseModel):
    """An edge in the document knowledge graph."""

    source: str = Field(..., description="Source node identifier")
    target: str = Field(..., description="Target node identifier")
    relation: str = Field(
        ..., description="Relationship type (e.g. 'has_revenue', 'discussed_in')"
    )
    page_ref: int = Field(default=0, description="Page where relationship found")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class DocumentKnowledgeGraph(BaseModel):
    """Lightweight knowledge graph over a document."""

    document_id: str = Field(..., description="Source document ID")
    entities: list[EntityMention] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdge] = Field(default_factory=list)
    cross_references: list[CrossReference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Query agent result
# ---------------------------------------------------------------------------

class AuditResult(BaseModel):
    """Result of an audit-mode verification."""

    claim: str = Field(..., description="The claim being verified")
    status: Literal["verified", "not_found", "unverifiable"] = Field(
        ..., description="Verification status"
    )
    supporting_evidence: list[ProvenanceCitation] = Field(
        default_factory=list, description="Supporting citations"
    )
    explanation: str = Field(
        default="", description="Explanation of verification result"
    )


class QueryResult(BaseModel):
    """Structured result from the Query Agent (Stage 5)."""

    answer: str = Field(..., description="Natural-language answer to the query")
    provenance: ProvenanceChain = Field(
        ..., description="Source citations backing the answer"
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="Tools invoked to produce this answer",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Agent's confidence in the answer [0, 1]",
    )


# ---------------------------------------------------------------------------
# Extraction ledger entry
# ---------------------------------------------------------------------------

class LedgerEntry(BaseModel):
    """A single extraction attempt record for the ledger."""

    document_id: str = Field(...)
    source_filename: str = Field(default="")
    strategy_used: str = Field(...)
    confidence_score: float = Field(default=0.0)
    cost_estimate: str = Field(default="unknown")
    processing_time_s: float = Field(default=0.0)
    fallback_trigger: str | None = Field(default=None)
    page_scope: list[int] | None = Field(default=None)
    failure_reason: str | None = Field(default=None)
    escalation_triggered: bool = Field(default=False)
    escalation_reason: str | None = Field(default=None)
    flagged_for_review: bool = Field(default=False)
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

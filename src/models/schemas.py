"""Pydantic v2 schemas for the Document Intelligence Refinery.

All core data models used across the five-stage pipeline are defined here.
Models are ordered by dependency — leaf models first, composite models last.
No implementation logic — pure data contracts only.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Spatial primitives
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Spatial coordinates of a region on a specific page.

    Coordinates follow the PDF convention: origin at top-left corner,
    x increases rightward, y increases downward (in points).
    """

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
    """A structured table extracted from a document page.

    Headers and rows are stored separately so downstream consumers
    can reconstruct the table without re-parsing.
    """

    headers: list[str] = Field(
        ..., description="Column header labels in reading order"
    )
    rows: list[list[str]] = Field(
        ..., description="Row data; each inner list aligns with headers"
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the full table")


class FigureObject(BaseModel):
    """A figure (image/chart/diagram) extracted from a document page.

    The caption is always stored as metadata of its parent figure,
    never split into a separate chunk (chunking rule #2).
    """

    caption: str = Field(
        default="",
        description="Figure caption text (may be empty if not detected)",
    )
    bbox: BoundingBox = Field(..., description="Bounding box of the figure")
    figure_type: str = Field(
        default="image",
        description="Type hint: image, chart, diagram, photograph, etc.",
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


class ExtractedDocument(BaseModel):
    """Normalized extraction output for an entire document.

    All three extraction strategies (Fast Text, Layout-Aware, Vision-Augmented)
    must produce an ``ExtractedDocument`` so that downstream stages
    (chunking, indexing, querying) are strategy-agnostic.
    """

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


# ---------------------------------------------------------------------------
# Triage / classification
# ---------------------------------------------------------------------------

class DocumentProfile(BaseModel):
    """Classification output produced by the Triage Agent (Stage 1).

    The profile governs which extraction strategy the downstream
    Structure Extraction Layer will use for this document.
    Stored at ``.refinery/profiles/{document_id}.json``.
    """

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


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------

class LDU(BaseModel):
    """Logical Document Unit — a RAG-ready, semantically coherent chunk.

    Produced by the Semantic Chunking Engine (Stage 3).  Each LDU
    preserves its structural context and spatial provenance so that
    query answers can cite the exact source location.
    """

    content: str = Field(..., description="Chunk text content")
    chunk_type: Literal[
        "paragraph", "table", "figure", "list", "section"
    ] = Field(..., description="Semantic type of this chunk")
    page_refs: list[int] = Field(
        ..., description="1-indexed page numbers this chunk spans"
    )
    bbox: BoundingBox | None = Field(
        default=None,
        description="Bounding box (None if chunk spans multiple discontiguous regions)",
    )
    parent_section: str | None = Field(
        default=None,
        description="Title of the parent section for hierarchical context",
    )
    token_count: int = Field(
        ..., ge=0, description="Approximate token count of content"
    )
    content_hash: str = Field(
        ...,
        description="SHA-256 hex digest of content for provenance verification",
    )


# ---------------------------------------------------------------------------
# PageIndex — hierarchical navigation
# ---------------------------------------------------------------------------

class PageIndexNode(BaseModel):
    """A node in the hierarchical PageIndex tree.

    Represents a document section with navigation metadata that allows
    an LLM to traverse the document without reading everything.
    """

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
        description="LLM-generated 2–3 sentence summary of section content",
    )
    data_types_present: list[str] = Field(
        default_factory=list,
        description="Types of data in section: tables, figures, equations, etc.",
    )


# Enable recursive model references
PageIndexNode.model_rebuild()


class PageIndex(BaseModel):
    """Hierarchical navigation structure for a document.

    The equivalent of a 'smart table of contents' that an LLM can
    traverse to locate information without reading the entire document.
    """

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
    """Ordered chain of provenance citations for an answer or extracted fact.

    Every answer emitted by the query agent must carry a ProvenanceChain
    so that claims are auditable back to their spatial source.
    """

    query: str = Field(..., description="The question or claim being sourced")
    citations: list[ProvenanceCitation] = Field(
        default_factory=list,
        description="Source citations in relevance order",
    )
    verified: bool = Field(
        default=False,
        description="Whether the chain has been verified against source",
    )

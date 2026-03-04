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


class ExtractedDocument(BaseModel):
    """Normalized extraction output for an entire document.

    All three extraction strategies (Fast Text, Layout-Aware, Vision-Augmented)
    must produce an ``ExtractedDocument`` so that downstream stages
    (chunking, indexing, querying) are strategy-agnostic.
    """

    document_id: str = Field(
        ..., description="Unique identifier for the source document"
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

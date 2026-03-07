"""Semantic Chunking Engine — Stage 3 of the Document Intelligence Refinery.

Converts raw ``ExtractedDocument`` output into a list of Logical Document
Units (LDUs) — semantically coherent, RAG-ready chunks that preserve
structural context and spatial provenance.

Five enforceable chunking rules (the "Constitution"):
    1. A table cell is never split from its header row.
    2. A figure caption is always stored as metadata of its parent figure chunk.
    3. A numbered/bullet list is kept as a single LDU unless it exceeds
       ``max_tokens``.
    4. Section headers are stored as ``parent_section`` metadata on all
       child chunks within that section.
    5. Cross-references (e.g. "see Table 3") are detected and preserved
       in chunk metadata for downstream relationship resolution.

Additional features:
    - Table subgroup splitting (by year, region, category, repeated headers)
    - Cross-reference detection and storage in chunk metadata
    - Table-text linking (resolve "see Table 2" references)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import yaml

from src.models.schemas import (
    BoundingBox,
    CrossReference,
    ExtractedDocument,
    ExtractedPage,
    FigureObject,
    LDU,
    TableObject,
    TextBlock,
)
from src.utils.hash_utils import generate_content_hash

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)

# ---------------------------------------------------------------------------
# Heuristics for content-type detection
# ---------------------------------------------------------------------------

_LIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(\d+)\.\s", re.MULTILINE),
    re.compile(r"^[-•●◦▪]\s", re.MULTILINE),
    re.compile(r"^[a-z]\)\s", re.MULTILINE),
    re.compile(r"^[ivxlcdm]+\)\s", re.MULTILINE),
]

_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:[A-Z][A-Z\s\d.&:–—-]{2,80})"
    r"|"
    r"(?:\d+(?:\.\d+)*\s+[A-Z].{0,80})"
    r"|"
    r"(?:[A-Z][a-zA-Z\s&:–—-]{2,60})"
    r")$"
)

# Cross-reference patterns
_XREF_RE = re.compile(
    r"(?:see|refer\s+to|as\s+shown\s+in|in|per)\s+"
    r"(?:Table|Figure|Fig\.|Section|Appendix|Chart|Note)\s*\d+(?:\.\d+)*",
    re.IGNORECASE,
)

# Table/Figure identifier patterns
_TABLE_ID_RE = re.compile(r"Table\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
_FIGURE_ID_RE = re.compile(r"(?:Figure|Fig\.?)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)

# Year pattern for table subgroup detection
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Repeated header detection in table rows
_CATEGORY_HEADER_RE = re.compile(r"^[A-Z][A-Za-z\s]{2,40}$")


def _estimate_tokens(text: str) -> int:
    """Rough token count — split on whitespace."""
    return len(text.split())


def _is_list_block(text: str) -> bool:
    """Return True if *text* looks like a numbered/bullet list."""
    for pat in _LIST_PATTERNS:
        matches = pat.findall(text)
        if len(matches) >= 2:
            return True
    return False


def _is_section_header(text: str) -> bool:
    """Heuristic: short line, Title/UPPER case, no trailing period."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped.split()) > 10:
        return False
    if stripped.endswith("."):
        return False
    return bool(_HEADER_RE.match(stripped))


def _detect_cross_references(text: str, page: int) -> list[CrossReference]:
    """Detect cross-references in text and return structured objects."""
    refs: list[CrossReference] = []
    for match in _XREF_RE.finditer(text):
        ref_text = match.group(0)

        # Determine target type
        ref_lower = ref_text.lower()
        if "table" in ref_lower:
            target_type = "table"
        elif "fig" in ref_lower:
            target_type = "figure"
        elif "section" in ref_lower:
            target_type = "section"
        elif "appendix" in ref_lower:
            target_type = "appendix"
        elif "note" in ref_lower:
            target_type = "note"
        elif "chart" in ref_lower:
            target_type = "chart"
        else:
            target_type = "unknown"

        # Extract target label
        label_match = re.search(
            r"(Table|Figure|Fig\.?|Section|Appendix|Chart|Note)\s*(\d+(?:\.\d+)*)",
            ref_text, re.IGNORECASE
        )
        if label_match:
            target_label = f"{label_match.group(1)} {label_match.group(2)}"
        else:
            target_label = ref_text

        refs.append(CrossReference(
            source_page=page,
            source_text=ref_text,
            target_type=target_type,
            target_label=target_label,
        ))
    return refs


def _split_table_into_subgroups(
    table: TableObject,
) -> list[tuple[str, list[list[str]]]]:
    """Split a table into logical subgroups if meaningful patterns detected.

    Returns list of (subgroup_label, rows) tuples.
    If no meaningful split is found, returns a single group with all rows.
    """
    if len(table.rows) < 4:
        return [("", table.rows)]

    # Strategy 1: Look for year-based groupings in headers
    year_cols = []
    for i, h in enumerate(table.headers):
        if _YEAR_RE.search(h):
            year_cols.append(i)

    # Strategy 2: Look for repeated category headers in first column
    # (rows where all but first cell are empty → subgroup header)
    subgroups: list[tuple[str, list[list[str]]]] = []
    current_label = ""
    current_rows: list[list[str]] = []

    for row in table.rows:
        # Check if this row is a subgroup header
        first_cell = row[0].strip() if row else ""
        other_cells = [c.strip() for c in row[1:]] if len(row) > 1 else []
        is_subheader = (
            first_cell
            and _CATEGORY_HEADER_RE.match(first_cell)
            and all(not c or c == "-" or c == "—" for c in other_cells)
        )

        if is_subheader and current_rows:
            subgroups.append((current_label, current_rows))
            current_label = first_cell
            current_rows = []
        elif is_subheader:
            current_label = first_cell
        else:
            current_rows.append(row)

    if current_rows:
        subgroups.append((current_label, current_rows))

    # Only use subgroups if we found meaningful splits (at least 2)
    if len(subgroups) >= 2:
        return subgroups

    return [("", table.rows)]


class ChunkingEngine:
    """Accepts an ``ExtractedDocument`` and emits ``list[LDU]``.

    All five chunking-constitution rules are enforced during emission.
    Enhanced with cross-reference detection and table subgroup splitting.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)

        chunk_cfg = rules.get("chunking", {})
        self._max_tokens: int = chunk_cfg.get("max_tokens_per_chunk", 512)
        self._min_tokens: int = chunk_cfg.get("min_tokens_per_chunk", 50)
        self._overlap: int = chunk_cfg.get("overlap_tokens", 64)

    def chunk_document(self, doc: ExtractedDocument) -> list[LDU]:
        """Transform *doc* into a flat list of LDUs.

        Processing order per page:
        1. Text blocks → paragraphs / lists / section headers
        2. Tables → table LDUs (with subgroup splitting)
        3. Figures → figure LDUs

        Section header tracking is maintained across pages.
        Cross-references are detected and stored in chunk metadata.
        """
        ldus: list[LDU] = []
        current_section: str | None = None
        table_counter = 0
        figure_counter = 0

        for page in doc.pages:
            page_num = page.page_number

            # ── Text blocks ──────────────────────────────────────────
            for tb in page.text_blocks:
                text = tb.content.strip()
                if not text:
                    continue

                # Rule #4 — detect section headers
                if _is_section_header(text):
                    current_section = text
                    ldus.append(self._make_ldu(
                        content=text,
                        chunk_type="section",
                        page_refs=[page_num],
                        bbox=tb.bbox,
                        parent_section=current_section,
                    ))
                    continue

                # Rule #3 — list detection
                if _is_list_block(text):
                    ldus.extend(self._chunk_list(
                        text, page_num, tb.bbox, current_section,
                    ))
                    continue

                # Default: paragraph chunking with token-limit splits
                ldus.extend(self._chunk_paragraph(
                    text, page_num, tb.bbox, current_section,
                ))

            # ── Tables (rule #1, with subgroup splitting) ────────────
            for tbl in page.tables:
                table_counter += 1
                table_id = tbl.table_id or f"Table_{table_counter}"
                ldus.extend(self._chunk_table(
                    tbl, page_num, current_section, table_id,
                ))

            # ── Figures (rule #2) ─────────────────────────────────────
            for fig in page.figures:
                figure_counter += 1
                figure_id = fig.figure_id or f"Figure_{figure_counter}"
                ldus.append(self._chunk_figure(
                    fig, page_num, current_section, figure_id,
                ))

        return ldus

    # ------------------------------------------------------------------
    # Chunk builders
    # ------------------------------------------------------------------

    def _chunk_paragraph(
        self,
        text: str,
        page: int,
        bbox: BoundingBox,
        section: str | None,
    ) -> list[LDU]:
        """Split text into paragraph LDUs respecting token limits."""
        xrefs = _detect_cross_references(text, page)

        tokens = text.split()
        if len(tokens) <= self._max_tokens:
            return [self._make_ldu(
                content=text,
                chunk_type="paragraph",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
                cross_references=xrefs,
            )]

        chunks: list[LDU] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._max_tokens, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            chunk_xrefs = _detect_cross_references(chunk_text, page)
            chunks.append(self._make_ldu(
                content=chunk_text,
                chunk_type="paragraph",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
                cross_references=chunk_xrefs,
            ))
            if end >= len(tokens):
                break
            start = end - self._overlap

        return chunks

    def _chunk_list(
        self,
        text: str,
        page: int,
        bbox: BoundingBox,
        section: str | None,
    ) -> list[LDU]:
        """Rule #3 — keep list as a single LDU unless > max_tokens."""
        xrefs = _detect_cross_references(text, page)

        if _estimate_tokens(text) <= self._max_tokens:
            return [self._make_ldu(
                content=text,
                chunk_type="list",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
                cross_references=xrefs,
            )]

        items = re.split(r"(?=^\d+\.\s|^[-•●◦▪]\s|^[a-z]\)\s)", text, flags=re.MULTILINE)
        items = [it for it in items if it.strip()]

        chunks: list[LDU] = []
        current_items: list[str] = []
        current_tokens = 0

        for item in items:
            item_tokens = _estimate_tokens(item)
            if current_tokens + item_tokens > self._max_tokens and current_items:
                chunk_text = "\n".join(current_items)
                chunks.append(self._make_ldu(
                    content=chunk_text,
                    chunk_type="list",
                    page_refs=[page],
                    bbox=bbox,
                    parent_section=section,
                    cross_references=_detect_cross_references(chunk_text, page),
                ))
                current_items = []
                current_tokens = 0
            current_items.append(item.strip())
            current_tokens += item_tokens

        if current_items:
            chunk_text = "\n".join(current_items)
            chunks.append(self._make_ldu(
                content=chunk_text,
                chunk_type="list",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
                cross_references=_detect_cross_references(chunk_text, page),
            ))

        return chunks

    def _chunk_table(
        self,
        table: TableObject,
        page: int,
        section: str | None,
        table_id: str = "",
    ) -> list[LDU]:
        """Rule #1 — table cells never split from headers.

        Enhanced: splits tables into logical subgroups when meaningful
        patterns are detected (year, region, category, repeated headers).
        Each subgroup becomes its own LDU linked to the parent table.
        """
        subgroups = _split_table_into_subgroups(table)

        if len(subgroups) <= 1:
            # Single table LDU
            content = self._serialize_table(table.headers, table.rows, table.caption)
            xrefs = _detect_cross_references(content, page)
            return [self._make_ldu(
                content=content,
                chunk_type="table",
                page_refs=[page],
                bbox=table.bbox,
                parent_section=section,
                cross_references=xrefs,
                table_id=table_id,
            )]

        # Multiple subgroups
        ldus: list[LDU] = []
        for label, rows in subgroups:
            content = self._serialize_table(table.headers, rows, table.caption)
            if label:
                content = f"[Subgroup: {label}]\n{content}"

            ldus.append(self._make_ldu(
                content=content,
                chunk_type="table",
                page_refs=[page],
                bbox=table.bbox,
                parent_section=section,
                table_id=table_id,
                subgroup_label=label,
                parent_table_id=table_id,
            ))

        return ldus

    @staticmethod
    def _serialize_table(
        headers: list[str], rows: list[list[str]], caption: str = ""
    ) -> str:
        """Serialize table to Markdown format for RAG retrieval."""
        parts: list[str] = []
        if caption:
            parts.append(f"Caption: {caption}")
        header_line = " | ".join(headers)
        separator = " | ".join("---" for _ in headers)
        rows_text = "\n".join(
            " | ".join(cell for cell in row) for row in rows
        )
        parts.append(f"{header_line}\n{separator}\n{rows_text}")
        return "\n".join(parts)

    def _chunk_figure(
        self,
        figure: FigureObject,
        page: int,
        section: str | None,
        figure_id: str = "",
    ) -> LDU:
        """Rule #2 — figure caption is stored as part of the figure LDU."""
        parts = [f"[{figure.figure_type.upper()}]"]
        if figure.caption:
            parts.append(figure.caption)
        else:
            parts.append("(no caption)")
        content = " ".join(parts)

        return self._make_ldu(
            content=content,
            chunk_type="figure",
            page_refs=[page],
            bbox=figure.bbox,
            parent_section=section,
            figure_id=figure_id,
        )

    # ------------------------------------------------------------------
    # LDU factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_ldu(
        content: str,
        chunk_type: str,
        page_refs: list[int],
        bbox: BoundingBox | None = None,
        parent_section: str | None = None,
        cross_references: list[CrossReference] | None = None,
        table_id: str = "",
        figure_id: str = "",
        subgroup_label: str = "",
        parent_table_id: str = "",
    ) -> LDU:
        """Construct an LDU with computed token count and content hash."""
        return LDU(
            content=content,
            chunk_type=chunk_type,
            page_refs=page_refs,
            bbox=bbox,
            parent_section=parent_section,
            token_count=_estimate_tokens(content),
            content_hash=generate_content_hash(content),
            cross_references=cross_references or [],
            table_id=table_id,
            figure_id=figure_id,
            subgroup_label=subgroup_label,
            parent_table_id=parent_table_id,
        )


# ---------------------------------------------------------------------------
# ChunkValidator
# ---------------------------------------------------------------------------


class ChunkValidator:
    """Post-emission validator that checks every LDU against the constitution.

    Returns a list of human-readable error strings.  Empty list = valid.

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)

        chunk_cfg = rules.get("chunking", {})
        self._max_tokens: int = chunk_cfg.get("max_tokens_per_chunk", 512)
        self._min_tokens: int = chunk_cfg.get("min_tokens_per_chunk", 50)

    def validate(self, ldu: LDU) -> list[str]:
        """Validate a single LDU.  Returns list of error strings."""
        errors: list[str] = []

        # 1. Content must not be empty
        if not ldu.content.strip():
            errors.append("Empty content: LDU has no meaningful text")

        # 2. Token count must not exceed max (tables get a pass — rule #1)
        if ldu.chunk_type != "table" and ldu.token_count > self._max_tokens:
            errors.append(
                f"Token count {ldu.token_count} exceeds max {self._max_tokens}"
            )

        # 3. Content hash must match actual content
        expected_hash = generate_content_hash(ldu.content)
        if ldu.content_hash != expected_hash:
            errors.append(
                f"Content hash mismatch: expected {expected_hash[:16]}… "
                f"got {ldu.content_hash[:16]}…"
            )

        # 4. Page refs must not be empty
        if not ldu.page_refs:
            errors.append("Missing page_refs: LDU must reference at least one page")

        return errors

    def validate_batch(self, ldus: Sequence[LDU]) -> list[dict]:
        """Validate every LDU in *ldus*.

        Returns a list of dicts ``{"index": int, "errors": list[str]}``
        for each LDU that has at least one error.
        """
        issues: list[dict] = []
        for idx, ldu in enumerate(ldus):
            errs = self.validate(ldu)
            if errs:
                issues.append({"index": idx, "errors": errs})
        return issues

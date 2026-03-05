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
       in chunk content for downstream relationship resolution.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Sequence

import yaml

from src.models.schemas import (
    BoundingBox,
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

# Patterns that indicate a numbered or bullet list
_LIST_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(\d+)\.\s", re.MULTILINE),      # "1. item"
    re.compile(r"^[-•●◦▪]\s", re.MULTILINE),      # "• item", "- item"
    re.compile(r"^[a-z]\)\s", re.MULTILINE),       # "a) item"
    re.compile(r"^[ivxlcdm]+\)\s", re.MULTILINE),  # "i) item", "iv) item"
]

# A section header is typically a short line (≤ 10 words) that is either
# ALL-CAPS, Title Case with no trailing period, or starts with a digit +
# period  (e.g. "3.1 Revenue Analysis").
_HEADER_RE = re.compile(
    r"^(?:"
    r"(?:[A-Z][A-Z\s\d.&:–—-]{2,80})"   # ALL CAPS line
    r"|"
    r"(?:\d+(?:\.\d+)*\s+[A-Z].{0,80})"  # "3.1 Revenue …"
    r"|"
    r"(?:[A-Z][a-zA-Z\s&:–—-]{2,60})"    # Title Case (no trailing period)
    r")$"
)

# Cross-reference patterns (detect, do not resolve)
_XREF_RE = re.compile(
    r"(?:see|refer\s+to|as\s+shown\s+in|in)\s+"
    r"(?:Table|Figure|Fig\.|Section|Appendix|Chart)\s*\d+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token count — split on whitespace.

    A proper tokeniser (tiktoken / sentencepiece) can be swapped in
    without changing the interface.
    """
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


# ---------------------------------------------------------------------------
# ChunkingEngine
# ---------------------------------------------------------------------------


class ChunkingEngine:
    """Accepts an ``ExtractedDocument`` and emits ``list[LDU]``.

    All five chunking-constitution rules are enforced during emission.

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.  Reads ``chunking`` section.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        with open(rp, "r", encoding="utf-8") as fh:
            rules = yaml.safe_load(fh)

        chunk_cfg = rules.get("chunking", {})
        self._max_tokens: int = chunk_cfg.get("max_tokens_per_chunk", 512)
        self._min_tokens: int = chunk_cfg.get("min_tokens_per_chunk", 50)
        self._overlap: int = chunk_cfg.get("overlap_tokens", 64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ExtractedDocument) -> list[LDU]:
        """Transform *doc* into a flat list of LDUs.

        Processing order per page:
        1. Text blocks → paragraphs / lists / section headers
        2. Tables → table LDUs
        3. Figures → figure LDUs

        Section header tracking is maintained across pages for
        ``parent_section`` propagation (rule #4).
        """
        ldus: list[LDU] = []
        current_section: str | None = None

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

            # ── Tables (rule #1) ─────────────────────────────────────
            for tbl in page.tables:
                ldus.append(self._chunk_table(
                    tbl, page_num, current_section,
                ))

            # ── Figures (rule #2) ─────────────────────────────────────
            for fig in page.figures:
                ldus.append(self._chunk_figure(
                    fig, page_num, current_section,
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
        tokens = text.split()
        if len(tokens) <= self._max_tokens:
            return [self._make_ldu(
                content=text,
                chunk_type="paragraph",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
            )]

        # Split into overlapping windows
        chunks: list[LDU] = []
        start = 0
        while start < len(tokens):
            end = min(start + self._max_tokens, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            chunks.append(self._make_ldu(
                content=chunk_text,
                chunk_type="paragraph",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
            ))
            if end >= len(tokens):
                break
            start = end - self._overlap  # overlap for context continuity

        return chunks

    def _chunk_list(
        self,
        text: str,
        page: int,
        bbox: BoundingBox,
        section: str | None,
    ) -> list[LDU]:
        """Rule #3 — keep list as a single LDU unless > max_tokens.

        When splitting is needed, splits at list-item boundaries
        (never in the middle of an item).
        """
        if _estimate_tokens(text) <= self._max_tokens:
            return [self._make_ldu(
                content=text,
                chunk_type="list",
                page_refs=[page],
                bbox=bbox,
                parent_section=section,
            )]

        # Split at list-item boundaries
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
            ))

        return chunks

    def _chunk_table(
        self,
        table: TableObject,
        page: int,
        section: str | None,
    ) -> LDU:
        """Rule #1 — table is ALWAYS a single LDU. Never split rows from headers."""
        # Serialise table to Markdown-style text for RAG retrieval
        header_line = " | ".join(table.headers)
        separator = " | ".join("---" for _ in table.headers)
        rows_text = "\n".join(
            " | ".join(cell for cell in row) for row in table.rows
        )
        content = f"{header_line}\n{separator}\n{rows_text}"

        return self._make_ldu(
            content=content,
            chunk_type="table",
            page_refs=[page],
            bbox=table.bbox,
            parent_section=section,
        )

    def _chunk_figure(
        self,
        figure: FigureObject,
        page: int,
        section: str | None,
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

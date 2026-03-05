"""PageIndex Builder — Stage 4 of the Document Intelligence Refinery.

Builds a hierarchical PageIndex tree from a flat ``list[LDU]`` produced
by the Chunking Engine (Stage 3).  The tree mirrors a "smart table of
contents" that an LLM can traverse to locate relevant sections without
embedding-searching the entire corpus.

No LLM is required — section detection, summary generation, and signal
tagging are all deterministic heuristics.  An LLM-backed summariser
can be swapped in later via the ``_generate_summary`` hook.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

from src.models.schemas import LDU, PageIndex, PageIndexNode
from src.utils.hash_utils import generate_content_hash

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)
_DEFAULT_PAGEINDEX_DIR = (
    Path(__file__).resolve().parents[2] / ".refinery" / "pageindex"
)
_DEFAULT_DB = (
    Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"
)

# ---------------------------------------------------------------------------
# Numeric-density heuristic
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\d[\d,._]*")


def _numeric_density(text: str) -> float:
    """Fraction of whitespace-separated tokens that contain digits."""
    tokens = text.split()
    if not tokens:
        return 0.0
    numeric_tokens = sum(1 for t in tokens if _NUMBER_RE.search(t))
    return numeric_tokens / len(tokens)


# ---------------------------------------------------------------------------
# Lightweight keyword overlap scorer
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase alpha-numeric tokens for bag-of-words comparison."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ---------------------------------------------------------------------------
# PageIndexBuilder
# ---------------------------------------------------------------------------


class PageIndexBuilder:
    """Constructs a :class:`PageIndex` tree from a list of LDUs.

    The builder groups LDUs by their ``parent_section`` label, computes
    page ranges, detects data-type signals (tables, figures, lists,
    numeric density), and generates a deterministic summary from the
    first paragraph(s) of each section.

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.  Currently reads the
        ``chunking`` section for consistency; future versions will
        read a dedicated ``pageindex`` section.
    summary_max_sentences : int
        Maximum number of sentences in the deterministic summary.
    numeric_density_threshold : float
        Minimum fraction of numeric tokens to tag ``numeric_dense``.
    """

    def __init__(
        self,
        rules_path: str | Path | None = None,
        summary_max_sentences: int = 3,
        numeric_density_threshold: float = 0.15,
    ) -> None:
        self._summary_max_sentences = summary_max_sentences
        self._numeric_density_threshold = numeric_density_threshold

        # Load rules (currently unused beyond consistency probe)
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as fh:
                self._rules = yaml.safe_load(fh) or {}
        else:
            self._rules = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        ldus: list[LDU],
        source_filename: str,
        document_id: str | None = None,
    ) -> PageIndex:
        """Build a PageIndex from *ldus*.

        Parameters
        ----------
        ldus : list[LDU]
            Flat list of Logical Document Units (output of ChunkingEngine).
        source_filename : str
            Original PDF filename for metadata.
        document_id : str | None
            Unique document identifier.  Auto-generated from
            *source_filename* if ``None``.

        Returns
        -------
        PageIndex
        """
        if document_id is None:
            document_id = self._auto_id(source_filename)

        if not ldus:
            return PageIndex(document_id=document_id, root_nodes=[])

        # 1. Group LDUs by section (preserving document order)
        sections = self._group_by_section(ldus)

        # 2. Build one PageIndexNode per section
        root_nodes: list[PageIndexNode] = []
        for section_title, section_ldus in sections.items():
            node = self._build_node(section_title, section_ldus)
            root_nodes.append(node)

        return PageIndex(document_id=document_id, root_nodes=root_nodes)

    def query(
        self,
        page_index: PageIndex,
        topic: str,
        top_n: int = 3,
    ) -> list[PageIndexNode]:
        """Return the *top_n* most relevant section nodes for *topic*.

        Uses lightweight bag-of-words overlap scoring (no embeddings).
        Scores are computed over: title, summary, and data_types_present.

        Parameters
        ----------
        page_index : PageIndex
            The index to search.
        topic : str
            Natural-language query topic.
        top_n : int
            Maximum results.

        Returns
        -------
        list[PageIndexNode]
            Sections ranked by relevance (highest first).
        """
        topic_tokens = _tokenise(topic)
        if not topic_tokens:
            return page_index.root_nodes[:top_n]

        scored: list[tuple[float, PageIndexNode]] = []
        for node in self._flatten_nodes(page_index.root_nodes):
            score = self._relevance_score(node, topic_tokens)
            scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:top_n]]

    # ------------------------------------------------------------------
    # Persistence — JSON
    # ------------------------------------------------------------------

    def save_json(
        self,
        page_index: PageIndex,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Save *page_index* as JSON to ``{output_dir}/{document_id}.json``.

        Returns the path to the written file.
        """
        out = Path(output_dir) if output_dir else _DEFAULT_PAGEINDEX_DIR
        out.mkdir(parents=True, exist_ok=True)

        fp = out / f"{page_index.document_id}.json"
        fp.write_text(
            page_index.model_dump_json(indent=2),
            encoding="utf-8",
        )
        log.info("PageIndex JSON saved → %s", fp)
        return fp

    # ------------------------------------------------------------------
    # Persistence — SQLite
    # ------------------------------------------------------------------

    def persist_to_db(
        self,
        page_index: PageIndex,
        source_filename: str,
        db_path: str | Path | None = None,
    ) -> None:
        """Upsert *page_index* into the ``page_indexes`` table.

        Uses ``INSERT OR REPLACE`` so repeated calls overwrite cleanly.
        """
        db = Path(db_path) if db_path else _DEFAULT_DB
        tree_json = json.dumps(
            [n.model_dump() for n in page_index.root_nodes],
            default=str,
        )
        node_count = self._count_nodes(page_index.root_nodes)
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(db))
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO page_indexes
                    (document_id, source_filename, tree_json, node_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (page_index.document_id, source_filename, tree_json, node_count, now),
            )
            conn.commit()
            log.info(
                "PageIndex persisted to DB (%d nodes) → %s",
                node_count,
                db,
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal — section grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_section(ldus: list[LDU]) -> "OrderedDict[str, list[LDU]]":
        """Group LDUs by ``parent_section``, preserving document order.

        LDUs without a ``parent_section`` are placed under a synthetic
        ``"(Untitled)"`` section.
        """
        sections: OrderedDict[str, list[LDU]] = OrderedDict()
        for ldu in ldus:
            key = ldu.parent_section or "(Untitled)"
            if key not in sections:
                sections[key] = []
            sections[key].append(ldu)
        return sections

    # ------------------------------------------------------------------
    # Internal — node construction
    # ------------------------------------------------------------------

    def _build_node(
        self,
        title: str,
        ldus: list[LDU],
    ) -> PageIndexNode:
        """Construct a single PageIndexNode from its constituent LDUs."""
        # Page range — computed from content LDUs only (skip section headers)
        all_pages: list[int] = []
        for ldu in ldus:
            if ldu.chunk_type != "section":
                all_pages.extend(ldu.page_refs)
        # Fall back to section-header page_refs if no content LDUs
        if not all_pages:
            for ldu in ldus:
                all_pages.extend(ldu.page_refs)
        page_start = min(all_pages) if all_pages else 1
        page_end = max(all_pages) if all_pages else 1

        # Data-type signals
        data_types = self._detect_data_types(ldus)

        # Deterministic summary
        summary = self._generate_summary(ldus)

        # Key entities (simple: capitalised multi-word sequences)
        key_entities = self._extract_key_entities(ldus)

        return PageIndexNode(
            title=title,
            page_start=page_start,
            page_end=page_end,
            children=[],
            key_entities=key_entities,
            summary=summary,
            data_types_present=data_types,
        )

    # ------------------------------------------------------------------
    # Internal — data-type detection
    # ------------------------------------------------------------------

    def _detect_data_types(self, ldus: list[LDU]) -> list[str]:
        """Detect which structural data types are present in *ldus*."""
        types: set[str] = set()
        combined_text = ""

        for ldu in ldus:
            if ldu.chunk_type == "table":
                types.add("tables")
            elif ldu.chunk_type == "figure":
                types.add("figures")
            elif ldu.chunk_type == "list":
                types.add("lists")

            if ldu.chunk_type != "section":
                combined_text += " " + ldu.content

        # Numeric density check
        if combined_text.strip():
            if _numeric_density(combined_text) >= self._numeric_density_threshold:
                types.add("numeric_dense")

        return sorted(types)

    # ------------------------------------------------------------------
    # Internal — deterministic summary
    # ------------------------------------------------------------------

    def _generate_summary(self, ldus: list[LDU]) -> str:
        """Build a 2-3 sentence summary from paragraph LDUs.

        Takes the first N sentences from the first paragraph-type LDU
        in the section.  Skips tables, figures, and section-header LDUs.

        This is the hook point for swapping in an LLM summariser later.
        """
        for ldu in ldus:
            if ldu.chunk_type in ("paragraph", "list"):
                return self._first_n_sentences(
                    ldu.content,
                    n=self._summary_max_sentences,
                )
        # Fallback: use any non-section LDU
        for ldu in ldus:
            if ldu.chunk_type != "section" and ldu.content.strip():
                return self._first_n_sentences(
                    ldu.content,
                    n=self._summary_max_sentences,
                )
        return ""

    @staticmethod
    def _first_n_sentences(text: str, n: int = 3) -> str:
        """Extract the first *n* sentences from *text*."""
        # Split on sentence-ending punctuation followed by a space or end
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        selected = sentences[:n]
        result = " ".join(selected)
        # Ensure it ends with a period if it doesn't already
        if result and not result.rstrip().endswith((".", "!", "?")):
            result = result.rstrip() + "."
        return result

    # ------------------------------------------------------------------
    # Internal — key entity extraction (lightweight)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_key_entities(ldus: list[LDU], max_entities: int = 10) -> list[str]:
        """Extract capitalised multi-word sequences as candidate entities.

        This is a cheap NER stand-in.  A proper NER model can replace
        this method without changing the interface.
        """
        entity_re = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
        counts: dict[str, int] = {}

        for ldu in ldus:
            if ldu.chunk_type == "section":
                continue
            for match in entity_re.finditer(ldu.content):
                ent = match.group(0)
                # Filter out very short or generic
                if len(ent) > 4:
                    counts[ent] = counts.get(ent, 0) + 1

        # Sort by frequency, return top-N
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [ent for ent, _ in ranked[:max_entities]]

    # ------------------------------------------------------------------
    # Internal — query scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _relevance_score(
        node: PageIndexNode,
        topic_tokens: set[str],
    ) -> float:
        """Bag-of-words overlap score for a node against topic tokens."""
        # Build node token set from title + summary + data_types + entities
        node_text = " ".join([
            node.title,
            node.summary,
            " ".join(node.data_types_present),
            " ".join(node.key_entities),
        ])
        node_tokens = _tokenise(node_text)

        if not node_tokens:
            return 0.0

        overlap = topic_tokens & node_tokens
        # Jaccard-like score weighted toward recall on topic tokens
        if not topic_tokens:
            return 0.0
        return len(overlap) / len(topic_tokens)

    # ------------------------------------------------------------------
    # Internal — utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_id(source_filename: str) -> str:
        """Generate a deterministic document ID from filename."""
        return hashlib.md5(source_filename.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _count_nodes(nodes: list[PageIndexNode]) -> int:
        """Recursively count all nodes in the tree."""
        total = 0
        for node in nodes:
            total += 1
            total += PageIndexBuilder._count_nodes(node.children)
        return total

    @staticmethod
    def _flatten_nodes(nodes: list[PageIndexNode]) -> list[PageIndexNode]:
        """Recursively flatten the tree into a list."""
        result: list[PageIndexNode] = []
        for node in nodes:
            result.append(node)
            result.extend(PageIndexBuilder._flatten_nodes(node.children))
        return result

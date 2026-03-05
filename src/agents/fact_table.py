"""FactTable Extractor — Stage 5 data layer.

Extracts structured key-value numerical facts from LDUs and persists
them into a SQLite ``fact_tables`` table for precise structured
queries over financial / numerical documents.

The extraction uses regex-based heuristics to identify key-value
pairs in both paragraph and table LDUs.  This is the "structured_query"
data source for the Query Agent.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.models.schemas import Fact, LDU
from src.utils.hash_utils import generate_content_hash

log = logging.getLogger(__name__)

_DEFAULT_DB = (
    Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"
)

# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# "Key: $4.2B" or "Key: 12.5%" or "Key: 500 million ETB"
_KV_PATTERN = re.compile(
    r"(?P<key>[A-Z][A-Za-z /&-]+?)"         # Key: starts with capital
    r"\s*[:=]\s*"                             # separator
    r"(?P<value>"
    r"[\$€£₤]?\s*[\d,._]+(?:\.\d+)?"        # numeric value (with optional currency prefix)
    r"(?:\s*(?:million|billion|thousand|[BMKTbmkt]))?"  # magnitude suffix
    r"(?:\s*%|(?:\s+[A-Z]{2,5}))?"           # unit: percent or currency code
    r")",
    re.MULTILINE,
)

# "Key | 4,200 | 3,800" — table row pattern
_TABLE_ROW_PATTERN = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z /&()-]+?)\s*\|\s*"
    r"(?P<value>[\d,._]+(?:\.\d+)?)"
    r"(?:\s*\|.*)?$",
    re.MULTILINE,
)


def _parse_unit(value_str: str) -> str:
    """Guess the unit from a value string."""
    if "%" in value_str:
        return "%"
    if value_str.strip().startswith("$"):
        return "USD"
    if value_str.strip().startswith("€"):
        return "EUR"
    if value_str.strip().startswith("£") or value_str.strip().startswith("₤"):
        return "GBP"
    # Check for trailing currency code like "ETB", "USD"
    m = re.search(r"\b([A-Z]{2,5})$", value_str.strip())
    if m:
        return m.group(1)
    return ""


class FactTableExtractor:
    """Extracts and persists structured key-value facts from LDUs.

    Usage
    -----
    >>> extractor = FactTableExtractor()
    >>> facts = extractor.extract(ldus, document_id="doc1")
    >>> extractor.persist_to_db(facts, db_path="path/to/db")
    >>> results = extractor.query_facts(document_id="doc1", db_path="path/to/db")
    """

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        ldus: list[LDU],
        document_id: str,
    ) -> list[Fact]:
        """Extract key-value numerical facts from *ldus*.

        Scans both paragraph text (``Key: value`` patterns) and table
        rows (``Key | value`` patterns).

        Parameters
        ----------
        ldus : list[LDU]
            Logical Document Units to scan.
        document_id : str
            Document identifier to attach to each fact.

        Returns
        -------
        list[Fact]
        """
        facts: list[Fact] = []
        for ldu in ldus:
            page = ldu.page_refs[0] if ldu.page_refs else 1
            content_hash = ldu.content_hash or generate_content_hash(ldu.content)

            # Key: Value patterns (paragraph / mixed text)
            for m in _KV_PATTERN.finditer(ldu.content):
                key = m.group("key").strip().rstrip(":")
                value = m.group("value").strip()
                if not key or not value:
                    continue
                unit = _parse_unit(value)
                facts.append(Fact(
                    key=key,
                    value=value,
                    unit=unit,
                    page_ref=page,
                    document_id=document_id,
                    content_hash=content_hash,
                ))

            # Table row patterns
            if ldu.chunk_type == "table":
                for m in _TABLE_ROW_PATTERN.finditer(ldu.content):
                    key = m.group("key").strip()
                    value = m.group("value").strip()
                    if not key or not value:
                        continue
                    facts.append(Fact(
                        key=key,
                        value=value,
                        unit="",
                        page_ref=page,
                        document_id=document_id,
                        content_hash=content_hash,
                    ))

        return facts

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist_to_db(
        self,
        facts: list[Fact],
        db_path: str | Path | None = None,
    ) -> int:
        """Insert *facts* into the ``fact_tables`` SQLite table.

        Returns the number of rows inserted.
        """
        db = Path(db_path) if db_path else _DEFAULT_DB
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(db))
        try:
            for f in facts:
                conn.execute(
                    """
                    INSERT INTO fact_tables
                        (document_id, key, value, unit, page_ref, content_hash, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (f.document_id, f.key, f.value, f.unit, f.page_ref, f.content_hash, now),
                )
            conn.commit()
            log.info("Persisted %d facts to %s", len(facts), db)
        finally:
            conn.close()

        return len(facts)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_facts(
        self,
        document_id: str,
        key_pattern: str | None = None,
        db_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Query facts from the ``fact_tables`` table.

        Parameters
        ----------
        document_id : str
            Filter by document.
        key_pattern : str | None
            Optional SQL LIKE pattern to filter by key (e.g. ``'%revenue%'``).
        db_path : str | Path | None
            Database path.

        Returns
        -------
        list[dict]
            Rows as dictionaries.
        """
        db = Path(db_path) if db_path else _DEFAULT_DB
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        try:
            if key_pattern:
                rows = conn.execute(
                    "SELECT * FROM fact_tables WHERE document_id = ? AND key LIKE ?",
                    (document_id, key_pattern),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM fact_tables WHERE document_id = ?",
                    (document_id,),
                ).fetchall()
        finally:
            conn.close()

        return [dict(r) for r in rows]

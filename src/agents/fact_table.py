"""FactTable Extractor — Stage 5 data layer (Production).

Hybrid fact extraction pipeline:
  1. **regex**       — fast KV / table-row patterns for structured text
  2. **table_parse** — column-aware parsing of pipe-separated table LDUs
  3. **llm_assisted**— local Ollama LLM call for complex / ambiguous facts
                       (always enabled; falls back silently if Ollama is down)

Enriched facts include entity, metric, period, section, extraction_method,
and confidence.  Persistence uses the extended ``fact_tables`` schema with
new columns auto-created via ALTER TABLE if missing.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

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
    r"[\$€£₤]?\s*[\d,._]+(?:\.\d+)?"        # numeric value (opt. currency prefix)
    r"(?:\s*(?:million|billion|thousand|[BMKTbmkt]))?"
    r"(?:\s*%|(?:\s+[A-Z]{2,5}))?"
    r")",
    re.MULTILINE,
)

# "Key: Some Text Value" — colon-separated with non-numeric value (lower confidence)
_KV_TEXT_PATTERN = re.compile(
    r"^(?P<key>[A-Z][A-Za-z /.&-]{2,40}?)"  # Key: starts with capital, meaningful length
    r"\s*:\s*"                                # colon separator
    r"(?P<value>[A-Z0-9][^\n]{3,120})"       # value: starts uppercase/digit, 3-120 chars
    r"$",
    re.MULTILINE,
)

# "Key  4,200" — no colon, whitespace-separated key-value
_KV_NOCOLON = re.compile(
    r"(?P<key>[A-Z][A-Za-z /&-]{3,40}?)"    # Key: starts with capital, 3-40 chars
    r"\s{2,}"                                # two or more spaces (no colon)
    r"(?P<value>"
    r"[\$€£₤]?\s*[\d,._]+(?:\.\d+)?"
    r"(?:\s*(?:million|billion|thousand|[BMKTbmkt]))?"
    r"(?:\s*%|(?:\s+[A-Z]{2,5}))?"
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

# Detect time-period mentions
_PERIOD_PATTERN = re.compile(
    r"(?:FY|CY|Q[1-4])?\s*\d{4}"            # FY2024, Q3 2023, 2024
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}"
    r"|(?:H[12])\s*\d{4}"                    # H1 2024
    r"|(?:first|second|third|fourth)\s+(?:quarter|half)",
    re.IGNORECASE,
)

# Section heading detector (e.g. "2.1 Revenue Analysis")
_SECTION_PATTERN = re.compile(
    r"^(?:\d+\.?\d*\.?\d*)\s+[A-Z].*$", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    m = re.search(r"\b([A-Z]{2,5})$", value_str.strip())
    if m:
        return m.group(1)
    return ""


def _detect_period(text: str) -> str:
    """Extract the first time-period reference from *text*."""
    m = _PERIOD_PATTERN.search(text)
    return m.group(0).strip() if m else ""


def _detect_section(text: str) -> str:
    """Extract a section heading from *text*."""
    m = _SECTION_PATTERN.search(text)
    return m.group(0).strip()[:120] if m else ""


def _guess_entity(key: str, context: str) -> str:
    """Simple heuristic to guess the entity a fact refers to.

    Looks for prominent proper nouns near the key mention.
    """
    # If key itself contains an entity-like pattern, return it
    # E.g. "Apple Revenue" → entity = "Apple"
    words = key.strip().split()
    if len(words) >= 2 and words[0][0].isupper() and words[-1][0].isupper():
        # Check if first word looks like an entity (not a common metric word)
        metric_words = {
            "Total", "Net", "Gross", "Operating", "Annual", "Monthly",
            "Quarterly", "Average", "Median", "Adjusted", "Non",
        }
        if words[0] not in metric_words:
            return words[0]
    return ""


def _confidence_for_method(method: str, value_str: str) -> float:
    """Assign a base confidence score based on extraction method."""
    base = {"regex": 0.70, "table_parse": 0.85, "llm_assisted": 0.75, "hybrid": 0.80}
    c = base.get(method, 0.5)
    # Bump confidence if the value looks very numeric / well-formed
    cleaned = re.sub(r"[\$€£₤,%\s]", "", value_str)
    cleaned = cleaned.replace(",", "").replace("_", "")
    try:
        float(cleaned)
        c += 0.05
    except ValueError:
        c -= 0.10
    return round(max(0.0, min(1.0, c)), 2)


# ---------------------------------------------------------------------------
# LLM-assisted extraction (optional)
# ---------------------------------------------------------------------------

def _llm_extract_facts(
    text: str,
    document_id: str,
    page: int,
    content_hash: str,
    ollama_base_url: str,
    model: str,
    section: str = "",
) -> list[Fact]:
    """Call local Ollama to extract structured facts from complex text.

    Returns an empty list if Ollama is unreachable or the feature is disabled.
    """
    if not model:
        return []

    prompt = (
        "Extract all numerical key-value facts from the following text. "
        "Return a JSON array of objects with keys: key, value, unit, entity, metric, period.\n"
        "If a field is unknown, use an empty string.\n\n"
        f"Text:\n{text[:3000]}"
    )

    try:
        resp = httpx.post(
            f"{ollama_base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1024,
                "temperature": 0.1,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        # Strip markdown fencing if present
        content = re.sub(r"^```(?:json)?\s*", "", content.strip())
        content = re.sub(r"\s*```$", "", content.strip())
        items = json.loads(content)
    except Exception as exc:
        log.warning("LLM fact extraction failed: %s", exc)
        return []

    facts: list[Fact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        k = str(item.get("key", "")).strip()
        v = str(item.get("value", "")).strip()
        if not k or not v:
            continue
        unit = str(item.get("unit", "")) or _parse_unit(v)
        facts.append(Fact(
            key=k,
            value=v,
            unit=unit,
            page_ref=page,
            document_id=document_id,
            content_hash=content_hash,
            entity=str(item.get("entity", "")),
            metric=str(item.get("metric", "")),
            period=str(item.get("period", "")) or _detect_period(text),
            section=section,
            extraction_method="llm_assisted",
            confidence=0.75,
        ))
    return facts


def _llm_extract_page_facts(
    page_text: str,
    document_id: str,
    page: int,
    ollama_base_url: str,
    model: str,
) -> list[Fact]:
    """Page-level LLM extraction for scanned financial documents.

    Reconstructs multi-column table rows from OCR text and extracts
    all key-value pairs with period context (current year / prior year).
    """
    if not model or not page_text.strip():
        return []

    content_hash = generate_content_hash(page_text[:500])
    prompt = (
        "You are extracting financial facts from a scanned audited financial statement. "
        "The text below comes from a single page and was produced by OCR. "
        "Rows may look like: 'Line Item | current_year_value | prior_year_value'.\n\n"
        "Extract EVERY numeric key-value pair. For rows with two values use period "
        "'current_year' and 'prior_year' to distinguish them.\n"
        "Return a JSON array where each object has: key, value, unit, period.\n"
        "Use empty string for unknown fields. Only return the JSON array — no prose.\n\n"
        f"Page text:\n{page_text[:4000]}"
    )

    try:
        resp = httpx.post(
            f"{ollama_base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048,
                "temperature": 0.05,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw.strip())
        items = json.loads(raw)
    except Exception as exc:
        log.warning("Page-level LLM fact extraction failed (page %d): %s", page, exc)
        return []

    facts: list[Fact] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        k = str(item.get("key", "")).strip()
        v = str(item.get("value", "")).strip()
        if not k or not v:
            continue
        unit = str(item.get("unit", "")) or _parse_unit(v)
        facts.append(Fact(
            key=k,
            value=v,
            unit=unit,
            page_ref=page,
            document_id=document_id,
            content_hash=content_hash,
            entity="",
            metric=k,
            period=str(item.get("period", "")),
            section="",
            extraction_method="llm_assisted",
            confidence=0.78,
        ))
    return facts


# ═══════════════════════════════════════════════════════════════════════════
# Main extractor class
# ═══════════════════════════════════════════════════════════════════════════

class FactTableExtractor:
    """Hybrid fact extractor: regex → table_parse → LLM-assisted.

    Usage
    -----
    >>> extractor = FactTableExtractor()
    >>> facts = extractor.extract(ldus, document_id="doc1")
    >>> extractor.persist_to_db(facts)
    >>> results = extractor.query_facts("doc1")
    """

    def __init__(
        self,
        *,
        enable_llm: bool = True,
        ollama_base_url: str = "http://localhost:11434/v1",
        ollama_model: str = "qwen3-coder:480b-cloud",
        budget_max_llm_calls: int = 20,
    ) -> None:
        self.enable_llm = enable_llm
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self._budget_max = budget_max_llm_calls
        self._llm_calls = 0

    @classmethod
    def from_config(cls) -> "FactTableExtractor":
        """Create an instance from the centralized config."""
        try:
            from src.config import get_settings
            cfg = get_settings()
            return cls(
                enable_llm=cfg.enable_llm_fact_extraction,
                ollama_base_url=cfg.ollama_base_url,
                ollama_model=cfg.ollama_model,
                budget_max_llm_calls=cfg.budget_max_llm_calls,
            )
        except Exception:
            return cls()

    # ------------------------------------------------------------------
    # High-level extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        ldus: list[LDU],
        document_id: str,
        origin: str = "",
    ) -> list[Fact]:
        """Extract enriched numerical facts from *ldus*.

        Pipeline per LDU:
        1. Regex patterns (KV with colon, KV without colon)
        2. Table-row parsing for table LDUs
        3. (Optional) LLM-assisted extraction for complex paragraphs

        For scanned-image documents an additional page-level LLM pass
        runs across all LDUs grouped by page, letting the LLM
        reconstruct multi-column table rows that regex cannot match.

        Deduplicates by (key_lower, value_lower) before returning.
        """
        seen: set[tuple[str, str]] = set()
        facts: list[Fact] = []

        def _add(fact: Fact) -> None:
            dedup = (fact.key.lower(), fact.value.lower())
            if dedup in seen:
                return
            seen.add(dedup)
            facts.append(fact)

        for ldu in ldus:
            page = ldu.page_refs[0] if ldu.page_refs else 1
            chash = ldu.content_hash or generate_content_hash(ldu.content)
            section = _detect_section(ldu.content)
            period = _detect_period(ldu.content)

            # ── Table LDUs → table_parse method ──────────────────────
            if ldu.chunk_type == "table":
                for m in _TABLE_ROW_PATTERN.finditer(ldu.content):
                    k = m.group("key").strip().rstrip(":")
                    v = m.group("value").strip()
                    if not k or not v:
                        continue
                    unit = _parse_unit(v)
                    _add(Fact(
                        key=k,
                        value=v,
                        unit=unit,
                        page_ref=page,
                        document_id=document_id,
                        content_hash=chash,
                        entity=_guess_entity(k, ldu.content),
                        metric=k,
                        period=period,
                        section=section,
                        extraction_method="table_parse",
                        confidence=_confidence_for_method("table_parse", v),
                    ))
            else:
                # ── Regex: KV with colon/equals ──────────────────────
                for m in _KV_PATTERN.finditer(ldu.content):
                    k = m.group("key").strip().rstrip(":")
                    v = m.group("value").strip()
                    if not k or not v:
                        continue
                    unit = _parse_unit(v)
                    _add(Fact(
                        key=k,
                        value=v,
                        unit=unit,
                        page_ref=page,
                        document_id=document_id,
                        content_hash=chash,
                        entity=_guess_entity(k, ldu.content),
                        metric=k,
                        period=period,
                        section=section,
                        extraction_method="regex",
                        confidence=_confidence_for_method("regex", v),
                    ))

                # ── Regex: KV text values (lower confidence) ─────────
                for m in _KV_TEXT_PATTERN.finditer(ldu.content):
                    k = m.group("key").strip().rstrip(":")
                    v = m.group("value").strip()
                    if not k or not v or len(v) < 3:
                        continue
                    # Skip if already captured by numeric pattern
                    if (k.lower(), v.lower()) in seen:
                        continue
                    _add(Fact(
                        key=k,
                        value=v,
                        unit="",
                        page_ref=page,
                        document_id=document_id,
                        content_hash=chash,
                        entity=_guess_entity(k, ldu.content),
                        metric="",
                        period=period,
                        section=section,
                        extraction_method="regex",
                        confidence=0.55,   # lower confidence for text KV
                    ))

                # ── Regex: KV without colon (whitespace gap) ─────────
                for m in _KV_NOCOLON.finditer(ldu.content):
                    k = m.group("key").strip().rstrip(":")
                    v = m.group("value").strip()
                    if not k or not v:
                        continue
                    unit = _parse_unit(v)
                    _add(Fact(
                        key=k,
                        value=v,
                        unit=unit,
                        page_ref=page,
                        document_id=document_id,
                        content_hash=chash,
                        entity=_guess_entity(k, ldu.content),
                        metric=k,
                        period=period,
                        section=section,
                        extraction_method="regex",
                        confidence=_confidence_for_method("regex", v) - 0.05,
                    ))

            # ── LLM-assisted (budget-guarded, local Ollama) ──────────
            if (
                self.enable_llm
                and self._llm_calls < self._budget_max
                and len(ldu.content) > 200  # only for substantial text
            ):
                self._llm_calls += 1
                llm_facts = _llm_extract_facts(
                    text=ldu.content,
                    document_id=document_id,
                    page=page,
                    content_hash=chash,
                    ollama_base_url=self.ollama_base_url,
                    model=self.ollama_model,
                    section=section,
                )
                for f in llm_facts:
                    _add(f)

        # ── Scanned-doc page-level LLM pass ──────────────────────────
        # For scanned_image origin, aggregate all LDUs per page and
        # call the LLM once per page to recover multi-column table rows
        # that the per-LDU regex pass cannot match.
        if self.enable_llm and origin == "scanned_image":
            pages_text: dict[int, list[str]] = {}
            for ldu in ldus:
                pg = ldu.page_refs[0] if ldu.page_refs else 1
                pages_text.setdefault(pg, []).append(ldu.content.strip())

            for pg, parts in sorted(pages_text.items()):
                if self._llm_calls >= self._budget_max:
                    break
                page_text = "\n".join(p for p in parts if p)
                if len(page_text) < 50:
                    continue
                self._llm_calls += 1
                page_facts = _llm_extract_page_facts(
                    page_text=page_text,
                    document_id=document_id,
                    page=pg,
                    ollama_base_url=self.ollama_base_url,
                    model=self.ollama_model,
                )
                for f in page_facts:
                    _add(f)
            log.info(
                "Scanned-page LLM pass complete (doc=%s, total_llm_calls=%d)",
                document_id, self._llm_calls,
            )

        log.info(
            "Extracted %d facts from %d LDUs (doc=%s, llm_calls=%d)",
            len(facts), len(ldus), document_id, self._llm_calls,
        )
        return facts

    # ------------------------------------------------------------------
    # Persistence (extended schema)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_extended_columns(conn: sqlite3.Connection) -> None:
        """Add enriched columns to fact_tables if they don't exist."""
        cursor = conn.execute("PRAGMA table_info(fact_tables)")
        existing = {row[1] for row in cursor.fetchall()}
        new_cols = {
            "entity": "TEXT DEFAULT ''",
            "metric": "TEXT DEFAULT ''",
            "period": "TEXT DEFAULT ''",
            "section": "TEXT DEFAULT ''",
            "bbox_json": "TEXT DEFAULT ''",
            "extraction_method": "TEXT DEFAULT 'regex'",
            "confidence": "REAL DEFAULT 0.0",
        }
        for col, typedef in new_cols.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE fact_tables ADD COLUMN {col} {typedef}")
                log.info("Added column fact_tables.%s", col)

    def persist_to_db(
        self,
        facts: list[Fact],
        db_path: str | Path | None = None,
    ) -> int:
        """Insert *facts* into the ``fact_tables`` SQLite table.

        Automatically adds enriched columns if they don't exist.
        Returns the number of rows inserted.
        """
        db = Path(db_path) if db_path else _DEFAULT_DB
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(str(db))
        try:
            self._ensure_extended_columns(conn)
            for f in facts:
                bbox_json = ""
                if f.bbox:
                    bbox_json = json.dumps(f.bbox.model_dump())
                conn.execute(
                    """
                    INSERT INTO fact_tables
                        (document_id, key, value, unit, page_ref, content_hash,
                         entity, metric, period, section, bbox_json,
                         extraction_method, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f.document_id, f.key, f.value, f.unit, f.page_ref,
                        f.content_hash, f.entity, f.metric, f.period,
                        f.section, bbox_json, f.extraction_method,
                        f.confidence, now,
                    ),
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
        entity: str | None = None,
        period: str | None = None,
        min_confidence: float = 0.0,
        db_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Query facts from the ``fact_tables`` table.

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
            clauses = ["document_id = ?"]
            params: list[Any] = [document_id]

            if key_pattern:
                clauses.append("key LIKE ?")
                params.append(key_pattern)
            if entity:
                clauses.append("entity LIKE ?")
                params.append(f"%{entity}%")
            if period:
                clauses.append("period LIKE ?")
                params.append(f"%{period}%")
            if min_confidence > 0:
                clauses.append("confidence >= ?")
                params.append(min_confidence)

            sql = f"SELECT * FROM fact_tables WHERE {' AND '.join(clauses)}"
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()

        return [dict(r) for r in rows]

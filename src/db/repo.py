"""SQLite repository — thin helper layer over `.refinery/refinery.db`.

All writes are upsert-safe (INSERT OR REPLACE) so re-running the
pipeline on the same document is idempotent.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.db.init_db import initialize_database

_DEFAULT_DB = Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"


class RefineryRepo:
    """Convenience wrapper for all database operations.

    Ensures the DB is initialised on first use and provides typed
    insert/query helpers for every table.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB
        initialize_database(self._db_path)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # documents
    # ------------------------------------------------------------------

    def upsert_document(
        self,
        document_id: str,
        source_filename: str,
        origin_type: str,
        layout_complexity: str,
        domain_hint: str,
        estimated_cost: str,
        page_count: int,
        total_chunks: int,
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO documents
            (document_id, source_filename, origin_type, layout_complexity,
             domain_hint, estimated_cost, processing_timestamp,
             page_count, total_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (
                document_id, source_filename, origin_type,
                layout_complexity, domain_hint, estimated_cost,
                ts, page_count, total_chunks,
            ))

    def get_document(self, document_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY processing_timestamp DESC").fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # chunks
    # ------------------------------------------------------------------

    def upsert_chunk(
        self,
        chunk_id: str,
        document_id: str,
        page_number: int,
        chunk_type: str,
        content: str,
        content_hash: str,
        section_path: str = "",
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO chunks
            (chunk_id, document_id, page_number, chunk_type,
             content, content_hash, section_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (
                chunk_id, document_id, page_number,
                chunk_type, content, content_hash, section_path,
            ))

    def upsert_chunks_batch(self, rows: list[dict]) -> int:
        """Bulk upsert chunks. Each dict must have the standard keys."""
        sql = """
        INSERT OR REPLACE INTO chunks
            (chunk_id, document_id, page_number, chunk_type,
             content, content_hash, section_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.executemany(sql, [
                (r["chunk_id"], r["document_id"], r["page_number"],
                 r["chunk_type"], r["content"], r["content_hash"],
                 r.get("section_path", ""))
                for r in rows
            ])
        return len(rows)

    def get_chunks(self, document_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY page_number",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # structured_tables
    # ------------------------------------------------------------------

    def upsert_table(
        self,
        table_id: str,
        document_id: str,
        page_number: int,
        table_json: str,
        bbox_json: str,
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO structured_tables
            (table_id, document_id, page_number, table_json, bbox_json)
        VALUES (?, ?, ?, ?, ?)
        """
        with self._conn() as conn:
            conn.execute(sql, (table_id, document_id, page_number, table_json, bbox_json))

    # ------------------------------------------------------------------
    # provenance_ledger
    # ------------------------------------------------------------------

    def append_provenance(
        self,
        document_id: str,
        action: str,
        chunk_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        sql = """
        INSERT INTO provenance_ledger (document_id, chunk_id, action, timestamp, metadata_json)
        VALUES (?, ?, ?, ?, ?)
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (
                document_id, chunk_id, action, ts,
                json.dumps(metadata) if metadata else None,
            ))

    # ------------------------------------------------------------------
    # page_indexes
    # ------------------------------------------------------------------

    def upsert_page_index(
        self,
        document_id: str,
        source_filename: str,
        tree_json: str,
        node_count: int,
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO page_indexes
            (document_id, source_filename, tree_json, node_count, created_at)
        VALUES (?, ?, ?, ?, ?)
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (document_id, source_filename, tree_json, node_count, ts))

    def get_page_index(self, document_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM page_indexes WHERE document_id = ?",
                (document_id,),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # fact_tables
    # ------------------------------------------------------------------

    def upsert_facts_batch(self, document_id: str, facts: list[dict]) -> int:
        """Insert fact-table rows. Deletes old facts for the document first."""
        with self._conn() as conn:
            conn.execute("DELETE FROM fact_tables WHERE document_id = ?", (document_id,))
            ts = datetime.now(timezone.utc).isoformat()
            conn.executemany(
                """INSERT INTO fact_tables
                   (document_id, key, value, unit, page_ref, content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (document_id, f["key"], f["value"],
                     f.get("unit", ""), f.get("page_ref", 0),
                     f.get("content_hash", ""), ts)
                    for f in facts
                ],
            )
        return len(facts)

    def get_facts(self, document_id: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM fact_tables WHERE document_id = ? ORDER BY key",
                (document_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # query_logs
    # ------------------------------------------------------------------

    def log_query(
        self, query_text: str, result_count: int,
        latency_ms: float, source_documents: str = "",
    ) -> None:
        sql = """
        INSERT INTO query_logs (query_text, timestamp, result_count, latency_ms, source_documents)
        VALUES (?, ?, ?, ?, ?)
        """
        ts = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(sql, (query_text, ts, result_count, latency_ms, source_documents))

    def get_query_history(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

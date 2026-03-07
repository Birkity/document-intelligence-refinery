"""Idempotent SQLite database initialiser.

Creates (or verifies) the Refinery governance database at
``.refinery/refinery.db`` using the DDL in ``schema.sql``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_DEFAULT_DB = Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"


_FACT_TABLE_MIGRATIONS: list[tuple[str, str]] = [
    ("entity", "TEXT DEFAULT ''"),
    ("metric", "TEXT DEFAULT ''"),
    ("period", "TEXT DEFAULT ''"),
    ("section", "TEXT DEFAULT ''"),
    ("bbox_json", "TEXT DEFAULT ''"),
    ("extraction_method", "TEXT DEFAULT 'regex'"),
    ("confidence", "REAL DEFAULT 1.0"),
]


def _migrate_fact_tables(conn: sqlite3.Connection) -> None:
    """Add any missing enriched columns to the fact_tables table.

    Skipped entirely when the table does not yet exist (fresh database);
    the DDL will create it with all columns.
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fact_tables'"
    )
    if cursor.fetchone() is None:
        return  # fresh DB — table will be created by the DDL

    cursor = conn.execute("PRAGMA table_info(fact_tables)")
    existing = {row[1] for row in cursor.fetchall()}
    for col_name, col_def in _FACT_TABLE_MIGRATIONS:
        if col_name not in existing:
            conn.execute(
                f"ALTER TABLE fact_tables ADD COLUMN {col_name} {col_def}"
            )


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create the SQLite database and run the schema DDL.

    The operation is idempotent — all tables use ``CREATE TABLE IF NOT EXISTS``.
    For existing databases, any missing enriched columns are added via ALTER TABLE.

    Parameters
    ----------
    db_path : str | Path | None
        Explicit database file path.  Defaults to ``.refinery/refinery.db``.

    Returns
    -------
    Path
        Absolute path to the database file.
    """
    db = Path(db_path) if db_path else _DEFAULT_DB
    db.parent.mkdir(parents=True, exist_ok=True)

    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")

    conn = sqlite3.connect(str(db))
    try:
        # Migrate existing DB first so the DDL indexes succeed on old schemas.
        _migrate_fact_tables(conn)
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()

    return db

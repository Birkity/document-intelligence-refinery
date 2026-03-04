"""Idempotent SQLite database initialiser.

Creates (or verifies) the Refinery governance database at
``.refinery/refinery.db`` using the DDL in ``schema.sql``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
_DEFAULT_DB = Path(__file__).resolve().parents[2] / ".refinery" / "refinery.db"


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create the SQLite database and run the schema DDL.

    The operation is idempotent — all tables use ``CREATE TABLE IF NOT EXISTS``.

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
        conn.executescript(ddl)
        conn.commit()
    finally:
        conn.close()

    return db

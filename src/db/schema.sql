-- SQLite schema for the Document Intelligence Refinery governance database.
-- All tables use IF NOT EXISTS for idempotent initialisation.

-- ── Documents table ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    origin_type TEXT NOT NULL,
    layout_complexity TEXT NOT NULL,
    domain_hint TEXT NOT NULL,
    estimated_cost TEXT NOT NULL,
    processing_timestamp TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL
);

-- ── Chunks table ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_type TEXT NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    section_path TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_number);

-- ── Structured tables ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS structured_tables (
    table_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    table_json TEXT NOT NULL,
    bbox_json TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_tables_document ON structured_tables(document_id);

-- ── Provenance ledger ────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS provenance_ledger (
    ledger_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    chunk_id TEXT,
    action TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    metadata_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(document_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id)
);

CREATE INDEX IF NOT EXISTS idx_ledger_document ON provenance_ledger(document_id);
CREATE INDEX IF NOT EXISTS idx_ledger_timestamp ON provenance_ledger(timestamp);

-- ── PageIndex storage ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS page_indexes (
    document_id TEXT PRIMARY KEY,
    source_filename TEXT NOT NULL,
    tree_json TEXT NOT NULL,
    node_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

-- ── Query logs ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    source_documents TEXT
);

CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON query_logs(timestamp);

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

-- ── Fact tables (structured key-value extraction) ─────────────────────────
CREATE TABLE IF NOT EXISTS fact_tables (
    fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    unit TEXT DEFAULT '',
    page_ref INTEGER NOT NULL,
    content_hash TEXT DEFAULT '',
    entity TEXT DEFAULT '',
    metric TEXT DEFAULT '',
    period TEXT DEFAULT '',
    section TEXT DEFAULT '',
    bbox_json TEXT DEFAULT '',
    extraction_method TEXT DEFAULT 'regex',
    confidence REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_document ON fact_tables(document_id);
CREATE INDEX IF NOT EXISTS idx_facts_key ON fact_tables(key);
CREATE INDEX IF NOT EXISTS idx_facts_entity ON fact_tables(entity);
CREATE INDEX IF NOT EXISTS idx_facts_period ON fact_tables(period);

-- ── Entity mentions ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS entity_mentions (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    mention_count INTEGER DEFAULT 0,
    aliases_json TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_document ON entity_mentions(document_id);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entity_mentions(entity_name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entity_mentions(entity_type);

-- ── Knowledge graph edges ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    relation TEXT NOT NULL,
    page_ref INTEGER DEFAULT 0,
    confidence REAL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(document_id)
);

CREATE INDEX IF NOT EXISTS idx_kgedges_document ON knowledge_graph_edges(document_id);
CREATE INDEX IF NOT EXISTS idx_kgedges_source ON knowledge_graph_edges(source_node);

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

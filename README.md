# The Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous PDF documents and emits structured, queryable, spatially-indexed knowledge. Built as a Forward Deployed Engineer (FDE) engagement tool for enterprise document intelligence.

## Architecture

```
PDF Corpus
    │
    ▼
┌──────────────┐     ┌────────────────────┐     ┌─────────────────┐
│ Triage Agent │────▶│ Extraction Router   │────▶│ Chunking Engine │
│ (Stage 1)    │     │ A → B → C escalate │     │ (Stage 3)       │
└──────────────┘     └────────────────────┘     └─────────────────┘
        │                     │                         │
  DocumentProfile    .refinery/ledger             List[LDU]
                                                ┌───────┴───────┐
                                                │               │
                                          ┌─────▼─────┐  ┌─────▼────┐
                                          │ PageIndex  │  │ ChromaDB │
                                          │ (Stage 4)  │  │ Vectors  │
                                          └─────┬─────┘  └─────┬────┘
                                                │               │
                                          ┌─────▼───────────────▼────┐
                                          │    Query Agent (Stage 5) │
                                          │  + FactTable + Audit     │
                                          └──────────────────────────┘
```

### Pipeline Stages

| Stage | Component | Status | Description |
|-------|-----------|--------|-------------|
| 1 | Triage Agent | Done | Classifies docs by origin, layout, domain, cost tier |
| 2 | Extraction Router | Done | Three strategies (A/B/C) with confidence-gated escalation |
| 3 | Chunking Engine | Done | Converts `ExtractedDocument` → `List[LDU]` with 5 enforced rules |
| 4 | PageIndex Builder | Done | Hierarchical section navigation for LLM traversal |
| 5 | Query Agent | Done | LangGraph agent with 3 tools, FactTable, Audit Mode |
| - | CLI Orchestrator | Done | Typer CLI with batch/sample-page support |
| - | SQLite + ChromaDB | Done | Full persistence layer with metadata filters |
| - | OCR Backends | Done | PaddleOCR / Tesseract with PyMuPDF rendering |

### Extraction Strategies

| Strategy | Cost | Backend | When Used |
|----------|------|---------|-----------|
| **A** — Fast Text | Low | pdfplumber | Native-digital, single-column PDFs |
| **B** — Layout-Aware | Medium | Docling + pdfplumber fallback | Multi-column, table-heavy documents |
| **C** — Vision/OCR | High | PaddleOCR / Tesseract / pdfplumber fallback | Scanned images, low-confidence docs |

## Quick Start

### 1. Install

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # Windows
# or: source venv/bin/activate   # Linux/Mac

pip install -e ".[dev]"

# Optional OCR backends:
pip install -e ".[ocr]"        # PaddleOCR
pip install -e ".[tesseract]"  # Tesseract
```

### 2. Initialize the database

```bash
python -m src.cli init-db
```

### 3. Process a single PDF

```bash
python -m src.cli run data/sample.pdf --sample-pages 3
```

### 4. Batch-process all PDFs

```bash
python -m src.cli batch data/ --sample-pages 3
```

### 5. Query indexed documents

```bash
python -m src.cli query "What is the total revenue?"
python -m src.cli query "Net profit" --doc-id abc123
```

### 6. Audit a document

```bash
python -m src.cli audit --doc-id abc123
```

### 7. Inspect artefacts

```bash
python -m src.cli show pageindex --doc-id abc123
python -m src.cli show facts --doc-id abc123
python -m src.cli list-docs
```
from src.agents.chunker import ChunkingEngine
from src.agents.pageindex import PageIndexBuilder

agent = TriageAgent()
router = ExtractionRouter()
chunker = ChunkingEngine()
builder = PageIndexBuilder()

profile = agent.generate_document_profile('data/CBE ANNUAL REPORT 2023-24.pdf')
doc, _ = router.route_and_extract(profile, 'data/CBE ANNUAL REPORT 2023-24.pdf')
ldus = chunker.chunk_document(doc)
pi = builder.build(ldus, source_filename=profile.source_filename, document_id=profile.document_id)

builder.save_json(pi)  # → .refinery/pageindex/{doc_id}.json
print(f'Sections: {len(pi.root_nodes)}')
for node in pi.root_nodes:
    print(f'  [{node.page_start}-{node.page_end}] {node.title}  ({', '.join(node.data_types_present) or 'text'})')
"
```

### 7. Query a document with provenance

```bash
python -c "
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter
from src.agents.chunker import ChunkingEngine
from src.agents.pageindex import PageIndexBuilder
from src.agents.fact_table import FactTableExtractor
from src.agents.query_agent import QueryAgent

agent = TriageAgent()
router = ExtractionRouter()
chunker = ChunkingEngine()
builder = PageIndexBuilder()

profile = agent.generate_document_profile('data/CBE ANNUAL REPORT 2023-24.pdf')
doc, _ = router.route_and_extract(profile, 'data/CBE ANNUAL REPORT 2023-24.pdf')
ldus = chunker.chunk_document(doc)
pi = builder.build(ldus, source_filename=profile.source_filename, document_id=profile.document_id)

# Build query agent
qa = QueryAgent()
qa.ingest_ldus(ldus, document_id=profile.document_id, source_filename=profile.source_filename)
qa.register_page_index(pi)

# Extract and persist facts
fact_ext = FactTableExtractor()
facts = fact_ext.extract(ldus, document_id=profile.document_id)
fact_ext.persist_to_db(facts)

# Ask a question
result = qa.answer('What is the total revenue?', document_id=profile.document_id)
print(f'Answer: {result.answer}')
print(f'Tools: {result.tools_used}')
for c in result.provenance.citations:
    print(f'  Source: {c.document_name} p.{c.page_number}')
"
```

### 8. Generate profile artifacts (batch)

```bash
python scripts/generate_profiles.py
python scripts/generate_ledger.py
```

### 9. Verify class coverage

```bash
python scripts/ensure_class_coverage.py   # Check and generate missing profiles
python scripts/generate_class_report.py   # Generate classification report
```

Ensures at least 3 documents per class (A/B/C/D).

### 10. Generate the interim report

```bash
python scripts/generate_interim_report.py
```

Produces `interim_submission.tex`.

### 11. Run tests

```bash
python -m pytest tests/ -v
```

**143 tests passing** across 9 test modules.

## Project Structure

```
├── rubric/
│   └── extraction_rules.yaml      # All thresholds — no hardcoding
├── src/
│   ├── cli.py                     # Typer CLI entry point (7 commands)
│   ├── models/
│   │   ├── __init__.py            # Re-exports all schemas
│   │   └── schemas.py             # Pydantic v2 schemas (14 models)
│   ├── agents/
│   │   ├── triage.py              # Stage 1 — Triage Agent
│   │   ├── extractor.py           # Stage 2 — ExtractionRouter (A→B→C)
│   │   ├── chunker.py             # Stage 3 — ChunkingEngine + ChunkValidator
│   │   ├── pageindex.py           # Stage 4 — PageIndexBuilder + query
│   │   ├── fact_table.py          # Stage 5 — FactTable extractor (SQLite)
│   │   └── query_agent.py         # Stage 5 — LangGraph Query Agent + Audit
│   ├── strategies/
│   │   ├── base.py                # BaseExtractor abstract interface
│   │   ├── fast_text.py           # Strategy A — pdfplumber (low cost)
│   │   ├── layout.py              # Strategy B — Docling / enhanced pdfplumber
│   │   └── vision.py              # Strategy C — OCR / VLM (high cost)
│   ├── vision/
│   │   ├── __init__.py
│   │   └── ocr_backends.py        # PaddleOCR + Tesseract backends
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py        # PipelineOrchestrator (end-to-end)
│   ├── utils/
│   │   └── hash_utils.py          # SHA-256 content hashing for provenance
│   └── db/
│       ├── schema.sql             # SQLite DDL (7 tables incl. fact_tables)
│       ├── init_db.py             # Idempotent DB init
│       ├── repo.py                # SQLite repository layer (upsert helpers)
│       └── vector_store.py        # ChromaDB wrapper with metadata filters
├── scripts/
│   ├── generate_profiles.py       # Batch profile generation (12 docs)
│   ├── generate_ledger.py         # Batch extraction + ledger
│   ├── generate_interim_report.py # LaTeX report generator
│   ├── ensure_class_coverage.py   # Class coverage checker
│   └── generate_class_report.py   # Classification report generator
├── tests/
│   ├── test_models.py             # Schema validation (11 tests)
│   ├── test_triage_origin.py      # Origin-type detection (6 tests)
│   ├── test_triage_layout.py      # Layout, domain, cost (18 tests)
│   ├── test_extraction_router.py  # Escalation logic + ledger (8 tests)
│   ├── test_db_and_schemas.py     # DB + PageIndex + Provenance (10 tests)
│   ├── test_chunking_engine.py    # Chunking rules + validator (29 tests)
│   ├── test_pageindex_builder.py  # PageIndex builder + query (21 tests)
│   ├── test_fact_table.py         # FactTable extraction + persistence (12 tests)
│   └── test_query_agent.py        # Query Agent tools + audit (19 tests)
├── .refinery/
│   ├── profiles/                  # DocumentProfile JSONs (12+, 3 per class)
│   ├── pageindex/                 # PageIndex tree JSONs per document
│   ├── extraction_ledger.jsonl    # Extraction audit trail
│   ├── chroma_store/              # ChromaDB vector store (LDU embeddings)
│   └── refinery.db                # SQLite governance DB (7 tables)
├── pyproject.toml
├── README.md
└── class_coverage_report.txt      # Document class verification
```

## Pydantic Schemas

All data contracts are defined in `src/models/schemas.py`:

| Model | Purpose |
|-------|---------|
| `BoundingBox` | Spatial coordinates (x1, y1, x2, y2, page) |
| `TextBlock` | Content + bounding box |
| `TableObject` | Headers + rows + bounding box |
| `FigureObject` | Caption + bounding box + figure type |
| `ExtractedPage` | Text blocks, tables, figures, reading order |
| `ExtractedDocument` | Full document extraction (strategy-agnostic) |
| `DocumentProfile` | Triage classification output |
| `LDU` | Logical Document Unit — RAG-ready chunk |
| `PageIndexNode` | Hierarchical section node |
| `PageIndex` | Smart table of contents for LLM navigation |
| `ProvenanceCitation` | Source citation (doc, page, bbox, hash) |
| `ProvenanceChain` | Ordered citations for auditable answers |
| `Fact` | Key-value numerical fact (for FactTable) |
| `QueryResult` | Agent answer with provenance + tools used |

## LLM Routing

| Component | LLM Backend | Notes |
|-----------|-------------|-------|
| Strategy D (Vision) | **OpenRouter** (`google/gemma-3-27b-it:free`) | Only for extreme OCR failures; budget-gated |
| PageIndex summaries | **Ollama local** (`qwen3-coder:480b-cloud`) | Integrated; deterministic fallback on failure |
| QueryAgent synthesis | **Ollama local** (`qwen3-coder:480b-cloud`) | Integrated; deterministic fallback on failure |
| Audit reasoning | **Ollama local** (`qwen3-coder:480b-cloud`) | Integrated; skipped if Ollama unreachable |
| FactTable extraction | **Ollama local** (`qwen3-coder:480b-cloud`) | Budget-gated (20 calls/doc); falls back silently |
| Triage, Extraction A/B/C, Chunking | None | Fully deterministic |

All Ollama calls use the OpenAI-compatible endpoint at `http://localhost:11434/v1`.
Ensure `ollama run qwen3-coder:480b-cloud` is running before executing the pipeline.

## Configuration

All thresholds are externalized in `rubric/extraction_rules.yaml`:

| Section | Key Parameters |
|---------|---------------|
| `origin_detection` | `min_char_density: 0.001`, `scanned_image_ratio: 0.7` |
| `layout_detection` | `table_area_ratio_threshold: 0.3`, `column_variance_threshold: 0.25` |
| `escalation` | `strategy_a_min_confidence: 0.6`, `strategy_b_min_confidence: 0.5` |
| `chunking` | `max_tokens_per_chunk: 512`, `min_tokens_per_chunk: 50`, `overlap_tokens: 64` |
| `review` | `flag_confidence_below: 0.4` |
| `domain_keywords` | financial, legal, technical, medical keyword lists |

A new document type can be onboarded by modifying only `extraction_rules.yaml` — no code changes required.

## Extraction Strategy Tiers

| Tier | Strategy | Cost | When Used |
|------|----------|------|-----------|
| A | FastTextExtractor | Low | `native_digital` + `single_column` |
| B | LayoutExtractor | Medium | `multi_column`, `table_heavy`, `mixed` origin |
| C | VisionExtractor | High | `scanned_image` (direct — skips A/B) or low-confidence fallback |

**Escalation Guard:** Strategy A confidence < 0.6 → escalate to B. Strategy B confidence < 0.5 → escalate to C. Scanned-image documents route directly to Strategy C (no pointless intermediate steps).

## Semantic Chunking Engine (Stage 3)

The `ChunkingEngine` converts `ExtractedDocument` → `List[LDU]` following five enforced chunking rules (the "Constitution"):

| # | Rule | Enforcement |
|---|------|-------------|
| 1 | A table cell is never split from its header row | Tables emit as single LDU with headers + rows |
| 2 | A figure caption is always stored as metadata of its parent figure chunk | `FigureObject.caption` embedded in figure LDU content |
| 3 | A numbered/bullet list is kept as a single LDU unless it exceeds `max_tokens` | List detection via regex; overflow split at item boundaries |
| 4 | Section headers propagate as `parent_section` on all child chunks | Header detection by length + capitalization heuristics |
| 5 | Cross-references (e.g. "see Table 3") are preserved in chunk content | References remain in the paragraph text verbatim |

Each LDU carries: `content`, `chunk_type`, `page_refs`, `bbox`, `parent_section`, `token_count`, and a SHA-256 `content_hash` for provenance verification.

The `ChunkValidator` checks every emitted LDU for: non-empty content, token limits, hash integrity, and valid page references.

## PageIndex Builder (Stage 4)

The `PageIndexBuilder` creates a hierarchical navigation tree from `List[LDU]` — the equivalent of a "smart table of contents" that an LLM can traverse to locate sections without embedding-searching the full corpus.

**`build()` executes four explicit phases:**

```
extract sections         (_group_by_section)
      ↓
build PageIndex tree     (_build_node × N  — deterministic summaries)
      ↓
LLM enriches summaries   (_enrich_summaries_llm  — local Ollama)
      ↓
store in PageIndex JSON  (save_json / persist_to_db)
```

Both summary types always run: the deterministic first-N-sentences summary is generated in Phase 2 as a guaranteed fallback. Phase 3 then walks every node and replaces it with an Ollama-generated summary where the call succeeds.

**Other features:**
- Groups LDUs by `parent_section`, preserving document order
- Computes page ranges (`page_start` / `page_end`) from content LDU page refs
- Detects data-type signals per section: `tables`, `figures`, `lists`, `numeric_dense`
- Extracts lightweight key entities via capitalised multi-word patterns
- **Query API**: `builder.query(pi, topic="revenue growth", top_n=3)` returns top-N matching sections via bag-of-words scoring
- **Persistence**: JSON to `.refinery/pageindex/` + SQLite `page_indexes` table (upsert)

| Signal | Detection Logic |
|--------|-----------------|
| `tables` | Any LDU with `chunk_type="table"` |
| `figures` | Any LDU with `chunk_type="figure"` |
| `lists` | Any LDU with `chunk_type="list"` |
| `numeric_dense` | ≥15% of tokens in section content contain digits |

## Query Agent & Provenance Layer (Stage 5)

The `QueryAgent` is a LangGraph agent with three tools and full provenance tracking:

### Three Tools

| Tool | Backend | Purpose |
|------|---------|--------|
| `pageindex_navigate` | PageIndex tree | Navigate to relevant sections by topic — avoids full-corpus search |
| `semantic_search` | ChromaDB vectors | Embedding-based retrieval over all ingested LDU chunks |
| `structured_query` | SQLite `fact_tables` | Precise key-value lookups for numerical facts (revenue, costs, rates) |

### FactTable Extractor

Hybrid fact extraction pipeline using three methods in order:
1. **Regex** — colon-separated, pipe-table, and whitespace-separated key-value patterns
2. **Table parse** — column-aware parsing of pipe-separated table LDUs
3. **LLM-assisted via local Ollama** — budget-gated (max 20 calls/doc), always enabled; falls back silently if Ollama is down

Facts are persisted to the `fact_tables` SQLite table with enriched fields: entity, metric, period, confidence, extraction_method.

### Audit Mode

Given a claim (e.g. "The report states revenue was $4.2B in Q3"), the system:
1. Searches for supporting evidence across all three tools
2. Computes token overlap (excluding stop words) to verify relevance
3. Returns a `ProvenanceChain` with `verified=True` + citations, or `verified=False` (unverifiable)

### Provenance

Every answer returns a `QueryResult` containing:
- `answer` — composed from retrieved evidence
- `provenance` — `ProvenanceChain` with per-citation `document_name`, `page_number`, `bbox`, `content_hash`
- `tools_used` — which tools contributed to the answer
- `confidence` — score based on evidence count

## Document Class Coverage

Validated across 4 document classes with 12+ profiled documents (minimum 3 per class):

| Class | Type | Example Documents |
|-------|------|-------------------|
| A | Native Digital Financial | CBE Annual Reports, financial statements |
| B | Scanned Government/Legal | DBE Audit Reports, audited financial statements |
| C | Technical Assessment | FTA Performance Survey, pharmaceutical assessments |
| D | Structured Data (table-heavy) | CPI reports, tax expenditure data |

Verify coverage: `python scripts/generate_class_report.py`

## Testing

**167 tests** across 10 modules, run with `python -m pytest tests/ -v`:

- **test_models.py** — Schema instantiation, validation, rejection of invalid data (11 tests)
- **test_triage_origin.py** — Origin-type detection (native digital, scanned, mixed) (6 tests)
- **test_triage_layout.py** — Layout complexity, domain hint, extraction cost estimation (18 tests)
- **test_extraction_router.py** — Strategy selection, escalation chains, ledger persistence (8 tests)
- **test_db_and_schemas.py** — Database init (7 tables), PageIndex, ProvenanceChain serialization (10 tests)
- **test_chunking_engine.py** — All 5 chunking rules, content hashing, validator, end-to-end mixed docs (29 tests)
- **test_pageindex_builder.py** — Section grouping, page ranges, data-type signals, summaries, JSON/DB persistence, query (21 tests)
- **test_fact_table.py** — FactTable extraction, persistence, key-pattern queries (12 tests)
- **test_query_agent.py** — 3 tools (pageindex/semantic/structured), answer provenance, audit mode, LangGraph graph (19 tests)
- **test_pipeline_and_cli.py** — Repo layer, VectorStore filters, sample-page selection, OCR backends, CLI smoke, orchestrator (24 tests)

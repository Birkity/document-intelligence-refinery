# Pipeline Testing Guide

## Overview

Comprehensive testing guide for the Document Intelligence Refinery — a five-stage agentic pipeline for PDF document extraction, chunking, indexing, and querying. Covers unit tests, integration tests, and manual validation of all pipeline components.

**Test suite: 190 tests across 12 test files.**

---

## Prerequisites

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# 2. Install dependencies (with dev tools)
pip install -e ".[dev]"

# 3. Copy and configure environment
cp .env.example .env
# Edit .env:
#   OPENROUTER_API_KEY  — required only for Strategy D (Vision LLM)
#   OLLAMA_BASE_URL     — local Ollama (default: http://localhost:11434/v1)
#   OLLAMA_MODEL        — local model name (default: qwen3-coder:480b-cloud)
```

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENROUTER_API_KEY` | For Vision tier only | `""` | OpenRouter API key (Strategy D) |
| `VISION_MODEL` | No | `google/gemma-3-27b-it:free` | Vision LLM model (OpenRouter) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Local Ollama server URL |
| `OLLAMA_MODEL` | No | `qwen3-coder:480b-cloud` | Local Ollama model name |
| `ENABLE_VISION_EXTRACTION` | No | `false` | Feature flag for vision tier |
| `ENABLE_LLM_FACT_EXTRACTION` | No | `true` | Feature flag for LLM fact extraction |
| `BUDGET_MAX_PER_DOCUMENT` | No | `0.50` | Max USD per document |
| `BUDGET_MAX_VISION_CALLS` | No | `5` | Max vision calls per doc |
| `BUDGET_MAX_LLM_CALLS` | No | `20` | Max Ollama LLM calls per doc |

> **LLM routing**: OpenRouter is used **only** for Strategy D vision extraction.
> All other LLM usage (PageIndex summaries, QueryAgent reasoning, FactTable extraction) routes to the **local Ollama** server.
> If Ollama is down, all three fall back to deterministic methods automatically.

See `.env.example` for the full list.

---

## Running Tests

### Full Suite (190 tests)
```bash
python -m pytest tests/ -v --tb=short
```

### Quick Smoke Test
```bash
python -m pytest tests/ -x --tb=short -q
```

### By Pipeline Stage

```bash
# ── Stage 1: Triage Agent ──
python -m pytest tests/test_triage_origin.py tests/test_triage_layout.py -v

# ── Stage 2: Extraction Router & Strategies ──
python -m pytest tests/test_extraction_router.py -v

# ── Stage 3: Chunking Engine ──
python -m pytest tests/test_chunking_engine.py -v

# ── Stage 4: PageIndex Builder ──
python -m pytest tests/test_pageindex_builder.py -v

# ── Stage 5a: FactTable Extractor ──
python -m pytest tests/test_fact_table.py -v

# ── Stage 5b: Entity Linker & Knowledge Graph ──
python -m pytest tests/test_entity_linker.py -v

# ── Stage 5c: Query Agent + Audit Mode ──
python -m pytest tests/test_query_agent.py -v

# ── Infrastructure: DB, Models, Schemas ──
python -m pytest tests/test_db_and_schemas.py tests/test_models.py -v

# ── Integration: Pipeline & CLI ──
python -m pytest tests/test_pipeline_and_cli.py -v
```

---

## Test File → Pipeline Stage Mapping

| Test File | Stage | Components Tested |
|---|---|---|
| `test_triage_origin.py` | 1 | `detect_origin_type()`, native/scanned/mixed classification |
| `test_triage_layout.py` | 1 | Layout complexity, domain hint, cost estimation, profile generation |
| `test_extraction_router.py` | 2 | 4-tier chain (A→B→C→D), escalation logic, ledger writing |
| `test_chunking_engine.py` | 3 | Paragraph/table/figure/list chunking, cross-ref detection, hash validation |
| `test_pageindex_builder.py` | 4 | Section indexing, data type detection, query, JSON/DB persistence |
| `test_fact_table.py` | 5a | Regex + table_parse extraction, enriched fields, SQLite persistence |
| `test_entity_linker.py` | 5b | Entity extraction (org/date/amount/metric/regulation), KG construction |
| `test_query_agent.py` | 5c | 3-tool agent, provenance chains, audit mode, LangGraph graph |
| `test_models.py` | Cross | Pydantic v2 model validation across all schemas |
| `test_db_and_schemas.py` | Infra | SQLite init, 9 tables, model serialisation, vector store |
| `test_pipeline_and_cli.py` | Integ | Pipeline orchestrator, repo, vector store, CLI commands |
| `test_docling.py` | 2 (opt) | Docling layout extractor (requires docling install) |

---

## Stage-by-Stage Testing Details

### Stage 1: Triage Agent

Tests validate origin classification (native_digital / scanned_image / mixed) with confidence scoring, layout complexity detection (single_column / multi_column / table_heavy / figure_heavy), domain hinting (financial / legal / technical / medical / general), and extraction cost estimation.

**No LLM required.** All classification is heuristic (char density, image ratio, keyword matching).

```python
profile = triage.generate_document_profile("data/report.pdf")
assert profile.origin_type == "native_digital"
assert profile.confidence >= 0.7
```

### Stage 2: Extraction Router

4-tier extraction ladder with confidence-gated escalation:

| Tier | Strategy | Backend | Trigger |
|---|---|---|---|
| A | `fast_text` | pdfplumber | Default for native + simple layout |
| B | `layout_aware` | Docling / pdfplumber+ | Complex layouts, table-heavy |
| C | `ocr_heavy` | RapidOCR → EasyOCR → PyMuPDF | Scanned pages |
| D | `vision_augmented` | OpenRouter Vision LLM | Extreme difficulty (budget-gated) |

Tests cover strategy selection, confidence-gated escalation (A→B→C→D), page-level escalation, budget guards, review flagging, and ledger persistence.

**LLM: Only in Strategy D (disabled by default).**

### Stage 3: Chunking Engine

Validates the 5 chunking rules ("Constitution"):

| Rule | Constraint |
|---|---|
| 1 | Table cells never split from header row |
| 2 | Figure captions stored as metadata of parent figure |
| 3 | Numbered/bullet lists kept as single LDU (unless > max_tokens) |
| 4 | Section headers tracked as `parent_section` on child chunks |
| 5 | Cross-references detected and stored in chunk metadata |

Additional: content-hash determinism (SHA-256), token-count enforcement (max 512), table subgroup splitting.

**No LLM required.**

### Stage 4: PageIndex Builder

Tests verify section detection, page-range tracking, data-type signals (tables, figures, lists, numeric-dense), summary generation, cross-page sections, and JSON + SQLite persistence.

**PageIndex build flow (4 phases):**
1. **Extract sections** — group LDUs by `parent_section`
2. **Build tree** — construct nodes with deterministic summaries (first N sentences)
3. **LLM enrichment** — walk every node, call local Ollama (`qwen3-coder:480b-cloud`) to replace deterministic summaries; keeps deterministic if Ollama unreachable
4. **Return** — fully-enriched `PageIndex` ready for `save_json` / `persist_to_db`

> Both summary types always run: deterministic is the fallback, LLM is the primary.

```bash
# Tests mock the Ollama endpoint — no running server required for tests
python -m pytest tests/test_pageindex_builder.py -v
```

### Stage 5a: FactTable Extractor

Hybrid extraction: regex → table_parse → LLM-assisted.

Tests cover all regex patterns, enriched fields (entity, metric, period, section, extraction_method, confidence), `ALTER TABLE` migration for new columns, persistence, and querying with filters.

**LLM: Integrated via local Ollama** — budget-gated (max 20 calls/doc), always enabled (`ENABLE_LLM_FACT_EXTRACTION=true`). Uses `qwen3-coder:480b-cloud` locally. Falls back silently if Ollama is unreachable.

### Stage 5b: Entity Linker & Knowledge Graph

Tests cover organisation/person/date/amount/metric/location/regulation detection, fact enrichment, knowledge graph edge types (`has_metric`, `reported_for_period`, `co_occurs_on_page`, `references_*`), and edge deduplication.

**No LLM required.** Uses regex/heuristic NER.

### Stage 5c: Query Agent + Audit Mode

Tests validate:
- **pageindex_navigate** — topic search via bag-of-words scoring
- **semantic_search** — ChromaDB vector retrieval
- **structured_query** — SQL fact lookups with entity/period/confidence filters
- **answer()** — multi-hop retrieval + LLM synthesis (falls back to deterministic)
- **audit()** — verification with status: `verified` / `not_found` / `unverifiable` + LLM reasoning
- **LangGraph** — graph construction (`retrieve` → `structured` → `synthesise`)

**LLM: Integrated via local Ollama** — always attempted for answer synthesis and audit reasoning. Falls back to deterministic composition if Ollama is unreachable.

### Vector Store Ingestion

ChromaDB vector store with cosine distance, tested via:
- `test_db_and_schemas.py` — basic CRUD operations
- `test_pipeline_and_cli.py` — end-to-end ingestion via orchestrator
- `test_query_agent.py` — semantic search over ingested chunks

---

## Integration Testing

### End-to-End Pipeline

```bash
# Initialise database
python -m src.cli init-db

# Process a single PDF (3-page sample)
python -m src.cli run data/sample.pdf --sample-pages 3 -v

# Process with explicit page range
python -m src.cli run data/report.pdf --page-range 7-20 -v

# Batch-process a directory
python -m src.cli batch data/ --sample-pages 5 -v

# Query the processed document
python -m src.cli query "What is the total revenue?" --doc-id <doc_id>

# Audit a claim
python -m src.cli audit --doc-id <doc_id> --claim "Revenue was $4.2B"

# List processed documents
python -m src.cli list-docs

# Inspect artefacts
python -m src.cli show pageindex --doc-id <doc_id>
python -m src.cli show facts --doc-id <doc_id>
python -m src.cli show ledger
```

### Sample Page Strategies

```bash
# Head/mid/tail (default) — distributed across document thirds
python -m src.cli run data/report.pdf --sample-pages 10 -v

# Head only — first N pages
python -m src.cli run data/report.pdf --sample-pages 10 --page-strategy head -v

# Uniform — evenly spaced
python -m src.cli run data/report.pdf --sample-pages 10 --page-strategy uniform -v

# Random — N random pages
python -m src.cli run data/report.pdf --sample-pages 10 --page-strategy random -v
```

### Artefact Verification

After a pipeline run, check `.refinery/` for:

| Path | Contents |
|---|---|
| `.refinery/profiles/{doc_id}.json` | Document profile (triage output) |
| `.refinery/pageindex/{doc_id}.json` | PageIndex tree (hierarchical sections) |
| `.refinery/extraction_ledger.jsonl` | Extraction audit trail |
| `.refinery/refinery.db` | SQLite (9 tables: documents, chunks, facts, etc.) |
| `.refinery/chroma_store/` | ChromaDB vector embeddings |
| `.refinery/runs/{run_id}/` | Per-run JSON artefacts (profile, ldus, facts, KG) |

---

## Test Design Principles

1. **No network calls in standard tests** — Vision LLM and OpenRouter are mocked or feature-flagged off
2. **Deterministic** — All tests use fixed inputs, `tmp_path` for isolation
3. **Fast** — Full suite runs in ~2 minutes
4. **Backward-compatible** — Enriched schemas use defaults for optional fields
5. **No vision models in unit tests** — Feature-flagged off by default

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pydantic_settings` | Missing dependency | `pip install pydantic-settings` |
| Vision tests timeout | No API key | Set `ENABLE_VISION_EXTRACTION=false` |
| ChromaDB lock errors on Windows | Concurrent access | Use `tmp_path` in tests |
| EasyOCR slow first run | Model download | Expected; subsequent runs are fast |
| `langgraph` import error | Missing dependency | `pip install langgraph langchain-core` |

## Coverage Goals

- **Unit test coverage**: >85% across all pipeline stages
- **Integration coverage**: Full pipeline run with 3-page sample
- **No hardcoded secrets**: All API keys via `.env` / env vars
- **No vision model in standard tests**: Feature-flagged off by default
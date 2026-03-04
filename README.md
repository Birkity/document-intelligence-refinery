# The Document Intelligence Refinery

A production-grade, multi-stage agentic pipeline that ingests heterogeneous PDF documents and emits structured, queryable, spatially-indexed knowledge.

## Architecture

```
PDF Corpus
    │
    ▼
┌──────────────┐     ┌────────────────────┐     ┌─────────────────┐
│ Triage Agent │────▶│ Extraction Router   │────▶│ Chunking Engine │
│ (Stage 1)    │     │ A → B → C escalate │     │ (Stage 3)       │
└──────────────┘     └────────────────────┘     └─────────────────┘
                              │                         │
                    .refinery/ledger            ┌───────┴───────┐
                                                │               │
                                          ┌─────▼─────┐  ┌─────▼────┐
                                          │ PageIndex  │  │ ChromaDB │
                                          │ (Stage 4)  │  │ Vectors  │
                                          └─────┬─────┘  └─────┬────┘
                                                │               │
                                          ┌─────▼───────────────▼────┐
                                          │    Query Agent (Stage 5) │
                                          └──────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# or: .\venv\Scripts\Activate.ps1   # Windows

pip install -e ".[dev]"
```

### 2. Initialize the database

```bash
python -c "from src.db.init_db import initialize_database; initialize_database()"
```

This creates `.refinery/refinery.db` with all governance tables.

### 3. Run triage on a document

```bash
python -c "
from src.agents.triage import TriageAgent
agent = TriageAgent()
profile = agent.generate_document_profile('data/CBE ANNUAL REPORT 2023-24.pdf')
print(profile.model_dump_json(indent=2))
"
```

The profile JSON is saved to `.refinery/profiles/{document_id}.json`.

### 4. Run extraction with escalation

```bash
python -c "
from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

agent = TriageAgent()
router = ExtractionRouter()

profile = agent.generate_document_profile('data/CBE ANNUAL REPORT 2023-24.pdf')
doc, ledger = router.route_and_extract(profile, 'data/CBE ANNUAL REPORT 2023-24.pdf')
print(f'Pages: {len(doc.pages)}, Strategies tried: {[e[\"strategy_used\"] for e in ledger]}')
"
```

Ledger entries are appended to `.refinery/extraction_ledger.jsonl`.

### 5. Generate profile artifacts (batch)

```bash
python scripts/generate_profiles.py
python scripts/generate_ledger.py
```

### 6. Verify class coverage

```bash
python scripts/ensure_class_coverage.py   # Check and generate missing profiles
python scripts/generate_class_report.py   # Generate classification report
```

Ensures at least 3 documents per class (A: native digital financial, B: scanned, C: technical assessment, D: table-heavy).

### 7. Generate the interim report

```bash
python scripts/generate_interim_report.py
```

Produces `interim_submission.tex`.

### 8. Run tests

```bash
python -m pytest tests/ -v
```

All 61 tests passing.

## Project Structure

```
├── rubric/
│   └── extraction_rules.yaml      # All thresholds — no hardcoding
├── src/
│   ├── models/
│   │   └── schemas.py             # Pydantic schemas with source_filename tracking
│   ├── agents/
│   │   ├── triage.py              # Triage Agent (Stage 1)
│   │   └── extractor.py           # ExtractionRouter with A→B→C escalation
│   ├── strategies/
│   │   ├── base.py                # BaseExtractor interface
│   │   ├── fast_text.py           # Strategy A — pdfplumber
│   │   ├── layout.py              # Strategy B — Docling / enhanced pdfplumber
│   │   └── vision.py              # Strategy C — VLM / OCR
│   └── db/
│       ├── schema.sql             # SQLite DDL with source_filename column
│       ├── init_db.py             # Idempotent DB init
│       └── vector_store.py        # ChromaDB wrapper
├── scripts/
│   ├── generate_profiles.py       # Batch profile generation
│   ├── generate_ledger.py         # Batch extraction + ledger
│   ├── generate_interim_report.py # LaTeX report generator
│   ├── ensure_class_coverage.py   # Class coverage checker
│   └── generate_class_report.py   # Classification report generator
├── tests/
│   ├── test_models.py
│   ├── test_triage_origin.py
│   ├── test_triage_layout.py
│   ├── test_extraction_router.py
│   └── test_db_and_schemas.py
├── .refinery/
│   ├── profiles/                  # 31 DocumentProfile JSONs (4 classes)
│   ├── extraction_ledger.jsonl    # Extraction audit trail with filenames
│   └── refinery.db                # SQLite governance DB
├── pyproject.toml
├── README.md
└── class_coverage_report.txt      # Document class verification
```

## Configuration

All thresholds are in `rubric/extraction_rules.yaml`:
- Origin detection thresholds
- Layout complexity thresholds
- Escalation confidence thresholds
- Domain keyword lists
- Chunk size rules
- Review trigger thresholds

**No hardcoded thresholds in code.**

## Extraction Strategy Tiers

| Tier | Strategy | Cost | Speed | When Used |
|------|----------|------|-------|-----------|
| A | FastTextExtractor | Negligible | Fast | native_digital + single_column |
| B | LayoutExtractor | Moderate | Medium | multi_column, table_heavy, mixed |
| C | VisionExtractor | High | Slow | scanned_image, low-confidence fallback |

Escalation: If Strategy A confidence < 0.6 → try B. If B confidence < 0.5 → try C.

## Document Class Coverage

The system has been validated across 4 document classes with 31 profiled documents:

- **Class A**: Native Digital Financial Reports (5 documents) — CBE Annual Reports, financial statements
- **Class B**: Scanned Government/Legal Documents (20 documents) — Audit Report 2023, Audited Financial Statements
- **Class C**: Technical Assessment Reports (3 documents) — FTA Performance Survey, technical assessments
- **Class D**: Structured Data Reports (3 documents) — Consumer Price Index, table-heavy fiscal data

**All schemas track `source_filename`** — Every DocumentProfile and ExtractedDocument includes the original PDF filename for traceability.

Verify coverage: `python scripts/generate_class_report.py`

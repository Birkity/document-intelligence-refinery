"""Typer CLI for the Document Intelligence Refinery.

Commands
--------
init-db      Create / verify the SQLite database.
run          Process a single PDF through the full pipeline.
batch        Process all PDFs in a directory.
query        Run a natural-language query against indexed documents.
audit        Show audit / provenance trail for a document.
show         Inspect pipeline artefacts (pageindex, ledger, facts).
list-docs    List all processed documents.

Usage::

    python -m src.cli init-db
    python -m src.cli run data/report.pdf --sample-pages 3
    python -m src.cli batch data/ --sample-pages 3
    python -m src.cli query "What is the total revenue?" --doc-id abc123
    python -m src.cli audit --doc-id abc123
    python -m src.cli show pageindex --doc-id abc123
    python -m src.cli list-docs
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="refinery",
    help="Document Intelligence Refinery — PDF extraction & querying pipeline.",
    add_completion=False,
)

# ── Logging setup ─────────────────────────────────────────────────────────

_LOG_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format=_LOG_FMT, level=level, force=True)


# ── Helpers ───────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root


def _echo_json(data, indent: int = 2) -> None:
    """Pretty-print JSON data."""
    typer.echo(json.dumps(data, indent=indent, default=str))


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════


@app.command("init-db")
def init_db(
    db_path: Optional[str] = typer.Option(None, help="Path to SQLite database."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialise (or verify) the SQLite governance database."""
    _setup_logging(verbose)
    from src.db.init_db import initialize_database

    p = initialize_database(db_path)
    typer.echo(f"Database ready at {p}")


@app.command("run")
def run_pipeline(
    pdf_path: str = typer.Argument(..., help="Path to a single PDF file."),
    sample_pages: Optional[int] = typer.Option(
        None, "--sample-pages", "-s",
        help="Process only N sample pages (head/mid/tail).",
    ),
    page_strategy: str = typer.Option(
        "head_mid_tail", "--page-strategy",
        help="Page sampling strategy: head_mid_tail, head, uniform.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run the full extraction pipeline on a single PDF."""
    _setup_logging(verbose)
    from src.pipeline.orchestrator import PipelineOrchestrator

    pdf = Path(pdf_path)
    if not pdf.exists():
        typer.echo(f"Error: file not found — {pdf}", err=True)
        raise typer.Exit(1)

    orch = PipelineOrchestrator()
    result = orch.run(
        pdf,
        sample_pages=sample_pages,
        page_sample_strategy=page_strategy,
    )

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Run ID       : {result.run_id}")
    typer.echo(f"  Document ID  : {result.document_id}")
    typer.echo(f"  Origin       : {result.profile.origin_type}")
    typer.echo(f"  Layout       : {result.profile.layout_complexity}")
    typer.echo(f"  Cost tier    : {result.profile.estimated_extraction_cost}")
    typer.echo(f"  Pages        : {len(result.extracted_doc.pages)}")
    typer.echo(f"  LDUs         : {len(result.ldus)}")
    typer.echo(f"  Facts        : {len(result.facts)}")
    typer.echo(f"  Sample pages : {result.sample_pages or 'all'}")
    typer.echo(f"  Time         : {result.elapsed_seconds:.2f}s")
    typer.echo(f"  Artefacts    : {result.artefact_dir}")
    typer.echo(f"{'='*60}\n")


@app.command("batch")
def batch_run(
    directory: str = typer.Argument(..., help="Directory containing PDFs."),
    sample_pages: Optional[int] = typer.Option(
        3, "--sample-pages", "-s",
        help="Pages per document (default: 3 for speed).",
    ),
    page_strategy: str = typer.Option("head_mid_tail", "--page-strategy"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process all PDFs in a directory."""
    _setup_logging(verbose)
    from src.pipeline.orchestrator import PipelineOrchestrator

    d = Path(directory)
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        typer.echo(f"No PDFs found in {d}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(pdfs)} PDFs in {d}")
    orch = PipelineOrchestrator()
    ok_count = 0
    err_count = 0

    for i, pdf in enumerate(pdfs, 1):
        typer.echo(f"\n[{i}/{len(pdfs)}] {pdf.name}")
        try:
            result = orch.run(
                pdf,
                sample_pages=sample_pages,
                page_sample_strategy=page_strategy,
            )
            typer.echo(
                f"  → {len(result.ldus)} LDUs, {len(result.facts)} facts, "
                f"{result.elapsed_seconds:.1f}s"
            )
            ok_count += 1
        except Exception as exc:
            typer.echo(f"  ✗ Error: {exc}", err=True)
            err_count += 1

    typer.echo(f"\nBatch complete: {ok_count} ok, {err_count} errors")


@app.command("query")
def query_docs(
    question: str = typer.Argument(..., help="Natural-language question."),
    doc_id: Optional[str] = typer.Option(None, "--doc-id", "-d", help="Scope to one document."),
    n_results: int = typer.Option(5, "--n", help="Number of results."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Semantic search over indexed document chunks."""
    _setup_logging(verbose)
    from src.db.vector_store import VectorStore

    vs = VectorStore()
    if doc_id:
        results = vs.query_by_document(question, doc_id, n_results=n_results)
    else:
        results = vs.query(question, n_results=n_results)

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    if not docs:
        typer.echo("No results found.")
        return

    for i, (text, meta, dist) in enumerate(zip(docs, metas, dists), 1):
        typer.echo(f"\n--- Result {i} (distance={dist:.4f}) ---")
        typer.echo(f"  doc_id: {meta.get('document_id', '?')}")
        typer.echo(f"  page:   {meta.get('page_number', '?')}")
        typer.echo(f"  type:   {meta.get('chunk_type', '?')}")
        preview = text[:200].replace("\n", " ")
        typer.echo(f"  text:   {preview}...")


@app.command("audit")
def audit(
    doc_id: str = typer.Option(..., "--doc-id", "-d", help="Document ID to audit."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Show provenance / audit trail for a document."""
    _setup_logging(verbose)
    from src.db.repo import RefineryRepo

    repo = RefineryRepo()
    doc = repo.get_document(doc_id)
    if not doc:
        typer.echo(f"Document {doc_id} not found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n=== Document: {doc['source_filename']} ===")
    typer.echo(f"  origin: {doc['origin_type']}")
    typer.echo(f"  layout: {doc['layout_complexity']}")
    typer.echo(f"  pages:  {doc['page_count']}")
    typer.echo(f"  chunks: {doc['total_chunks']}")
    typer.echo(f"  cost:   {doc['estimated_cost']}")
    typer.echo(f"  time:   {doc['processing_timestamp']}")

    chunks = repo.get_chunks(doc_id)
    if chunks:
        typer.echo(f"\n--- Chunks ({len(chunks)}) ---")
        for c in chunks[:10]:
            preview = c["content"][:80].replace("\n", " ")
            typer.echo(f"  [{c['chunk_type']}] p{c['page_number']}: {preview}")
        if len(chunks) > 10:
            typer.echo(f"  ... and {len(chunks) - 10} more")

    facts = repo.get_facts(doc_id)
    if facts:
        typer.echo(f"\n--- Facts ({len(facts)}) ---")
        for f in facts[:15]:
            typer.echo(f"  {f['key']}: {f['value']} {f['unit']}  (p{f['page_ref']})")
        if len(facts) > 15:
            typer.echo(f"  ... and {len(facts) - 15} more")


@app.command("show")
def show_artefact(
    artefact: str = typer.Argument(
        ..., help="Artefact type: pageindex, ledger, facts, profile.",
    ),
    doc_id: str = typer.Option(..., "--doc-id", "-d", help="Document ID."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Inspect a pipeline artefact for a document."""
    _setup_logging(verbose)
    from src.db.repo import RefineryRepo

    repo = RefineryRepo()

    if artefact == "pageindex":
        pi = repo.get_page_index(doc_id)
        if pi:
            _echo_json(json.loads(pi["tree_json"]))
        else:
            typer.echo("PageIndex not found.", err=True)
    elif artefact == "facts":
        facts = repo.get_facts(doc_id)
        _echo_json(facts)
    elif artefact == "profile":
        doc = repo.get_document(doc_id)
        _echo_json(doc)
    elif artefact == "ledger":
        # Read from JSONL file
        ledger_path = _PROJECT_ROOT / ".refinery" / "extraction_ledger.jsonl"
        if ledger_path.exists():
            entries = []
            for line in ledger_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry.get("document_id") == doc_id:
                    entries.append(entry)
            _echo_json(entries)
        else:
            typer.echo("Ledger file not found.", err=True)
    else:
        typer.echo(f"Unknown artefact: {artefact}. Use: pageindex, ledger, facts, profile.", err=True)
        raise typer.Exit(1)


@app.command("list-docs")
def list_documents(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """List all processed documents in the database."""
    _setup_logging(verbose)
    from src.db.repo import RefineryRepo

    repo = RefineryRepo()
    docs = repo.list_documents()
    if not docs:
        typer.echo("No documents processed yet. Run 'refinery run <pdf>' first.")
        return

    typer.echo(f"\n{'ID':<14} {'Filename':<40} {'Origin':<16} {'Pages':>5} {'Chunks':>6}")
    typer.echo("-" * 85)
    for d in docs:
        typer.echo(
            f"{d['document_id'][:12]:<14} "
            f"{d['source_filename'][:38]:<40} "
            f"{d['origin_type']:<16} "
            f"{d['page_count']:>5} "
            f"{d['total_chunks']:>6}"
        )
    typer.echo(f"\nTotal: {len(docs)} documents")


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()

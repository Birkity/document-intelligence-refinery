"""Generate the interim submission LaTeX report.

Usage:
    python scripts/generate_interim_report.py

Produces: interim_submission.tex in the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
OUTPUT = _REPO / "interim_submission.tex"


def _latex_escape(text: str) -> str:
    """Escape characters that are special in LaTeX."""
    for ch, esc in [
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
        ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(ch, esc)
    return text


def build_latex() -> str:
    """Return the full LaTeX document as a string."""

    return r"""\documentclass[12pt,a4paper]{article}

% ─── Packages ──────────────────────────────────────────────────────────────
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\geometry{margin=2.5cm}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{listings}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{longtable}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
    citecolor=blue!60!black,
}

\lstset{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    backgroundcolor=\color{gray!10},
}

\title{%
    \textbf{The Document Intelligence Refinery} \\[0.5em]
    \Large Interim Submission Report \\[0.3em]
    \large Week 3 --- TRP1 Challenge
}
\author{Forward Deployed Engineer}
\date{\today}

\begin{document}
\maketitle
\tableofcontents
\newpage

% ═══════════════════════════════════════════════════════════════════════════
% SECTION 1 — Domain Notes (Phase 0)
% ═══════════════════════════════════════════════════════════════════════════
\section{Domain Notes (Phase 0)}

\subsection{Extraction Strategy Decision Tree}

The triage agent classifies each document along three dimensions and uses
the result to select the cheapest strategy that can extract with acceptable
quality.  The decision tree operates as follows:

\begin{enumerate}[leftmargin=2em]
    \item \textbf{Origin Type Detection}
    \begin{itemize}
        \item Compute character density = total\_chars / total\_page\_area.
        \item Compute image ratio = total\_image\_area / total\_page\_area.
        \item If char\_density $<$ \texttt{min\_char\_density} AND image\_ratio $>$ \texttt{scanned\_image\_ratio} $\Rightarrow$ \textbf{scanned\_image}.
        \item If char\_density $\geq$ threshold AND image\_ratio $\leq$ threshold $\Rightarrow$ \textbf{native\_digital}.
        \item Otherwise $\Rightarrow$ \textbf{mixed}.
    \end{itemize}

    \item \textbf{Layout Complexity Detection}
    \begin{itemize}
        \item If table area ratio $>$ \texttt{table\_area\_ratio\_threshold} $\Rightarrow$ \textbf{table\_heavy}.
        \item If image ratio $>$ \texttt{figure\_ratio\_threshold} AND char density $>$ \texttt{figure\_char\_density\_min} $\Rightarrow$ \textbf{figure\_heavy}.
        \item If coefficient of variation of word x-midpoints $>$ \texttt{column\_variance\_threshold} $\Rightarrow$ \textbf{multi\_column}.
        \item Otherwise $\Rightarrow$ \textbf{single\_column}.
    \end{itemize}

    \item \textbf{Cost Tier Selection}
    \begin{itemize}
        \item \texttt{scanned\_image} $\Rightarrow$ \textbf{needs\_vision\_model} (Strategy C).
        \item \texttt{native\_digital} + \texttt{single\_column} $\Rightarrow$ \textbf{fast\_text\_sufficient} (Strategy A).
        \item All other combinations $\Rightarrow$ \textbf{needs\_layout\_model} (Strategy B).
    \end{itemize}

    \item \textbf{Escalation Guard}
    \begin{itemize}
        \item After extraction, compute confidence score.
        \item If Strategy A confidence $<$ 0.6 $\Rightarrow$ escalate to Strategy B.
        \item If Strategy B confidence $<$ 0.5 $\Rightarrow$ escalate to Strategy C.
        \item If final confidence $<$ 0.4 $\Rightarrow$ flag for manual review.
    \end{itemize}
\end{enumerate}

All thresholds are loaded from \texttt{rubric/extraction\_rules.yaml}.
No values are hardcoded in source code.

\subsection{Failure Modes Observed}

\begin{description}[leftmargin=2em]
    \item[Structure Collapse]
    Traditional OCR flattens multi-column layouts, merges table rows, and drops
    hierarchical headers.  For example, the CBE Annual Report's two-column
    financial commentary gets concatenated into a single paragraph, destroying
    reading order and making downstream RAG answers nonsensical.

    \item[Context Poverty]
    Naive token-count chunking severs logical units.  A financial table split
    across two 512-token chunks loses its header row in the second chunk,
    causing the LLM to hallucinate column labels.  The FTA assessment report's
    numbered findings become isolated sentences without their parent section
    context.

    \item[Provenance Blindness]
    Without bounding-box coordinates and page references tied to every
    extracted fact, answers cannot be audited.  When a user asks
    ``What was CBE's net income in 2024?'' the system can produce a number
    but cannot point to the exact cell in the income statement table on
    page 47.  This makes the system unsuitable for regulated environments.
\end{description}

\subsection{Pipeline Diagram (Mermaid)}

The following Mermaid diagram describes the five-stage pipeline with
feedback loops.  It can be rendered using any Mermaid-compatible tool.

\begin{lstlisting}[language={},caption={Mermaid Pipeline Diagram}]
graph TD

%% =========================
%% Stage 1: Triage Agent
%% =========================
subgraph Stage1["Stage 1: Triage Agent"]
A1["PDF Input"] --> A2["detect_origin_type"]
A2 --> A3["detect_layout_complexity"]
A3 --> A4["detect_domain_hint"]
A4 --> A5["estimate_extraction_cost"]
A5 --> A6["DocumentProfile + source_filename"]
end

%% =========================
%% Stage 2: Extraction Router
%% =========================
subgraph Stage2["Stage 2: Extraction Router (Confidence-Gated)"]
B1["ExtractionRouter"] --> B2{"Cost Tier?"}

B2 -->|fast_text_sufficient| B3["Strategy A: FastTextExtractor"]
B2 -->|needs_layout_model| B4["Strategy B: LayoutExtractor (Docling)"]
B2 -->|needs_vision_model| B5["Strategy C: VisionExtractor (HF VLM)"]

B3 -->|conf < A_to_B_threshold| B4
B4 -->|conf < B_to_C_threshold| B5

B3 --> B6["ExtractedDocument + source_filename"]
B4 --> B6
B5 --> B6

B6 --> B7["Append extraction_ledger.jsonl"]
B5 -->|conf < review_threshold| B8["Flag for Manual Review"]
end

%% =========================
%% Stage 3: Chunking Engine
%% =========================
subgraph Stage3["Stage 3: Chunking Engine (Rules + Validator)"]
C1["ExtractedDocument"] --> C2["ChunkValidator"]
C2 -->|rules violated| C1
C2 -->|valid| C3["LDU List"]
end

%% =========================
%% Stage 4: Indexing
%% =========================
subgraph Stage4["Stage 4: Indexing"]
D1["LDU List"] --> D2["ChromaDB Embeddings"]
D1 --> D3["SQLite Structured Facts"]
D1 --> D4["PageIndex Tree"]
end

%% =========================
%% Stage 5: Query Agent
%% =========================
subgraph Stage5["Stage 5: Query Agent"]
E1["User Query"] --> E2["pageindex_navigate"]
E1 --> E3["semantic_search"]
E1 --> E4["structured_query (SQL)"]
E2 --> E5["Answer + ProvenanceChain"]
E3 --> E5
E4 --> E5
E5 -->|unverifiable| E6["Audit Flag"]
end

%% =========================
%% Pipeline Connections
%% =========================
A6 --> B1
B6 --> C1
C3 --> D1
D2 --> E3
D3 --> E4
D4 --> E2
\end{lstlisting}


% ═══════════════════════════════════════════════════════════════════════════
% SECTION 2 — Architecture Diagram
% ═══════════════════════════════════════════════════════════════════════════
\section{Architecture Diagram}

\subsection{Full Five-Stage Pipeline}

The Document Intelligence Refinery implements a non-sequential pipeline
with confidence-gated escalation, feedback loops, and a dual-store
architecture (SQLite + ChromaDB).

\subsubsection{Stage 1: Triage Agent}
\begin{itemize}
    \item Analyses the first $N$ pages (configurable via \texttt{max\_pages\_to\_sample}).
    \item Produces a \texttt{DocumentProfile} Pydantic model with \texttt{source\_filename}.
    \item Persists profiles to \texttt{.refinery/profiles/\{doc\_id\}.json}.
    \item Classification dimensions: origin type, layout complexity, domain hint, extraction cost.
\end{itemize}

\subsubsection{Stage 2: Extraction Router}
\begin{itemize}
    \item Reads the \texttt{DocumentProfile} and selects starting strategy.
    \item \textbf{Strategy A} (FastTextExtractor): pdfplumber text extraction, negligible compute cost.
    \item \textbf{Strategy B} (LayoutExtractor): Docling layout-aware parser, moderate compute cost.
    \item \textbf{Strategy C} (VisionExtractor): HuggingFace vision-language model, high compute cost.
    \item \textbf{Escalation Guard}: confidence thresholds from \texttt{extraction\_rules.yaml}.
    \item Every attempt logged to \texttt{.refinery/extraction\_ledger.jsonl} with source filename tracking.
    \item Documents with confidence $<$ \texttt{flag\_confidence\_below} trigger review queue entry.
\end{itemize}

\subsubsection{Stage 3: Semantic Chunking Engine}
\begin{itemize}
    \item Converts \texttt{ExtractedDocument} into \texttt{List[LDU]}.
    \item Enforces chunking constitution: tables never split from headers, figure captions stay with figures, lists kept whole.
    \item Each LDU carries \texttt{content\_hash} for provenance verification.
\end{itemize}

\subsubsection{Stage 4: Indexing Layer}
\begin{itemize}
    \item \textbf{ChromaDB}: stores chunk embeddings with document\_id, page\_number, section\_path metadata.
    \item \textbf{SQLite}: stores structured tables, provenance ledger, query logs with source filenames.
    \item \textbf{PageIndex}: hierarchical navigation tree for section-level traversal.
\end{itemize}

\subsubsection{Stage 5: Query Agent}
\begin{itemize}
    \item Three tools: \texttt{pageindex\_navigate}, \texttt{semantic\_search}, \texttt{structured\_query}.
    \item Every answer carries a \texttt{ProvenanceChain} with document name, page number, bounding box, and content hash.
    \item Unverifiable claims flagged via audit mode.
\end{itemize}

\subsection{Config-Driven Architecture}

All tunable parameters are externalised to \texttt{rubric/extraction\_rules.yaml}:

\begin{itemize}
    \item \texttt{origin\_detection}: \texttt{min\_char\_density}, \texttt{scanned\_image\_ratio}, \texttt{max\_pages\_to\_sample}
    \item \texttt{layout\_detection}: \texttt{table\_area\_ratio\_threshold}, \texttt{figure\_ratio\_threshold}, \texttt{column\_variance\_threshold}
    \item \texttt{escalation}: \texttt{strategy\_a\_min\_confidence}, \texttt{strategy\_b\_min\_confidence}, \texttt{max\_escalation\_depth}
    \item \texttt{chunking}: \texttt{max\_tokens\_per\_chunk}, \texttt{min\_tokens\_per\_chunk}, \texttt{overlap\_tokens}
    \item \texttt{review}: \texttt{flag\_confidence\_below}, \texttt{max\_image\_ratio\_fast\_text}
    \item \texttt{domain\_keywords}: financial, legal, technical, medical keyword lists
\end{itemize}

\subsection{Separation of Storage}

\begin{center}
\begin{tabular}{lll}
\toprule
\textbf{Store} & \textbf{Technology} & \textbf{Purpose} \\
\midrule
Governance DB    & SQLite   & Documents, chunks, structured tables, provenance, query logs \\
Vector Store     & ChromaDB & Chunk embeddings for semantic search \\
Profile Store    & JSON     & 31 DocumentProfile artifacts with source filename tracking \\
Extraction Ledger & JSONL   & Audit trail with document ID and filename per attempt \\
\bottomrule
\end{tabular}
\end{center}

\subsection{Document Class Coverage}

The corpus includes 31 profiled documents across 4 classes (minimum 3 per class):

\begin{itemize}
    \item \textbf{Class A} (Native Digital Financial Reports): 5 documents
    \begin{itemize}
        \item CBE ANNUAL REPORT 2023-24.pdf, CBE Annual Report 2016-17.pdf, etc.
    \end{itemize}
    \item \textbf{Class B} (Scanned Government/Legal): 20 documents
    \begin{itemize}
        \item Audit Report - 2023.pdf, 2018-2022 Audited Financial Statements, etc.
    \end{itemize}
    \item \textbf{Class C} (Technical Assessment Reports): 3 documents
    \begin{itemize}
        \item fta\_performance\_survey\_final\_report\_2022.pdf, etc.
    \end{itemize}
    \item \textbf{Class D} (Table-Heavy Structured Data): 3 documents
    \begin{itemize}
        \item Consumer Price Index reports, Pharmaceutical Manufacturing report
    \end{itemize}
\end{itemize}


% ═══════════════════════════════════════════════════════════════════════════
% SECTION 3 — Cost Analysis
% ═══════════════════════════════════════════════════════════════════════════
\section{Cost Analysis}

\subsection{Per-Strategy Cost Estimates}

\subsubsection{Strategy A --- FastTextExtractor (pdfplumber)}
\begin{description}
    \item[Cost:] Negligible compute (CPU-only text extraction, \$0 direct spend).
    \item[Latency:] $\sim$0.5--2s per document (depends on page count).
    \item[Risk:] Structure collapse on multi-column or table-heavy documents.
    \item[When Used:] \texttt{native\_digital} + \texttt{single\_column} documents.
\end{description}

\subsubsection{Strategy B --- LayoutExtractor (Docling)}
\begin{description}
    \item[Cost:] Moderate compute (CPU-intensive layout analysis, \$0 direct spend using local Docling).
    \item[Latency:] $\sim$5--15s per document.
    \item[Risk:] Manageable --- structural accuracy significantly higher than A.
    \item[When Used:] \texttt{multi\_column}, \texttt{table\_heavy}, \texttt{mixed} origin documents.
\end{description}

\subsubsection{Strategy C --- VisionExtractor (HuggingFace Vision-Language Model)}
\begin{description}
    \item[Cost:] High compute (\$0 direct spend using free HF models, but GPU time significant).
    \item[Latency:] $\sim$15--60s per document (CPU) or 5--10s (GPU).
    \item[Risk:] OCR structural drift; handwriting may still fail.
    \item[When Used:] \texttt{scanned\_image} documents, or escalation fallback.
\end{description}

\subsection{Summary Table}

\begin{center}
\begin{tabular}{lllll}
\toprule
\textbf{Tier} & \textbf{Compute Cost} & \textbf{Speed (CPU)} & \textbf{Structure Quality} & \textbf{When Used} \\
\midrule
A (pdfplumber) & Negligible & $<$2s & Low--Medium & native\_digital + single\_column \\
B (Docling)    & Moderate   & 5--15s & Medium--High & multi\_column, table\_heavy, mixed \\
C (HF Vision)  & High       & 15--60s & High & scanned\_image, low-confidence fallback \\
\bottomrule
\end{tabular}
\end{center}

\textit{Note: All costs measured in compute time. Direct spend = \$0 using free open-source models.}

\subsection{Cost Optimisation Strategy}

The escalation chain ensures the system defaults to the cheapest strategy
and only incurs higher costs when quality demands it.  For a typical
heterogeneous corpus of 50 documents:

\begin{itemize}
    \item $\sim$30\% native digital $\Rightarrow$ Strategy A (negligible compute).
    \item $\sim$40\% mixed/complex layout $\Rightarrow$ Strategy B (moderate compute via Docling).
    \item $\sim$30\% scanned $\Rightarrow$ Strategy C (high compute via HF vision model).
    \item Estimated total: \textbf{\$0 direct spend} using free/open-source models (pdfplumber, Docling, HuggingFace). Compute cost measured in CPU/GPU time rather than API fees.
\end{itemize}


% ═══════════════════════════════════════════════════════════════════════════
% SECTION 4 — Repository Structure
% ═══════════════════════════════════════════════════════════════════════════
\section{Repository Structure}

\begin{lstlisting}[language={},caption={Project Layout}]
document-intelligence-refinery/
|-- rubric/
|   +-- extraction_rules.yaml
|-- src/
|   |-- models/
|   |   +-- schemas.py          # DocumentProfile, ExtractedDocument,
|   |                           # LDU, PageIndex, ProvenanceChain
|   |                           # (includes source_filename tracking)
|   |-- agents/
|   |   |-- triage.py           # Triage Agent (Stage 1)
|   |   +-- extractor.py        # ExtractionRouter (Stage 2)
|   |-- strategies/
|   |   |-- base.py             # BaseExtractor interface
|   |   |-- fast_text.py        # Strategy A
|   |   |-- layout.py           # Strategy B
|   |   +-- vision.py           # Strategy C
|   +-- db/
|       |-- schema.sql           # SQLite DDL (with source_filename)
|       |-- init_db.py           # Idempotent DB initialisation
|       +-- vector_store.py      # ChromaDB wrapper
|-- scripts/
|   |-- generate_profiles.py
|   |-- generate_ledger.py
|   |-- generate_interim_report.py
|   |-- ensure_class_coverage.py # Class coverage checker
|   +-- generate_class_report.py # Classification report
|-- tests/
|   |-- test_models.py
|   |-- test_triage_origin.py
|   |-- test_triage_layout.py
|   |-- test_extraction_router.py
|   +-- test_db_and_schemas.py
|-- .refinery/
|   |-- profiles/                # 31 DocumentProfile JSONs (4 classes)
|   |-- extraction_ledger.jsonl  # Extraction audit trail
|   +-- refinery.db              # SQLite governance DB
|-- pyproject.toml
|-- README.md
+-- class_coverage_report.txt    # Document class verification
\end{lstlisting}


% ═══════════════════════════════════════════════════════════════════════════
% SECTION 5 — Test Coverage
% ═══════════════════════════════════════════════════════════════════════════
\section{Test Coverage}

61 unit tests across five test modules:

\begin{center}
\begin{tabular}{lr}
\toprule
\textbf{Module} & \textbf{Tests} \\
\midrule
\texttt{test\_models.py}            & 14 \\
\texttt{test\_triage\_origin.py}    & 5 \\
\texttt{test\_triage\_layout.py}    & 22 \\
\texttt{test\_extraction\_router.py}& 8 \\
\texttt{test\_db\_and\_schemas.py}  & 12 \\
\midrule
\textbf{Total}                      & \textbf{61} \\
\bottomrule
\end{tabular}
\end{center}

Key areas tested:
\begin{itemize}
    \item Origin type detection (native, scanned, mixed)
    \item Layout complexity (single-column, multi-column, table-heavy, figure-heavy)
    \item Domain keyword classification (financial, legal, technical, medical, general)
    \item Extraction cost estimation
    \item Full document profile generation with source filename tracking
    \item JSON serialisation and persistence
    \item Escalation chain: A $\to$ B on low confidence
    \item Escalation chain: B $\to$ C on low confidence
    \item No escalation when confidence is sufficient
    \item Review flag on very low confidence
    \item Ledger entry persistence with document metadata
    \item Database creation, idempotency, and CRUD operations
    \item PageIndex and ProvenanceChain schema validation
    \item Schema updates (DocumentProfile and ExtractedDocument with source\_filename)
\end{itemize}


\end{document}
"""


def main() -> None:
    tex = build_latex()
    OUTPUT.write_text(tex, encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Size: {OUTPUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()

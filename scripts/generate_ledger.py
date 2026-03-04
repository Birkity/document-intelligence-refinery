"""Generate extraction ledger entries by running the ExtractionRouter on corpus docs.

Usage:
    python -m scripts.generate_ledger
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.agents.triage import TriageAgent
from src.agents.extractor import ExtractionRouter

CORPUS = [
    "CBE ANNUAL REPORT 2023-24.pdf",
    "Audit Report - 2023.pdf",
    "fta_performance_survey_final_report_2022.pdf",
    "tax_expenditure_ethiopia_2021_22.pdf",
    "Annual_Report_JUNE-2023.pdf",
    "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    "Consumer Price Index March 2025.pdf",
    "2013-E.C-Audit-finding-information.pdf",
    "EthSwitch-10th-Annual-Report-202324.pdf",
    "Company_Profile_2024_25.pdf",
    "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "2013-E.C-Assigned-regular-budget-and-expense.pdf",
]

DATA_DIR = _REPO / "data"


def main() -> None:
    agent = TriageAgent()
    router = ExtractionRouter()

    for fname in CORPUS:
        pdf_path = DATA_DIR / fname
        if not pdf_path.exists():
            print(f"  SKIP  {fname}")
            continue

        try:
            profile = agent.generate_document_profile(str(pdf_path))
            doc, entries = router.route_and_extract(profile, str(pdf_path))
            n_pages = len(doc.pages)
            strategies = [e["strategy_used"] for e in entries]
            print(
                f"  OK    {fname}  pages={n_pages}  "
                f"strategies={strategies}  "
                f"confidence={entries[-1]['confidence_score']:.4f}"
            )
        except Exception as exc:
            print(f"  FAIL  {fname}: {exc}")

    ledger = _REPO / ".refinery" / "extraction_ledger.jsonl"
    n_lines = sum(1 for _ in open(ledger, encoding="utf-8"))
    print(f"\n{n_lines} ledger entries in {ledger}")


if __name__ == "__main__":
    main()

"""Generate DocumentProfile JSON artifacts for at least 12 corpus documents.

Runs the TriageAgent against selected PDFs from data/ and writes profiles
to .refinery/profiles/ plus ledger entries to .refinery/extraction_ledger.jsonl.

Usage:
    python -m scripts.generate_profiles
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.agents.triage import TriageAgent

# Select at least 12 representative documents across the four classes.
# Class A — native digital annual reports
# Class B — scanned / image-based audit reports
# Class C — mixed technical assessment reports
# Class D — table-heavy structured fiscal data
CORPUS: list[str] = [
    # Class A: native digital (financial annual reports)
    "CBE ANNUAL REPORT 2023-24.pdf",
    "Annual_Report_JUNE-2023.pdf",
    "Annual_Report_JUNE-2022.pdf",
    "EthSwitch-10th-Annual-Report-202324.pdf",
    # Class B: scanned image
    "Audit Report - 2023.pdf",
    "2013-E.C-Audit-finding-information.pdf",
    "2013-E.C-Procurement-information.pdf",
    # Class C: mixed (technical / assessment)
    "fta_performance_survey_final_report_2022.pdf",
    "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",
    "Security_Vulnerability_Disclosure_Standard_Procedure_1.pdf",
    "Company_Profile_2024_25.pdf",
    # Class D: table-heavy fiscal data
    "tax_expenditure_ethiopia_2021_22.pdf",
    "Consumer Price Index March 2025.pdf",
    "2013-E.C-Assigned-regular-budget-and-expense.pdf",
]

DATA_DIR = _REPO / "data"


def main() -> None:
    agent = TriageAgent()
    generated = 0

    for fname in CORPUS:
        pdf_path = DATA_DIR / fname
        if not pdf_path.exists():
            print(f"  SKIP  {fname} (not found)")
            continue
        try:
            profile = agent.generate_document_profile(str(pdf_path))
            print(
                f"  OK    {fname}\n"
                f"        origin={profile.origin_type}  "
                f"layout={profile.layout_complexity}  "
                f"domain={profile.domain_hint}  "
                f"cost={profile.estimated_extraction_cost}"
            )
            generated += 1
        except Exception as exc:
            print(f"  FAIL  {fname}: {exc}")

    print(f"\n{generated} profiles generated in .refinery/profiles/")


if __name__ == "__main__":
    main()

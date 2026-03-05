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

# Exactly 12 documents — 3 per class, selected to reliably hit each class
# after the corrected origin-type thresholds in extraction_rules.yaml.
#
# Class A — native-digital annual financial reports
#   char_density >= 0.001, image_ratio < 0.7  → origin = native_digital
#   domain = financial, layout = multi_column  → cost = needs_layout_model
#
# Class B — scanned government / legal documents
#   char_density ≈ 0,  image_ratio ≈ 1.0      → origin = scanned_image
#
# Class C — technical assessment / mixed-content reports
#   origin = native_digital or mixed, domain in (technical, general)
#   identified by filename pattern in classify_profile()
#
# Class D — table-heavy structured fiscal data
#   origin = native_digital, layout = table_heavy
CORPUS: list[str] = [
    # ── Class A: native-digital financial annual reports ──────────────
    "CBE ANNUAL REPORT 2023-24.pdf",       # cd≈0.0019, ir≈0.40
    "CBE Annual Report 2016-17.pdf",       # cd≈0.0022, ir≈0.30
    "CBE Annual Report 2018-19.pdf",       # cd≈0.0016, ir≈0.36
    # ── Class B: scanned image documents ─────────────────────────────
    "Audit Report - 2023.pdf",             # cd≈0.00005, ir≈0.80
    "2013-E.C-Audit-finding-information.pdf",   # cd≈0,  ir≈1.0
    "2013-E.C-Procurement-information.pdf",     # cd≈0,  ir≈1.0
    # ── Class C: technical / assessment reports ───────────────────────
    "fta_performance_survey_final_report_2022.pdf",           # native_digital
    "Company_Profile_2024_25.pdf",                           # native_digital
    "20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf",  # mixed
    # ── Class D: table-heavy structured fiscal data ───────────────────
    "tax_expenditure_ethiopia_2021_22.pdf",     # native_digital, table_heavy
    "Consumer Price Index March 2025.pdf",      # native_digital, table_heavy
    "Consumer Price Index August 2025.pdf",     # native_digital, table_heavy
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

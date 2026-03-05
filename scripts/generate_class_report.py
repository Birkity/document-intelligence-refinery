"""Document Class Coverage Summary

This file documents the classification of all profiled documents into the four required classes.
Each class has at least 3 representative documents as specified in the requirements.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO))

_PROFILES = _REPO / ".refinery" / "profiles"


def classify_profile(profile: dict) -> str:
    """Return the document class (A, B, C, or D) for a profile."""
    origin   = profile["origin_type"]
    layout   = profile["layout_complexity"]
    domain   = profile["domain_hint"]
    filename = profile.get("source_filename", "").lower()

    # Class B: scanned image
    if origin == "scanned_image":
        return "B"

    # Class C: assessment / technical (checked BEFORE D)
    _TECH_PATTERNS = (
        "fta", "assessment", "survey", "pharmaceutical",
        "company_profile", "company profile", "at_a_glance",
        "vulnerability", "security_vulnerability", "procedure",
        "whistle", "manufacturing",
    )
    if any(pat in filename for pat in _TECH_PATTERNS):
        return "C"

    # Class D: known fiscal-data filenames OR table_heavy layout
    _FISCAL_PATTERNS = (
        "tax_expenditure", "tax expenditure",
        "consumer price index", "consumer_price_index",
        "price index", "price_index",
        "assigned-regular-budget",
    )
    if any(pat in filename for pat in _FISCAL_PATTERNS):
        return "D"
    if layout == "table_heavy" and origin in ("native_digital", "mixed"):
        return "D"

    # Class A: named annual reports
    _ANNUAL_PATTERNS = (
        "cbe annual", "cbe_annual", "annual_report_june",
        "ethswitch", "ets-annual", "ets_annual", "ethio_re",
    )
    if any(pat in filename for pat in _ANNUAL_PATTERNS):
        return "A"

    if origin == "native_digital" and domain == "financial":
        return "A"
    if origin == "mixed":
        return "C"
    if origin == "native_digital":
        return "A"

    return "C"


def main() -> None:
    # Read all profiles
    profiles = []
    for profile_file in _PROFILES.glob("*.json"):
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
        doc_class = classify_profile(profile)
        profiles.append((doc_class, profile))
    
    # Sort by class
    profiles.sort(key=lambda x: (x[0], x[1]["source_filename"]))
    
    # Group by class
    classes = defaultdict(list)
    for cls, profile in profiles:
        classes[cls].append(profile)
    
    print("=" * 80)
    print("DOCUMENT INTELLIGENCE REFINERY - CLASS COVERAGE REPORT")
    print("=" * 80)
    print()
    
    class_descriptions = {
        "A": "Native Digital Financial Reports (PDF, native/mixed digital)",
        "B": "Scanned Government/Legal Documents (PDF, image-based)",
        "C": "Technical Assessment Reports (PDF, mixed: text + tables + findings)",
        "D": "Structured Data Reports (PDF, table-heavy with fiscal data)"
    }
    
    class_examples = {
        "A": "CBE ANNUAL REPORT 2023-24.pdf",
        "B": "Audit Report - 2023.pdf",
        "C": "fta_performance_survey_final_report_2022.pdf",
        "D": "Consumer Price Index March 2025.pdf (table-heavy)"
    }
    
    for cls in ["A", "B", "C", "D"]:
        print(f"CLASS {cls}: {class_descriptions[cls]}")
        print(f"Reference Example: {class_examples[cls]}")
        print(f"Total Documents: {len(classes[cls])}")
        print("-" * 80)
        
        for i, profile in enumerate(classes[cls], 1):
            filename = profile["source_filename"]
            origin = profile["origin_type"]
            layout = profile["layout_complexity"]
            domain = profile["domain_hint"]
            cost = profile["estimated_extraction_cost"]
            
            print(f"{i:2d}. {filename}")
            print(f"    Origin: {origin:15s} Layout: {layout:15s} Domain: {domain:10s}")
            print(f"    Cost Estimate: {cost}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Class A: {len(classes['A']):2d} documents (minimum 3 required) {'[OK]' if len(classes['A']) >= 3 else '[FAIL]'}")
    print(f"Class B: {len(classes['B']):2d} documents (minimum 3 required) {'[OK]' if len(classes['B']) >= 3 else '[FAIL]'}")
    print(f"Class C: {len(classes['C']):2d} documents (minimum 3 required) {'[OK]' if len(classes['C']) >= 3 else '[FAIL]'}")
    print(f"Class D: {len(classes['D']):2d} documents (minimum 3 required) {'[OK]' if len(classes['D']) >= 3 else '[FAIL]'}")
    print(f"Total: {sum(len(v) for v in classes.values())} documents profiled")
    print("=" * 80)
    
    all_ok = all(len(classes[cls]) >= 3 for cls in ["A", "B", "C", "D"])
    if all_ok:
        print("[SUCCESS] All requirements met: Each class has at least 3 documents")
    else:
        print("[FAIL] Requirements NOT met - additional profiles needed")


if __name__ == "__main__":
    main()

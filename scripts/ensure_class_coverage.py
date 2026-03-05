"""Ensure we have at least 3 documents from each of the 4 classes.

Document Classes:
- Class A: Native digital financial reports (native_digital + financial domain)
- Class B: Scanned government/legal documents (scanned_image origin)
- Class C: Technical assessment reports (mixed + technical, or complex layout)
- Class D: Table-heavy structured data (table_heavy layout)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO))

from src.agents.triage import TriageAgent

_DATA = _REPO / "data"
_PROFILES = _REPO / ".refinery" / "profiles"


def classify_profile(profile: dict) -> str:
    """Return the document class (A, B, C, or D) for a profile.

    Priority order:
      B  → scanned_image origin
      C  → filename matches assessment/technical patterns  (BEFORE D so that
             Pharmaceutical doesn't get mis-routed by its table_heavy layout)
      D  → known fiscal-data filename  OR  table_heavy layout
      A  → annual-report filename  OR  native_digital + financial
      C  → fallback (mixed / unrecognised)
    """
    origin   = profile["origin_type"]
    layout   = profile["layout_complexity"]
    domain   = profile["domain_hint"]
    filename = profile.get("source_filename", "").lower()

    # ── Class B ───────────────────────────────────────────────────────────
    if origin == "scanned_image":
        return "B"

    # ── Class C (before D) ────────────────────────────────────────────────
    # Assessment / technical / company-profile docs are always Class C even
    # if pdfplumber detects a high table area ratio.
    _TECH_PATTERNS = (
        "fta", "assessment", "survey", "pharmaceutical",
        "company_profile", "company profile", "at_a_glance",
        "vulnerability", "security_vulnerability", "procedure",
        "whistle", "manufacturing",
    )
    if any(pat in filename for pat in _TECH_PATTERNS):
        return "C"

    # ── Class D ───────────────────────────────────────────────────────────
    # Explicitly-known fiscal data filenames (pdfplumber may miss tables).
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

    # ── Class A ───────────────────────────────────────────────────────────
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
    # Read existing profiles
    existing_profiles = {}
    for profile_file in _PROFILES.glob("*.json"):
        profile = json.loads(profile_file.read_text(encoding="utf-8"))
        existing_profiles[profile["source_filename"]] = profile
    
    # Categorize existing profiles
    classes = defaultdict(list)
    for filename, profile in existing_profiles.items():
        doc_class = classify_profile(profile)
        classes[doc_class].append(filename)
    
    print("=" * 70)
    print("CURRENT CLASS DISTRIBUTION")
    print("=" * 70)
    for cls in ["A", "B", "C", "D"]:
        print(f"\nClass {cls}: {len(classes[cls])} documents")
        for fname in classes[cls][:5]:  # Show first 5
            print(f"  - {fname}")
        if len(classes[cls]) > 5:
            print(f"  ... and {len(classes[cls]) - 5} more")
    
    # Identify which classes need more documents
    needed = {}
    for cls in ["A", "B", "C", "D"]:
        if len(classes[cls]) < 3:
            needed[cls] = 3 - len(classes[cls])
    
    if not needed:
        print("\n" + "=" * 70)
        print("✓ All classes have at least 3 documents!")
        print("=" * 70)
        return
    
    print("\n" + "=" * 70)
    print("CLASSES NEEDING MORE DOCUMENTS")
    print("=" * 70)
    for cls, count in needed.items():
        print(f"Class {cls}: Need {count} more")
    
    # Find candidate PDFs that haven't been profiled yet
    all_pdfs = [p.name for p in _DATA.glob("*.pdf")]
    profiled_pdfs = set(existing_profiles.keys())
    unprofiled_pdfs = [p for p in all_pdfs if p not in profiled_pdfs]
    
    print(f"\n{len(unprofiled_pdfs)} unprofiled PDFs available")
    
    # Generate profiles for promising candidates
    if needed.get("A"):  # Need native digital financial reports
        print("\nSearching for Class A candidates (native digital financial)...")
        candidates = [
            p for p in unprofiled_pdfs
            if any(x in p.lower() for x in [
                "cbe annual report 2016", "cbe annual report 2017", "cbe annual report 2018",
                "ethswitch-annual-report-2019", "ethswitch-annual-report-2020",
                "ets_annual_report", "ets-annual-report"
            ])
        ]
        print(f"Found {len(candidates)} candidates: {candidates[:10]}")
        
        agent = TriageAgent()
        generated = 0
        for pdf_name in candidates[:needed["A"] + 5]:  # Generate extra to find native_digital
            pdf_path = _DATA / pdf_name
            if not pdf_path.exists():
                continue
            try:
                print(f"  Generating profile for {pdf_name}...")
                profile = agent.generate_document_profile(str(pdf_path))
                doc_class = classify_profile(profile.model_dump())
                print(f"    → Classified as Class {doc_class} (origin: {profile.origin_type})")
                generated += 1
                if doc_class == "A":
                    needed["A"] -= 1
                    if needed["A"] <= 0:
                        break
            except Exception as e:
                print(f"    ERROR: {e}")
    
    if needed.get("D"):  # Need table-heavy documents
        print("\nSearching for Class D candidates (table-heavy data)...")
        candidates = [
            p for p in unprofiled_pdfs
            if any(x in p.lower() for x in [
                "consumer price index", "cpi", "expenditure",
                "budget", "expense", "procurement"
            ])
        ]
        print(f"Found {len(candidates)} candidates: {candidates}")
        
        if not 'agent' in locals():
            agent = TriageAgent()
        
        for pdf_name in candidates[:needed["D"] + 2]:
            pdf_path = _DATA / pdf_name
            if not pdf_path.exists():
                continue
            try:
                print(f"  Generating profile for {pdf_name}...")
                profile = agent.generate_document_profile(str(pdf_path))
                doc_class = classify_profile(profile.model_dump())
                print(f"    → Classified as Class {doc_class} (layout: {profile.layout_complexity})")
                if doc_class == "D":
                    needed["D"] -= 1
                    if needed["D"] <= 0:
                        break
            except Exception as e:
                print(f"    ERROR: {e}")
    
    if needed.get("C"):  # Need technical assessment reports
        print("\nSearching for Class C candidates (technical/assessment)...")
        candidates = [
            p for p in unprofiled_pdfs
            if any(x in p.lower() for x in [
                "pharmaceutical", "manufacturing", "assessment",
                "standard_procedure", "vulnerability", "ethio_re",
                "profile", "glance", "switch"
            ]) and not any(x in p.lower() for x in [
                "consumer price", "audited", "audit report"
            ])
        ]
        print(f"Found {len(candidates)} candidates: {candidates}")
        
        if not 'agent' in locals():
            agent = TriageAgent()
        
        for pdf_name in candidates[:needed["C"] + 2]:
            pdf_path = _DATA / pdf_name
            if not pdf_path.exists():
                continue
            try:
                print(f"  Generating profile for {pdf_name}...")
                profile = agent.generate_document_profile(str(pdf_path))
                doc_class = classify_profile(profile.model_dump())
                print(f"    → Classified as Class {doc_class} (origin: {profile.origin_type}, layout: {profile.layout_complexity}, domain: {profile.domain_hint})")
                if doc_class == "C":
                    needed["C"] -= 1
                    if needed["C"] <= 0:
                        break
            except Exception as e:
                print(f"    ERROR: {e}")
    
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE - Rerun to check coverage")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""Triage Agent — Stage 1 of the Document Intelligence Refinery.

Classifies incoming documents by origin type, layout complexity,
language, and domain so that downstream extraction stages can select
the most cost-effective strategy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Literal

import pdfplumber
import yaml

from src.models.schemas import DocumentProfile


# Type aliases for clarity
OriginType = Literal["native_digital", "scanned_image", "mixed"]
LayoutComplexity = Literal[
    "single_column", "multi_column", "table_heavy", "figure_heavy", "mixed"
]
DomainHint = Literal["financial", "legal", "technical", "medical", "general"]
ExtractionCost = Literal[
    "fast_text_sufficient", "needs_layout_model", "needs_vision_model"
]

# Default path to the extraction rules configuration
_DEFAULT_RULES_PATH = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


class TriageAgent:
    """Classifies a PDF document to guide extraction strategy selection.

    Parameters
    ----------
    rules_path : str | Path | None
        Path to ``extraction_rules.yaml``.  Falls back to the repo default
        if not supplied.
    """

    def __init__(self, rules_path: str | Path | None = None) -> None:
        rules_file = Path(rules_path) if rules_path else _DEFAULT_RULES_PATH
        with open(rules_file, "r", encoding="utf-8") as fh:
            self._rules = yaml.safe_load(fh)

        # Origin detection config
        origin_cfg = self._rules.get("origin_detection", {})
        self.min_char_density: float = origin_cfg.get("min_char_density", 0.01)
        self.scanned_image_ratio: float = origin_cfg.get(
            "scanned_image_ratio", 0.5
        )
        self.max_pages_to_sample: int = origin_cfg.get("max_pages_to_sample", 5)

        # Layout detection config
        layout_cfg = self._rules.get("layout_detection", {})
        self.layout_max_pages: int = layout_cfg.get("max_pages_to_sample", 5)
        self.table_area_ratio_threshold: float = layout_cfg.get(
            "table_area_ratio_threshold", 0.3
        )
        self.figure_ratio_threshold: float = layout_cfg.get(
            "figure_ratio_threshold", 0.25
        )
        self.figure_char_density_min: float = layout_cfg.get(
            "figure_char_density_min", 0.005
        )
        self.column_variance_threshold: float = layout_cfg.get(
            "column_variance_threshold", 0.25
        )

        # Domain keywords config
        self.domain_keywords: dict[str, list[str]] = self._rules.get(
            "domain_keywords", {}
        )

    # ------------------------------------------------------------------
    # Origin-type detection
    # ------------------------------------------------------------------

    def detect_origin_type(self, pdf_path: str) -> tuple[OriginType, float]:
        """Determine whether a PDF is native-digital, scanned, or mixed.

        Analyses the first *N* pages (configured via
        ``max_pages_to_sample``) using pdfplumber to compute:

        * **character_density** — total characters / total page area
        * **image_ratio** — total image area / total page area

        Returns
        -------
        tuple[OriginType, float]
            A ``(origin_type, confidence)`` pair where confidence ∈ [0, 1].
        """
        total_chars = 0
        total_image_area = 0.0
        total_page_area = 0.0

        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = pdf.pages[: self.max_pages_to_sample]

            for page in pages_to_check:
                page_width = float(page.width)
                page_height = float(page.height)
                page_area = page_width * page_height

                total_page_area += page_area

                # Character count
                text = page.extract_text() or ""
                total_chars += len(text)

                # Image area
                for img in page.images:
                    img_width = float(img["x1"]) - float(img["x0"])
                    img_height = float(img["bottom"]) - float(img["top"])
                    total_image_area += abs(img_width * img_height)

        # Guard against empty / zero-area documents
        if total_page_area == 0:
            return ("scanned_image", 0.5)

        char_density = total_chars / total_page_area
        image_ratio = total_image_area / total_page_area

        # ------ Classification logic ------
        is_low_chars = char_density < self.min_char_density
        is_high_images = image_ratio > self.scanned_image_ratio

        if is_low_chars and is_high_images:
            origin_type: OriginType = "scanned_image"
        elif (not is_low_chars) and (not is_high_images):
            origin_type = "native_digital"
        else:
            origin_type = "mixed"

        # ------ Confidence score ------
        char_distance = abs(char_density - self.min_char_density) / max(
            self.min_char_density, 1e-9
        )
        image_distance = abs(image_ratio - self.scanned_image_ratio) / max(
            self.scanned_image_ratio, 1e-9
        )
        raw_confidence = (
            min(char_distance, 1.0) + min(image_distance, 1.0)
        ) / 2.0
        confidence = 0.5 + raw_confidence * 0.5
        confidence = min(confidence, 1.0)

        return (origin_type, round(confidence, 4))

    # ------------------------------------------------------------------
    # Layout complexity detection
    # ------------------------------------------------------------------

    def detect_layout_complexity(self, pdf_path: str) -> LayoutComplexity:
        """Analyse the first N pages to classify layout complexity.

        Uses pdfplumber to detect multi-column layouts, table-heavy pages,
        and figure-heavy pages via simple heuristics.

        Returns
        -------
        LayoutComplexity
            One of ``"multi_column"``, ``"table_heavy"``,
            ``"figure_heavy"``, or ``"single_column"``.
        """
        total_page_area = 0.0
        total_table_area = 0.0
        total_image_area = 0.0
        total_chars = 0
        all_x_midpoints: list[float] = []

        with pdfplumber.open(pdf_path) as pdf:
            pages = pdf.pages[: self.layout_max_pages]

            for page in pages:
                page_width = float(page.width)
                page_height = float(page.height)
                page_area = page_width * page_height
                total_page_area += page_area

                # --- Character count ---
                text = page.extract_text() or ""
                total_chars += len(text)

                # --- Collect x-midpoints of individual words ---
                words = page.extract_words() if hasattr(page, "extract_words") else []
                for w in words:
                    x0 = float(w["x0"])
                    x1 = float(w["x1"])
                    mid = (x0 + x1) / 2.0
                    # Normalise to [0, 1] relative to page width
                    all_x_midpoints.append(mid / page_width if page_width else 0)

                # --- Table area ---
                tables = (
                    page.find_tables()
                    if hasattr(page, "find_tables")
                    else []
                )
                for tbl in tables:
                    bbox = tbl.bbox  # (x0, top, x1, bottom)
                    tbl_w = float(bbox[2]) - float(bbox[0])
                    tbl_h = float(bbox[3]) - float(bbox[1])
                    total_table_area += abs(tbl_w * tbl_h)

                # --- Image area ---
                for img in page.images:
                    img_w = float(img["x1"]) - float(img["x0"])
                    img_h = float(img["bottom"]) - float(img["top"])
                    total_image_area += abs(img_w * img_h)

        if total_page_area == 0:
            return "single_column"

        # --- Decision: table-heavy ---
        table_ratio = total_table_area / total_page_area
        if table_ratio > self.table_area_ratio_threshold:
            return "table_heavy"

        # --- Decision: figure-heavy ---
        image_ratio = total_image_area / total_page_area
        char_density = total_chars / total_page_area
        if (
            image_ratio > self.figure_ratio_threshold
            and char_density > self.figure_char_density_min
        ):
            return "figure_heavy"

        # --- Decision: multi-column ---
        if len(all_x_midpoints) >= 10:
            # Coefficient of variation of normalised x-midpoints
            avg = mean(all_x_midpoints)
            if avg > 0:
                sd = stdev(all_x_midpoints) if len(all_x_midpoints) > 1 else 0.0
                cv = sd / avg
                if cv > self.column_variance_threshold:
                    return "multi_column"

        return "single_column"

    # ------------------------------------------------------------------
    # Domain hint detection
    # ------------------------------------------------------------------

    def detect_domain_hint(self, text_sample: str) -> DomainHint:
        """Classify the document's subject-matter domain via keyword matching.

        Keywords are loaded from ``domain_keywords`` in
        ``extraction_rules.yaml``.  The first domain whose keywords
        appear in *text_sample* (case-insensitive) wins.  Falls back
        to ``"general"`` if no match is found.

        Parameters
        ----------
        text_sample : str
            A representative text excerpt (e.g. first 2 pages).

        Returns
        -------
        DomainHint
            One of ``"financial"``, ``"legal"``, ``"technical"``,
            ``"medical"``, or ``"general"``.
        """
        lower_text = text_sample.lower()

        best_domain: DomainHint = "general"
        best_count = 0

        for domain, keywords in self.domain_keywords.items():
            count = sum(1 for kw in keywords if kw.lower() in lower_text)
            if count > best_count:
                best_count = count
                best_domain = domain  # type: ignore[assignment]

        return best_domain

    # ------------------------------------------------------------------
    # Estimated extraction cost
    # ------------------------------------------------------------------

    def estimate_extraction_cost(
        self,
        origin_type: str,
        layout_complexity: str,
    ) -> ExtractionCost:
        """Estimate the computational tier needed for extraction.

        Parameters
        ----------
        origin_type : str
            Result of :meth:`detect_origin_type`.
        layout_complexity : str
            Result of :meth:`detect_layout_complexity`.

        Returns
        -------
        ExtractionCost
            One of ``"fast_text_sufficient"``, ``"needs_layout_model"``,
            or ``"needs_vision_model"``.
        """
        if origin_type == "scanned_image":
            return "needs_vision_model"

        if (
            origin_type == "native_digital"
            and layout_complexity == "single_column"
        ):
            return "fast_text_sufficient"

        return "needs_layout_model"

    # ------------------------------------------------------------------
    # Full document profiling
    # ------------------------------------------------------------------

    def generate_document_profile(self, pdf_path: str) -> DocumentProfile:
        """Run all triage classifiers and return a complete DocumentProfile.

        Orchestrates:

        1. ``detect_origin_type``
        2. ``detect_layout_complexity``
        3. Text extraction for domain hint (first 2 pages)
        4. ``detect_domain_hint``
        5. ``estimate_extraction_cost``

        The resulting profile is also persisted to
        ``.refinery/profiles/{document_id}.json``.

        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.

        Returns
        -------
        DocumentProfile
        """
        # --- 1. origin type ---
        origin_type, _origin_conf = self.detect_origin_type(pdf_path)

        # --- 2. layout complexity ---
        layout_complexity = self.detect_layout_complexity(pdf_path)

        # --- 3. extract text sample for domain detection (first 2 pages) ---
        text_sample = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                text_sample += (page.extract_text() or "") + "\n"

        # --- 4. domain hint ---
        domain_hint = self.detect_domain_hint(text_sample)

        # --- 5. extraction cost ---
        extraction_cost = self.estimate_extraction_cost(
            origin_type, layout_complexity
        )

        # --- Build document id from file name hash ---
        file_name = Path(pdf_path).stem
        source_filename = Path(pdf_path).name
        doc_id = hashlib.sha256(file_name.encode()).hexdigest()[:12]

        profile = DocumentProfile(
            document_id=doc_id,
            source_filename=source_filename,
            origin_type=origin_type,
            layout_complexity=layout_complexity,
            language="en",  # placeholder — language detection not yet wired
            language_confidence=0.5,
            domain_hint=domain_hint,
            estimated_extraction_cost=extraction_cost,
        )

        # --- Persist to .refinery/profiles/ ---
        self._save_profile(profile)

        return profile

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_profile(profile: DocumentProfile) -> Path:
        """Serialise *profile* to ``.refinery/profiles/{document_id}.json``.

        Creates the directory tree if it does not exist.

        Returns
        -------
        Path
            Absolute path to the written JSON file.
        """
        profiles_dir = (
            Path(__file__).resolve().parents[2] / ".refinery" / "profiles"
        )
        profiles_dir.mkdir(parents=True, exist_ok=True)

        out_path = profiles_dir / f"{profile.document_id}.json"
        out_path.write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )
        return out_path

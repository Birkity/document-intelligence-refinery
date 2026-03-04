"""Triage Agent — Stage 1 of the Document Intelligence Refinery.

Classifies incoming documents by origin type, layout complexity,
language, and domain so that downstream extraction stages can select
the most cost-effective strategy.

Only origin-type detection is implemented in this iteration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pdfplumber
import yaml


# Type aliases for clarity
OriginType = Literal["native_digital", "scanned_image", "mixed"]

# Default path to the extraction rules configuration
_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"


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
            rules = yaml.safe_load(fh)

        origin_cfg = rules.get("origin_detection", {})
        self.min_char_density: float = origin_cfg.get("min_char_density", 0.01)
        self.scanned_image_ratio: float = origin_cfg.get("scanned_image_ratio", 0.5)
        self.max_pages_to_sample: int = origin_cfg.get("max_pages_to_sample", 5)

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
        # The further the signals are from the decision boundary the more
        # confident we are.  Normalise each signal's distance then average.
        char_distance = abs(char_density - self.min_char_density) / max(
            self.min_char_density, 1e-9
        )
        image_distance = abs(image_ratio - self.scanned_image_ratio) / max(
            self.scanned_image_ratio, 1e-9
        )
        raw_confidence = (min(char_distance, 1.0) + min(image_distance, 1.0)) / 2.0
        # Clamp to [0.5, 1.0] for clear-cut cases; floor at 0.5 for edge cases
        confidence = 0.5 + raw_confidence * 0.5
        confidence = min(confidence, 1.0)

        return (origin_type, round(confidence, 4))

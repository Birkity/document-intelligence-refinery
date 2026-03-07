"""OCRExtractor — Strategy C (OCR-heavy fallback using RapidOCR).

Uses RapidOCR for scanned/image pages or low-confidence extraction.
Falls back to EasyOCR or PyMuPDF text extraction if RapidOCR is
unavailable.

This is the OCR-heavy path in the extraction budget ladder:
  Cheap → pdfplumber (FastText)
  Medium → Docling (Layout)
  Hard → RapidOCR (this)
  Extreme → Vision LLM (VisionExtractor)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.models.schemas import (
    BoundingBox,
    ExtractedDocument,
    ExtractedPage,
    TextBlock,
)
from src.strategies.base import BaseExtractor

log = logging.getLogger(__name__)

_DEFAULT_RULES = (
    Path(__file__).resolve().parents[2] / "rubric" / "extraction_rules.yaml"
)


def _render_page_to_numpy(pdf_path: str, page_idx: int, dpi: int = 300) -> Any:
    """Render a single PDF page to a numpy array using PyMuPDF."""
    import fitz
    import numpy as np

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_idx]
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        return img
    finally:
        doc.close()


def _deduplicate_blocks(text_blocks: list[TextBlock]) -> list[TextBlock]:
    """Remove near-duplicate OCR blocks within a page.

    Keeps the first occurrence of each normalised text string.  This
    eliminates phantom repetitions produced when RapidOCR detects the
    same text region multiple times (e.g., overlapping render passes,
    decorative stamps).
    """
    seen: set[str] = set()
    result: list[TextBlock] = []
    for blk in text_blocks:
        key = " ".join(blk.content.lower().split())
        if key and key not in seen:
            seen.add(key)
            result.append(blk)
    return result


def _spatial_row_reconstruct(
    text_blocks: list[TextBlock],
    page_height: float,
) -> list[TextBlock]:
    """Reassemble OCR text blocks into logical table rows.

    OCR engines read columnar financial-statement tables
    column-by-column, producing isolated blocks like:
        Block A: "Revenue from mobile broadband"  (x=50,  y=200)
        Block B: "5,594,483,615"                  (x=380, y=200)
        Block C: "5,554,991,960"                  (x=520, y=200)

    This function groups blocks whose y-centres fall within a tight
    vertical tolerance into a single row and joins them left-to-right
    with " | ", yielding:
        "Revenue from mobile broadband | 5,594,483,615 | 5,554,991,960"

    The resulting text is directly parseable by the table-row regex in
    ``FactTableExtractor`` and gives the LLM proper context.

    Blocks that are already single-column (no horizontal peers on the
    same row) are returned unchanged.
    """
    if len(text_blocks) <= 1:
        return text_blocks

    # Row tolerance in page-coordinate units (≈ 8 pt ≈ 2.8 mm)
    row_tol = max(8.0, page_height * 0.012)

    # Sort top-to-bottom by y-centre
    def _yc(b: TextBlock) -> float:
        return (b.bbox.y1 + b.bbox.y2) / 2.0

    sorted_blks = sorted(text_blocks, key=_yc)

    # Greedily group into rows
    rows: list[list[TextBlock]] = []
    current: list[TextBlock] = [sorted_blks[0]]
    current_yc = _yc(sorted_blks[0])

    for blk in sorted_blks[1:]:
        byc = _yc(blk)
        if abs(byc - current_yc) <= row_tol:
            current.append(blk)
        else:
            rows.append(current)
            current = [blk]
            current_yc = byc
    rows.append(current)

    # Merge each multi-block row left-to-right
    result: list[TextBlock] = []
    for row in rows:
        if len(row) == 1:
            result.append(row[0])
            continue

        # Sort left-to-right
        row.sort(key=lambda b: (b.bbox.x1 + b.bbox.x2) / 2.0)

        # Only merge if blocks are clearly in different horizontal zones
        # (gap between consecutive blocks > min_gap_pt to avoid
        #  accidentally merging tightly-spaced word tokens in prose)
        min_gap_pt = 20.0
        first_x2 = row[0].bbox.x2
        second_x1 = row[1].bbox.x1
        if second_x1 - first_x2 < min_gap_pt and len(row) == 2:
            # Prose words on the same line — keep separate so the chunker
            # can decide whether to merge them as a paragraph
            result.extend(row)
            continue

        combined = " | ".join(b.content for b in row)
        result.append(
            TextBlock(
                content=combined,
                bbox=BoundingBox(
                    x1=min(b.bbox.x1 for b in row),
                    y1=min(b.bbox.y1 for b in row),
                    x2=max(b.bbox.x2 for b in row),
                    y2=max(b.bbox.y2 for b in row),
                    page_number=row[0].bbox.page_number,
                ),
            )
        )
    return result


class OCRExtractor(BaseExtractor):
    """Strategy C — OCR-heavy extraction via RapidOCR.

    Pipeline:
      1. Render each page to a 300-dpi image (PyMuPDF).
      2. Run RapidOCR on the image.
      3. Normalise OCR boxes → TextBlock with bounding boxes.
      4. Fall back to EasyOCR or pdfplumber if RapidOCR is unavailable.
    """

    strategy_name: str = "ocr_heavy"

    def __init__(self, rules_path: str | Path | None = None) -> None:
        super().__init__()
        rp = Path(rules_path) if rules_path else _DEFAULT_RULES
        if rp.exists():
            with open(rp, "r", encoding="utf-8") as fh:
                rules = yaml.safe_load(fh)
        else:
            rules = {}
        esc = rules.get("escalation", {})
        self._min_confidence: float = esc.get("strategy_c_min_confidence", 0.4)

        # Probe for OCR backends
        self._ocr_backend = self._detect_ocr_backend()

    def _detect_ocr_backend(self) -> str:
        """Detect available OCR backend: rapidocr > easyocr > none."""
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: F401
            return "rapidocr"
        except ImportError:
            pass
        try:
            import easyocr  # noqa: F401
            return "easyocr"
        except ImportError:
            pass
        log.warning("No OCR backend available (install rapidocr-onnxruntime or easyocr)")
        return "none"

    def extract(
        self,
        pdf_path: str,
        document_id: str,
        *,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract content using OCR."""
        if self._ocr_backend == "rapidocr":
            return self._extract_rapidocr(pdf_path, document_id, page_numbers)
        elif self._ocr_backend == "easyocr":
            return self._extract_easyocr(pdf_path, document_id, page_numbers)
        else:
            return self._extract_fitz_fallback(pdf_path, document_id, page_numbers)

    def _extract_rapidocr(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract using RapidOCR."""
        from rapidocr_onnxruntime import RapidOCR

        ocr = RapidOCR()
        import fitz

        pages_out: list[ExtractedPage] = []
        total_chars = 0
        total_ocr_conf = 0.0
        n_blocks = 0

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                pnum = page_idx + 1
                if page_numbers and pnum not in page_numbers:
                    continue

                page = doc[page_idx]
                pw, ph = float(page.rect.width), float(page.rect.height)

                # Render to numpy
                try:
                    img = _render_page_to_numpy(pdf_path, page_idx)
                except Exception as exc:
                    log.warning("Failed to render page %d: %s", pnum, exc)
                    pages_out.append(ExtractedPage(page_number=pnum))
                    continue

                # Run RapidOCR
                result, _elapse = ocr(img)
                text_blocks: list[TextBlock] = []

                if result:
                    img_h, img_w = img.shape[:2]
                    scale_x = pw / img_w
                    scale_y = ph / img_h

                    for item in result:
                        # item: [box_coords, text, confidence]
                        box, text, conf = item
                        if not text or not text.strip():
                            continue

                        total_chars += len(text)
                        total_ocr_conf += float(conf)
                        n_blocks += 1

                        # box is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        text_blocks.append(
                            TextBlock(
                                content=text.strip(),
                                bbox=BoundingBox(
                                    x1=min(xs) * scale_x,
                                    y1=min(ys) * scale_y,
                                    x2=max(xs) * scale_x,
                                    y2=max(ys) * scale_y,
                                    page_number=pnum,
                                ),
                            )
                        )

                # Post-process: deduplicate then reconstruct table rows
                text_blocks = _deduplicate_blocks(text_blocks)
                text_blocks = _spatial_row_reconstruct(text_blocks, ph)

                pages_out.append(
                    ExtractedPage(
                        page_number=pnum,
                        text_blocks=text_blocks,
                        extraction_strategy="ocr_heavy",
                        extraction_confidence=round(
                            total_ocr_conf / max(n_blocks, 1), 4
                        ),
                    )
                )
        finally:
            doc.close()

        # Overall confidence
        if n_blocks > 0:
            avg_conf = total_ocr_conf / n_blocks
            self.confidence_score = round(min(avg_conf, 1.0), 4)
        else:
            self.confidence_score = 0.1

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
            strategies_used=["ocr_heavy"],
        )

    def _extract_easyocr(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Extract using EasyOCR as fallback."""
        import easyocr
        import fitz

        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        pages_out: list[ExtractedPage] = []
        total_chars = 0
        total_conf = 0.0
        n_blocks = 0

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                pnum = page_idx + 1
                if page_numbers and pnum not in page_numbers:
                    continue

                page = doc[page_idx]
                pw, ph = float(page.rect.width), float(page.rect.height)

                try:
                    img = _render_page_to_numpy(pdf_path, page_idx)
                except Exception:
                    pages_out.append(ExtractedPage(page_number=pnum))
                    continue

                results = reader.readtext(img)
                text_blocks: list[TextBlock] = []
                img_h, img_w = img.shape[:2]
                scale_x = pw / img_w
                scale_y = ph / img_h

                for (box, text, conf) in results:
                    if not text.strip():
                        continue
                    total_chars += len(text)
                    total_conf += float(conf)
                    n_blocks += 1

                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    text_blocks.append(
                        TextBlock(
                            content=text.strip(),
                            bbox=BoundingBox(
                                x1=min(xs) * scale_x,
                                y1=min(ys) * scale_y,
                                x2=max(xs) * scale_x,
                                y2=max(ys) * scale_y,
                                page_number=pnum,
                            ),
                        )
                    )

                # Post-process: deduplicate then reconstruct table rows
                text_blocks = _deduplicate_blocks(text_blocks)
                text_blocks = _spatial_row_reconstruct(text_blocks, ph)

                pages_out.append(
                    ExtractedPage(
                        page_number=pnum,
                        text_blocks=text_blocks,
                        extraction_strategy="ocr_heavy",
                    )
                )
        finally:
            doc.close()

        if n_blocks > 0:
            self.confidence_score = round(min(total_conf / n_blocks, 1.0), 4)
        else:
            self.confidence_score = 0.1

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
            strategies_used=["ocr_heavy"],
        )

    def _extract_fitz_fallback(
        self,
        pdf_path: str,
        document_id: str,
        page_numbers: list[int] | None = None,
    ) -> ExtractedDocument:
        """Last-resort: PyMuPDF text extraction when no OCR is available."""
        import fitz

        pages_out: list[ExtractedPage] = []
        total_chars = 0

        doc = fitz.open(pdf_path)
        try:
            for page_idx in range(doc.page_count):
                pnum = page_idx + 1
                if page_numbers and pnum not in page_numbers:
                    continue

                page = doc[page_idx]
                blocks = page.get_text("blocks")
                text_blocks: list[TextBlock] = []

                for blk in blocks:
                    text = blk[4].strip() if len(blk) > 4 else ""
                    if not text:
                        continue
                    total_chars += len(text)
                    text_blocks.append(
                        TextBlock(
                            content=text,
                            bbox=BoundingBox(
                                x1=float(blk[0]),
                                y1=float(blk[1]),
                                x2=float(blk[2]),
                                y2=float(blk[3]),
                                page_number=pnum,
                            ),
                        )
                    )

                pages_out.append(
                    ExtractedPage(page_number=pnum, text_blocks=text_blocks)
                )
        finally:
            doc.close()

        n_pages = len(pages_out) or 1
        avg_chars = total_chars / n_pages
        self.confidence_score = round(
            min(avg_chars / 500, 1.0) * 0.5, 4  # capped lower since this is fallback
        )

        return ExtractedDocument(
            document_id=document_id,
            source_filename=Path(pdf_path).name,
            pages=pages_out,
            strategies_used=["ocr_heavy_fallback"],
        )

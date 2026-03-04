"""BaseExtractor — abstract interface for all extraction strategies.

Every extraction strategy (Fast Text, Layout-Aware, Vision-Augmented)
must subclass BaseExtractor and implement ``extract()``.  This ensures
a uniform contract: all strategies produce an ``ExtractedDocument``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.schemas import ExtractedDocument


class BaseExtractor(ABC):
    """Abstract base class for document extraction strategies.

    Attributes
    ----------
    confidence_score : float
        Confidence in the quality of the last extraction, ∈ [0, 1].
        Updated after each ``extract()`` call.
    strategy_name : str
        Human-readable label for ledger logging.
    """

    strategy_name: str = "base"

    def __init__(self) -> None:
        self.confidence_score: float = 0.0

    @abstractmethod
    def extract(self, pdf_path: str, document_id: str) -> ExtractedDocument:
        """Extract structured content from *pdf_path*.

        Implementations must:
        1. Populate ``self.confidence_score`` before returning.
        2. Return an ``ExtractedDocument`` with at least page-level data.

        Parameters
        ----------
        pdf_path : str
            Absolute path to the PDF file.
        document_id : str
            Unique document identifier for the output schema.

        Returns
        -------
        ExtractedDocument
        """
        ...

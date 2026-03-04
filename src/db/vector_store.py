"""ChromaDB vector store wrapper for the Document Intelligence Refinery.

Stores chunk embeddings with document / page / section metadata.
Uses a free local embedding model (default: ``all-MiniLM-L6-v2``
via chromadb's built-in sentence-transformers integration).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_PERSIST = (
    Path(__file__).resolve().parents[2] / ".refinery" / "chroma_store"
)
_COLLECTION_NAME = "refinery_chunks"


class VectorStore:
    """Thin wrapper around a local ChromaDB collection.

    Parameters
    ----------
    persist_dir : str | Path | None
        Directory where Chroma persists data.
    collection_name : str
        Name of the Chroma collection.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        collection_name: str = _COLLECTION_NAME,
    ) -> None:
        import chromadb

        pd = str(Path(persist_dir) if persist_dir else _DEFAULT_PERSIST)
        self._client = chromadb.PersistentClient(path=pd)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks into the vector collection.

        Parameters
        ----------
        ids : list[str]
            Unique IDs for each chunk (e.g. ``{doc_id}_{chunk_idx}``).
        documents : list[str]
            Text content of each chunk (Chroma will embed automatically).
        metadatas : list[dict]
            Per-chunk metadata — must include ``document_id``,
            ``page_number``, and ``section_path``.
        """
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self, query_text: str, n_results: int = 5
    ) -> dict[str, Any]:
        """Semantic search over stored chunks.

        Returns the raw ChromaDB query result dict.
        """
        return self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )

    @property
    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

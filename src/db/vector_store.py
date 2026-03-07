"""ChromaDB vector store wrapper for the Document Intelligence Refinery.

Stores chunk embeddings with document / page / section metadata.
Uses a free local embedding model (default: ``all-MiniLM-L6-v2``
via chromadb's built-in sentence-transformers integration).

Enhanced metadata filters: query by document_id, page range, chunk_type.
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

        Recommended metadata keys::

            document_id, page_number, section_path, chunk_type,
            content_hash, parent_section
        """
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        *,
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Semantic search over stored chunks.

        Parameters
        ----------
        where : dict | None
            ChromaDB metadata filter, e.g.
            ``{"document_id": "abc123"}`` or
            ``{"$and": [{"document_id": "abc"}, {"page_number": {"$gte": 3}}]}``
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def query_by_document(
        self,
        query_text: str,
        document_id: str,
        n_results: int = 5,
    ) -> dict[str, Any]:
        """Convenience: semantic search scoped to a single document."""
        return self.query(
            query_text,
            n_results=n_results,
            where={"document_id": document_id},
        )

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks for *document_id* from the collection."""
        self._collection.delete(where={"document_id": document_id})

    @property
    def count(self) -> int:
        """Number of chunks currently stored."""
        return self._collection.count()

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Chunk:
    id: str
    doc_id: str
    chunk_id: int
    text: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk]) -> None: ...

    def search(self, query_vec: list[float], top_k: int = 4) -> list[SearchResult]: ...


class InMemoryStore:
    """Flat in-memory vector index using numpy cosine similarity.

    Fine up to tens of thousands of chunks; swap for a real vector DB
    (pgvector, Qdrant, etc.) past that by implementing the same Protocol.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._normalized: np.ndarray | None = None

    def upsert(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id!r} has no embedding")
            self._chunks[chunk.id] = chunk
        self._rebuild_index()

    def search(self, query_vec: list[float], top_k: int = 4) -> list[SearchResult]:
        # Nothing has been ingested yet.
        if self._normalized is None or not self._ids:
            return []

        # Normalize the query vector too, so the dot product below is a
        # true cosine similarity (both sides length 1).
        query = np.asarray(query_vec, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        # One matrix-vector multiply scores every stored chunk against the
        # query at once: cosine_similarity(a, b) == dot(a, b) when a and b
        # are both unit vectors.
        scores = self._normalized @ query

        # argpartition is an O(n) "get me the top k, unsorted" — cheaper
        # than fully sorting all scores when we only need the best few.
        # Then argsort just those k to put them in ranked order.
        k = min(top_k, len(self._ids))
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            SearchResult(chunk=self._chunks[self._ids[i]], score=float(scores[i]))
            for i in top_indices
        ]

    def _rebuild_index(self) -> None:
        # Rebuilds the whole matrix from scratch on every upsert. Simple
        # and correct; if upserts get frequent/large this is the first
        # thing to optimize (append instead of rebuild).
        self._ids = list(self._chunks.keys())
        vectors = np.array([self._chunks[i].embedding for i in self._ids], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid divide-by-zero for an all-zero embedding
        self._normalized = vectors / norms

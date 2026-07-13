import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import Settings


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


# Fixed namespace for deriving Qdrant point IDs — arbitrary but must stay
# constant, since changing it would remap every existing chunk to a new ID.
_QDRANT_ID_NAMESPACE = uuid.UUID("5b1f9b0a-6b8b-4f2a-9d1a-2b6f0e7c8a11")


def chunk_point_id(chunk_id: str) -> str:
    """Derives a stable Qdrant point ID from our chunk id.

    Qdrant point IDs must be an unsigned integer or a UUID — arbitrary
    strings like "doc::3" aren't valid. uuid5 is deterministic (same
    chunk_id -> same point id every time), so re-ingesting a chunk
    overwrites its existing point instead of creating a duplicate.
    """
    return str(uuid.uuid5(_QDRANT_ID_NAMESPACE, chunk_id))


class QdrantStore:
    """VectorStore backed by a Qdrant collection.

    Unlike InMemoryStore, the actual indexing and similarity search happen
    inside Qdrant — this class is a thin translation layer between our
    Chunk/SearchResult types and Qdrant's point/payload model.
    """

    def __init__(self, url: str, collection: str, vector_size: int) -> None:
        self._client = QdrantClient(url=url)
        self._collection = collection
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            return
        # Distance.COSINE matches InMemoryStore's cosine similarity, so scores
        # from either backend are comparable and "higher is better" holds for both.
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk]) -> None:
        points = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"chunk {chunk.id!r} has no embedding")
            points.append(
                PointStruct(
                    id=chunk_point_id(chunk.id),
                    vector=chunk.embedding,
                    # Qdrant only stores vector + payload; everything needed to
                    # reconstruct a Chunk on the way out has to live in payload.
                    payload={
                        "chunk_id_str": chunk.id,
                        "doc_id": chunk.doc_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    },
                )
            )
        self._client.upsert(collection_name=self._collection, points=points)

    def search(self, query_vec: list[float], top_k: int = 4) -> list[SearchResult]:
        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for hit in response.points:
            payload = hit.payload or {}
            chunk = Chunk(
                id=payload.get("chunk_id_str", str(hit.id)),
                doc_id=payload["doc_id"],
                chunk_id=payload["chunk_id"],
                text=payload["text"],
                metadata=payload.get("metadata", {}),
            )
            results.append(SearchResult(chunk=chunk, score=hit.score))
        return results


def create_store(settings: Settings) -> VectorStore:
    if settings.vector_store == "qdrant":
        return QdrantStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            vector_size=settings.qdrant_vector_size,
        )
    return InMemoryStore()

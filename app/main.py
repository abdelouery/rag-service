import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import Settings
from .embeddings import EmbeddingClient
from .ingest import ingest_directory
from .llm import LLMClient, create_llm_client
from .schemas import QueryRequest, QueryResponse
from .store import InMemoryStore, VectorStore

settings = Settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_llm: LLMClient | None = None
_embedder: EmbeddingClient | None = None
_store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _llm, _embedder, _store
    _llm = create_llm_client(settings)
    _embedder = EmbeddingClient(settings)
    _store = InMemoryStore()

    data_dir = Path(settings.data_dir)
    if data_dir.exists():
        count = await ingest_directory(
            data_dir, _embedder, _store, settings.chunk_tokens, settings.chunk_overlap
        )
        print(f"Ingested {count} chunks from {data_dir}")

    yield


app = FastAPI(title="RAG Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    assert _llm is not None
    assert _embedder is not None
    assert _store is not None

    query_vec = await _embedder.embed(req.question)
    results = _store.search(query_vec, top_k=req.top_k)

    context = "\n\n".join(
        f"[{r.chunk.doc_id}#{r.chunk.chunk_id}]\n{r.chunk.text}" for r in results
    )
    system = (
        "You are a helpful assistant. Answer the question using only the "
        "provided context. If the context doesn't contain the answer, say so "
        "instead of guessing."
    )
    user = f"Context:\n{context}\n\nQuestion: {req.question}" if results else req.question

    answer = await _llm.generate(system=system, user=user)
    sources = [
        {"doc_id": r.chunk.doc_id, "chunk_id": r.chunk.chunk_id, "score": r.score}
        for r in results
    ]
    return QueryResponse(answer=answer, sources=sources, grounded=bool(results))

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import Settings
from .embeddings import EmbeddingClient
from .ingest import ingest_directory
from .llm import LLMClient, create_llm_client
from .schemas import QueryRequest, QueryResponse, Source
from .store import InMemoryStore, VectorStore

# Sentinel the model is instructed to emit verbatim when the context doesn't
# answer the question. Detecting it is a plain string check, not a semantic
# judgment of whether the answer is actually correct — see the /query handler.
UNGROUNDED_MESSAGE = "I don't know based on the provided context."

settings = Settings()
logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
_llm: LLMClient | None = None
_embedder: EmbeddingClient | None = None
_store: VectorStore | None = None

logger = logging.getLogger(__name__)

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

    # 1. Embed the question so it can be compared against stored chunk vectors.
    query_vec = await _embedder.embed(req.question)
    
    # 2. Retrieve the top-k most similar chunks from the vector store.
    results = _store.search(query_vec, top_k=req.top_k)

    # 3. Augment: stitch retrieved chunks into a single context block, tagged
    # with their doc/chunk id so the model (and our `sources` field) can cite them.
    context = "\n\n".join(
        f"[{r.chunk.doc_id}#{r.chunk.chunk_id}]\n{r.chunk.text}" for r in results
    )
    system = (
        "You are a helpful assistant. Answer the question using ONLY the "
        "information in the context chunks below — do not use outside knowledge, "
        "even if you know the answer. Every claim you make must be supported by "
        "the context. Cite the chunks you used inline right after the claim they "
        "support, like [doc_id#chunk_id].\n\n"
        f'If the context does not contain enough information to answer, respond '
        f'with exactly this sentence and nothing else: "{UNGROUNDED_MESSAGE}"'
    )
    # Fall back to the sentinel directly if nothing was retrieved — there's no
    # context to send the model, so skip the round trip.
    if results:
        user = f"Context:\n{context}\n\nQuestion: {req.question}"
        answer = await _llm.generate(system=system, user=user)
    else:
        answer = UNGROUNDED_MESSAGE

    # Detecting "ungrounded" is a case-insensitive prefix match on the sentinel
    # we told the model to emit — not a semantic check of the answer's
    # correctness. This is fragile: it breaks if the model paraphrases the
    # refusal instead of quoting it verbatim, wraps it in extra text (e.g. "I
    # don't know based on the provided context, but generally..."), or emits
    # it after giving a partial answer anyway. A more robust approach would
    # use structured output (e.g. a `{"answered": bool, "answer": str}` schema)
    # so groundedness is a field the model fills in rather than text we parse.
    grounded = bool(results) and not answer.strip().lower().startswith(
        UNGROUNDED_MESSAGE.lower()
    )
    sources = [
        Source(doc_id=r.chunk.doc_id, chunk_id=r.chunk.chunk_id, score=r.score, text=r.chunk.text)
        for r in results
    ]
    return QueryResponse(answer=answer, sources=sources, grounded=grounded)

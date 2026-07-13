import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import Settings
from .embeddings import EmbeddingClient
from .ingest import ingest_directory
from .llm import LLMClient, create_llm_client
from .schemas import LLMAnswer, QueryRequest, QueryResponse, Source
from .store import VectorStore, create_store

# Message returned when the store has nothing to search — there's no context
# to send the model, so we skip the round trip entirely.
NO_CONTEXT_MESSAGE = "I don't know — no documents have been ingested to answer from."

# Constrains the model's response shape via output_config.format: "answered"
# is a field the model fills in directly, not text we pattern-match after the
# fact (see the older sentinel-based approach this replaced).
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answered": {
            "type": "boolean",
            "description": "true only if the provided context contains enough information to answer the question",
        },
        "answer": {
            "type": "string",
            "description": "the answer grounded in the context, or a brief note on what's missing if answered is false",
        },
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "doc_id#chunk_id identifiers of the chunks actually used to support the answer; empty if answered is false",
        },
    },
    "required": ["answered", "answer", "citations"],
    "additionalProperties": False,
}

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
    _store = create_store(settings)

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

    logger.info('query_vec=%s', query_vec)
    
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
        "the context. Set `answered` to false if the context doesn't contain "
        "enough information to answer, rather than guessing or partially "
        "answering. List the doc_id#chunk_id of every chunk you actually used "
        "in `citations`."
    )
    if results:
        user = f"Context:\n{context}\n\nQuestion: {req.question}"
        raw = await _llm.generate_structured(system=system, user=user, schema=ANSWER_SCHEMA)
        # Validated against LLMAnswer rather than trusted as-is: output_config.format
        # guarantees the JSON matches the schema's types, but doesn't guarantee the
        # model didn't, say, hallucinate a citation string in some other shape.
        llm_answer = LLMAnswer.model_validate(raw)
        answer = llm_answer.answer
        grounded = llm_answer.answered
    else:
        answer = NO_CONTEXT_MESSAGE
        grounded = False

    sources = [
        Source(doc_id=r.chunk.doc_id, chunk_id=r.chunk.chunk_id, score=r.score, text=r.chunk.text)
        for r in results
    ]
    return QueryResponse(answer=answer, sources=sources, grounded=grounded)

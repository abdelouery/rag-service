from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from .config import Settings
from .llm import LLMClient, create_llm_client
from .schemas import QueryRequest, QueryResponse

settings = Settings()
_llm: LLMClient | None = None

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _llm
    _llm = create_llm_client(settings)
    yield


app = FastAPI(title="RAG Service", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    assert _llm is not None
    system = "You are a helpful assistant. Answer the question clearly and concisely."
    answer = await _llm.generate(system=system, user=req.question)
    return QueryResponse(answer=answer, sources=[], grounded=True)

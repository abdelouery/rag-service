import asyncio

from app import main
from app.schemas import QueryRequest
from app.store import Chunk, SearchResult


class FakeEmbedder:
    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query_vec: list[float], top_k: int = 4) -> list[SearchResult]:
        return self._results[:top_k]


class FakeLLM:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def generate(self, system: str, user: str) -> str:
        return self._answer


def test_query_answers_from_context_when_present(monkeypatch):
    chunk = Chunk(id="rag::0", doc_id="rag", chunk_id=0, text="RAG chunks are ~600 tokens with ~80 overlap.")
    monkeypatch.setattr(main, "_embedder", FakeEmbedder())
    monkeypatch.setattr(main, "_store", FakeStore([SearchResult(chunk=chunk, score=0.91)]))
    monkeypatch.setattr(main, "_llm", FakeLLM("Chunks are ~600 tokens with ~80 overlap [rag#0]."))

    resp = asyncio.run(main.query(QueryRequest(question="How big are the chunks?", top_k=4)))

    assert resp.grounded is True
    assert "[rag#0]" in resp.answer
    assert [s.model_dump() for s in resp.sources] == [
        {"doc_id": "rag", "chunk_id": 0, "score": 0.91, "text": chunk.text}
    ]


def test_query_reports_ungrounded_when_context_lacks_answer(monkeypatch):
    chunk = Chunk(id="rag::0", doc_id="rag", chunk_id=0, text="Unrelated content about chunking.")
    monkeypatch.setattr(main, "_embedder", FakeEmbedder())
    monkeypatch.setattr(main, "_store", FakeStore([SearchResult(chunk=chunk, score=0.15)]))
    monkeypatch.setattr(main, "_llm", FakeLLM(main.UNGROUNDED_MESSAGE))

    resp = asyncio.run(main.query(QueryRequest(question="What is the capital of France?", top_k=4)))

    assert resp.grounded is False
    assert resp.answer == main.UNGROUNDED_MESSAGE
    # Retrieved chunks are still surfaced even when the model couldn't answer,
    # so a caller can see what was (unsuccessfully) considered.
    assert len(resp.sources) == 1


def test_query_ungrounded_when_store_is_empty(monkeypatch):
    monkeypatch.setattr(main, "_embedder", FakeEmbedder())
    monkeypatch.setattr(main, "_store", FakeStore([]))
    monkeypatch.setattr(main, "_llm", FakeLLM("should not be called"))

    resp = asyncio.run(main.query(QueryRequest(question="Anything?", top_k=4)))

    assert resp.grounded is False
    assert resp.answer == main.UNGROUNDED_MESSAGE
    assert resp.sources == []

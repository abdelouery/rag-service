from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class Source(BaseModel):
    doc_id: str
    chunk_id: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    grounded: bool = True

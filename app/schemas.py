from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    grounded: bool = True

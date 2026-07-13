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


class LLMAnswer(BaseModel):
    """Shape the LLM is constrained to return via output_config.format.

    Groundedness is a field the model fills in directly, rather than text
    we pattern-match after the fact.
    """

    answered: bool
    answer: str
    citations: list[str] = []

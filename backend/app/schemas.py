from typing import Optional, List
from pydantic import BaseModel, Field

class SearchResult(BaseModel):
    id: str
    title: str
    authors: Optional[str] = None
    venue: Optional[str] = None
    year: Optional[int] = None
    n_citation: Optional[int] = None
    abstract: str
    score: float = Field(..., description="Retrieval score")

class SearchResponse(BaseModel):
    query: str
    count: int
    retrieval_mode: str
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    bm25_indexed_documents: int
    semantic_indexed_documents: int

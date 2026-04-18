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
    score: float = Field(..., description="BM25 score")

class SearchResponse(BaseModel):
    query: str
    count: int
    results: List[SearchResult]

class HealthResponse(BaseModel):
    status: str
    indexed_documents: int

import os

from app.config import DEFAULT_TOP_K, MAX_TOP_K, PROCESSED_DATA_PATH
from app.preprocess import preprocess_csv
from app.search_engine import SearchEngine
from app.semantic_search_engine import SemanticSearchEngine
from app.schemas import HealthResponse, SearchResponse, SearchResult
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query

search_engine = SearchEngine()
semantic_search_engine = SemanticSearchEngine()

def processed_file_has_content(path: str) -> bool:
    """
    Return True if the processed file exists and has non-empty content.
    """
    return os.path.exists(path) and os.path.getsize(path) > 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Preprocess CSV into JSONL is needed.
    Build BM25 index.
    """

    if not processed_file_has_content(PROCESSED_DATA_PATH):
        print("Processed file not found. Starting preprocessing: ")
        written = preprocess_csv()
        print(f"Preprocessing complete. Wrote {written} records.")
    else:
        print("Processed file found, skipping preprocessing.")

    bm25_count = search_engine.initialize()
    print(f"BM25 index ready with {bm25_count} documents.")

    semantic_count = semantic_search_engine.initialize()
    print(f"Semantic index ready with {semantic_count} documents.")

    yield

app = FastAPI(
    title="Search Ranking Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        bm25_indexed_documents=len(search_engine.documents),
        semantic_indexed_documents=len(semantic_search_engine.metadata),
    )

@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="Search Query"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    results = search_engine.search(query=query, top_k=top_k)

    return SearchResponse(
        query=query,
        count=len(results),
        retrieval_mode="bm25",
        results=[SearchResult(**result) for result in results]
    )

@app.get("/search/semantic", response_model=SearchResponse)
def semantic_search(
    q: str = Query(..., min_length=1, description="semantic search query"),
    top_k: int = Query(DEFAULT_TOP_K, ge=1, le=MAX_TOP_K),
) -> SearchResponse:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        results = semantic_search_engine.search(query=query, top_k=top_k)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Semantic seach failed: {str(exc)}"
        ) from exc
    
    return SearchResponse(
        query=query,
        count=len(results),
        retrieval_mode="semantic",
        results=[SearchResult(**result) for result in results]
    )
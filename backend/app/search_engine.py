import json
import os
import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from app.config import PROCESSED_DATA_PATH

TOKEN_RE = re.compile(r"\b\w+\b")

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    Lowercases and extracts word tokens.
    """
    if not text:
        return []
    return TOKEN_RE.findall(text.lower())

class SearchEngine():
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: BM25Okapi | None = None
    
    def load_documents(self) -> int:
        """
        Load preprocessed documents from JSONL.
        """
        if not os.path.exists(PROCESSED_DATA_PATH):
            raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_DATA_PATH}")
        
        documents: List[Dict[str, Any]] = []

        with open(PROCESSED_DATA_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                documents.append(json.loads(line))
        
        self.documents = documents
        return len(self.documents)
    
    def build_index(self) -> int:
        """
        Build BM25 index from loaded documents.
        """
        if not self.documents:
            raise ValueError("No documents loaded. Cannot build BM25 index.")
        
        self.tokenized_corpus = [
            tokenize(doc.get("search_text", ""))
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return len(self.tokenized_corpus)
    
    def initialize(self) -> int:
        """
        Load documents and build BM25 index.
        """
        self.load_documents()
        return self.build_index()
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search the BM25 index and return the tok_k results.
        """
        if self.bm25 is None:
            raise RuntimeError("Search engine is not initialized.")
        
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        
        scores = self.bm25.get_scores(query_tokens)

        scored_docs = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            doc = self.documents[index].copy()
            doc["score"] = float(score)
            scored_docs.append(doc)
        
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]
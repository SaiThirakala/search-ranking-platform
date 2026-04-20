import json
import os
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    EMBEDDINGS_PATH,
    EMBEDDING_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)

class SemanticSearchEngine:
    """
    Semantic retrieval engine for dense vector search.

    It loads document embeddings and aligned metadata, builds a FAISS
    index for near-neighbor search, encodes user quries into the same
    embedding space, and returns the top_k nearest documents with
    similarity scores.
    """

    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None
        self.metadata: List[Dict[str, Any]] = []
        self.index: faiss.Index | None = None
        self.model: SentenceTransformer | None = None
        self.embeddings_dim: int | None = None
    
    def load_embeddings(self) -> np.ndarray:
        """
        Load the saved embedding matrix from /data/processed.
        
        Expected shapre: (number_of_documents, embedding_dimension)
        """
        if not os.path.exists(EMBEDDINGS_PATH):
            raise FileNotFoundError(
                f"Embeddings files not found at {EMBEDDINGS_PATH}"
            )
        
        embeddings = np.load(EMBEDDINGS_PATH)

        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D embedding matrix, got shape {embeddings.shape}"
            )
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        self.embeddings = embeddings
        self.embeddings_dim = embeddings.shape[1]
        return embeddings

    def load_metadata(self) -> List[Dict[str, Any]]:
        """
        Load the aligned metadata from the /data/processed
        
        Each line corresponds to one row in the embeddings matrix.
        """
        if not os.path.exists(EMBEDDING_METADATA_PATH):
            raise FileNotFoundError(
                f"Metadata file not found at {EMBEDDING_METADATA_PATH}"
            )
        
        metadata: List[Dict[str, Any]] = []

        with open(EMBEDDING_METADATA_PATH, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                metadata.append(json.loads(line))
        
        self.metadata = metadata
        return metadata

    def validate_alignment(self) -> None:
        """
        Ensure embeddings and metadata are aligned.
        
        The number of embedding rows must match the number of
        metadata rows.
        """
        if self.embeddings is None:
            raise ValueError("EMbeddings must be loaded before alignment validation.")
        
        if not self.metadata:
            raise ValueError("Metadata must be loaded before alignment validation.")
        
        if self.embeddings.shape[0] != len(self.metadata):
            raise ValueError(
                "Embeddings/metadata alignment mismatch: "
                f"{self.embeddings.shpae[0]} embedding rows vs"
                f"{len(self.metadata)} metadata records."
            )

    def load_model(self) -> SentenceTransformer:
        """
        Load the sentence-transformer model used for embedding operation
        
        It is important to use the same model for both document embeddings 
        and query embeddings.
        """
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self.model

    def build_index(self) -> faiss.Index:
        """
        Build an in-memory FAISS index for dense retrieval.
        
        Inner product similarity can be used to approximate cosine similarity
        because embeddings were normalizes during generation.
        """
        if self.embeddings is None:
            raise ValueError("Embeddings must be loaded before building index.")
        
        if self.embeddings_dim is None:
            raise ValueError("Embedding dimenstion is unknown.")
        
        index = faiss.IndexFlatIP(self.embeddings_dim)
        index.add(self.embeddings)

        self.index = index
        return index

    def initialize(self) -> int:
        """
        Full semantic engine initialization
        
        1. Load embeddings
        2. Load metadata
        3. Validate alignment
        4. Load embedding model
        5. Build FAISS index
        
        Returns number of indexed documents.
        """
        self.load_embeddings()
        self.load_metadata()
        self.validate_alignment()
        self.load_model()
        self.build_index()

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert a user query into a normalized embedding vector.
        
        Returns shape (1, embedding_dimension)
        """
        if self.model is None:
            raise RuntimeError("Semantic model is not initialized.")
        
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty.")
        
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        return query_embedding

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic retrieval for a user query.
        
        Returns the top_k nearest documents with similarity scores.
        """
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized.")
        
        query_embedding = self.encode_query(query)

        scores, indices = self.index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            doc = self.metadata[index].copy()
            doc["score"] = float(score)
            results.append(doc)
        
        return results

import json
import os
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import (
    PROCESSED_DATA_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE,
)

def load_processed_docs() -> List[Dict[str, Any]]:
    """
    Load processed documents from the JSONL file created during dataset preprocessing.
    
    Each Line in the file is one JSON object representing a research paper.
    This functio reads all non-empty lines and parses them into Python dicts.
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

    return documents

def ensure_output_dir_exists() -> None:
    """
    Make sure the directories for the embedding artifacts exist before writing.
    """
    embeddings_dir = os.path.dirname(EMBEDDINGS_PATH)
    metadata_dir = os.path.dirname(EMBEDDING_METADATA_PATH)

    if embeddings_dir:
        os.makedirs(embeddings_dir, exist_ok=True)

    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)

def extract_texts_and_metadata(documents: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Split documents into texts for embedding and metadata aligned to those embeddings.
    
    The returns lists must stay in the exact same order so that metadata[i] corresponds to
    embeddings[i]
    """
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for doc in documents:
        search_text = doc.get("search_text", "").strip()
        if not search_text:
            continue

        texts.append(search_text)
        metadata.append(
            {
                "id": doc.get("id"),
                "title": doc.get("title"),
                "authors": doc.get("authors"),
                "venue": doc.get("venue"),
                "year": doc.get("year"),
                "n_citation": doc.get("n_citation"),
                "abstract": doc.get("abstract"),
                "search_text": search_text,
            }
        )
    
    return texts, metadata

def save_metadata(metadata: List[Dict[str, Any]]) -> None:
    """
    Save aligned metadata as JSONL.
    
    Each line corresponds to one embedding row in the .npy file.
    """
    with open(EMBEDDING_METADATA_PATH, "w", encoding="utf-8") as file:
        for record in metadata:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate dense embeddings for the provided texts using a sentence-transformer.

    Returns a NumPy array with shape (number_of_texts, embedding_dimension)
    """
    if not texts:
        raise ValueError("No texts available for embedding")
    
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)

def save_embeddings(embeddings: np.ndarray) -> None:
    """
    Save the emdedding matrix to disk as a Numpy .npy file
    """
    np.save(EMBEDDINGS_PATH, embeddings)

def build_embedding_artifacts() -> Dict[str, Any]:
    """
    End-to-end pipeline for embedding generations.
    
    First load processed documents from preprocessing stage. Then extract
    searchable text and aligned metadata. Generate dense embeddings. Save 
    embeddings and metadata to disk.
    
    Returns summary imformation for logging + debugging.
    """
    ensure_output_dir_exists()

    documents = load_processed_docs()
    texts, metadata = extract_texts_and_metadata(documents)

    embeddings = generate_embeddings(texts)

    save_embeddings(embeddings)
    save_metadata(metadata)

    return {
        "document_count": len(metadata),
        "embedding_shape": tuple(embeddings.shape),
        "embeddings_path": EMBEDDINGS_PATH,
        "metadata_path": EMBEDDING_METADATA_PATH,
        "model_name": EMBEDDING_MODEL_NAME,
    }


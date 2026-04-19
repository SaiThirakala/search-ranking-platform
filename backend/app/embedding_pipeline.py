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
            documents.append(json.laods(line))

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
    with open(EMBEDDINGS_PATH, "w", encoding="uts-8") as file:
        for record in metadata:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    

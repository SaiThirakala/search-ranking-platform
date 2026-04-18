import json
import os
import re
from typing import Any, Dict, Optional

import pandas as pd

from app.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, MAX_RECORDS, MIN_ABSTRACT_LENGTH

WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(value: Any) -> str:
    """
    Convert any input value into a cleaned single-line string.
    """
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    
    text = str(value).strip()
    text = WHITESPACE_RE.sub(" ", text)
    return text

def safe_int(value: Any) -> Optional[int]:
    """
    Converdt value to int is posisble, otherwise return None.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    if not text:
        return None
    
    try:
        return int(float(text))
    except ValueError:
        return None
    
def build_search_text(title: str, abstract: str) -> str:
    """
    Combines article title and abstract into a single searchable field.
    """
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract

def normalize_record(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize one CSV row into the processed search document schems.
    Return None if the row should be skipped.
    """
    paper_id = normalize_text(row.get("id"))
    title = normalize_text(row.get("title"))
    authors = normalize_text(row.get("authors"))
    venue = normalize_text(row.get("venue"))
    year = safe_int(row.get("year"))
    abstract = normalize_text(row.get("abstract"))
    n_citation = safe_int(row.get("n_citation"))

    # Omit any entries that do not contain an id, title, or abstract. Also omit entries that short abstracts.
    if not paper_id:
        return None
    if not title:
        return None
    if not abstract:
        return None
    if len(abstract) < MIN_ABSTRACT_LENGTH:
        return None
    
    search_text = build_search_text(title, abstract)

    return {
        "id": paper_id,
        "title": title,
        "authors": authors or None,
        "venue": venue or None,
        "year": year,
        "n_citation": n_citation,
        "abstract": abstract,
        "search_text": search_text,
    }
    
def ensure_processed_dir_exists() -> None:
    """
    Ensure the output directory exists.
    """
    processed_dir = os.path.dirname(PROCESSED_DATA_PATH)
    if processed_dir:
        os.makedirs(processed_dir, exist_ok=True)

def preprocess_csv() -> int:
    """
    Read raw CSV, normalize records and write processed JSON:.
    Returns the number of written records.
    """

    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")
    
    ensure_processed_dir_exists()

    usecols = ["id", "title", "authors", "venue", "year", "n_citation", "abstract"]
    df = pd.read_csv(
        RAW_DATA_PATH,
        usecols=usecols,
        nrows=MAX_RECORDS,
        low_memory=False
    )

    written = 0
    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as outfile:
        for row in df.to_dict(orient="records"):
            normalized = normalize_record(row)
            if normalized is None:
                continue

            outfile.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            written += 1
    
    return written


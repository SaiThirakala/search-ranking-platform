import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "/app/data/raw/dblp-v10.csv")
PROCESSED_DATA_PATH = os.getenv(
    "PROCESSED_DATA_PATH",
    "/app/data/processed/papers_processed.jsonl",
)

MAX_RECORDS = int(os.getenv("MAX_RECORDS", "10000"))
MIN_ABSTRACT_LENGTH = int(os.getenv("MIN_ABSTRACT_LENGTH", "40"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "10"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDINGS_PATH = os.getenv(
    "EMBEDDINGS_PATH",
    "/app/data/processed/paper_embeddings.npy",
)
EMBEDDING_METADATA_PATH = os.getenv(
    "EMBEDDING_METADATA_PATH",
    "/app/data/processed/paper_metadata.jsonl",
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
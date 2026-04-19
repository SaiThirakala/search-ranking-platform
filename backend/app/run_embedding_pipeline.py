from pprint import pprint
from app.embedding_pipeline import build_embedding_artifacts

if __name__ == "__main__":
    summary = build_embedding_artifacts()
    pprint(summary)

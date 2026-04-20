from pprint import pprint

from app.semantic_search_engine import SemanticSearchEngine

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    count = engine.initialize()

    print(f"Semantic engine initialized with {count} documents.")

    results = engine.search("graph neural networks", top_k=5)

    for idx, result in enumerate(results, start=1):
        print(f"\nResult {idx}")
        pprint(result)
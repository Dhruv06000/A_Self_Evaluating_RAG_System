from sentence_transformers import CrossEncoder

from src.config import DEVICE, RERANKER_MODEL_NAME, RERANK_SCORE_THRESHOLD, RERANK_TOP_K


def setup_reranker():
    return CrossEncoder(RERANKER_MODEL_NAME, device=DEVICE)


def rerank_documents(query: str, docs: list, reranker, debug: bool = False) -> list:
    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    if debug:
        print("\n--- RERANKED SCORES ---")
        for score, doc in scored[:RERANK_TOP_K]:
            print(f"  {score:.3f} | {doc.page_content[:80]!r}")

    filtered = [doc for score, doc in scored if score > RERANK_SCORE_THRESHOLD]

    if not filtered:
        return [doc for score, doc in scored[:3]]

    return filtered[:RERANK_TOP_K]

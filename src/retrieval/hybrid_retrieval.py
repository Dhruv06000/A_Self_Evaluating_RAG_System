from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.config import DEVICE, EMBEDDING_MODEL_NAME, CHROMA_DB_DIR, SEMANTIC_TOP_K, BM25_TOP_K


def setup_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


def setup_db(embedding_model):
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embedding_model,
    )


def setup_retrievers(db):
    semantic = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEMANTIC_TOP_K},
    )

    raw = db.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=d, metadata=m or {})
        for d, m in zip(raw["documents"], raw["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = BM25_TOP_K

    return semantic, bm25


def hybrid_retrieve(query: str, semantic_retriever, bm25_retriever, debug: bool = False) -> list:
    semantic_docs = semantic_retriever.invoke(query)
    keyword_docs = bm25_retriever.invoke(query)

    seen = set()
    unique_docs = []
    for doc in semantic_docs + keyword_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    if debug:
        print(f"\n--- RAW RETRIEVAL: {len(unique_docs)} unique docs ---")
        for doc in unique_docs[:5]:
            print(f"  {doc.page_content[:100]!r}")

    return unique_docs

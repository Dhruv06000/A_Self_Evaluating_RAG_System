from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import DEVICE, EMBEDDING_MODEL_NAME, MODEL_NAME, CHROMA_DB_DIR

load_dotenv()


class BasicRAG:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.db = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": "cosine"},
        )

        self.retriever = self.db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": 0.5,
                "k": 3,
            },
        )

        self.model = ChatGroq(model=MODEL_NAME, temperature=0.2)

    def ask_question(self, query: str) -> dict:
        relevant_docs = self.retriever.invoke(query)

        combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{chr(10).join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
"""

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]

        result = self.model.invoke(messages)

        return {
            "question": query,
            "query": query,
            "rewritten_query": query,
            "answer": result.content,
            "contexts": [doc.page_content for doc in relevant_docs],
            "sources": [doc.metadata.get("title", "Unknown") for doc in relevant_docs],
            "eval": None,
            "used_web": False,
        }


def baseline_answer(query: str) -> dict:
    rag = BasicRAG()
    return rag.ask_question(query)


def main():
    rag = BasicRAG()

    query = "What is Retrieval-Augmented Generation?"
    result = rag.ask_question(query)

    print(f"Query: {query}\n")

    print("--- Context ---")
    for i, doc in enumerate(result["contexts"], 1):
        print(f"Document {i}:\n{doc}\n")

    print("\n--- Generated Response ---")
    print(result["answer"])


if __name__ == "__main__":
    main()

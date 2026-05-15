# Self-Evaluating RAG: hybrid retrieval + reranking + faithfulness retry loop
# Minimal refactor: original SelfRAG behavior preserved, helper logic moved to modules.

import time
import json
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document

from src.config import FAITHFULNESS_MIN_SCORE, MAX_RETRIES
from src.generation.prompt_templates import QUERY_REWRITE_SYSTEM_PROMPT, FAITHFULNESS_SYSTEM_PROMPT, FAITHFULNESS_PROMPT_TEMPLATE
from src.generation.answer_generator import setup_models, build_prompt, generate_answer
from src.retrieval.hybrid_retrieval import setup_embeddings, setup_db, setup_retrievers, hybrid_retrieve
from src.retrieval.reranker import setup_reranker, rerank_documents

load_dotenv()


class SelfRAG:
    def __init__(self):
        self.chat_history = []

        self.embedding_model = setup_embeddings()
        self.db = setup_db(self.embedding_model)
        self.semantic_retriever, self.bm25_retriever = setup_retrievers(self.db)
        self.reranker = setup_reranker()
        self.rewrite_model, self.answer_model, self.faithfulness_model = setup_models()
        self.tavily_client = self._setup_tavily()

    def _setup_tavily(self):
        """
        Tavily web search client.
        Returns None if TAVILY_API_KEY not set.
        Currently disabled for evaluation, same as your original code.
        """
        print("  Tavily web search fallback disabled for evaluation.")
        return None
        try:
            from tavily import TavilyClient
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                print("  ⚠ TAVILY_API_KEY not set — web search fallback disabled.")
                return None
            print("  Tavily web search fallback enabled.")
            return TavilyClient(api_key=api_key)
        except ImportError:
            print("  ⚠ tavily-python not installed — web search fallback disabled.")
            print("    Run: pip install tavily-python")
            return None

    def _safe_invoke(self, model, messages, delay: float = 0.2):
        """Invoke a model with basic error handling and rate-limit delay."""
        try:
            time.sleep(delay)
            return model.invoke(messages)
        except Exception as e:
            print(f"\nModel Error: {e}")
            return AIMessage(content="Error generating response.")

    def _is_ambiguous(self, question: str) -> bool:
        return len(question.strip().split()) <= 1

    def _rewrite_query(self, question: str) -> str:
        if len(question.split()) < 4:
            return question

        messages = [
            SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
            *self.chat_history[-4:],
            HumanMessage(content=question),
        ]

        response = self._safe_invoke(self.rewrite_model, messages)
        return response.content.strip() if response else question

    def _retrieve_docs(self, query: str, debug: bool = False) -> list:
        """Hybrid retrieval followed by cross-encoder reranking."""
        unique_docs = hybrid_retrieve(query, self.semantic_retriever, self.bm25_retriever, debug=debug)
        return rerank_documents(query, unique_docs, self.reranker, debug=debug)

    def _format_chunks(self, docs: list) -> str:
        formatted = []
        for i, doc in enumerate(docs):
            content = doc.page_content.strip().replace("\n", " ")
            preview = content[:300] + ("..." if len(content) > 300 else "")
            source = doc.metadata.get("title", "Unknown")
            doc_type = doc.metadata.get("type", "")
            type_label = " [WEB]" if doc_type == "web" else ""
            formatted.append(f"[Chunk {i+1}] ({source}){type_label}\n{preview}")
        return "\n\n".join(formatted)

    def _build_prompt(self, question: str, docs: list, retry_note: str = "") -> str:
        return build_prompt(question, docs, retry_note)

    def _generate_answer(self, prompt: str) -> str:
        return generate_answer(self.answer_model, prompt, self.chat_history, self._safe_invoke)

    def _check_faithfulness(self, question: str, answer: str, docs: list) -> dict:
        context = "\n\n".join(doc.page_content.strip()[:600] for doc in docs)
        prompt = FAITHFULNESS_PROMPT_TEMPLATE.format(
            question=question,
            context=context,
            answer=answer,
        )

        messages = [
            SystemMessage(content=FAITHFULNESS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        respond = self._safe_invoke(self.faithfulness_model, messages)
        try:
            raw = respond.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(raw)
        except Exception as e:
            print(f"\n[Faithfulness check failed: {e}]")
            return {"faithful": True, "score": -1, "reason": "Eval parse error - skipped."}

    def ask_question(self, question: str, debug: bool = False) -> dict | None:
        if self._is_ambiguous(question):
            print("\nQuestion too vague. Please be more specific.\n")
            return None

        query = self._rewrite_query(question)

        if debug:
            print(f"\n--- REWRITTEN QUERY: {query!r} ---")

        docs = self._retrieve_docs(query, debug=debug)

        using_web = False
        if not docs:
            print("\n  Local knowledge base has no relevant chunks.")
            web_docs = self._tavily_search(query)

            if web_docs:
                docs = web_docs
                using_web = True
                print(f"  Using {len(docs)} web results as context.")
            else:
                print("\nNo relevant information found in local DB or web.\n")
                return None

        print("\n=========== RETRIEVED CHUNKS ===========\n")
        if using_web:
            print("  [Source: Tavily Web Search]\n")
        print(self._format_chunks(docs))

        answer = None
        eval_result = None

        for attempt in range(1, MAX_RETRIES + 1):
            if attempt > 1:
                retry_note = (
                    "Your previous answer may have included unsupported or overly broad information. "
                    "Regenerate the answer using ONLY the provided context. "
                    "Remove any claim that cannot be directly supported by the context."
                )
            else:
                retry_note = ""

            prompt = self._build_prompt(question, docs, retry_note=retry_note)
            answer = self._generate_answer(prompt)

            eval_result = self._check_faithfulness(question, answer, docs)
            print(f"\n--- FAITHFULNESS CHECK (attempt {attempt}) ---")
            print(f"  Faithful : {eval_result.get('faithful')}")
            print(f"  Score    : {eval_result.get('score')}/10")
            print(f"  Reason   : {eval_result.get('reason')}")

            if eval_result.get("faithful") and eval_result.get("score", 0) >= FAITHFULNESS_MIN_SCORE:
                break
            if attempt < MAX_RETRIES:
                print(f"\n  Answer failed faithfulness check. Retrying ({attempt}/{MAX_RETRIES})...")

        if not eval_result.get("faithful") or eval_result.get("score", 10) < FAITHFULNESS_MIN_SCORE:
            print("\n Warning: Answer may contain information outside the provided context.")

        print("\n=========== FINAL ANSWER ===========\n")
        print(answer)

        print("\n=========== EVALUATION ===========")
        print(f"Faithfulness Score : {eval_result.get('score')}/10")
        print(f"Faithful           : {eval_result.get('faithful')}")
        print(f"Reason             : {eval_result.get('reason')}")
        if using_web:
            print("Source Type        : Web (Tavily fallback)")

        print("\n=========== SOURCES USED ===========")
        sources = list(set([doc.metadata.get("title", "Unknown") for doc in docs]))
        for i, src in enumerate(sources):
            src_type = " [WEB]" if any(
                doc.metadata.get("type") == "web" and doc.metadata.get("title") == src
                for doc in docs
            ) else ""
            print(f"[{i+1}] {src}{src_type}")

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        self.chat_history = self.chat_history[-4:]

        return {
            "question": question,
            "rewritten_query": query,
            "answer": answer,
            "contexts": [doc.page_content for doc in docs],
            "sources": [doc.metadata.get("title", "Unknown") for doc in docs],
            "eval": eval_result,
            "used_web": using_web,
        }

    def _tavily_search(self, query: str) -> list:
        if self.tavily_client is None:
            return []

        try:
            print("\n  🌐 No local chunks found — trying Tavily web search...")
            results = self.tavily_client.search(
                query=query,
                max_results=5,
                search_depth="basic",
            )

            web_docs = []
            for result in results.get("results", []):
                content = result.get("content", "").strip()
                url = result.get("url", "")
                title = result.get("title", "Web result")

                if content:
                    web_docs.append(Document(
                        page_content=content,
                        metadata={
                            "title": title,
                            "source": url,
                            "type": "web",
                        },
                    ))

            print(f"  🌐 Tavily returned {len(web_docs)} web results.")
            return web_docs

        except Exception as e:
            print(f"\n  [Tavily search failed: {e}]")
            return []

    def start_chat(self):
        print("Self-Evaluating RAG Assistant ready. Type 'exit' to quit.")

        while True:
            question = input("\nYour question: ").strip()

            if question.lower() == "exit":
                break

            if question:
                self.ask_question(question, debug=False)

            time.sleep(0.2)


if __name__ == "__main__":
    bot = SelfRAG()
    bot.start_chat()

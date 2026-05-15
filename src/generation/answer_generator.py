from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MODEL_NAME, REWRITE_MODEL_NAME, MAX_CONTEXT_CHARS,FAITHFULNESS_MODEL_NAME
from src.generation.prompt_templates import ANSWER_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE


def setup_models():
    rewrite = ChatGroq(model=REWRITE_MODEL_NAME, temperature=0.0)
    answer = ChatGroq(model=MODEL_NAME, temperature=0.0)
    faithfulness = ChatGroq(model=FAITHFULNESS_MODEL_NAME, temperature=0)
    return rewrite, answer, faithfulness


def build_prompt(question: str, docs: list, retry_note: str = "") -> str:
    context = ""
    used_sources = []

    for doc in docs:
        title = doc.metadata.get("title", "Unknown")
        chunk = f"{doc.page_content.strip()}\n\n"
        if len(context) + len(chunk) > MAX_CONTEXT_CHARS:
            break
        context += chunk
        if title not in used_sources:
            used_sources.append(title)

    return RAG_PROMPT_TEMPLATE.format(
        context=context.strip(),
        question=question,
        retry_note=retry_note,
    )


def generate_answer(answer_model, prompt: str, chat_history: list, safe_invoke) -> str:
    messages = [
        SystemMessage(content=ANSWER_SYSTEM_PROMPT),
        *chat_history[-4:],
        HumanMessage(content=prompt),
    ]

    response = safe_invoke(answer_model, messages)
    return response.content.strip() if response else "Error generating answer."

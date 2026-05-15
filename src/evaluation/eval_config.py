# eval_config.py
# DeepEval evaluation configuration
# Judge model: Local Ollama Mistral (runs on your RTX 4050)
import os
# os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "600"

from deepeval.models import OllamaModel


def get_judge(model_name: str = "mistral:latest") -> OllamaModel:
    return OllamaModel(model=model_name)


METRIC_THRESHOLDS = {
    "answer_relevancy": 0.6,
    "faithfulness": 0.6,
    "contextual_precision": 0.6,
    "contextual_recall": 0.5,
}


EVAL_SETTINGS = {
    "results_dir": "src/evaluation/results",
    "output_file": "src/evaluation/results/self_eval/self_eval_final_results.json",
    "questions_file": "src/evaluation/benchmark_questions.json",
    "skip_on_none": True,
    "include_categories": [
        "rag_concepts",
        "retrieval",
        "embeddings",
        "langchain_docs",
        "chroma",
        "transformer_pdf",
        "multi_hop",
        "edge_case",
    ],
}

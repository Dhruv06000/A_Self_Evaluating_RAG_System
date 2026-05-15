# run_baseline_evaluation.py
import os
# os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "60"

import sys
import json
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from deepeval.test_case import LLMTestCase

from src.evaluation.eval_config import get_judge, METRIC_THRESHOLDS, EVAL_SETTINGS
from src.evaluation.metrics import build_metrics
from src.retrieval.retriever import BasicRAG


def load_questions(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_single_question(rag: BasicRAG, q_data: dict) -> dict | None:
    question = q_data["question"]

    print(f"\n{'=' * 60}")
    print(f"[Q{q_data['id']}] {question}")
    print(f"Category: {q_data['category']}")
    print(f"{'=' * 60}")

    backoff = [5, 15, 30, 60]

    for attempt in range(len(backoff)):
        try:
            start_time = time.perf_counter()
            result = rag.ask_question(question)
            latency = time.perf_counter() - start_time

            if result is not None:
                result["latency_seconds"] = round(latency, 4)

            if result is None:
                print("  ⚠ Basic RAG returned None — no context found.")

            return result

        except Exception as e:
            err = str(e)

            if "429" in err or "rate_limit" in err.lower():
                wait = backoff[attempt]
                print(f"  ⚠ Groq 429 — waiting {wait}s attempt {attempt + 1}")
                time.sleep(wait)
            else:
                print(f"  ✗ Error: {err[:120]}")
                return None

    print("  ✗ All retries exhausted.")
    return None


def score_test_case(test_case: LLMTestCase, metrics: list) -> dict:
    scores = {}

    for metric in metrics:
        name = metric.__class__.__name__
        print(f"    → {name}...", end="", flush=True)

        try:
            metric.measure(test_case)
            score = round(metric.score, 4)
            passed = metric.is_successful()
            reason = getattr(metric, "reason", "") or ""

            print(f" {score:.4f} {'✓' if passed else '✗'}")

            if reason:
                print(f"       {reason[:110]}")

            scores[name] = {
                "score": score,
                "passed": passed,
                "reason": reason,
            }

        except Exception as e:
            print(f" ✗ failed: {str(e)[:80]}")
            scores[name] = {
                "score": -1,
                "passed": False,
                "reason": str(e)[:200],
            }

        time.sleep(0.2)

    return scores


def print_summary(all_results: list) -> dict:
    metric_names = [
        "AnswerRelevancyMetric",
        "FaithfulnessMetric",
        "ContextualPrecisionMetric",
        "ContextualRecallMetric",
    ]

    category_scores = {}

    for r in all_results:
        cat = r["category"]

        if cat not in category_scores:
            category_scores[cat] = {m: [] for m in metric_names}

        for m in metric_names:
            s = r["scores"].get(m, {}).get("score", -1)
            if s >= 0:
                category_scores[cat][m].append(s)

    print("\n\n" + "=" * 70)
    print("  EVALUATION SUMMARY — BASIC RAG BASELINE")
    print("=" * 70)

    overall = {m: [] for m in metric_names}

    for cat, mscores in sorted(category_scores.items()):
        print(f"\n  Category: {cat}")

        for m in metric_names:
            vals = mscores[m]

            if vals:
                avg = sum(vals) / len(vals)
                overall[m].extend(vals)
                print(f"    {m:<35} avg={avg:.4f}  n={len(vals)}")

    print("\n" + "-" * 70)
    print("  OVERALL AVERAGES")
    print("-" * 70)

    summary = {}

    for m in metric_names:
        vals = overall[m]

        if vals:
            avg = sum(vals) / len(vals)
            passed = sum(1 for v in vals if v >= 0.5)

            summary[m] = {
                "avg": round(avg, 4),
                "n": len(vals),
                "passed": passed,
            }

            print(f"  {m:<35} avg={avg:.4f}  passed={passed}/{len(vals)}")

    latencies = [
        r.get("latency_seconds")
        for r in all_results
        if r.get("latency_seconds") is not None
    ]

    if latencies:
        avg_latency = sum(latencies) / len(latencies)

        summary["latency"] = {
            "avg_seconds": round(avg_latency, 4),
            "min_seconds": round(min(latencies), 4),
            "max_seconds": round(max(latencies), 4),
            "n": len(latencies),
        }

        print("\n" + "-" * 70)
        print("  LATENCY")
        print("-" * 70)
        print(f"  Avg latency : {avg_latency:.4f}s")
        print(f"  Min latency : {min(latencies):.4f}s")
        print(f"  Max latency : {max(latencies):.4f}s")

    print("=" * 70)
    return summary


def save_results(all_results: list, summary: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "run_timestamp": datetime.now().isoformat(),
        "system_version": "basic_rag_baseline_mistral_judge",
        "total_questions": len(all_results),
        "summary": summary,
        "results": all_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Results saved to: {output_path}")


def main():
    print("\n" + "=" * 70)
    print("  Basic RAG Baseline — DeepEval Evaluation")
    print("  Judge: mistral:latest via Ollama")
    print("=" * 70)

    settings = EVAL_SETTINGS
    questions = load_questions(settings["questions_file"])

    print(f"\n  Loaded {len(questions)} questions from dataset.")
    print("  Using Basic RAG baseline.")
    print("  Initializing Basic RAG system...")
    rag = BasicRAG()

    print("  Initializing judge model mistral:latest...")
    judge = get_judge("mistral:latest")
    metrics = build_metrics(judge, METRIC_THRESHOLDS)

    print(f"  Metrics   : {[m.__class__.__name__ for m in metrics]}")
    print(f"  Thresholds: {METRIC_THRESHOLDS}")
    print(f"  Categories: {settings['include_categories']}")
    print("\n  Starting baseline evaluation...\n")

    all_results = []
    skipped = 0

    for q_data in questions: 
        if (
            settings["include_categories"]
            and q_data["category"] not in settings["include_categories"]
        ):
            continue

        rag_result = run_single_question(rag, q_data)

        if rag_result is None:
            skipped += 1
            if settings["skip_on_none"]:
                continue

        test_case = LLMTestCase(
            input=q_data["question"],
            actual_output=rag_result["answer"],
            expected_output=q_data["ground_truth"],
            retrieval_context=rag_result["contexts"],
        )

        print("\n  Scoring...")
        scores = score_test_case(test_case, metrics)

        all_results.append(
            {
                "id": q_data["id"],
                "category": q_data["category"],
                "question": q_data["question"],
                "ground_truth": q_data["ground_truth"],
                "answer": rag_result["answer"],
                "rewritten_query": rag_result.get("rewritten_query", q_data["question"]),
                "contexts": rag_result["contexts"],
                "sources": rag_result["sources"],
                "rag_eval": rag_result.get("eval"),
                "latency_seconds": rag_result.get("latency_seconds"),
                "scores": scores,
            }
        )

        time.sleep(0.5)

    print(f"\n  Done: {len(all_results)} evaluated, {skipped} skipped.")

    summary = print_summary(all_results)

    output_path = "src/evaluation/results/baseline/basic_rag_results.json"
    save_results(all_results, summary, output_path)


if __name__ == "__main__":
    main()
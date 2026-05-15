from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)


def build_metrics(judge, thresholds: dict) -> list:
    return [
        AnswerRelevancyMetric(
            threshold=thresholds["answer_relevancy"],
            model=judge,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=thresholds["faithfulness"],
            model=judge,
            include_reason=True,
        ),
        ContextualPrecisionMetric(
            threshold=thresholds["contextual_precision"],
            model=judge,
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=thresholds["contextual_recall"],
            model=judge,
            include_reason=True,
        ),
    ]
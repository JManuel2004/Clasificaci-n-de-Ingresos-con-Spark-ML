"""Holdout metrics for the positive income class."""

from __future__ import annotations

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _curve_metric(predictions: DataFrame, metric_name: str) -> float | None:
    classes = predictions.select("label_index").distinct().count()
    if classes < 2:
        return None
    evaluator = BinaryClassificationEvaluator(
        labelCol="label_index",
        rawPredictionCol="rawPrediction",
        metricName=metric_name,
    )
    value = float(evaluator.evaluate(predictions))
    if value != value:
        return None
    return value


def classification_metrics(predictions: DataFrame) -> dict:
    """Precision, recall, and F1 are for the >50K class (label_index 1)."""
    counts = {
        (float(row["label_index"]), float(row["prediction"])): int(row["count"])
        for row in predictions.groupBy("label_index", "prediction").count().collect()
    }
    true_positive = counts.get((1.0, 1.0), 0)
    true_negative = counts.get((0.0, 0.0), 0)
    false_positive = counts.get((0.0, 1.0), 0)
    false_negative = counts.get((1.0, 0.0), 0)
    total = true_positive + true_negative + false_positive + false_negative
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    return {
        "auc_roc": _curve_metric(predictions, "areaUnderROC"),
        "auc_pr": _curve_metric(predictions, "areaUnderPR"),
        "accuracy": _safe_divide(true_positive + true_negative, total),
        "precision_gt_50k": precision,
        "recall_gt_50k": recall,
        "f1_gt_50k": _safe_divide(2 * precision * recall, precision + recall),
        "support": total,
        "confusion_matrix": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
    }

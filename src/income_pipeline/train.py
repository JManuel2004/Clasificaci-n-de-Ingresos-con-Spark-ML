"""Train the income pipeline on a holdout split and persist it."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from pyspark.sql import functions as F

from income_pipeline.constants import EXCLUDED_FEATURES, EXCLUSION_REASON
from income_pipeline.errors import DataQualityError
from income_pipeline.features import build_pipeline
from income_pipeline.holdout import attach_class_weights, train_test_split
from income_pipeline.io_utils import json_ready, write_json
from income_pipeline.labels import with_label_index
from income_pipeline.metrics import classification_metrics
from income_pipeline.quality import clean_frame
from income_pipeline.schema import read_income_csv
from income_pipeline.session import build_session, configure_logging

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/raw/adult_income_sample.csv")
DEFAULT_MODEL_DIR = Path("models/income_lr")
DEFAULT_METRICS = Path("artifacts/metrics.json")
DEFAULT_QUALITY = Path("artifacts/quality_report.json")


@dataclass(frozen=True)
class TrainSettings:
    input_path: Path
    model_dir: Path
    metrics_path: Path
    quality_report_path: Path
    test_fraction: float = 0.2
    seed: int = 42
    max_iter: int = 100
    reg_param: float = 0.01
    elastic_net: float = 0.0
    max_invalid_fraction: float = 0.05


def _positive_rate(frame) -> float:
    rate = frame.agg(F.avg("label_index").alias("rate")).first()["rate"]
    if rate is None:
        return 0.0
    return float(rate)


def _class_weight_report(train) -> dict:
    rows = (
        train.groupBy("label_index")
        .agg(F.avg("class_weight").alias("weight"), F.count("*").alias("rows"))
        .collect()
    )
    return {
        str(int(round(float(row["label_index"])))): {
            "rows": int(row["rows"]),
            "weight": float(row["weight"]),
        }
        for row in rows
    }


def train_dataframe(spark, frame, settings: TrainSettings) -> dict:
    """Fit on a cleaned, labeled frame and return the metrics payload."""
    labeled = with_label_index(frame).filter(F.col("label_index").isNotNull())
    train, test = train_test_split(labeled, settings.test_fraction, settings.seed)
    train = attach_class_weights(train).cache()
    train_rows = train.count()
    test_rows = test.count()
    if train_rows == 0 or test_rows == 0:
        raise ValueError("train and test folds must both be non-empty")

    pipeline = build_pipeline(
        max_iter=settings.max_iter,
        reg_param=settings.reg_param,
        elastic_net=settings.elastic_net,
        seed=settings.seed,
    )
    model = pipeline.fit(train)
    settings.model_dir.parent.mkdir(parents=True, exist_ok=True)
    model.write().overwrite().save(str(settings.model_dir))

    iterations = None
    summary = getattr(model.stages[-1], "summary", None)
    if summary is not None:
        iterations = int(summary.totalIterations)

    payload = {
        "algorithm": "logistic_regression",
        "spark_version": spark.version,
        "seed": settings.seed,
        "test_fraction": settings.test_fraction,
        "hyperparameters": {
            "max_iter": settings.max_iter,
            "reg_param": settings.reg_param,
            "elastic_net": settings.elastic_net,
            "standardization": False,
            "threshold": 0.5,
        },
        "rows": {"train": int(train_rows), "test": int(test_rows)},
        "positive_rate": {
            "train": _positive_rate(train),
            "test": _positive_rate(test),
        },
        "class_weights": _class_weight_report(train),
        "solver_iterations": iterations,
        "features": {
            "numeric": ["age", "hours_per_week", "education_ordinal", "works_overtime"],
            "categorical": ["sex", "workclass"],
            "excluded": list(EXCLUDED_FEATURES),
            "exclusion_reason": EXCLUSION_REASON,
            "label_encoding": {">50K": 1, "<=50K": 0},
        },
        "metrics": {
            "train": classification_metrics(model.transform(train)),
            "test": classification_metrics(model.transform(test)),
        },
        "data": {
            "source": "synthetic_adult_style",
            "warning": (
                "Holdout metrics measure how well the model recovers the "
                "synthetic labeling rule. They are not an estimate of "
                "performance on real census data."
            ),
        },
    }
    return json_ready(payload)


def train_from_csv(spark, settings: TrainSettings) -> dict:
    raw = read_income_csv(spark, settings.input_path, include_label=True)
    try:
        cleaned = clean_frame(
            raw,
            supervised=True,
            max_invalid_fraction=settings.max_invalid_fraction,
        )
    except DataQualityError as exc:
        write_json(settings.quality_report_path, json_ready(exc.report))
        raise
    write_json(settings.quality_report_path, json_ready(cleaned.report))
    return train_dataframe(spark, cleaned.frame, settings)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the income classification pipeline.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--reg-param", type=float, default=0.01)
    parser.add_argument("--max-invalid-fraction", type=float, default=0.05)
    return parser.parse_args(argv)


def settings_from_args(args: argparse.Namespace) -> TrainSettings:
    return TrainSettings(
        input_path=args.input,
        model_dir=args.model_dir,
        metrics_path=args.metrics,
        quality_report_path=args.quality_report,
        test_fraction=args.test_fraction,
        seed=args.seed,
        max_iter=args.max_iter,
        reg_param=args.reg_param,
        max_invalid_fraction=args.max_invalid_fraction,
    )


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    settings = settings_from_args(parse_args(argv))
    spark = build_session("income-train")
    try:
        try:
            payload = train_from_csv(spark, settings)
        except DataQualityError as exc:
            logger.error("training stopped: %s", exc)
            raise SystemExit(1) from exc
        write_json(settings.metrics_path, payload)
    finally:
        spark.stop()
    test_metrics = payload["metrics"]["test"]
    logger.info(
        "saved model to %s | test auc_roc=%s f1_gt_50k=%s",
        settings.model_dir,
        test_metrics["auc_roc"],
        test_metrics["f1_gt_50k"],
    )


if __name__ == "__main__":
    main()

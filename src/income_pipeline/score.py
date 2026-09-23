"""Score a CSV of people with a saved pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from income_pipeline.constants import NEGATIVE_LABEL, POSITIVE_LABEL
from income_pipeline.errors import DataQualityError
from income_pipeline.io_utils import json_ready, write_csv, write_json
from income_pipeline.quality import clean_frame
from income_pipeline.schema import read_income_csv
from income_pipeline.session import build_session, configure_logging

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/samples/applicants.csv")
DEFAULT_MODEL_DIR = Path("models/income_lr")
DEFAULT_OUTPUT = Path("data/scored/applicants.csv")


def with_prediction_columns(scored: DataFrame) -> DataFrame:
    """Map raw Spark outputs back to the income labels.

    Label 1 is always >50K because training encodes that class explicitly.
    """
    probability = vector_to_array(F.col("probability")).getItem(1)
    return scored.withColumn("prob_gt_50k", F.round(probability, 6)).withColumn(
        "predicted_label",
        F.when(F.col("prediction") == F.lit(1.0), F.lit(POSITIVE_LABEL)).otherwise(
            F.lit(NEGATIVE_LABEL)
        ),
    )


def score_frame(model: PipelineModel, frame: DataFrame) -> DataFrame:
    scored = with_prediction_columns(model.transform(frame))
    columns = [
        column
        for column in (
            "age",
            "sex",
            "workclass",
            "fnlwgt",
            "education",
            "hours_per_week",
            "predicted_label",
            "prob_gt_50k",
        )
        if column in scored.columns
    ]
    return scored.select(*columns)


def score_csv(
    spark,
    *,
    input_path: Path,
    model_dir: Path,
    output_path: Path,
) -> dict:
    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    raw = read_income_csv(spark, input_path, include_label=False)
    cleaned = clean_frame(raw, supervised=False, max_invalid_fraction=1.0)
    model = PipelineModel.load(str(model_dir))
    predictions = score_frame(model, cleaned.frame)
    written = write_csv(predictions, output_path)
    rejected_rows = int(cleaned.report["invalid_rows"])
    if rejected_rows:
        reject_path = output_path.with_name(f"{output_path.stem}.rejected.csv")
        write_csv(cleaned.rejected, reject_path)
    else:
        reject_path = None
    report_path = output_path.with_name(f"{output_path.stem}.quality.json")
    report = {
        **cleaned.report,
        "scored_rows": written,
        "rejected_output": str(reject_path) if reject_path else None,
        "predictions_output": str(output_path),
    }
    write_json(report_path, json_ready(report))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score new income records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    configure_logging()
    args = parse_args(argv)
    spark = build_session("income-score")
    try:
        try:
            report = score_csv(
                spark,
                input_path=args.input,
                model_dir=args.model_dir,
                output_path=args.output,
            )
        except DataQualityError as exc:
            logger.error("scoring stopped: %s", exc)
            raise SystemExit(1) from exc
    finally:
        spark.stop()
    logger.info(
        "scored %s rows to %s",
        report["scored_rows"],
        report["predictions_output"],
    )


if __name__ == "__main__":
    main()

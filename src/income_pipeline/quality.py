"""Data-quality gate applied before training and before scoring."""

from __future__ import annotations

from typing import NamedTuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from income_pipeline.constants import (
    EDUCATION_ORDINAL,
    LABELS,
    MAX_AGE,
    MAX_HOURS,
    MIN_AGE,
    MIN_HOURS,
    SEX_VALUES,
    WORKCLASS_VALUES,
)
from income_pipeline.errors import DataQualityError


class CleanResult(NamedTuple):
    frame: DataFrame
    rejected: DataFrame
    report: dict


def _reject_reason(supervised: bool):
    reason = (
        F.when(
            F.col("age").isNull() | (F.col("age") < MIN_AGE) | (F.col("age") > MAX_AGE),
            F.lit("age"),
        )
        .when(
            F.col("hours_per_week").isNull()
            | (F.col("hours_per_week") < MIN_HOURS)
            | (F.col("hours_per_week") > MAX_HOURS),
            F.lit("hours_per_week"),
        )
        .when(
            F.col("sex").isNull() | ~F.col("sex").isin(list(SEX_VALUES)),
            F.lit("sex"),
        )
        .when(
            F.col("workclass").isNull() | ~F.col("workclass").isin(list(WORKCLASS_VALUES)),
            F.lit("workclass"),
        )
        .when(
            F.col("education").isNull() | ~F.col("education").isin(list(EDUCATION_ORDINAL)),
            F.lit("education"),
        )
        .when(F.col("fnlwgt").isNotNull() & (F.col("fnlwgt") <= 0), F.lit("fnlwgt"))
    )
    if supervised:
        reason = reason.when(
            F.col("label").isNull() | ~F.col("label").isin(list(LABELS)),
            F.lit("label"),
        )
    return reason


def _string_columns(supervised: bool) -> tuple[str, ...]:
    columns = ["sex", "workclass", "education"]
    if supervised:
        columns.append("label")
    return tuple(columns)


def clean_frame(
    frame: DataFrame,
    *,
    supervised: bool,
    max_invalid_fraction: float,
) -> CleanResult:
    """Trim text, reject rows outside the contract, and drop duplicates.

    Training fails closed when the invalid share is above the limit or when
    either income class is missing. Scoring keeps going and returns the
    rejected rows so the caller can persist them.
    """
    if not 0 <= max_invalid_fraction <= 1:
        raise ValueError("max_invalid_fraction must be between 0 and 1")

    prepared = frame
    for column in _string_columns(supervised):
        prepared = prepared.withColumn(column, F.trim(F.col(column)))

    prepared = prepared.withColumn("_reject_reason", _reject_reason(supervised)).cache()
    input_rows = prepared.count()
    rejected = prepared.filter(F.col("_reject_reason").isNotNull())
    invalid_rows = rejected.count()
    reason_counts = {
        row["_reject_reason"]: int(row["count"])
        for row in rejected.groupBy("_reject_reason").count().collect()
    }

    valid = prepared.filter(F.col("_reject_reason").isNull()).drop("_reject_reason")
    duplicate_keys = [
        "age",
        "sex",
        "workclass",
        "fnlwgt",
        "education",
        "hours_per_week",
    ]
    if supervised:
        duplicate_keys.append("label")
    before_dedup = valid.count()
    deduped = valid.dropDuplicates(duplicate_keys)
    kept_rows = deduped.count()
    duplicate_rows = before_dedup - kept_rows

    label_counts: dict[str, int] = {}
    if supervised and kept_rows:
        label_counts = {
            row["label"]: int(row["count"])
            for row in deduped.groupBy("label").count().collect()
        }

    invalid_fraction = (invalid_rows / input_rows) if input_rows else 1.0
    report = {
        "input_rows": int(input_rows),
        "invalid_rows": int(invalid_rows),
        "invalid_fraction": invalid_fraction,
        "duplicate_rows": int(duplicate_rows),
        "kept_rows": int(kept_rows),
        "max_invalid_fraction": max_invalid_fraction,
        "rejection_reasons": reason_counts,
        "label_counts": label_counts,
        "errors": [],
    }

    errors: list[str] = []
    if input_rows == 0:
        errors.append("input is empty")
    if supervised and input_rows and invalid_fraction > max_invalid_fraction:
        errors.append(
            "invalid fraction "
            f"{invalid_fraction:.2%} exceeds limit {max_invalid_fraction:.2%}"
        )
    if kept_rows == 0:
        errors.append("no rows left after cleaning")
    if supervised and kept_rows and set(label_counts) != set(LABELS):
        found = sorted(label_counts)
        errors.append(f"training data must include both labels, found {found}")
    report["errors"] = errors

    rejected_out = rejected.withColumnRenamed("_reject_reason", "reject_reason")
    if errors and supervised:
        prepared.unpersist()
        raise DataQualityError(report)

    if errors and not supervised and (input_rows == 0 or kept_rows == 0):
        prepared.unpersist()
        raise DataQualityError(report)

    prepared.unpersist()
    return CleanResult(frame=deduped, rejected=rejected_out, report=report)

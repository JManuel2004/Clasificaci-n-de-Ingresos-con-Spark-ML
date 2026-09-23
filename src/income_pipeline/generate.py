"""Reproducible Adult-style sample.

The labels are drawn from a documented rule, not from the UCI Adult census
file. The pipeline can be audited end to end without an external download.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from income_pipeline.constants import (
    EDUCATION_ORDINAL,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_COLUMNS,
    SEX_VALUES,
    WORKCLASS_VALUES,
)

DEFAULT_ROWS = 20_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("data/raw/adult_income_sample.csv")


def positive_probability(
    age: int,
    education: str,
    hours_per_week: int,
    workclass: str,
) -> float:
    """Probability of the >50K label under the synthetic rule."""
    probability = 0.12
    if education in {"Bachelors", "Masters", "Doctorate"}:
        probability += 0.28
    if education in {"Masters", "Doctorate"}:
        probability += 0.12
    if 30 <= age <= 55:
        probability += 0.16
    if hours_per_week >= 40:
        probability += 0.12
    if workclass == "Private":
        probability += 0.05
    return min(probability, 0.95)


def build_records(n_rows: int, seed: int) -> list[dict]:
    if n_rows < 1:
        raise ValueError("n_rows must be at least 1")
    rng = random.Random(seed)
    education_levels = list(EDUCATION_ORDINAL)
    records: list[dict] = []
    for _ in range(n_rows):
        age = rng.randint(18, 74)
        sex = rng.choice(SEX_VALUES)
        workclass = rng.choice(WORKCLASS_VALUES)
        education = rng.choice(education_levels)
        hours_per_week = rng.randint(1, 80)
        probability = positive_probability(age, education, hours_per_week, workclass)
        label = POSITIVE_LABEL if rng.random() < probability else NEGATIVE_LABEL
        records.append(
            {
                "age": age,
                "sex": sex,
                "workclass": workclass,
                "fnlwgt": rng.randint(12_000, 1_499_999),
                "education": education,
                "hours_per_week": hours_per_week,
                "label": label,
            }
        )
    return records


def write_records(records: list[dict], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RAW_COLUMNS))
        writer.writeheader()
        writer.writerows(records)
    return destination


def records_to_frame(spark, records: list[dict]):
    from income_pipeline.schema import income_schema

    return spark.createDataFrame(records, schema=income_schema(include_label=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a reproducible synthetic income CSV."
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    records = build_records(args.rows, args.seed)
    destination = write_records(records, args.output)
    print(f"wrote {len(records)} rows to {destination}")


if __name__ == "__main__":
    main()

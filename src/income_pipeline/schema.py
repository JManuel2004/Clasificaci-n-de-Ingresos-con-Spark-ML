"""Explicit CSV schema. Types are not inferred from a sample of rows."""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import IntegerType, StringType, StructField, StructType


def income_schema(*, include_label: bool) -> StructType:
    fields = [
        StructField("age", IntegerType(), True),
        StructField("sex", StringType(), True),
        StructField("workclass", StringType(), True),
        StructField("fnlwgt", IntegerType(), True),
        StructField("education", StringType(), True),
        StructField("hours_per_week", IntegerType(), True),
    ]
    if include_label:
        fields.append(StructField("label", StringType(), True))
    return StructType(fields)


def read_income_csv(
    spark: SparkSession,
    path: Path | str,
    *,
    include_label: bool,
) -> DataFrame:
    """Read a headered CSV with the income schema.

    A value that cannot be cast (for example a non-numeric age) becomes null.
    Quality checks decide whether that row is rejected.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"input file not found: {source}")
    return (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .option("nullValue", "")
        .option("encoding", "UTF-8")
        .schema(income_schema(include_label=include_label))
        .csv(str(source))
    )

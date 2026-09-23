"""Class-aware holdout split.

`randomSplit` does not preserve class shares. Drawing one uniform number per
row and cutting at `1 - test_fraction` keeps each class near the requested
ratio without fitting an encoder on the test fold.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def train_test_split(
    frame: DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[DataFrame, DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")

    keyed = frame.withColumn("_split_rand", F.rand(seed)).cache()
    keyed.count()
    threshold = 1.0 - test_fraction
    train = keyed.filter(F.col("_split_rand") < F.lit(threshold)).drop("_split_rand")
    test = keyed.filter(F.col("_split_rand") >= F.lit(threshold)).drop("_split_rand")
    train = train.cache()
    test = test.cache()
    train.count()
    test.count()
    keyed.unpersist()
    return train, test


def attach_class_weights(train: DataFrame) -> DataFrame:
    """Inverse-frequency weights computed on the training fold only."""
    counts = {
        int(round(float(row["label_index"]))): int(row["count"])
        for row in train.groupBy("label_index").count().collect()
    }
    if counts.keys() != {0, 1}:
        raise ValueError("training fold must contain both classes to assign weights")
    total = sum(counts.values())
    negative_weight = total / (2 * counts[0])
    positive_weight = total / (2 * counts[1])
    weight = (
        F.when(F.col("label_index") == F.lit(1.0), F.lit(float(positive_weight)))
        .otherwise(F.lit(float(negative_weight)))
    )
    return train.withColumn("class_weight", weight)

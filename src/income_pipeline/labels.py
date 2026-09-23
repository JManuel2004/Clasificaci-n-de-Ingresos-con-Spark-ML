"""Fixed label encoding.

StringIndexer assigns 0 to the most frequent class. That makes
`prediction == 1` mean a different income class whenever the class balance
flips. The positive class is therefore mapped explicitly.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from income_pipeline.constants import NEGATIVE_LABEL, POSITIVE_LABEL


def with_label_index(frame: DataFrame) -> DataFrame:
    return frame.withColumn(
        "label_index",
        F.when(F.col("label") == POSITIVE_LABEL, F.lit(1.0))
        .when(F.col("label") == NEGATIVE_LABEL, F.lit(0.0))
        .otherwise(F.lit(None))
        .cast("double"),
    )

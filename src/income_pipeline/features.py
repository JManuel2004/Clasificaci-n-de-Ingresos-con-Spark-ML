"""Spark ML pipeline shared by training and scoring.

Numeric columns are scaled on their own. LogisticRegression standardization
is left off so one-hot columns are not rescaled together with them.
"""

from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import (
    Imputer,
    OneHotEncoder,
    SQLTransformer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)

from income_pipeline.constants import (
    CATEGORICAL_COLUMNS,
    EDUCATION_ORDINAL,
    NUMERIC_MODEL_COLUMNS,
)


def education_ordinal_statement() -> str:
    branches = " ".join(
        f"WHEN education = '{level}' THEN {rank}"
        for level, rank in EDUCATION_ORDINAL.items()
    )
    return (
        "SELECT *, CASE "
        f"{branches} "
        "ELSE NULL END AS education_ordinal "
        "FROM __THIS__"
    )


def overtime_statement() -> str:
    return (
        "SELECT *, CASE WHEN hours_per_week >= 40 THEN 1 ELSE 0 END "
        "AS works_overtime FROM __THIS__"
    )


def build_pipeline(
    *,
    max_iter: int = 100,
    reg_param: float = 0.01,
    elastic_net: float = 0.0,
    seed: int = 42,
) -> Pipeline:
    # The holdout split consumes `seed`. Binomial logistic regression in
    # Spark 4 uses L-BFGS and does not accept a seed of its own.
    del seed
    indexers = [
        StringIndexer(
            inputCol=column,
            outputCol=f"{column}_idx",
            handleInvalid="keep",
            stringOrderType="alphabetAsc",
        )
        for column in CATEGORICAL_COLUMNS
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{column}_idx" for column in CATEGORICAL_COLUMNS],
        outputCols=[f"{column}_oh" for column in CATEGORICAL_COLUMNS],
        dropLast=True,
        handleInvalid="keep",
    )
    numeric = VectorAssembler(
        inputCols=list(NUMERIC_MODEL_COLUMNS),
        outputCol="numeric_features",
        handleInvalid="error",
    )
    scaler = StandardScaler(
        inputCol="numeric_features",
        outputCol="numeric_scaled",
        withMean=True,
        withStd=True,
    )
    features = VectorAssembler(
        inputCols=["numeric_scaled", "sex_oh", "workclass_oh"],
        outputCol="features",
    )
    classifier = LogisticRegression(
        featuresCol="features",
        labelCol="label_index",
        weightCol="class_weight",
        predictionCol="prediction",
        probabilityCol="probability",
        rawPredictionCol="rawPrediction",
        maxIter=max_iter,
        regParam=reg_param,
        elasticNetParam=elastic_net,
        standardization=False,
        family="binomial",
    )
    stages = [
        SQLTransformer(statement=education_ordinal_statement()),
        Imputer(
            inputCols=["age", "hours_per_week", "education_ordinal"],
            outputCols=["age", "hours_per_week", "education_ordinal"],
            strategy="median",
        ),
        SQLTransformer(statement=overtime_statement()),
        *indexers,
        encoder,
        numeric,
        scaler,
        features,
        classifier,
    ]
    return Pipeline(stages=stages)

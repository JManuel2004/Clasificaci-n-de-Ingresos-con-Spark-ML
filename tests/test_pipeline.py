from pathlib import Path

import pytest
from pyspark.ml.feature import VectorAssembler

from income_pipeline.features import build_pipeline
from income_pipeline.generate import build_records, records_to_frame
from income_pipeline.quality import clean_frame
from income_pipeline.score import score_frame
from income_pipeline.train import TrainSettings, parse_args, train_dataframe

from tests.helpers import person


def test_feature_vector_does_not_include_the_sampling_weight():
    assemblers = [
        stage
        for stage in build_pipeline().getStages()
        if isinstance(stage, VectorAssembler)
    ]
    used = {column for stage in assemblers for column in stage.getInputCols()}
    assert "fnlwgt" not in used
    assert {"age", "hours_per_week", "education_ordinal", "works_overtime"} <= used


def test_train_parser_defaults():
    args = parse_args([])
    assert args.input == Path("data/raw/adult_income_sample.csv")
    assert args.test_fraction == 0.2
    assert args.seed == 42


@pytest.fixture(scope="module")
def trained(spark, tmp_path_factory):
    records = build_records(1200, seed=7)
    cleaned = clean_frame(
        records_to_frame(spark, records),
        supervised=True,
        max_invalid_fraction=0.0,
    )
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    settings = TrainSettings(
        input_path=Path("unused.csv"),
        model_dir=tmp_path_factory.mktemp("model"),
        metrics_path=artifact_dir / "metrics.json",
        quality_report_path=artifact_dir / "quality.json",
        seed=7,
        max_iter=50,
    )
    payload = train_dataframe(spark, cleaned.frame, settings)
    return {"spark": spark, "payload": payload, "model_dir": settings.model_dir}


def test_holdout_metrics_recover_the_synthetic_rule(trained):
    test_metrics = trained["payload"]["metrics"]["test"]
    assert test_metrics["auc_roc"] > 0.7
    assert test_metrics["f1_gt_50k"] > 0.5
    assert trained["payload"]["features"]["label_encoding"] == {">50K": 1, "<=50K": 0}
    assert trained["payload"]["features"]["excluded"] == ["fnlwgt"]


def test_saved_model_scores_without_labels_or_weights(trained):
    from pyspark.ml import PipelineModel

    spark = trained["spark"]
    model = PipelineModel.load(str(trained["model_dir"]))
    doctorate = person(age=55, education="Doctorate", hours_per_week=60, workclass="Private")
    preschool = person(age=19, education="Preschool", hours_per_week=12, workclass="Gov")
    unseen = person(age=36, education="Bachelors", hours_per_week=40, workclass="Never-worked")
    rows = []
    for row in (doctorate, preschool, unseen):
        row.pop("label")
        rows.append(row)
    scored = score_frame(model, spark.createDataFrame(rows)).collect()
    by_education = {row.education: row for row in scored}
    assert float(by_education["Doctorate"].prob_gt_50k) > float(
        by_education["Preschool"].prob_gt_50k
    )
    assert by_education["Never-worked"].predicted_label in {">50K", "<=50K"}


def test_engineered_columns_follow_the_source_row(trained):
    from pyspark.ml import PipelineModel

    spark = trained["spark"]
    model = PipelineModel.load(str(trained["model_dir"]))
    row = person(age=55, education="Doctorate", hours_per_week=20, label=">50K")
    transformed = model.transform(spark.createDataFrame([row])).first()
    assert float(transformed.education_ordinal) == 11.0
    assert int(transformed.works_overtime) == 0

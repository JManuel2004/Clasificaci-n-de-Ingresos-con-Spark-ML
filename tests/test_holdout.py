import pytest
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from income_pipeline.generate import build_records, records_to_frame
from income_pipeline.holdout import attach_class_weights, train_test_split
from income_pipeline.labels import with_label_index


def test_positive_class_stays_one_when_it_is_the_majority(spark):
    rows = [{"label": ">50K"}] * 8 + [{"label": "<=50K"}] * 2
    encoded = with_label_index(spark.createDataFrame(rows))
    pairs = {
        (row.label, row.label_index)
        for row in encoded.select("label", "label_index").distinct().collect()
    }
    assert pairs == {(">50K", 1.0), ("<=50K", 0.0)}


def test_split_is_disjoint_and_keeps_both_classes(spark):
    labeled = with_label_index(records_to_frame(spark, build_records(800, seed=3)))
    keyed = labeled.withColumn(
        "row_id",
        F.row_number().over(Window.orderBy(F.monotonically_increasing_id())),
    ).cache()
    total = keyed.count()
    train, test = train_test_split(keyed, test_fraction=0.2, seed=11)
    assert train.join(test, "row_id", "inner").count() == 0
    assert train.count() + test.count() == total
    ratio = test.count() / total
    assert 0.12 < ratio < 0.28
    for fold in (train, test):
        present = {row.label for row in fold.select("label").distinct().collect()}
        assert present == {">50K", "<=50K"}
    keyed.unpersist()


def test_class_weights_favor_the_smaller_class(spark):
    labeled = with_label_index(records_to_frame(spark, build_records(600, seed=5)))
    train, _ = train_test_split(labeled, test_fraction=0.2, seed=5)
    weighted = attach_class_weights(train)
    rows = weighted.groupBy("label_index").agg(
        F.avg("class_weight").alias("weight"),
        F.count("*").alias("rows"),
    ).collect()
    by_label = {int(round(float(row.label_index))): row for row in rows}
    smaller = min(by_label, key=lambda label: by_label[label].rows)
    larger = max(by_label, key=lambda label: by_label[label].rows)
    assert by_label[smaller].weight > by_label[larger].weight


def test_split_rejects_an_invalid_fraction(spark):
    frame = spark.createDataFrame([{"label": ">50K", "label_index": 1.0}])
    with pytest.raises(ValueError):
        train_test_split(frame, test_fraction=0.0, seed=1)

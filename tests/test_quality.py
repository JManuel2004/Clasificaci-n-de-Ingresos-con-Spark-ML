import pytest

from income_pipeline.errors import DataQualityError
from income_pipeline.quality import clean_frame
from income_pipeline.schema import read_income_csv

from tests.helpers import person


def _frame(spark, rows):
    return spark.createDataFrame(rows)


def test_rejects_bad_values_and_keeps_a_null_sampling_weight(spark):
    rows = [
        person(age=33, fnlwgt=None, label="<=50K"),
        person(age=50, fnlwgt=-5, label=">50K", education="Masters"),
        person(age=51, fnlwgt=20, label=">50K", education="Doctorate", sex="Male"),
    ]
    result = clean_frame(_frame(spark, rows), supervised=True, max_invalid_fraction=0.5)
    assert result.report["invalid_rows"] == 1
    assert result.report["kept_rows"] == 2
    assert result.report["rejection_reasons"] == {"fnlwgt": 1}
    assert {row.fnlwgt for row in result.frame.select("fnlwgt").collect()} == {None, 20}


def test_drops_exact_duplicates(spark):
    repeated = person(age=30, label="<=50K")
    other = person(age=52, label=">50K", education="Masters", sex="Male")
    result = clean_frame(
        _frame(spark, [repeated, repeated, other]),
        supervised=True,
        max_invalid_fraction=0.0,
    )
    assert result.report["duplicate_rows"] == 1
    assert result.report["kept_rows"] == 2


def test_training_fails_when_too_many_rows_are_invalid(spark):
    rows = [
        person(age=10, label="<=50K"),
        person(age=11, label=">50K"),
        person(age=40, label=">50K", education="Masters"),
    ]
    with pytest.raises(DataQualityError) as caught:
        clean_frame(_frame(spark, rows), supervised=True, max_invalid_fraction=0.05)
    assert "invalid fraction" in str(caught.value)


def test_training_fails_when_one_class_is_missing(spark):
    rows = [person(age=30, label="<=50K"), person(age=31, label="<=50K", sex="Male")]
    with pytest.raises(DataQualityError) as caught:
        clean_frame(_frame(spark, rows), supervised=True, max_invalid_fraction=0.0)
    assert "both labels" in str(caught.value)


def test_missing_csv_names_the_file(spark, tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError, match="input file not found"):
        read_income_csv(spark, missing, include_label=True)


def test_csv_type_mismatch_is_rejected(spark, tmp_path):
    source = tmp_path / "people.csv"
    source.write_text(
        "age,sex,workclass,fnlwgt,education,hours_per_week,label\n"
        "abc,Female,Private,1000,Bachelors,40,<=50K\n"
        "30,Female,Private,1000,Bachelors,40,>50K\n"
        "31,Male,Gov,1000,HS-grad,20,<=50K\n",
        encoding="utf-8",
    )
    loaded = read_income_csv(spark, source, include_label=True)
    result = clean_frame(loaded, supervised=True, max_invalid_fraction=0.5)
    assert result.report["invalid_rows"] == 1
    assert result.report["kept_rows"] == 2
    assert result.report["rejection_reasons"]["age"] == 1

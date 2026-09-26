"""Spark session defaults for local runs and tests."""

from __future__ import annotations

import logging
import os
import sys

from income_pipeline.spark_runtime import assert_java_available, prepare_spark_builder


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_session(app_name: str, master: str | None = None):
    from pyspark.sql import SparkSession

    assert_java_available()
    # Windows venvs do not provide a `python3` executable. Spark workers
    # default to that name, so pin both sides to the interpreter that
    # launched the job. HADOOP_HOME has to be set before the JVM starts.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    chosen_master = master or os.environ.get("SPARK_MASTER", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(chosen_master)
        .config(
            "spark.sql.shuffle.partitions",
            os.environ.get("SPARK_SHUFFLE_PARTITIONS", "8"),
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )
    spark = prepare_spark_builder(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

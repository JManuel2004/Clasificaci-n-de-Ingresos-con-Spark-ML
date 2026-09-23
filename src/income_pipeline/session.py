"""Spark session defaults for local runs and tests."""

from __future__ import annotations

import logging
import os
import shutil
import sys


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def assert_java_available() -> None:
    if shutil.which("java") or os.environ.get("JAVA_HOME"):
        return
    raise RuntimeError(
        "Java 17 or newer is required to run PySpark. "
        "Install Eclipse Temurin 17 and set JAVA_HOME."
    )


def build_session(app_name: str, master: str | None = None):
    from pyspark.sql import SparkSession

    assert_java_available()
    # Windows venvs do not provide a `python3` executable. Spark workers
    # default to that name, so pin both sides to the interpreter that
    # launched the job.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    chosen_master = master or os.environ.get("SPARK_MASTER", "local[*]")
    spark = (
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
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark

"""Small writers that keep job outputs easy to open outside Spark."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pyspark.sql import DataFrame


def json_ready(value):
    """Round floats so metric files stay readable and stable."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path | str, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(frame: DataFrame, path: Path | str) -> int:
    """Write one headered CSV file.

    The score command is a local batch tool, so it consolidates rows on the
    driver. A lake job should write partitioned Parquet instead.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = frame.columns
    written = 0
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in frame.toLocalIterator():
            writer.writerow([row[column] for column in columns])
            written += 1
    return written

import os
import subprocess
from pathlib import Path

import pytest

from income_pipeline.spark_runtime import (
    PLAIN_FILESYSTEM,
    WINUTILS_SOURCE,
    java_major_from_text,
    prepare_spark_builder,
)


def test_java_version_text_reads_modern_and_legacy_majors():
    assert java_major_from_text('openjdk version "17.0.20" 2026-08-18') == 17
    assert java_major_from_text('java version "1.8.0_402"') == 8
    assert java_major_from_text("no version here") is None


def test_winutils_ls_line_matches_the_hadoop_tokenizer():
    assert 'Console.WriteLine("-rwxrwxrwx|1|user|group|0|0|stub");' in WINUTILS_SOURCE


def test_prepare_keeps_linux_builder_unchanged():
    if os.name == "nt":
        pytest.skip("the plain filesystem is only installed on Windows")

    class Builder:
        def __init__(self):
            self.configs = {}

        def config(self, key, value):
            self.configs[key] = value
            return self

    builder = Builder()
    assert prepare_spark_builder(builder) is builder
    assert builder.configs == {}


class _Builder:
    def __init__(self):
        self.configs = {}

    def config(self, key, value):
        self.configs[key] = value
        return self


@pytest.mark.skipif(os.name != "nt", reason="winutils is only built on Windows")
def test_windows_stub_answers_ls_and_chmod(tmp_path, monkeypatch):
    monkeypatch.delenv("HADOOP_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    builder = prepare_spark_builder(_Builder())
    home = Path(os.environ["HADOOP_HOME"])
    binary = home / "bin" / "winutils.exe"
    assert builder.configs["spark.hadoop.fs.file.impl"] == PLAIN_FILESYSTEM
    assert Path(builder.configs["spark.driver.extraClassPath"]).is_file()
    listed = subprocess.run(
        [str(binary), "ls", "-F", "anywhere"],
        capture_output=True,
        text=True,
        check=False,
    )
    chmod = subprocess.run(
        [str(binary), "chmod", "0644", "anywhere"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0
    assert listed.stdout.splitlines()[0].split("|")[:4] == [
        "-rwxrwxrwx",
        "1",
        "user",
        "group",
    ]
    assert chmod.returncode == 0
    assert chmod.stdout == ""
    assert binary.is_file()


def test_plain_filesystem_class_name_stays_in_sync_with_the_java_source():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "income_pipeline"
        / "hadoop"
        / "PlainLocalFileSystem.java"
    )
    text = source.read_text(encoding="utf-8")
    assert f"class {PLAIN_FILESYSTEM.rsplit('.', 1)[-1]}" in text
    assert "package income.pipeline.hadoop;" in text

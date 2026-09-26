"""Local Spark startup, including the Windows filesystem workaround.

PySpark writes models through Hadoop's local filesystem. On Windows that
filesystem runs ``winutils.exe`` for permissions and ``NativeIO.Windows``
while listing directories. A JDK does not include either binary. This module
builds a no-op winutils and a small filesystem that lists directories with
``java.io.File``, then points the session at both before the JVM starts.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

PLAIN_FILESYSTEM = "income.pipeline.hadoop.PlainLocalFileSystem"

# Hadoop on Windows tokenizes `winutils ls -F` on '|' and expects
# permission|links|owner|group|... The other commands only need exit code 0.
WINUTILS_SOURCE = """\
using System;
class Winutils {
    static int Main(string[] args) {
        if (args.Length > 0 && args[0] == "ls") {
            Console.WriteLine("-rwxrwxrwx|1|user|group|0|0|stub");
        }
        return 0;
    }
}
"""


def java_major_from_text(text: str) -> int | None:
    """Read a major version from `java -version` output."""
    match = re.search(r'version "(?:1\.)?(\d+)', text)
    if match is None:
        return None
    return int(match.group(1))


def java_binary() -> str | None:
    home = os.environ.get("JAVA_HOME")
    if home:
        name = "java.exe" if os.name == "nt" else "java"
        candidate = Path(home) / "bin" / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which("java")


def assert_java_available() -> None:
    binary = java_binary()
    if binary is None:
        raise RuntimeError(
            "Java 17 or newer is required to run PySpark. "
            "Install Eclipse Temurin 17 and set JAVA_HOME."
        )
    completed = subprocess.run(
        [binary, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    reported = f"{completed.stderr or ''}{completed.stdout or ''}"
    major = java_major_from_text(reported)
    if major is not None and major < 17:
        raise RuntimeError(
            f"Java {major} is too old for this pipeline. Install Java 17 or newer."
        )


def prepare_spark_builder(builder):
    """Attach Windows filesystem fixes. Other platforms are unchanged."""
    if os.name != "nt":
        return builder
    home = _ensure_winutils_home()
    os.environ["HADOOP_HOME"] = str(home)
    jar_path = _ensure_plain_filesystem_jar()
    logger.info("using Windows Hadoop home at %s", home)
    return (
        builder.config("spark.driver.extraClassPath", str(jar_path))
        .config("spark.executor.extraClassPath", str(jar_path))
        .config("spark.hadoop.fs.file.impl", PLAIN_FILESYSTEM)
    )


def _cache_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
    return Path(base) / "income-pipeline" / "hadoop"


def _source_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stamp_matches(destination: Path, source_hash: str) -> bool:
    stamp = destination.with_name(destination.name + ".sha256")
    if not destination.is_file() or not stamp.is_file():
        return False
    return stamp.read_text(encoding="utf-8").strip() == source_hash


def _write_stamp(destination: Path, source_hash: str) -> None:
    stamp = destination.with_name(destination.name + ".sha256")
    stamp.write_text(source_hash + "\n", encoding="utf-8")


def _winutils_is_present(home: str | None) -> bool:
    if not home:
        return False
    return (Path(home) / "bin" / "winutils.exe").is_file()


def _ensure_winutils_home() -> Path:
    current = os.environ.get("HADOOP_HOME")
    if _winutils_is_present(current):
        return Path(current)
    if current:
        logger.warning(
            "HADOOP_HOME=%s has no bin/winutils.exe; building a local stub",
            current,
        )
    home = _cache_dir()
    binary = home / "bin" / "winutils.exe"
    payload = WINUTILS_SOURCE.encode("utf-8")
    source_hash = _source_hash(payload)
    if not _stamp_matches(binary, source_hash):
        _compile_winutils(binary, payload)
        _write_stamp(binary, source_hash)
    return home


def _csc_path() -> Path | None:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    for relative in (
        Path("Microsoft.NET") / "Framework64" / "v4.0.30319" / "csc.exe",
        Path("Microsoft.NET") / "Framework" / "v4.0.30319" / "csc.exe",
    ):
        candidate = windir / relative
        if candidate.is_file():
            return candidate
    return None


def _compile_winutils(binary: Path, payload: bytes) -> None:
    compiler = _csc_path()
    if compiler is None:
        raise RuntimeError(
            "Spark on Windows needs bin\\winutils.exe under HADOOP_HOME. "
            "The .NET Framework compiler csc.exe was not found, so the local "
            "stub could not be built. Install .NET Framework 4.x, or set "
            "HADOOP_HOME to a directory that already contains bin\\winutils.exe."
        )
    binary.parent.mkdir(parents=True, exist_ok=True)
    source = binary.with_suffix(".cs")
    source.write_bytes(payload)
    completed = subprocess.run(
        [
            str(compiler),
            "/nologo",
            "/optimize+",
            "/target:winexe",
            f"/out:{binary}",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0 or not binary.is_file():
        detail = (completed.stdout or "") + (completed.stderr or "")
        raise RuntimeError(f"could not compile the Windows winutils stub:\n{detail}")


def _java_source() -> Path:
    source = Path(__file__).resolve().parent / "hadoop" / "PlainLocalFileSystem.java"
    if not source.is_file():
        raise FileNotFoundError(
            f"Windows filesystem helper source is missing: {source}"
        )
    return source


def _javac_path() -> Path | None:
    home = os.environ.get("JAVA_HOME")
    if home:
        candidate = Path(home) / "bin" / ("javac.exe" if os.name == "nt" else "javac")
        if candidate.is_file():
            return candidate
    found = shutil.which("javac")
    return Path(found) if found else None


def _jar_tool() -> Path | None:
    home = os.environ.get("JAVA_HOME")
    if home:
        candidate = Path(home) / "bin" / ("jar.exe" if os.name == "nt" else "jar")
        if candidate.is_file():
            return candidate
    found = shutil.which("jar")
    return Path(found) if found else None


def _hadoop_api_jar() -> Path:
    import pyspark

    jars = Path(pyspark.__file__).resolve().parent / "jars"
    matches = sorted(jars.glob("hadoop-client-api-*.jar"))
    if not matches:
        raise RuntimeError(f"hadoop-client-api jar not found under {jars}")
    return matches[-1]


def _ensure_plain_filesystem_jar() -> Path:
    source = _java_source()
    payload = source.read_bytes()
    source_hash = _source_hash(payload)
    jar_path = _cache_dir() / "plain-local-fs.jar"
    if _stamp_matches(jar_path, source_hash):
        return jar_path
    javac = _javac_path()
    jar_tool = _jar_tool()
    if javac is None or jar_tool is None:
        raise RuntimeError(
            "JAVA_HOME must point at a JDK. javac is required on Windows "
            "to build the local filesystem helper used when saving models."
        )
    classes = _cache_dir() / "plain-local-fs-classes"
    if classes.exists():
        shutil.rmtree(classes)
    classes.mkdir(parents=True, exist_ok=True)
    compiled = subprocess.run(
        [
            str(javac),
            "--release",
            "8",
            "-cp",
            str(_hadoop_api_jar()),
            "-d",
            str(classes),
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode != 0:
        detail = (compiled.stdout or "") + (compiled.stderr or "")
        raise RuntimeError(f"could not compile the Windows filesystem helper:\n{detail}")
    packaged = subprocess.run(
        [str(jar_tool), "cf", str(jar_path), "-C", str(classes), "."],
        capture_output=True,
        text=True,
        check=False,
    )
    if packaged.returncode != 0 or not jar_path.is_file():
        detail = (packaged.stdout or "") + (packaged.stderr or "")
        raise RuntimeError(f"could not package the Windows filesystem helper:\n{detail}")
    _write_stamp(jar_path, source_hash)
    return jar_path

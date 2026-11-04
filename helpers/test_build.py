"""Checks for the build system (SPEC section 1), criteria 1-4."""
from __future__ import annotations

import importlib
import py_compile
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from _testlib import REPO_ROOT, load_pipe_module

BUILD = REPO_ROOT / "helpers" / "build_gemini_pipe.py"
PYTHON = sys.executable


def _import_build_module():
    spec = importlib.util.spec_from_file_location(
        "build_gemini_pipe_under_test", BUILD
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_build_script(*args):
    return subprocess.run(
        [PYTHON, str(BUILD), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )


def test_build_produces_valid_artifact():
    # criterion 1
    result = _run_build_script()
    assert result.returncode == 0, result.stdout + result.stderr
    pipe_file = REPO_ROOT / "gemini_pipe.py"
    min_file = REPO_ROOT / "gemini_pipe.min.py"
    assert pipe_file.exists()
    assert min_file.exists()
    for path in (pipe_file, min_file):
        py_compile.compile(str(path), doraise=True)
    module = load_pipe_module()
    docstring = module.__doc__ or ""
    assert "version: 0.1.0" in docstring
    assert hasattr(module.Pipe, "pipes")


def test_check_passes_after_build_and_flags_stale_artifact(tmp_path):
    # criterion 2
    ok = _run_build_script("--check")
    assert ok.returncode == 0, ok.stdout + ok.stderr

    # A temp copy of the repo with the artifact altered must fail, naming it.
    for entry in ("src", "helpers", "gemini_pipe.py", "gemini_pipe.min.py"):
        source = REPO_ROOT / entry
        if source.is_dir():
            shutil.copytree(source, tmp_path / entry)
        else:
            shutil.copy2(source, tmp_path / entry)
    stale = tmp_path / "gemini_pipe.py"
    stale.write_text(stale.read_text(encoding="utf-8") + "# stale\n", encoding="utf-8")
    result = subprocess.run(
        [PYTHON, str(tmp_path / "helpers" / "build_gemini_pipe.py"), "--check"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    output = result.stdout + result.stderr
    assert "gemini_pipe.py" in output


def test_guard_functions_raise_build_error():
    # criterion 3
    build = _import_build_module()

    stream = "def f():\n    import x\n    return x\n"
    with pytest.raises(build.BuildError) as exc:
        build.guard_stream_source("fake/stream.py", stream)
    assert "fake/stream.py:" in str(exc.value)

    group_a = "class _A:\n    def foo(self):\n        return 1\n"
    group_b = "class _B:\n    def foo(self):\n        return 2\n"
    with pytest.raises(build.BuildError) as exc:
        build.guard_duplicate_methods({"a.py": group_a, "b.py": group_b})
    message = str(exc.value)
    assert "a.py" in message and "b.py" in message

    group = (
        "class _G:\n"
        "    def ok(self):\n"
        "        return 1\n"
        "\n"
        "def helper():\n"
        "    return 2\n"
    )
    with pytest.raises(build.BuildError) as exc:
        build.guard_method_group("g.py", group)
    assert "g.py:" in str(exc.value)


def test_stream_source_without_marker_pair_fails(monkeypatch):
    # criterion 4
    build = _import_build_module()
    original = build.STREAM_SOURCES
    monkeypatch.setattr(
        build,
        "STREAM_SOURCES",
        [("no.marker.section", "shared/models.py")],
    )
    with pytest.raises(build.BuildError, match="no.marker.section"):
        build.build_text()

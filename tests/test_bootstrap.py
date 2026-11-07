"""The entry shims re-exec into the repo venv when the deps are missing."""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_bootstrap", Path(__file__).resolve().parent.parent / "_bootstrap.py"
)
_bootstrap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bootstrap)


def _point_bootstrap_at(monkeypatch, root):
    monkeypatch.setattr(_bootstrap, "__file__", str(root / "_bootstrap.py"))


def test_noop_without_repo_venv(monkeypatch, tmp_path):
    _point_bootstrap_at(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == []


def test_noop_when_already_the_venv_interpreter(monkeypatch, tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    _point_bootstrap_at(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "executable", str(venv_python))
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == []


def test_reexecs_with_venv_python(monkeypatch, tmp_path):
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("")
    _point_bootstrap_at(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "executable", "/opt/homebrew/bin/python3")
    monkeypatch.setattr(sys, "argv", ["./generate.py", "--version"])
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == [(str(venv_python), [str(venv_python), "./generate.py", "--version"])]

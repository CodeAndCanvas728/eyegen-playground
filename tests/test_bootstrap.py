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


def _make_venv(root, symlink_to=None):
    """A repo venv whose bin/python is a real file, or a symlink like real venvs."""
    venv_python = root / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    if symlink_to is None:
        venv_python.write_text("")
    else:
        venv_python.symlink_to(symlink_to)
    return venv_python


def test_noop_without_repo_venv(monkeypatch, tmp_path):
    _point_bootstrap_at(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == []


def test_noop_when_already_in_the_venv(monkeypatch, tmp_path):
    venv_python = _make_venv(tmp_path)
    _point_bootstrap_at(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "executable", str(venv_python))
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == []


def test_reexecs_with_venv_python(monkeypatch, tmp_path):
    venv_python = _make_venv(tmp_path)
    _point_bootstrap_at(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "prefix", "/opt/homebrew")
    monkeypatch.setattr(sys, "executable", "/opt/homebrew/bin/python3")
    monkeypatch.setattr(sys, "argv", ["./generate.py", "--version"])
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == [(str(venv_python), [str(venv_python), "./generate.py", "--version"])]


def test_reexecs_even_when_venv_python_symlinks_to_the_running_interpreter(monkeypatch, tmp_path):
    """venv/bin/python is a symlink to the base interpreter; resolving it would
    collapse both sides to the same path and wrongly skip the re-exec."""
    base_python = tmp_path / "base" / "python3"
    base_python.parent.mkdir(parents=True)
    base_python.write_text("")
    venv_python = _make_venv(tmp_path, symlink_to=base_python)
    _point_bootstrap_at(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "base"))
    monkeypatch.setattr(sys, "executable", str(base_python))
    monkeypatch.setattr(sys, "argv", ["./generate.py", "--version"])
    calls = []
    monkeypatch.setattr(_bootstrap.os, "execv", lambda *a: calls.append(a))
    _bootstrap.reexec_into_repo_venv()
    assert calls == [(str(venv_python), [str(venv_python), "./generate.py", "--version"])]

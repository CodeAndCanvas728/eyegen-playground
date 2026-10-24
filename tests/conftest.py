"""Fixtures that keep the tests away from the developer's own files.

The tool keeps the user's models in files next to the configuration, and
the user's models are the developer's. A test that reads those files would
read the developer's on this machine and none in CI — the file is
gitignored, so a test that passes there fails here, and the other way
round. Every test therefore runs against files of its own.
"""

import os

import pytest

from eyegen import model_catalogue, ollama_registry

# The path guard trusts the temporary directories while the tests run. It is
# set here, before any test module is imported, so that a test file run on
# its own behaves like the whole suite.
os.environ.setdefault("EYEGEN_TEST_ROOTS", "1")


@pytest.fixture(autouse=True)
def _the_users_own_files(tmp_path, monkeypatch):
    """Point the tool's own model files at a directory of the test's own."""
    monkeypatch.setattr(model_catalogue, "MODELS_FILE", tmp_path / "models.json")
    monkeypatch.setattr(
        ollama_registry, "OLLAMA_REGISTRY_FILE", tmp_path / "ollamadiffuser-models.json"
    )
    monkeypatch.setenv("OLLAMADIFFUSER_MODEL_CONFIG", str(tmp_path / "ollamadiffuser-models.json"))

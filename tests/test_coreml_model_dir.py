"""Tests for where a CoreML bundle is kept.

A bundle is named after the repository it came from, and the name is the
directory the bundle is written to. A name with a path in it would put it
somewhere else.
"""

import pytest

from eyegen.backends import coreml


class TestCoremlModelDir:
    def test_a_name_with_a_path_in_it_is_refused(self, tmp_path, monkeypatch):
        """The name is the directory the bundle is written to."""
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "models"))

        with pytest.raises(ValueError, match="not a path"):
            coreml.get_coreml_model_dir("../../etc")

    def test_the_repository_segment_names_the_directory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COREML_MODELS_DIR", str(tmp_path / "models"))

        assert coreml.get_coreml_model_dir("coreml-stable-diffusion-2-1-base") == (
            tmp_path / "models" / "coreml-stable-diffusion-2-1-base"
        )

"""Tests for the key the GUI keeps its loaded pipeline under.

The catalogue is read when a model is looked up and never cached, so a
model the user has just described differently is a different pipeline even
when the name is the same. These check that the specification is part of
the key.
"""

import pytest

from eyegen.config import Backend, EyeGenConfig
from eyegen.model_specs import ModelSource, ModelSpec


def _loader():
    """The loader, where the GUI's dependencies are installed."""
    try:
        from eyegen.gui import pipeline_loader
    except ImportError:
        pytest.skip("PySide6 not available")
    return pipeline_loader


def _spec(steps: int = 16) -> ModelSpec:
    return ModelSpec(
        name="klein",
        backend=Backend.SDCLI,
        sources={"diffusion_model": ModelSource(repo="owner/repo", file="model.gguf")},
        steps=steps,
    )


def test_a_specification_that_changed_is_a_different_pipeline():
    """Re-editing the catalogue must not reuse the pipeline it described."""
    loader = _loader()
    config = EyeGenConfig(model="klein")

    before = loader._make_cache_key(Backend.SDCLI, config, None, True, spec=_spec(steps=16))
    after = loader._make_cache_key(Backend.SDCLI, config, None, True, spec=_spec(steps=32))

    assert before != after


def test_the_same_specification_is_the_same_pipeline():
    loader = _loader()
    config = EyeGenConfig(model="klein")

    first = loader._make_cache_key(Backend.SDCLI, config, None, True, spec=_spec())
    second = loader._make_cache_key(Backend.SDCLI, config, None, True, spec=_spec())

    assert first == second


def test_a_model_without_a_specification_has_no_fingerprint():
    loader = _loader()
    config = EyeGenConfig(model="flux-3-image-klein-gguf")

    key = loader._make_cache_key(Backend.SDCLI, config, None, True, spec=None)

    assert loader._spec_fingerprint(None) == ""
    assert key[-1] == ""

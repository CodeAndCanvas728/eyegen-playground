"""Tests for the catalogue file: where the user's own models are kept.

The catalogue is a file next to the configuration, and it is the user's.
These check that it is written atomically, that it is read as it is, and
that a file which cannot be read says where it is rather than looking
like an empty catalogue.
"""

import pytest

from eyegen.config import Backend
from eyegen.model_catalogue import (
    catalogue_path,
    find_spec,
    load_catalogue,
    remove_spec,
    upsert_spec,
)
from eyegen.model_specs import ModelSource, ModelSpec


def _spec(name: str = "klein") -> ModelSpec:
    return ModelSpec(
        name=name,
        backend=Backend.SDCLI,
        sources={
            "diffusion_model": ModelSource(repo="unsloth/flux", file="transformer.gguf"),
        },
        steps=16,
    )


def test_the_catalogue_is_written_and_read():
    upsert_spec(_spec())
    assert catalogue_path().is_file()
    assert load_catalogue()["klein"].backend == Backend.SDCLI


def test_a_catalogue_that_is_not_there_is_empty():
    assert load_catalogue() == {}


def test_a_catalogue_that_cannot_be_parsed_says_where():
    catalogue_path().write_text("{not json")
    with pytest.raises(ValueError, match="models.json"):
        load_catalogue()


def test_a_catalogue_that_is_not_an_object_says_so():
    catalogue_path().write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="must hold an object"):
        load_catalogue()


def test_a_removed_spec_is_gone():
    upsert_spec(_spec())
    assert remove_spec("klein") is True
    assert load_catalogue() == {}
    assert remove_spec("klein") is False


def test_a_spec_is_found_by_name():
    upsert_spec(_spec())
    assert find_spec("klein") is not None
    assert find_spec("not-there") is None


def test_the_catalogue_is_written_in_one_piece(tmp_path):
    """A file that is half written is worse than a file that was not written."""
    upsert_spec(_spec())
    upsert_spec(_spec(name="second"))
    written = catalogue_path().read_text()
    assert "klein" in written
    assert "second" in written
    # No temporary file is left behind.
    assert [p.name for p in catalogue_path().parent.iterdir()] == [catalogue_path().name]

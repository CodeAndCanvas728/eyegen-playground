"""Tests for the steps and guidance the GUI shows a model.

A model is not a window: the pair of settings a model was last run with must
not follow the user to the next model. These check the reading and writing of
that per-model table, and that a model with no pair of its own is given what
its family asks for.
"""

from eyegen.backends.sdcli.constants import family_settings
from eyegen.config import Backend
from eyegen.gui.state import backend_value, settings_by_model, settings_for_model


def test_a_legacy_states_pair_belongs_to_its_own_model():
    """The one pair an older state file holds is not every model's."""
    saved = settings_by_model({"model": "walkyrie", "steps": 1, "guidance": 4.0})

    assert saved == {"walkyrie": {"steps": 1, "guidance": 4.0}}
    assert settings_for_model(saved, "qwen-image-2.1-UC-Q4_K_M", (20, 6.0)) == (20, 6.0)


def test_each_model_keeps_its_own_pair():
    state = {
        "settings_by_model": {
            "walkyrie": {"steps": 1, "guidance": 4.0},
            "qwen-image-2.1-UC-Q4_K_M": {"steps": 24, "guidance": 5.5},
        }
    }
    saved = settings_by_model(state)

    assert settings_for_model(saved, "walkyrie", (20, 6.0)) == (1, 4.0)
    assert settings_for_model(saved, "qwen-image-2.1-UC-Q4_K_M", (20, 6.0)) == (24, 5.5)


def test_an_entry_that_is_not_a_pair_is_ignored():
    """The file is the user's to edit, so a hand-written entry may be wrong."""
    state = {
        "model": "walkyrie",
        "settings_by_model": {"broken": {"steps": "many"}, "walkyrie": None},
    }

    assert settings_by_model(state) == {}


def test_a_model_with_no_pair_of_its_own_gets_its_familys():
    family = family_settings("qwen-image-2.1-UC-Q4_K_M")
    defaults = (family["steps"], family["guidance"])

    assert settings_for_model({}, "qwen-image-2.1-UC-Q4_K_M", defaults) == (20, 6.0)


def test_the_backend_is_written_as_a_name():
    """The combo holds a plain string.

    Reading it as an enum raised at close, and a state file that cannot be
    written is a state file that keeps the values it had — which is how a
    one-step experiment outlived the summer.
    """
    assert backend_value(Backend.AUTO) == "auto"
    assert backend_value("auto") == "auto"

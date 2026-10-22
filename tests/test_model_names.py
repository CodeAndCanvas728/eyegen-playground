"""Tests for the model names the tool prints.

A name in a message or a help string that no backend can resolve is a
name that sends the reader into a failure: `pull` would hand it to
OllamaDiffuser, and `generate` would reject it. These check that every
model name the tool offers is one the tool can actually run.
"""

import re

import pytest

from eyegen.backends import _format_unsupported_error, detect_backend
from eyegen.backends.sdcli.constants import SDCLI_MODELS
from eyegen.cli.models import pull
from eyegen.config import Backend

# A name that follows a pull command: "pull flux-3-image-klein-gguf".
_PULL = re.compile(r"pull\s+([a-z0-9][a-z0-9._-]*)")
# Words are not model names; a model name has a dash or a dot in it, and
# the ones in these texts that end in a file extension are file names.
_NOT_A_MODEL = (".md", ".py", ".sh", ".json", ".cpp", ".yaml", ".yml")
# The names in "e.g. one, or two".
_EXAMPLES = re.compile(r"e\.g\.,?\s*(.*?)\)")


def _pulled_names(text: str) -> list[str]:
    """The model names a text tells the reader to pull."""
    return [
        name
        for name in _PULL.findall(text)
        if ("-" in name or "." in name) and not name.endswith(_NOT_A_MODEL)
    ]


def _example_names(text: str) -> list[str]:
    """The model names a text gives as examples."""
    names = []
    for group in _EXAMPLES.findall(text):
        for part in group.split(","):
            part = part.strip()
            if part.startswith("or "):
                part = part[3:].strip()
            if part and ("-" in part or "." in part) and not part.endswith(_NOT_A_MODEL):
                names.append(part)
    return names


def _assert_resolvable(name: str, where: str) -> None:
    try:
        detect_backend(name, Backend.AUTO)
    except ValueError:
        pytest.fail(f"{where} names '{name}', which no backend resolves")


def test_the_catalogue_names_resolve():
    for name in SDCLI_MODELS:
        assert detect_backend(name, Backend.AUTO) == Backend.SDCLI


def test_the_error_message_names_models_that_resolve():
    """The message must not send a reader to a name nothing can run."""
    message = _format_unsupported_error("not-a-model", "auto-detected")
    suggested = [name for name in _pulled_names(message) if name != "not-a-model"]
    suggested += [
        name for name in re.findall(r"'([a-z0-9][a-z0-9._/-]*)'", message) if name != "not-a-model"
    ]
    assert suggested, "the message should name some models"
    for name in suggested:
        _assert_resolvable(name, "the error message")


def test_the_pull_help_names_a_model_that_resolves():
    """The pull command's help names a model, and it has to be a real one."""
    help_text = pull.__defaults__[0].help or ""
    examples = _example_names(help_text)
    assert examples, "the pull command should say what a model name looks like"
    for name in examples:
        _assert_resolvable(name, "the pull help")


def test_the_lifecycle_instructions_name_a_model_that_resolves():
    """The backend's own instructions name a model, and it has to be a real one."""
    from eyegen.backends import sdcli

    text = (sdcli.__doc__ or "") + (sdcli.pipeline.__doc__ or "")
    names = _pulled_names(text)
    assert names, "the instructions should say what to pull"
    for name in names:
        _assert_resolvable(name, "the instructions")

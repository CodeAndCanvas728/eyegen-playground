"""Tests for the backend update check."""

from unittest.mock import patch

import pytest

from eyegen.backends_update import (
    BackendState,
    Requirement,
    _version_tuple,
    format_report,
    format_state,
    read_requirements,
    survey,
)


def test_read_requirements_skips_comments_and_blank_lines(tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text(
        "# a comment\n"
        "\n"
        "mflux>=0.23.0\n"
        "Pillow>=10.0.0,<13.0.0\n"
        "typer>=0.9.0  # with a trailing comment\n"
    )
    requirements = read_requirements(path)
    assert [r.name for r in requirements] == ["mflux", "Pillow", "typer"]


def test_a_requirement_keeps_its_floor_and_ceiling():
    requirement = Requirement(name="Pillow", constraint=">=10.0.0,<13.0.0", raw="Pillow")
    assert requirement.floor == "10.0.0"
    assert requirement.ceiling == "13.0.0"


def test_a_requirement_keeps_no_comment(tmp_path):
    """pip is given the line verbatim, so the comment must not be in it."""
    path = tmp_path / "requirements.txt"
    path.write_text("typer>=0.9.0  # with a trailing comment\n")
    requirement = read_requirements(path)[0]
    assert requirement.raw == "typer>=0.9.0"
    assert "#" not in requirement.raw


def test_a_requirement_without_a_bound_has_no_floor():
    requirement = Requirement(name="typer", constraint="", raw="typer")
    assert requirement.floor is None
    assert requirement.ceiling is None


def test_versions_compare_by_their_numbers():
    assert _version_tuple("0.23.0") > _version_tuple("0.17.5")
    assert _version_tuple("2.10.0") > _version_tuple("2.9.0")
    assert _version_tuple("1.0.0rc1") == _version_tuple("1.0.0")


def _state(name="mflux", constraint=">=0.23.0", installed="0.17.5", latest="0.23.0"):
    return BackendState(
        requirement=Requirement(name=name, constraint=constraint, raw=name),
        installed=installed,
        latest=latest,
    )


def test_a_backend_below_its_floor_is_flagged():
    state = _state(installed="0.17.5", latest="0.23.0")
    assert state.behind_floor
    assert state.newer_available


def test_a_backend_at_its_floor_is_not_behind():
    state = _state(installed="0.23.0", latest="0.23.0")
    assert not state.behind_floor
    assert not state.newer_available


def test_a_newer_release_beyond_the_ceiling_is_noted():
    state = _state(
        name="Pillow", constraint=">=10.0.0,<13.0.0", installed="12.3.0", latest="13.0.0"
    )
    assert state.newer_available
    assert not state.within_ceiling


def test_the_report_says_when_everything_is_current():
    state = _state(installed="0.23.0", latest="0.23.0")
    report = format_report([state])
    assert "up to date" in report
    assert "✓" in format_state(state)


def test_the_report_names_what_is_behind():
    report = format_report([_state(installed="0.17.5", latest="0.23.0")])
    assert "Below what the requirements ask for" in report
    assert "mflux" in report


def test_the_survey_asks_pypi_for_each_requirement():
    requirements = [Requirement(name="mflux", constraint=">=0.23.0", raw="mflux>=0.23.0")]
    with patch("eyegen.backends_update.latest_version", return_value="0.23.0") as asked:
        states = survey(requirements)
    asked.assert_called_once_with("mflux")
    assert states[0].latest == "0.23.0"


def test_a_package_that_cannot_be_reached_is_not_an_error():
    requirements = [Requirement(name="mflux", constraint=">=0.23.0", raw="mflux>=0.23.0")]
    with patch("eyegen.backends_update.latest_version", return_value=None):
        states = survey(requirements)
    assert states[0].latest is None
    assert states[0].error


@pytest.mark.parametrize(
    "line,expected",
    [
        ("mflux>=0.23.0", "mflux"),
        ("Pillow>=10.0.0,<13.0.0", "Pillow"),
        ("diffusionkit>=0.1.0", "diffusionkit"),
    ],
)
def test_the_requirement_pattern(line, expected):
    requirements = read_requirements_from_text(line)
    assert requirements[0].name == expected


def read_requirements_from_text(text):
    import tempfile
    from pathlib import Path

    from eyegen.backends_update import read_requirements

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "requirements.txt"
        path.write_text(text + "\n")
        return read_requirements(path)

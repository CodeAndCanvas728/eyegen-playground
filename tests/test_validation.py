"""Tests for eyegen.validation."""

from pathlib import Path

import pytest

from eyegen.validation import (
    sanitize_prompt,
    validate_dimensions,
    validate_image_path,
)


def test_validate_dimensions_valid():
    assert validate_dimensions(512, 512) is None


def test_validate_dimensions_invalid():
    assert validate_dimensions(100, 512) is not None
    assert validate_dimensions(0, 512) is not None
    assert validate_dimensions(-8, 512) is not None
    assert validate_dimensions(512, 0) is not None
    assert validate_dimensions(512, -8) is not None


def test_validate_dimensions_zero():
    assert validate_dimensions(0, 1024) is not None


def test_validate_dimensions_negative():
    assert validate_dimensions(-8, 1024) is not None


def test_validate_image_path_missing():
    assert validate_image_path("/nonexistent/image.png") is not None


def test_validate_image_path_unsupported_format(tmp_path: Path):
    p = tmp_path / "image.gif"
    p.write_text("fake")
    assert validate_image_path(str(p)) is not None


def test_validate_image_path_supported_format(tmp_path: Path):
    p = tmp_path / "image.png"
    p.write_text("fake")
    assert validate_image_path(str(p)) is None


def test_sanitize_prompt_replaces_smart_quotes():
    prompt = "“hello” ‘world’ — …"
    cleaned = sanitize_prompt(prompt)
    assert '"hello"' in cleaned
    assert "'world'" in cleaned
    assert "—" not in cleaned


def test_validate_safe_path_traversal():
    from eyegen.validation import validate_safe_path
    with pytest.raises(ValueError, match="Directory traversal"):
        validate_safe_path("/safe/path/../../unsafe", "test_path")


def test_validate_safe_path_not_under_root():
    from eyegen.validation import validate_safe_path
    with pytest.raises(ValueError, match="not under any expected root"):
        validate_safe_path("/usr/bin/somefile", "test_path")

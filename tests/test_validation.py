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


def test_validate_safe_path_allows_home():
    from eyegen.validation import validate_safe_path

    p = validate_safe_path(str(Path.home()), "home_path")
    assert p == Path.home().resolve()


def test_validate_safe_path_allows_tmp():
    from eyegen.validation import validate_safe_path

    p = validate_safe_path("/tmp", "tmp_path")  # noqa: S108
    assert p == Path("/tmp").resolve()  # noqa: S108


def test_validate_safe_path_rejects_var():
    from eyegen.validation import validate_safe_path

    with pytest.raises(ValueError, match="not under any expected root"):
        validate_safe_path("/var", "var_path")


def test_validate_safe_path_rejects_volumes():
    from eyegen.validation import validate_safe_path

    with pytest.raises(ValueError, match="not under any expected root"):
        validate_safe_path("/Volumes", "volumes_path")


def test_validate_safe_path_subdir_of_allowed():
    from eyegen.validation import validate_safe_path

    sub = Path.home() / "Documents" / "test_projects" / "mygen-playground"
    p = validate_safe_path(str(sub), "project_root")
    assert p == sub.resolve()


def test_validate_safe_path_rejects_explicit_traversal():
    from eyegen.validation import validate_safe_path

    with pytest.raises(ValueError, match="Directory traversal"):
        validate_safe_path("/safe/../../etc/passwd", "bad_path")


def test_validate_safe_path_allows_var_folders():
    from eyegen.validation import validate_safe_path

    p = validate_safe_path("/private/var/folders/some/temp/path", "temp_path")
    assert p == Path("/private/var/folders/some/temp/path").resolve()


def test_validate_safe_path_rejects_var_raw():
    from eyegen.validation import validate_safe_path

    with pytest.raises(ValueError, match="not under any expected root"):
        validate_safe_path("/var", "var_raw")


def test_validate_safe_path_rejects_private_var():
    from eyegen.validation import validate_safe_path

    with pytest.raises(ValueError, match="not under any expected root"):
        validate_safe_path("/private/var/log", "private_var_log")


def test_theme_py_is_synchronized():  # noqa: PLR0915
    """Verify that theme.py is up-to-date with design-system/palette.json."""
    import json
    from pathlib import Path

    root = Path(__file__).parent.parent
    palette_path = root / "design-system" / "palette.json"
    theme_path = root / "eyegen" / "gui" / "style" / "theme.py"

    assert palette_path.exists(), "design-system/palette.json must exist"
    assert theme_path.exists(), "theme.py must exist"

    # Execute theme.py in a clean namespace to avoid importing PySide6
    namespace = {}
    exec(theme_path.read_text(), namespace)  # noqa: S102
    _DARK = namespace["_DARK"]
    _LIGHT = namespace["_LIGHT"]

    with open(palette_path, "r") as f:
        palette = json.load(f)

    accent = palette["accent_ramp"]
    neutral = palette["neutral_ramp"]
    text_contrast = palette["text_contrast"]

    # Verify key tokens in _DARK
    assert _DARK["bg"] == neutral["900"]
    assert _DARK["surface"] == neutral["700"]
    assert _DARK["border"] == neutral["700"]
    assert _DARK["border-light"] == neutral["600"]
    assert _DARK["text-primary"] == text_contrast["ground"]["recommended_text"]
    assert _DARK["text-secondary"] == neutral["300"]
    assert _DARK["accent"] == accent["600"]
    assert _DARK["accent-hover"] == accent["400"]
    assert _DARK["accent-light"] == accent["200"]
    assert _DARK["input-border"] == neutral["700"]
    assert _DARK["input-focus"] == accent["600"]
    assert _DARK["slider-handle"] == accent["600"]
    assert _DARK["slider-track"] == neutral["700"]
    assert _DARK["preview-gradient-start"] == accent["700"]
    assert _DARK["preview-gradient-end"] == neutral["900"]

    # Verify key tokens in _LIGHT
    assert _LIGHT["bg"] == neutral["50"]
    assert _LIGHT["surface"] == neutral["100"]
    assert _LIGHT["surface-hover"] == accent["100"]
    assert _LIGHT["border"] == neutral["200"]
    assert _LIGHT["border-light"] == neutral["100"]
    assert _LIGHT["text-primary"] == accent["900"]
    assert _LIGHT["text-secondary"] == neutral["600"]
    assert _LIGHT["accent"] == accent["600"]
    assert _LIGHT["accent-hover"] == accent["500"]
    assert _LIGHT["input-border"] == neutral["200"]
    assert _LIGHT["input-focus"] == accent["600"]
    assert _LIGHT["slider-handle"] == accent["600"]
    assert _LIGHT["slider-track"] == neutral["200"]
    assert _LIGHT["preview-gradient-start"] == accent["200"]
    assert _LIGHT["preview-gradient-end"] == neutral["50"]

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


def _assert_value(d, key, expected, label=""):
    assert d[key] == expected, f"{label}[{key!r}]: expected {expected!r}, got {d[key]!r}"


def test_theme_py_is_synchronized():  # noqa: PLR0915
    """Verify that theme.py is up-to-date with design-system/palette.json."""
    import json
    from pathlib import Path

    root = Path(__file__).parent.parent
    palette_path = root / "design-system" / "palette.json"
    theme_path = root / "eyegen" / "gui" / "style" / "theme.py"

    assert palette_path.exists(), "design-system/palette.json must exist"
    assert theme_path.exists(), "theme.py must exist"

    namespace = {}
    exec(theme_path.read_text(), namespace)  # noqa: S102
    _DARK = namespace["_DARK"]
    _LIGHT = namespace["_LIGHT"]

    with open(palette_path) as f:
        palette = json.load(f)

    accent = palette["accent_ramp"]
    neutral = palette["neutral_ramp"]
    text_contrast = palette["text_contrast"]
    surfaces = palette["surfaces"]

    # --- _DARK palette-derived tokens ---
    _assert_value(_DARK, "bg", neutral["900"], "_DARK")
    _assert_value(_DARK, "surface", surfaces["dark"]["card"], "_DARK")
    _assert_value(_DARK, "surface-hover", neutral["600"], "_DARK")
    _assert_value(_DARK, "card", surfaces["dark"]["card"], "_DARK")
    _assert_value(_DARK, "border", neutral["700"], "_DARK")
    _assert_value(_DARK, "border-light", neutral["600"], "_DARK")
    _assert_value(_DARK, "text-primary", text_contrast["ground"]["recommended_text"], "_DARK")
    _assert_value(_DARK, "text-secondary", neutral["300"], "_DARK")
    _assert_value(_DARK, "text-muted", neutral["400"], "_DARK")
    _assert_value(_DARK, "accent", accent["600"], "_DARK")
    _assert_value(_DARK, "accent-hover", accent["400"], "_DARK")
    _assert_value(_DARK, "accent-light", accent["200"], "_DARK")
    _assert_value(_DARK, "btn-pressed", accent["700"], "_DARK")
    _assert_value(_DARK, "scrollbar-hover", neutral["400"], "_DARK")
    _assert_value(_DARK, "input-bg", neutral["900"], "_DARK")
    _assert_value(_DARK, "input-border", neutral["700"], "_DARK")
    _assert_value(_DARK, "input-focus", accent["600"], "_DARK")
    _assert_value(_DARK, "slider-handle", accent["600"], "_DARK")
    _assert_value(_DARK, "slider-track", neutral["700"], "_DARK")
    _assert_value(_DARK, "preview-bg", neutral["900"], "_DARK")
    _assert_value(_DARK, "preview-gradient-start", accent["700"], "_DARK")
    _assert_value(_DARK, "preview-gradient-end", neutral["900"], "_DARK")

    # --- _LIGHT palette-derived tokens ---
    _assert_value(_LIGHT, "bg", neutral["50"], "_LIGHT")
    _assert_value(_LIGHT, "surface", neutral["100"], "_LIGHT")
    _assert_value(_LIGHT, "surface-hover", accent["100"], "_LIGHT")
    _assert_value(_LIGHT, "border", neutral["200"], "_LIGHT")
    _assert_value(_LIGHT, "border-light", neutral["100"], "_LIGHT")
    _assert_value(_LIGHT, "text-primary", accent["900"], "_LIGHT")
    _assert_value(_LIGHT, "text-secondary", neutral["600"], "_LIGHT")
    _assert_value(_LIGHT, "text-muted", neutral["500"], "_LIGHT")
    _assert_value(_LIGHT, "accent", accent["600"], "_LIGHT")
    _assert_value(_LIGHT, "accent-hover", accent["500"], "_LIGHT")
    _assert_value(_LIGHT, "accent-light", accent["200"], "_LIGHT")
    _assert_value(_LIGHT, "btn-pressed", accent["700"], "_LIGHT")
    _assert_value(_LIGHT, "scrollbar-hover", neutral["500"], "_LIGHT")
    _assert_value(_LIGHT, "input-border", neutral["200"], "_LIGHT")
    _assert_value(_LIGHT, "input-focus", accent["600"], "_LIGHT")
    _assert_value(_LIGHT, "slider-handle", accent["600"], "_LIGHT")
    _assert_value(_LIGHT, "slider-track", neutral["200"], "_LIGHT")
    _assert_value(_LIGHT, "preview-gradient-start", accent["200"], "_LIGHT")
    _assert_value(_LIGHT, "preview-gradient-end", neutral["50"], "_LIGHT")

    # Static (non-palette) tokens
    _assert_value(_DARK, "card", "#4C453E", "_DARK")
    _assert_value(_DARK, "success", "#10B981", "_DARK")
    _assert_value(_DARK, "warning", "#F59E0B", "_DARK")
    _assert_value(_DARK, "danger", "#EF4444", "_DARK")
    _assert_value(_DARK, "shadow", "rgba(0,0,0,0.5)", "_DARK")
    _assert_value(_LIGHT, "card", "#FFFFFF", "_LIGHT")
    _assert_value(_LIGHT, "input-bg", "#FFFFFF", "_LIGHT")
    _assert_value(_LIGHT, "preview-bg", "#F2EDE6", "_LIGHT")
    _assert_value(_LIGHT, "success", "#059669", "_LIGHT")
    _assert_value(_LIGHT, "warning", "#D97706", "_LIGHT")
    _assert_value(_LIGHT, "danger", "#DC2626", "_LIGHT")
    _assert_value(_LIGHT, "shadow", "rgba(0,0,0,0.06)", "_LIGHT")

    # Shared design tokens (radii, spacing, typography)
    _SHARED = {
        "radius-sm": "6px",
        "radius-md": "10px",
        "radius-lg": "14px",
        "radius-xl": "20px",
        "font-size-sm": "11px",
        "font-size-md": "13px",
        "font-size-lg": "16px",
        "font-size-xl": "20px",
        "font-size-2xl": "24px",
        "spacing-xs": "8px",
        "spacing-sm": "8px",
        "spacing-md": "16px",
        "spacing-lg": "16px",
        "spacing-xl": "24px",
    }
    for key, val in _SHARED.items():
        _assert_value(_DARK, key, val, "_DARK")
        _assert_value(_LIGHT, key, val, "_LIGHT")

    # Verify count to catch any newly-added untested keys
    assert len(_DARK) == 40, f"_DARK has {len(_DARK)} keys, expected 40"
    assert len(_LIGHT) == 40, f"_LIGHT has {len(_LIGHT)} keys, expected 40"

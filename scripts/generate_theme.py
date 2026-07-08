#!/usr/bin/env python3
"""Regenerate theme.py from design-system/palette.json."""

import json
from pathlib import Path

# ruff: noqa: T201 — CLI script, print is the intended output mechanism


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


def _fmt(d: dict) -> str:
    return ",\n    ".join(f'"{k}": "{v}"' for k, v in d.items())


def main():
    root = Path(__file__).parent.parent
    palette_path = root / "design-system" / "palette.json"
    theme_path = root / "eyegen" / "gui" / "style" / "theme.py"

    if not palette_path.exists():
        print(f"Error: {palette_path} does not exist.")
        exit(1)

    with open(palette_path) as f:
        palette = json.load(f)

    accent = palette["accent_ramp"]
    neutral = palette["neutral_ramp"]
    text_contrast = palette["text_contrast"]
    surfaces = palette["surfaces"]

    dark = {
        "bg": neutral["900"],
        "surface": surfaces["dark"]["card"],
        "surface-hover": neutral["600"],
        "card": surfaces["dark"]["card"],
        "border": neutral["700"],
        "border-light": neutral["600"],
        "text-primary": text_contrast["ground"]["recommended_text"],
        "text-secondary": neutral["300"],
        "text-muted": neutral["400"],
        "accent": accent["600"],
        "accent-hover": accent["400"],
        "accent-light": accent["200"],
        "btn-pressed": accent["700"],
        "scrollbar-hover": neutral["400"],
        "success": "#10B981",
        "warning": "#F59E0B",
        "danger": "#EF4444",
        "input-bg": neutral["900"],
        "input-border": neutral["700"],
        "input-focus": accent["600"],
        "slider-handle": accent["600"],
        "slider-track": neutral["700"],
        "preview-bg": neutral["900"],
        "preview-gradient-start": accent["700"],
        "preview-gradient-end": neutral["900"],
        "shadow": "rgba(0,0,0,0.5)",
    }

    light = {
        "bg": neutral["50"],
        "surface": neutral["100"],
        "surface-hover": accent["100"],
        "card": "#FFFFFF",
        "border": neutral["200"],
        "border-light": neutral["100"],
        "text-primary": accent["900"],
        "text-secondary": neutral["600"],
        "text-muted": neutral["500"],
        "accent": accent["600"],
        "accent-hover": accent["500"],
        "accent-light": accent["200"],
        "btn-pressed": accent["700"],
        "scrollbar-hover": neutral["500"],
        "success": "#059669",
        "warning": "#D97706",
        "danger": "#DC2626",
        "input-bg": "#FFFFFF",
        "input-border": neutral["200"],
        "input-focus": accent["600"],
        "slider-handle": accent["600"],
        "slider-track": neutral["200"],
        "preview-bg": "#F2EDE6",
        "preview-gradient-start": accent["200"],
        "preview-gradient-end": neutral["50"],
        "shadow": "rgba(0,0,0,0.06)",
    }

    shared_str = _fmt(_SHARED)
    dark_str = _fmt(dark)
    light_str = _fmt(light)

    theme_content = f'''"""Design tokens and theme loader for EyeGen GUI themes.

Automatically generated from design-system/palette.json. Do not edit manually.
"""

import string
from pathlib import Path


_SHARED = {{
    {shared_str},
}}


_DARK = {{
    {dark_str},
    **_SHARED,
}}


_LIGHT = {{
    {light_str},
    **_SHARED,
}}


class LazyThemeDict(dict):
    """Lazy-loads QSS on first access, substituting tokens from the theme dict."""

    def __init__(self):
        super().__init__()
        self._cache = {{}}

    def __getitem__(self, key: str) -> str:
        if key not in self._cache:
            if key == "dark":
                self._cache[key] = _load_qss("dark.qss", _DARK)
            elif key == "light":
                self._cache[key] = _load_qss("light.qss", _LIGHT)
            else:
                raise KeyError(key)
        return self._cache[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return key in self._cache

    def keys(self):
        return list(self._cache.keys())


def _load_qss(name: str, tokens: dict) -> str:
    p = Path(__file__).parent / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required stylesheet '{{name}}' is missing in '{{p.parent}}'. "
            "Please ensure the application is correctly installed and all assets are present."
        )
    return string.Template(p.read_text()).safe_substitute(tokens)


ALL_THEMES = LazyThemeDict()


def __getattr__(name: str):
    if name == "DARK_QSS":
        return ALL_THEMES["dark"]
    if name == "LIGHT_QSS":
        return ALL_THEMES["light"]
    raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")
'''

    with open(theme_path, "w") as f:
        f.write(theme_content)
    print(f"Regenerated {theme_path} successfully.")


if __name__ == "__main__":
    main()

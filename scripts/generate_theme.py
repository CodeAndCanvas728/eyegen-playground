#!/usr/bin/env python3
"""Script to regenerate theme.py from design-system/palette.json."""

import json
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    palette_path = root / "design-system" / "palette.json"
    theme_path = root / "eyegen" / "gui" / "style" / "theme.py"

    if not palette_path.exists():
        print(f"Error: {palette_path} does not exist.")  # noqa: T201
        exit(1)

    with open(palette_path, "r") as f:
        palette = json.load(f)

    accent = palette["accent_ramp"]
    neutral = palette["neutral_ramp"]
    text_contrast = palette["text_contrast"]

    theme_content = f"""\"\"\"Design tokens and theme loader for EyeGen GUI themes.

Automatically generated from design-system/palette.json. Do not edit manually.
\"\"\"

from pathlib import Path

_DARK = {{
    "bg": "{neutral["900"]}",
    "surface": "{neutral["700"]}",
    "surface-hover": "#5A534C",
    "card": "#36312C",
    "border": "{neutral["700"]}",
    "border-light": "{neutral["600"]}",
    "text-primary": "{text_contrast["ground"]["recommended_text"]}",
    "text-secondary": "{neutral["300"]}",
    "text-muted": "#A8A098",
    "accent": "{accent["600"]}",
    "accent-hover": "{accent["400"]}",
    "accent-light": "{accent["200"]}",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "input-bg": "#1E1C1A",
    "input-border": "{neutral["700"]}",
    "input-focus": "{accent["600"]}",
    "slider-handle": "{accent["600"]}",
    "slider-track": "{neutral["700"]}",
    "preview-bg": "#1A1816",
    "preview-gradient-start": "{accent["700"]}",
    "preview-gradient-end": "{neutral["900"]}",
    "shadow": "rgba(0,0,0,0.5)",
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
}}

_LIGHT = {{
    "bg": "{neutral["50"]}",
    "surface": "{neutral["100"]}",
    "surface-hover": "{accent["100"]}",
    "card": "#FFFFFF",
    "border": "{neutral["200"]}",
    "border-light": "{neutral["100"]}",
    "text-primary": "{accent["900"]}",
    "text-secondary": "{neutral["600"]}",
    "text-muted": "#756962",
    "accent": "{accent["600"]}",
    "accent-hover": "{accent["500"]}",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "input-bg": "#FFFFFF",
    "input-border": "{neutral["200"]}",
    "input-focus": "{accent["600"]}",
    "slider-handle": "{accent["600"]}",
    "slider-track": "{neutral["200"]}",
    "preview-bg": "#F2EDE6",
    "preview-gradient-start": "{accent["200"]}",
    "preview-gradient-end": "{neutral["50"]}",
    "shadow": "rgba(0,0,0,0.06)",
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
}}


class LazyThemeDict(dict):
    def __init__(self):
        super().__init__()
        self._cache = {{}}

    def __getitem__(self, key: str) -> str:
        if key not in self._cache:
            if key in ("dark", "light"):
                self._cache[key] = _load_qss(f"{{key}}.qss")
            else:
                raise KeyError(key)
        return self._cache[key]

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key) -> bool:
        return key in ("dark", "light")


def _load_qss(name: str) -> str:
    p = Path(__file__).parent / name
    if not p.exists():
        raise FileNotFoundError(
            f"Required stylesheet '{{name}}' is missing in '{{p.parent}}'. "
            "Please ensure the application is correctly installed and all assets are present."
        )
    return p.read_text()


ALL_THEMES = LazyThemeDict()


def __getattr__(name: str):
    if name == "DARK_QSS":
        return ALL_THEMES["dark"]
    if name == "LIGHT_QSS":
        return ALL_THEMES["light"]
    raise AttributeError(f"module {{__name__!r}} has no attribute {{name!r}}")
"""

    with open(theme_path, "w") as f:
        f.write(theme_content)
    print(f"Regenerated {theme_path} successfully.")  # noqa: T201


if __name__ == "__main__":
    main()

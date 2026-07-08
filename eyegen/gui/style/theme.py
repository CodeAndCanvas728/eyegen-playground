"""Design tokens and theme loader for EyeGen GUI themes.

Automatically generated from design-system/palette.json. Do not edit manually.
"""

import string
from pathlib import Path

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


_DARK = {
    "bg": "#2D2925",
    "surface": "#4C453E",
    "surface-hover": "#BCB5AE",
    "card": "#4C453E",
    "border": "#4C453E",
    "border-light": "#6B6157",
    "text-primary": "#F7F3EE",
    "text-secondary": "#BCB5AE",
    "text-muted": "#9F9489",
    "accent": "#853D4F",
    "accent-hover": "#BD6A80",
    "accent-light": "#E0C3CA",
    "btn-pressed": "#5D2D39",
    "scrollbar-hover": "#9F9489",
    "success": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "input-bg": "#2D2925",
    "input-border": "#4C453E",
    "input-focus": "#853D4F",
    "slider-handle": "#853D4F",
    "slider-track": "#4C453E",
    "preview-bg": "#2D2925",
    "preview-gradient-start": "#5D2D39",
    "preview-gradient-end": "#2D2925",
    "shadow": "rgba(0,0,0,0.5)",
    **_SHARED,
}


_LIGHT = {
    "bg": "#F8F7F7",
    "surface": "#EAE8E6",
    "surface-hover": "#877A6E",
    "card": "#FFFFFF",
    "border": "#D6D1CD",
    "border-light": "#EAE8E6",
    "text-primary": "#361C23",
    "text-secondary": "#6B6157",
    "text-muted": "#877A6E",
    "accent": "#853D4F",
    "accent-hover": "#AC4962",
    "accent-light": "#E0C3CA",
    "btn-pressed": "#5D2D39",
    "scrollbar-hover": "#877A6E",
    "success": "#059669",
    "warning": "#D97706",
    "danger": "#DC2626",
    "input-bg": "#FFFFFF",
    "input-border": "#D6D1CD",
    "input-focus": "#853D4F",
    "slider-handle": "#853D4F",
    "slider-track": "#D6D1CD",
    "preview-bg": "#F2EDE6",
    "preview-gradient-start": "#E0C3CA",
    "preview-gradient-end": "#F8F7F7",
    "shadow": "rgba(0,0,0,0.06)",
    **_SHARED,
}


class LazyThemeDict(dict):
    """Lazy-loads QSS on first access, substituting tokens from the theme dict."""

    def __init__(self):
        super().__init__()
        self._cache = {}

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
            f"Required stylesheet '{name}' is missing in '{p.parent}'. "
            "Please ensure the application is correctly installed and all assets are present."
        )
    return string.Template(p.read_text()).safe_substitute(tokens)


ALL_THEMES = LazyThemeDict()


def __getattr__(name: str):
    if name == "DARK_QSS":
        return ALL_THEMES["dark"]
    if name == "LIGHT_QSS":
        return ALL_THEMES["light"]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

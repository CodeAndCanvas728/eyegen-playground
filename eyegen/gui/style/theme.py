"""Design tokens and theme loader for EyeGen GUI themes."""

from pathlib import Path

_DARK = {
    "bg": "#121218",
    "surface": "#1e1e28",
    "surface-hover": "#28283a",
    "card": "#181823",
    "border": "#2e2e42",
    "border-light": "#3a3a50",
    "text-primary": "#e8e8ef",
    "text-secondary": "#9a9ab5",
    "text-muted": "#6c6c88",
    "accent": "#8b5cf6",
    "accent-hover": "#a78bfa",
    "accent-gradient-start": "#7c3aed",
    "accent-gradient-end": "#c026d3",
    "accent-secondary": "#ec4899",
    "accent-secondary-hover": "#f472b6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "input-bg": "#12121c",
    "input-border": "#2e2e42",
    "input-focus": "#8b5cf6",
    "slider-handle": "#8b5cf6",
    "slider-track": "#2e2e42",
    "preview-bg": "#0e0e16",
    "preview-gradient-start": "#2d1b4e",
    "preview-gradient-end": "#181823",
    "shadow": "rgba(0,0,0,0.4)",
    "radius-sm": "6px",
    "radius-md": "10px",
    "radius-lg": "14px",
    "radius-xl": "20px",
    "font-size-sm": "11px",
    "font-size-md": "13px",
    "font-size-lg": "15px",
    "font-size-xl": "18px",
    "font-size-2xl": "22px",
    "spacing-xs": "4px",
    "spacing-sm": "8px",
    "spacing-md": "12px",
    "spacing-lg": "16px",
    "spacing-xl": "24px",
}

_LIGHT = {
    "bg": "#f0f0f4",
    "surface": "#ffffff",
    "surface-hover": "#f5f5fa",
    "card": "#ffffff",
    "border": "#dcdce6",
    "border-light": "#e6e6ed",
    "text-primary": "#1e1e2d",
    "text-secondary": "#5a5a72",
    "text-muted": "#9191a8",
    "accent": "#7c3aed",
    "accent-hover": "#8b5cf6",
    "accent-gradient-start": "#7c3aed",
    "accent-gradient-end": "#c026d3",
    "accent-secondary": "#ec4899",
    "accent-secondary-hover": "#f472b6",
    "success": "#059669",
    "warning": "#d97706",
    "danger": "#dc2626",
    "input-bg": "#ffffff",
    "input-border": "#dcdce6",
    "input-focus": "#7c3aed",
    "slider-handle": "#7c3aed",
    "slider-track": "#dcdce6",
    "preview-bg": "#e4e4ec",
    "preview-gradient-start": "#e9d5ff",
    "preview-gradient-end": "#fae8ff",
    "shadow": "rgba(0,0,0,0.06)",
    "radius-sm": "6px",
    "radius-md": "10px",
    "radius-lg": "14px",
    "radius-xl": "20px",
    "font-size-sm": "11px",
    "font-size-md": "13px",
    "font-size-lg": "15px",
    "font-size-xl": "18px",
    "font-size-2xl": "22px",
    "spacing-xs": "4px",
    "spacing-sm": "8px",
    "spacing-md": "12px",
    "spacing-lg": "16px",
    "spacing-xl": "24px",
}


def _load_qss(name: str) -> str:
    """Load a .qss file from the style directory."""
    p = Path(__file__).parent / name
    return p.read_text()


DARK_QSS = _load_qss("dark.qss")
LIGHT_QSS = _load_qss("light.qss")

ALL_THEMES = {"dark": DARK_QSS, "light": LIGHT_QSS}

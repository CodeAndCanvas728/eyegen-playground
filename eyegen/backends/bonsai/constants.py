"""Bonsai backend constants."""

from pathlib import Path

DEFAULT_BONSAI_DIR = Path.home() / "models" / "eyegen" / "bonsai-demo"

SUPPORTED_VARIANTS = ("ternary-mlx", "binary-mlx")
DEFAULT_VARIANT = "ternary-mlx"

BONSAI_DEFAULT_STEPS = 4
BONSAI_DEFAULT_GUIDANCE = 1.0

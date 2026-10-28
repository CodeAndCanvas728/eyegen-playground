"""Reading and writing the configuration file."""

from __future__ import annotations

import json
import logging

import eyegen.config as _config
from eyegen.config import EyeGenConfig
from eyegen.config_recovery import backup_corrupted_config, handle_parse_error

log = logging.getLogger(__name__)
_config_warnings: list[str] = []


def pop_config_warnings() -> list[str]:
    """Return accumulated config-load warnings and clear them."""
    warnings = list(_config_warnings)
    _config_warnings.clear()
    return warnings


def _write_default_config() -> None:
    """Write a fresh config, after the old one turned out to be unusable."""
    save_config(EyeGenConfig())


def load_config() -> EyeGenConfig:
    """Load configuration from config.json, migrate/normalize it, or return defaults."""
    if not _config.CONFIG_FILE.exists():
        return EyeGenConfig()

    try:
        with open(_config.CONFIG_FILE, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        backup_corrupted_config(
            e, _config.CONFIG_FILE, _write_default_config, _config_warnings.append
        )
        return EyeGenConfig()

    if not isinstance(data, dict):
        log.warning("config.json root is not a dictionary. Resetting to defaults.")
        handle_parse_error(
            ValueError("Root is not a dictionary"),
            _write_default_config,
            _config_warnings.append,
        )
        return EyeGenConfig()

    try:
        cfg = EyeGenConfig.from_dict(data)
        normalized_dict = cfg.to_dict()
        if normalized_dict != data:
            log.warning("Config required migration/normalization; saving config.")
            try:
                save_config(cfg)
            except (OSError, PermissionError, ValueError) as save_err:
                log.error("Failed to save migrated config: %s", save_err)
        return cfg
    except (ValueError, TypeError) as e:
        handle_parse_error(e, _write_default_config, _config_warnings.append)
        return EyeGenConfig()


def save_config(config: EyeGenConfig):
    """Save configuration to config.json after running EyeGenConfig validation."""
    errors = config.validate()
    if errors:
        raise ValueError("; ".join(errors))
    with open(_config.CONFIG_FILE, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

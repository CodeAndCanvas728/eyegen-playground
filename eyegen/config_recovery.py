"""Recovery when the configuration file cannot be read.

Both helpers take the decision out of ``load_config`` so that the reading
of a file is not tangled with what to do when the file is broken.
"""

from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


def backup_corrupted_config(
    exc: Exception,
    config_file,
    write_default: Callable[[], None],
    warn: Callable[[str], None],
) -> None:
    """Move a broken config file aside and write a default one."""
    backup_file = config_file.with_suffix(".json.bak")
    msg = f"Corrupted config.json backed up to {backup_file} and reset to defaults."
    warn(msg)
    log.error(
        "Invalid JSON in config.json: %s. Backing up to %s and resetting to defaults.",
        exc,
        backup_file,
        exc_info=True,
    )
    try:
        if config_file.exists():
            config_file.replace(backup_file)
        write_default()
    except (OSError, PermissionError, ValueError) as write_err:
        log.error("Failed to write default config: %s", write_err)


def handle_parse_error(
    exc: Exception,
    write_default: Callable[[], None],
    warn: Callable[[str], None],
) -> None:
    """Reset a config file whose values do not make sense."""
    msg = "Config file had invalid values and was reset to defaults."
    warn(msg)
    log.error(
        "Failed to parse config.json, resetting to defaults: %s",
        exc,
        exc_info=True,
    )
    try:
        write_default()
    except (OSError, PermissionError, ValueError) as write_err:
        log.error("Failed to write default config: %s", write_err)

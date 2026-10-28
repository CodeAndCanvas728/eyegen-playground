"""stable-diffusion.cpp CLI installation and model discovery."""

from __future__ import annotations

import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eyegen.model_sources import ModelSource
from eyegen.model_specs import ModelSpec
from eyegen.paths import sdcli_model_dir
from eyegen.validation import validate_model_name

from .constants import (
    DEFAULT_SDCLI_BIN,
    DEFAULT_SDCLI_TIMEOUT,
)

log = logging.getLogger(__name__)


@dataclass
class SdCliInstallStatus:
    """Whether the backend can run, and what is missing if it cannot."""

    installed: bool
    binary: Optional[Path]
    model_dir: Optional[Path]
    missing_files: list[str]
    message: str


def get_sdcli_bin() -> Optional[Path]:
    """Find the sd-cli binary: the environment wins, then the usual place, then PATH."""
    override = os.environ.get("SDCLI_BIN")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_file() else None

    if DEFAULT_SDCLI_BIN.is_file():
        return DEFAULT_SDCLI_BIN

    found = shutil.which("sd-cli")
    return Path(found) if found else None


def get_sdcli_model_dir(model: str, config=None) -> Path:
    """The directory holding one model's files.

    Each model has a directory of its own: two models have files with the
    same names — ``ae.safetensors`` is a VAE whichever model it belongs to
    — and one directory for both would mean one overwriting the other.

    The model names the directory, so the name is checked before it becomes
    a path.
    """
    override = os.environ.get("SDCLI_MODEL_DIR")
    if override:
        return Path(override).expanduser() / validate_model_name(model)
    return sdcli_model_dir(model, config)


def get_sdcli_timeout() -> float:
    """How long a render may take, in seconds."""
    override = os.environ.get("SDCLI_TIMEOUT")
    if override:
        try:
            value = float(override)
        except ValueError:
            log.warning("SDCLI_TIMEOUT is not a number: %s", override)
        else:
            if value > 0:
                return value
    return float(DEFAULT_SDCLI_TIMEOUT)


def source_is_here(source: ModelSource, model_dir: Optional[Path]) -> bool:
    """Whether *source*'s file is where the backend uses it from.

    A file that is already on this machine is used where it is; a file that
    is downloaded is used from the model's own directory. One rule answers
    both "what is missing" and "is it installed", so the two cannot say
    different things about the same file.
    """
    if source.is_local:
        return Path(source.path).expanduser().is_file()
    return model_dir is not None and (model_dir / source.name).is_file()


def missing_model_files(
    model: str,
    model_dir: Optional[Path],
    spec: Optional[ModelSpec] = None,
) -> list[str]:
    """The files of *model* that are not where they should be."""
    if spec is None:
        from .models import sdcli_specs

        spec = sdcli_specs().get(model)
    if spec is None:
        return []
    missing = []
    for source in spec.sources.values():
        if source_is_here(source, model_dir):
            continue
        if source.is_local:
            missing.append(str(Path(source.path).expanduser()))
        else:
            missing.append(source.name)
    return missing


def validate_sdcli_install(
    model: str,
    model_dir: Optional[Path] = None,
    spec: Optional[ModelSpec] = None,
    config=None,
) -> SdCliInstallStatus:
    """Check that the binary and the model files are in place."""
    binary = get_sdcli_bin()
    directory = model_dir
    if directory is None and not (spec is not None and spec.is_local):
        directory = get_sdcli_model_dir(model, config)
    missing = missing_model_files(model, directory, spec=spec)

    if binary is None:
        return SdCliInstallStatus(
            installed=False,
            binary=None,
            model_dir=directory,
            missing_files=missing,
            message=(
                "sd-cli not found. Build it from https://github.com/leejet/stable-diffusion.cpp "
                "with -DGGML_METAL=ON and put the binary at "
                f"{DEFAULT_SDCLI_BIN} (or set SDCLI_BIN)."
            ),
        )

    if missing:
        where = str(directory) if directory is not None else "the model"
        return SdCliInstallStatus(
            installed=False,
            binary=binary,
            model_dir=directory,
            missing_files=missing,
            message=(
                f"Missing model files in {where}: {', '.join(missing)}. "
                f"Fetch them with: ./generate.py pull {model}"
            ),
        )

    return SdCliInstallStatus(
        installed=True,
        binary=binary,
        model_dir=directory,
        missing_files=[],
        message=f"sd-cli ready at {binary}",
    )


def _sysctl_hw_memsize() -> Optional[int]:
    """The machine's physical memory via ``sysctl hw.memsize``, or None."""
    import ctypes

    # The full path matters: a bare name ("libc") is not in the dyld
    # cache of every Python that could run the tool, while the absolute
    # path always resolves. sysctlbyname lives in libSystem.
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    libc.sysctlbyname.restype = ctypes.c_int
    libc.sysctlbyname.argtypes = [
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    value = ctypes.c_ulonglong()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(b"hw.memsize", ctypes.byref(value), ctypes.byref(size), None, 0) == 0:
        return int(value.value)
    return None


def _sysconf_physical_pages_bytes() -> Optional[int]:
    """Physical memory via sysconf, or None when the platform does not answer."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        return None


def physical_memory() -> int:
    """The machine's physical memory in bytes.

    macOS reports it through ``sysctl hw.memsize``; the generic path is
    sysconf. When neither can be read, the run is refused rather than
    guessed at: a guard that answers "an exabyte" when it cannot see would
    open wide and let a model that freezes the machine through.
    """
    if sys.platform == "darwin":
        size = _sysctl_hw_memsize()
        if size is not None:
            return size
    size = _sysconf_physical_pages_bytes()
    if size is None:
        raise RuntimeError(
            "Could not determine this machine's memory, so it cannot be checked that "
            "the model fits into it. Refusing to run a model sd-cli loads whole rather "
            "than risk a freeze."
        )
    return size


def diffusion_path(spec) -> str:
    """The transformer's own path, for the message that names the model."""
    source = spec.sources.get("diffusion_model")
    return source.path if source is not None else ""

"""Keeping the backends up to date.

The backends are Python packages in the venv, and a new model arrives
when one of them is released: FLUX.3 came with mflux 0.23. Nothing here
knows that on its own, so the tool can be asked.

The requirements file is the source of truth for what a good version is.
It carries the floors — mflux 0.23 for FLUX.3 — and the ceilings that
come from a broken release, like Pillow under 13, which removed the
module mflux imports. Upgrading means moving within those bounds, not
to the newest thing on PyPI.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from eyegen.config import PROJECT_ROOT

log = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/{package}/json"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"

# A requirement like "Pillow>=10.0.0,<13.0.0" or "mflux>=0.23.0".
_REQUIREMENT = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*"
    r"(?P<spec>\[[^\]]*\])?\s*"
    r"(?P<constraint>[<>=!~][^#]*)?\s*(?:#.*)?$"
)


@dataclass
class Requirement:
    """One line of the requirements file."""

    name: str
    constraint: str
    raw: str

    @property
    def floor(self) -> Optional[str]:
        """The lowest version this requirement allows, if it says."""
        match = re.search(r">=\s*([0-9][^,]*?)(?:\s*,|$)", self.constraint)
        return match.group(1).strip() if match else None

    @property
    def ceiling(self) -> Optional[str]:
        """The highest version this requirement allows, if it says."""
        match = re.search(r"<\s*([0-9][^,]*?)(?:\s*,|$)", self.constraint)
        return match.group(1).strip() if match else None


@dataclass
class BackendState:
    """What is installed, what is asked for, and what is out there."""

    requirement: Requirement
    installed: Optional[str]
    latest: Optional[str]
    error: Optional[str] = None

    @property
    def name(self) -> str:
        return self.requirement.name

    @property
    def behind_floor(self) -> bool:
        """True when the installed version is below what the file asks for."""
        floor = self.requirement.floor
        if not floor or not self.installed:
            return False
        return _version_tuple(self.installed) < _version_tuple(floor)

    @property
    def newer_available(self) -> bool:
        """True when PyPI has something newer than what is installed."""
        if not self.latest or not self.installed:
            return False
        return _version_tuple(self.latest) > _version_tuple(self.installed)

    @property
    def within_ceiling(self) -> bool:
        """True when the newest release is still allowed by the file."""
        ceiling = self.requirement.ceiling
        if not ceiling or not self.latest:
            return True
        return _version_tuple(self.latest) < _version_tuple(ceiling)


def _version_tuple(version: str) -> tuple:
    """A comparable tuple for a version string, ignoring anything odd."""
    parts = []
    for chunk in re.split(r"[.\-+]", version):
        match = re.match(r"^(\d+)", chunk)
        parts.append(int(match.group(1)) if match else 0)
    return tuple(parts) if parts else (0,)


def read_requirements(path: Optional[Path] = None) -> list[Requirement]:
    """Read the requirements file into a list of requirements."""
    source = path or REQUIREMENTS_FILE
    requirements: list[Requirement] = []
    if not source.is_file():
        return requirements
    for line in source.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _REQUIREMENT.match(stripped)
        if not match:
            continue
        # The line without its comment: pip is given this verbatim, and
        # a trailing comment would arrive as an argument of its own.
        without_comment = stripped.split("#", 1)[0].strip()
        requirements.append(
            Requirement(
                name=match.group("name"),
                constraint=(match.group("constraint") or "").strip(),
                raw=without_comment,
            )
        )
    return requirements


def installed_version(package: str) -> Optional[str]:
    """The version of *package* in this environment, if it is installed."""
    try:
        from importlib.metadata import version

        return version(package)
    except Exception:  # noqa: BLE001 - a missing package is not an error here
        return None


def latest_version(package: str, timeout: float = 10.0) -> Optional[str]:
    """The newest release of *package* on PyPI, if it can be reached."""
    url = PYPI_URL.format(package=package)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - https, from a fixed template
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError, TimeoutError) as exc:
        log.debug("Could not reach PyPI for %s: %s", package, exc)
        return None
    return (data.get("info") or {}).get("version")


def survey(requirements: Optional[list[Requirement]] = None) -> list[BackendState]:
    """The state of every requirement: installed, asked for, and newest."""
    states = []
    for requirement in requirements if requirements is not None else read_requirements():
        installed = installed_version(requirement.name)
        latest = latest_version(requirement.name)
        error = None
        if latest is None:
            error = "PyPI could not be reached"
        states.append(
            BackendState(
                requirement=requirement,
                installed=installed,
                latest=latest,
                error=error,
            )
        )
    return states


def venv_python() -> Optional[Path]:
    """The interpreter of the project venv, if there is one."""
    candidate = PROJECT_ROOT / "venv" / "bin" / "python"
    return candidate if candidate.is_file() else None


def run_pip(arguments: list[str], timeout: float = 1800.0) -> tuple[int, str]:
    """Run pip in the project venv and return its status and output.

    pip has a timeout of its own: a download that stalls does not finish,
    and a command that waits forever is worse than one that fails.
    """
    python = venv_python()
    if python is None:
        return 1, "No project venv found at venv/. Run ./setup.sh first."
    try:
        result = subprocess.run(  # noqa: S603
            [str(python), "-m", "pip", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 1, f"pip did not finish within {timeout:.0f} seconds"
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def apply_updates(requirements: Optional[list[Requirement]] = None) -> tuple[bool, str]:
    """Upgrade the requirements in the venv, within the bounds the file sets."""
    wanted = requirements if requirements is not None else read_requirements()
    if not wanted:
        return False, "No requirements to install."
    arguments = ["install", "--upgrade"]
    for requirement in wanted:
        arguments.append(requirement.raw)
    status, output = run_pip(arguments)
    if status != 0:
        return False, output.strip().splitlines()[-1] if output.strip() else "pip failed"
    return True, output


def format_state(state: BackendState) -> str:
    """One line describing a backend's state."""
    installed = state.installed or "not installed"
    wanted = state.requirement.floor or state.requirement.constraint or "any"
    newest = state.latest or "unknown"
    if state.behind_floor:
        mark = "⚠"
    elif state.newer_available:
        mark = "↑"
    else:
        mark = "✓"
    return f"  {mark} {state.name:16} {installed:12} wants ≥{wanted:10} newest {newest}"


def format_report(states: list[BackendState]) -> str:
    """The whole report, as the command prints it."""
    lines = ["The backends and what is out there:", ""]
    lines.extend(format_state(state) for state in states)
    behind = [s for s in states if s.behind_floor]
    newer = [s for s in states if s.newer_available]
    lines.append("")
    if behind:
        names = ", ".join(s.name for s in behind)
        lines.append(f"Below what the requirements ask for: {names}")
    if newer:
        names = ", ".join(s.name for s in newer)
        lines.append(f"A newer release is available: {names}")
    if not behind and not newer:
        lines.append("Everything is up to date.")
    for state in states:
        if state.latest and not state.within_ceiling:
            lines.append(
                f"Note: {state.name} {state.latest} is above the ceiling the "
                f"requirements set ({state.requirement.constraint})."
            )
    return "\n".join(lines)

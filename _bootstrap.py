"""Re-exec an entry shim with the repo's venv when the deps are missing.

``./generate.py`` runs under the system python3 (``#!/usr/bin/env python3``),
which has no typer. The documented flow is ``source venv/bin/activate``, but
the bare command should work too: when the ``eyegen`` import fails and a repo
``venv/`` exists, the shim re-execs itself with ``venv/bin/python``. It never
loops — once the venv interpreter is the running one, the guard is a no-op.
"""

import os
import sys
from pathlib import Path


def reexec_into_repo_venv() -> None:
    """Replace this process with the repo venv's interpreter, if there is one."""
    venv_python = Path(__file__).resolve().parent / "venv" / "bin" / "python"
    if not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), sys.argv[0], *sys.argv[1:]])  # noqa: S606

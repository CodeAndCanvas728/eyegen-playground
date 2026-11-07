#!/usr/bin/env python3
"""EyeGen CLI entry-point shim."""

if __name__ == "__main__":
    try:
        from eyegen.cli import main

        main()
    except ModuleNotFoundError:
        import _bootstrap

        _bootstrap.reexec_into_repo_venv()  # no-op when there is no repo venv
        raise

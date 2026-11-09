#!/usr/bin/env python3
"""EyeGen GUI entry-point shim."""

if __name__ == "__main__":
    try:
        # eyegen.gui imports lazily, so import the module that pulls PySide6
        # here, inside the guarded try.
        from eyegen.gui.main import main
    except ModuleNotFoundError:
        import _bootstrap

        _bootstrap.reexec_into_repo_venv()  # no-op when there is no repo venv
        raise

    # Outside the guarded try: a ModuleNotFoundError raised lazily during
    # main() must not re-exec and run the command twice.
    main()

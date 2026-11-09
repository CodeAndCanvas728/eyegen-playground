#!/usr/bin/env python3
"""EyeGen CLI entry-point shim."""

if __name__ == "__main__":
    try:
        from eyegen.cli import main
    except ModuleNotFoundError:
        import _bootstrap

        _bootstrap.reexec_into_repo_venv()  # no-op when there is no repo venv
        raise

    # Outside the guarded try: a ModuleNotFoundError raised lazily during
    # main() must not re-exec and run the command twice.
    main()

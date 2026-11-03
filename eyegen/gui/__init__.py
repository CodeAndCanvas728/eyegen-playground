"""EyeGen PySide6 GUI."""

__all__ = ["main"]


def main():
    """Start the GUI.

    Imported here rather than at the top of the package so that the pure
    helpers beside it — persisting and restoring state — can be read and
    tested on a machine where PySide6 is not installed.
    """
    from eyegen.gui.main import main as _main

    return _main()

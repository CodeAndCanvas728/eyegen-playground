"""Bonsai/CoreML-specific background worker threads for the EyeGen GUI."""

import subprocess

from PySide6.QtCore import QThread, Signal

from eyegen.backends import bonsai, coreml


class _ScriptWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            r = subprocess.run(  # noqa: S603
                [self.script_path],
                capture_output=True,
                text=True,
                timeout=900.0,
            )
            if r.returncode == 0:
                self.finished.emit(True, self._success_message())
            else:
                msg = (r.stderr or r.stdout or "Unknown error").strip().splitlines()[-5:]
                self.finished.emit(False, f"Setup failed (exit {r.returncode}): " + "\n".join(msg))
        except subprocess.TimeoutExpired as exc:
            self.finished.emit(
                False,
                f"Setup error: Script timed out after {exc.timeout} seconds. "
                "Please check your internet connection or run the script manually."
            )
        except (OSError, ValueError, RuntimeError) as exc:
            self.finished.emit(False, f"Setup error: {exc}")

    def _success_message(self) -> str:
        return "Script completed"


class BonsaiSetupWorker(_ScriptWorker):
    def _success_message(self) -> str:
        return "Bonsai installed successfully"


class BonsaiDownloadWorker(QThread):
    finished = Signal(bool, str)
    status = Signal(str)

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant

    def run(self):
        try:
            ok = bonsai.download_bonsai_model(
                self.variant,
                progress_callback=lambda m: self.status.emit(m),
            )
            if ok:
                self.finished.emit(True, f"Bonsai model '{self.variant}' is ready")
            else:
                self.finished.emit(False, f"Failed to download bonsai variant '{self.variant}'")
        except (OSError, ValueError, RuntimeError) as exc:
            self.finished.emit(False, f"Download error: {exc}")


class CoreMLSetupWorker(_ScriptWorker):
    def _success_message(self) -> str:
        return "CoreML sidecar venv installed"


class CoreMLDownloadWorker(QThread):
    finished = Signal(bool, str)
    status = Signal(str)

    def __init__(self, alias: str):
        super().__init__()
        self.alias = alias

    def run(self):
        try:
            target = coreml.pull_preconverted_coreml_model(
                self.alias,
                progress_callback=lambda m: self.status.emit(m),
            )
            if target:
                self.finished.emit(True, f"CoreML model downloaded to {target}")
            else:
                self.finished.emit(False, f"Failed to download CoreML model '{self.alias}'")
        except (OSError, ValueError, RuntimeError) as exc:
            self.finished.emit(False, f"Download error: {exc}")

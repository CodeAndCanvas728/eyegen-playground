"""Bonsai/CoreML/Idol-specific background worker threads for the EyeGen GUI."""

import subprocess
import threading

from PySide6.QtCore import QThread, Signal

from eyegen.backends import bonsai, idol


class _ScriptWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, script_path: str):
        super().__init__()
        self.script_path = script_path

    def run(self):
        try:
            from eyegen.config import load_config

            cfg = load_config()
            timeout = float(cfg.subprocess_timeout)
            r = subprocess.run([self.script_path], capture_output=True, text=True, timeout=timeout)  # noqa: S603
            if r.returncode == 0:
                self.finished.emit(True, self._success_message())
            else:
                msg = (r.stderr or r.stdout or "Unknown error").strip().splitlines()[-5:]
                self.finished.emit(False, f"Setup failed (exit {r.returncode}): " + "\n".join(msg))
        except subprocess.TimeoutExpired:
            self.finished.emit(False, "Setup timed out after 10 minutes.")
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


class To3DWorker(QThread):
    """Runs one image through the Idol sidecar (image → 3D Gaussian splat).

    The runner is kept on the instance so cancel() can terminate the
    subprocess mid-run. It is written by the worker thread and read by the
    GUI thread, so access goes through a lock.
    """

    finished = Signal(object, str)
    error = Signal(str)
    status = Signal(str)

    def __init__(self, image, out_dir, mode, steps, num_gaussians, seed, device, config):
        super().__init__()
        self.image = image
        self.out_dir = out_dir
        self.mode = mode
        self.steps = steps
        self.num_gaussians = num_gaussians
        self.seed = seed
        self.device = device
        self.config = config
        self._runner_lock = threading.Lock()
        self._runner = None

    def run(self):
        try:
            runner = idol.IdolRunner(self.config)
            with self._runner_lock:
                self._runner = runner
            result = runner.run(
                self.image,
                self.out_dir,
                mode=self.mode,
                steps=self.steps,
                num_gaussians=self.num_gaussians,
                seed=self.seed,
                device=self.device,
                progress_callback=lambda line: self.status.emit(line),
            )
            self.finished.emit(result, f"3D splat written: {result.ply}")
        except (RuntimeError, OSError, ValueError) as exc:
            self.error.emit(str(exc))

    def cancel(self):
        with self._runner_lock:
            runner = self._runner
        if runner is not None:
            runner.cancel()

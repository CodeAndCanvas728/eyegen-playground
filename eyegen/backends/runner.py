"""Common subprocess runner base class for EyeGen backends."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import suppress
from typing import Callable, List, Optional, Tuple

from eyegen.config import EyeGenConfig

log = logging.getLogger(__name__)


class BaseSubprocessRunner:
    """Base class for running and managing subprocesses with timeouts and cancellation."""

    def __init__(self, config: EyeGenConfig):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False
        self.timeout = float(config.subprocess_timeout)

    def _validate_cmd_args(self, cmd: List[str]) -> None:
        """Validate cmd arguments to prevent option/argument injection."""

        ALLOWED_FLAGS = {
            "-m",
            "--model",
            "--prompt",
            "--size",
            "--steps",
            "--output",
            "--seed",
            "--compute-unit",
            "--num-inference-steps",
            "--guidance-scale",
            "--image-height",
            "--image-width",
            "-i",
            "-o",
            "--negative-prompt",
            "--attention-implementation",
            "--quantize-nbits",
            "--convert-unet",
            "--convert-text-encoder",
            "--convert-vae-decoder",
            "--convert-safety-checker",
            "--model-version",
        }
        for arg in cmd[1:]:
            if arg.startswith("-") and arg not in ALLOWED_FLAGS:
                raise ValueError(f"Unsafe subprocess flag detected: {arg}")

    def _read_stream_target(
        self,
        stream,
        container: List[str],
        exit_event: threading.Event,
        log_prefix: str,
        progress_callback: Optional[Callable[[str], None]],
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                if exit_event.is_set():
                    break
                line_stripped = line.rstrip()
                log.info("[%s] %s", log_prefix, line_stripped)
                container.append(line)
                if progress_callback:
                    progress_callback(line_stripped)
        except (OSError, ValueError) as e:
            log.debug("Error reading %s stream: %s", log_prefix, e)
        finally:
            with suppress(OSError):
                stream.close()

    def _prep_env_and_pipes(
        self,
        env: Optional[dict],
        stream_stdout: bool,
        stream_stderr: bool,
    ) -> Tuple[dict, int, int]:
        run_env = env.copy() if env else os.environ.copy()
        run_env.pop("PYTHONHOME", None)
        stdout_pipe = subprocess.PIPE if stream_stdout else subprocess.DEVNULL
        stderr_pipe = subprocess.PIPE if stream_stderr else subprocess.DEVNULL
        return run_env, stdout_pipe, stderr_pipe

    def _execute_subprocess(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        stream_stdout: bool = True,
        stream_stderr: bool = True,
        log_prefix: str = "",
        progress_callback: Optional[Callable[[str], None]] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[int, List[str], List[str]]:
        """Run command as subprocess, stream logs, handle timeout/cancellation."""

        self._validate_cmd_args(cmd)
        self._cancelled = False
        run_timeout = timeout if timeout is not None else self.timeout

        run_env, stdout_pipe, stderr_pipe = self._prep_env_and_pipes(
            env, stream_stdout, stream_stderr
        )

        self._proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=cwd,
            env=run_env,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
            text=True,
            bufsize=1,
        )

        stdout_lines: List[str] = []
        stderr_lines: List[str] = []
        threads: List[threading.Thread] = []
        exit_event = threading.Event()

        if stream_stdout and self._proc.stdout:
            t_out = threading.Thread(
                target=self._read_stream_target,
                args=(
                    self._proc.stdout,
                    stdout_lines,
                    exit_event,
                    log_prefix,
                    progress_callback,
                ),
                daemon=True,
            )
            t_out.start()
            threads.append(t_out)

        if stream_stderr and self._proc.stderr:
            t_err = threading.Thread(
                target=self._read_stream_target,
                args=(
                    self._proc.stderr,
                    stderr_lines,
                    exit_event,
                    log_prefix,
                    progress_callback,
                ),
                daemon=True,
            )
            t_err.start()
            threads.append(t_err)

        try:
            try:
                self._proc.wait(timeout=run_timeout)
            except subprocess.TimeoutExpired as exc:
                log.warning(
                    "%s subprocess timed out after %.1f seconds. Terminating pid=%s...",
                    log_prefix,
                    run_timeout,
                    self._proc.pid,
                )
                self._terminate_proc()
                raise RuntimeError(
                    f"{log_prefix} subprocess timed out after {run_timeout} seconds"
                ) from exc

            returncode = self._proc.returncode
        except (subprocess.SubprocessError, OSError):
            self._terminate_proc()
            raise
        finally:
            exit_event.set()
            # Explicitly close streams if wait timed out or failed to unblock reader threads
            if self._proc:
                for stream in (self._proc.stdout, self._proc.stderr):
                    if stream:
                        with suppress(OSError):
                            stream.close()
            self._proc = None
            for t in threads:
                t.join(timeout=1.0)

        if self._cancelled:
            raise RuntimeError(f"{log_prefix} generation was cancelled by the user.")

        return returncode, stdout_lines, stderr_lines

    def _terminate_proc(self, grace: float = 5.0) -> None:
        if self._proc is None:
            return
        proc = self._proc
        log.info("Terminating process pid=%s", proc.pid)
        try:
            proc.terminate()
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                log.warning(
                    "Process did not terminate within grace period, killing pid=%s",
                    proc.pid,
                )
                proc.kill()
                proc.wait(timeout=grace)
        except (subprocess.SubprocessError, OSError) as exc:
            log.warning("Error during process termination: %s", exc)

    def cancel(self) -> None:
        self._cancelled = True
        self._terminate_proc(grace=2.0)

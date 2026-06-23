"""Common subprocess runner base class for EyeGen backends."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
from typing import Optional, List, Tuple

log = logging.getLogger(__name__)


class BaseSubprocessRunner:
    """Base class for running and managing subprocesses with timeouts and cancellation."""

    def __init__(self, config: dict):
        self.config = config
        self._proc: Optional[subprocess.Popen] = None
        self._cancelled = False
        # Make timeout configurable. Default to 300.0.
        self.timeout = float(config.get("subprocess_timeout", 300.0))

    def _execute_subprocess(
        self,
        cmd: List[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        stream_stdout: bool = True,
        stream_stderr: bool = True,
        log_prefix: str = "",
    ) -> Tuple[int, List[str], List[str]]:
        """Run a command as a subprocess, stream output to log, and handle timeouts and cancellation."""
        self._cancelled = False

        run_env = env.copy() if env else os.environ.copy()
        run_env.pop("PYTHONHOME", None)

        stdout_pipe = subprocess.PIPE if stream_stdout else subprocess.DEVNULL
        stderr_pipe = subprocess.PIPE if stream_stderr else subprocess.DEVNULL

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

        def read_stream(stream, container):
            try:
                for line in iter(stream.readline, ""):
                    if exit_event.is_set():
                        break
                    line_stripped = line.rstrip()
                    log.info("[%s] %s", log_prefix, line_stripped)
                    container.append(line)
            except Exception as e:
                log.debug("Error reading %s stream: %s", log_prefix, e)
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        if stream_stdout and self._proc.stdout:
            t_out = threading.Thread(
                target=read_stream,
                args=(self._proc.stdout, stdout_lines),
                daemon=True,
            )
            t_out.start()
            threads.append(t_out)

        if stream_stderr and self._proc.stderr:
            t_err = threading.Thread(
                target=read_stream,
                args=(self._proc.stderr, stderr_lines),
                daemon=True,
            )
            t_err.start()
            threads.append(t_err)

        try:
            try:
                self._proc.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as exc:
                log.warning(
                    "%s subprocess timed out after %.1f seconds. Terminating pid=%s...",
                    log_prefix,
                    self.timeout,
                    self._proc.pid,
                )
                self._terminate_proc()
                raise RuntimeError(
                    f"{log_prefix} subprocess timed out after {self.timeout} seconds"
                ) from exc

            returncode = self._proc.returncode
        except Exception:
            self._terminate_proc()
            raise
        finally:
            exit_event.set()
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
                log.warning("Process did not terminate within grace period, killing pid=%s", proc.pid)
                proc.kill()
                proc.wait(timeout=grace)
        except Exception as exc:
            log.warning("Error during process termination: %s", exc)

    def cancel(self) -> None:
        self._cancelled = True
        self._terminate_proc(grace=2.0)

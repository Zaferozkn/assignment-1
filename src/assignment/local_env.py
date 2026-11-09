"""Host-process execution backend for development without Modal.

This backend is intentionally for local development and smoke tests. It does
not provide the isolation or reproducibility of the assignment's Modal images.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any


class LocalEnvironment:
    """Execute agent commands on the host in a chosen working directory."""

    def __init__(self, cwd: str | Path = "."):
        self.cwd = Path(cwd).resolve()
        if not self.cwd.is_dir():
            raise FileNotFoundError(f"Local working directory does not exist: {self.cwd}")
        self.env_defaults: dict[str, str] = {}
        self.system = platform.system()
        self.release = platform.release()
        self.version = platform.version()
        self.machine = platform.machine()

    def execute(
        self,
        command: str | list[str],
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        shell: bool | None = True,
        check: bool = False,
    ) -> dict[str, Any]:
        """Run a host command and return the same result shape as Environment."""
        shell = True if shell is None else shell
        working_dir = Path(cwd).resolve() if cwd else self.cwd
        try:
            completed = subprocess.run(
                command,
                cwd=working_dir,
                env={**os.environ, **self.env_defaults, **(env or {})},
                shell=shell,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=check,
            )
            output = completed.stdout + completed.stderr
            return {
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "output": output,
                "returncode": completed.returncode,
                "exception_info": "",
            }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            detail = f"Command timed out after {timeout} seconds."
            return {
                "stdout": stdout,
                "stderr": stderr + detail,
                "output": stdout + stderr + detail,
                "returncode": -1,
                "exception_info": detail,
                "extra": {"exception_type": type(exc).__name__, "exception": detail},
            }
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            return {
                "stdout": "",
                "stderr": detail,
                "output": detail,
                "returncode": -1,
                "exception_info": detail,
                "extra": {"exception_type": type(exc).__name__, "exception": str(exc)},
            }

    def stop(self, timeout: float = 10) -> None:
        """Host processes are short-lived per command; nothing to tear down."""

    def __enter__(self) -> LocalEnvironment:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

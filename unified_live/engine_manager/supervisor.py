from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import ProtocolError, read_message, write_message


class BackendError(RuntimeError):
    pass


class BackendSupervisor:
    """One request at a time, framed child IPC, bounded wait and explicit failure."""

    def __init__(self, engine_id: str, kind: str, data_dir: Path, options: dict | None = None,
                 python: str | None = None, module: str | None = None, root: str | None = None,
                 timeout: float = 2.0, startup_timeout: float = 120.0):
        self.engine_id, self.kind = engine_id, kind
        self.data_dir = Path(data_dir)
        self.options = options or {}
        self.python = python or sys.executable
        self.module = module
        self.root = root
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        if timeout <= 0 or startup_timeout <= 0:
            raise ValueError("Timeouts must be positive")
        self._lock = threading.RLock()
        self._inbox: queue.Queue = queue.Queue(maxsize=2)
        self._process: subprocess.Popen | None = None
        self._sequence = 0
        self._generation = 0
        self.state = "stopped"
        self.error: str | None = None
        self.requests = 0
        self.restarts = 0
        self.last_latency_ms: float | None = None
        self.algorithmic_delay_ms = 0.0
        self.warming_up = False
        self.capabilities: dict = {}
        self.processed_faces: int | None = None

    def status(self) -> dict[str, Any]:
        # Snapshot must not wait behind a slow inference holding the request lock.
        p = self._process
        code = p.poll() if p else None
        if self.state == "running" and p and code is not None:
            self.state, self.error = "failed", f"Worker exited with code {code}"
        return {"state": self.state, "engine_id": self.engine_id,
                "pid": p.pid if p and code is None else None,
                "error": self.error, "requests": self.requests,
                "restarts": self.restarts, "last_latency_ms": self.last_latency_ms,
                "algorithmic_delay_ms": self.algorithmic_delay_ms, "warming_up": self.warming_up,
                "capabilities": dict(self.capabilities)}

    def _read_loop(self, process: subprocess.Popen, generation: int) -> None:
        try:
            while True:
                response = read_message(process.stdout)
                if generation == self._generation:
                    self._inbox.put(response, timeout=1)
        except Exception as exc:
            if generation == self._generation:
                try:
                    self._inbox.put_nowait(exc)
                except queue.Full:
                    pass

    def _stderr_loop(self, process: subprocess.Popen) -> None:
        logger = logging.getLogger(f"unified_live.worker.{self.kind}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logdir = self.data_dir / "logs"
            logdir.mkdir(parents=True, exist_ok=True)
            logger.addHandler(RotatingFileHandler(logdir / f"{self.kind}.log", maxBytes=1_000_000, backupCount=2))
        for line in process.stderr:
            logger.info("%s: %s", self.engine_id, line.decode("utf-8", "replace").rstrip()[:1000])

    def _terminate(self) -> None:
        p = self._process
        self._process = None
        self._generation += 1
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=1)
        if p:
            for stream in (p.stdin, p.stdout, p.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self._inbox = queue.Queue(maxsize=2)

    def _fail(self, error: str) -> None:
        self.state, self.error = "failed", error
        self._terminate()

    def _request(self, op: str, array: np.ndarray | None = None, timeout: float | None = None) -> tuple[dict, np.ndarray | None]:
        p = self._process
        if p is None or p.poll() is not None:
            self._fail("Worker is not running")
            raise BackendError(self.error)
        self._sequence += 1
        ident = self._sequence
        meta = {"id": ident, "kind": "control" if array is None else ("video" if self.kind == "face" else "audio"), "op": op}
        if op == "init":
            meta["options"] = self.options
        limit = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + limit
        writes: queue.Queue = queue.Queue(maxsize=1)

        def send() -> None:
            try:
                write_message(p.stdin, meta, array)
                writes.put_nowait(None)
            except Exception as exc:
                writes.put_nowait(exc)

        try:
            threading.Thread(target=send, daemon=True).start()
            sent = writes.get(timeout=max(0, deadline - time.monotonic()))
            if isinstance(sent, Exception):
                raise sent
            reply = self._inbox.get(timeout=max(0, deadline - time.monotonic()))
            if isinstance(reply, Exception):
                raise reply
            response, output = reply
            if response["id"] != ident:
                raise ProtocolError("Response ID mismatch")
            if response["kind"] == "error":
                raise BackendError(str(response.get("error", "Backend error")))
            expected = "control" if array is None else meta["kind"]
            if response["kind"] != expected:
                raise ProtocolError("Response kind mismatch")
            if array is not None and (output is None or output.shape != array.shape or output.dtype != array.dtype):
                raise ProtocolError("Response array format mismatch")
            self.requests += 1
            if "latency_ms" in response:
                self.last_latency_ms = float(response["latency_ms"])
                self.algorithmic_delay_ms = max(0.0, float(response.get("algorithmic_delay_ms", 0)))
                self.warming_up = bool(response.get("warming_up", False))
                self.processed_faces = response.get("processed_faces")
            if isinstance(response.get("capabilities"), dict):
                self.capabilities = response["capabilities"]
            return response, output
        except (queue.Empty, BrokenPipeError, OSError, EOFError, ProtocolError, BackendError) as exc:
            if isinstance(exc, queue.Empty):
                reason = "Worker timeout"
            elif isinstance(exc, EOFError):
                reason = f"Worker IPC closed (exit code {p.poll()})"
            else:
                reason = str(exc)
            self._fail(reason)
            raise BackendError(reason) from exc

    def start(self) -> None:
        with self._lock:
            if self.state == "running" and self._process and self._process.poll() is None:
                return
            self._terminate()
            self.state, self.error = "starting", None
            self.algorithmic_delay_ms = 0.0
            self.warming_up = False
            env = os.environ.copy()
            package_root = str(Path(__file__).resolve().parents[2])
            paths = [package_root]
            if self.root:
                paths.append(str(Path(self.root).resolve()))
            if env.get("PYTHONPATH"):
                paths.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(paths)
            try:
                p = subprocess.Popen([self.python, "-m", "unified_live.engine_manager.worker", self.engine_id, self.module or "-"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=self.root or None, env=env, bufsize=0)
                self._process = p
                generation = self._generation
                threading.Thread(target=self._read_loop, args=(p, generation), daemon=True).start()
                threading.Thread(target=self._stderr_loop, args=(p,), daemon=True).start()
                self._request("init", timeout=self.startup_timeout)
                self._request("start", timeout=self.startup_timeout)
                self.state, self.error = "running", None
            except Exception as exc:
                self._fail(str(exc))
                raise BackendError(str(exc)) from exc

    def process(self, array: np.ndarray) -> np.ndarray:
        with self._lock:
            if self.state != "running":
                raise BackendError(self.error or "Backend stopped")
            return self._request("process", array)[1]

    def health(self) -> bool:
        with self._lock:
            if self.state != "running":
                return False
            try:
                return bool(self._request("health")[0].get("running"))
            except BackendError:
                return False

    def reset(self) -> None:
        with self._lock:
            if self.state == "running":
                self._request("reset")

    def cancel(self) -> None:
        """Interrupt an in-flight request without waiting for the request lock."""
        p = self._process
        if p is None:
            return
        try:
            self._inbox.put_nowait(BackendError("Worker cancelled"))
        except queue.Full:
            pass
        if p.poll() is None:
            try:
                p.kill()
            except OSError:
                pass

    def stop(self) -> None:
        with self._lock:
            if self.state == "running":
                try:
                    self._request("stop")
                except BackendError:
                    pass
            self._terminate()
            self.state, self.error = "stopped", None

    def restart(self) -> None:
        with self._lock:
            self.stop()
            self.restarts += 1
            self.start()

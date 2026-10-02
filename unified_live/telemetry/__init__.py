"""Best-effort hardware telemetry. Missing measurements stay None."""

from __future__ import annotations

import threading
import time

import psutil


class Telemetry:
    def __init__(self, gpu: int = 0):
        self.gpu = gpu
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._state = {}
        self._thread = threading.Thread(target=self._run, name="telemetry", daemon=True)
        self._thread.start()

    def _run(self):
        nvml = None
        error = None
        try:
            import pynvml

            pynvml.nvmlInit()
            nvml = pynvml
        except Exception as exc:
            error = str(exc)
        psutil.cpu_percent()
        try:
            while not self._stop.is_set():
                mem = psutil.virtual_memory()
                state = {
                    "cpu_percent": psutil.cpu_percent(),
                    "ram_percent": mem.percent,
                    "ram_used_gb": mem.used / 1024**3,
                    "gpu_name": None,
                    "gpu_utilization": None,
                    "vram_used_gb": None,
                    "vram_total_gb": None,
                    "gpu_error": error,
                    "measured_at": time.time(),
                }
                if nvml:
                    try:
                        handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu)
                        name = nvml.nvmlDeviceGetName(handle)
                        vram = nvml.nvmlDeviceGetMemoryInfo(handle)
                        state.update(
                            gpu_name=name.decode() if isinstance(name, bytes) else name,
                            gpu_utilization=nvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                            vram_used_gb=vram.used / 1024**3,
                            vram_total_gb=vram.total / 1024**3,
                            gpu_error=None,
                        )
                    except Exception as exc:
                        state["gpu_error"] = str(exc)
                with self._lock:
                    self._state = state
                self._stop.wait(1.0)
        finally:
            if nvml:
                nvml.nvmlShutdown()

    def snapshot(self):
        with self._lock:
            return self._state.copy()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)

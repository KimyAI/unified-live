from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from unified_live.engines import EngineRegistry
from unified_live.profiles import ProfileStore
from unified_live.settings import SettingsStore

from .supervisor import BackendError, BackendSupervisor


class EngineManager:
    """Qt-free application API. A media runtime can attach behind these façades."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings = SettingsStore(self.data_dir / "settings.json")
        self.profiles = ProfileStore(self.data_dir / "profiles")
        self.registry = EngineRegistry(self.settings.current.backends)
        self._supervisors: dict[str, BackendSupervisor | None] = {"face": None, "voice": None}
        self._runtime: Any = None
        self._settings_lock = threading.RLock()
        self._kind_locks = {"face": threading.RLock(), "voice": threading.RLock()}

    def attach_runtime(self, runtime: Any) -> None:
        self._runtime = runtime

    def _kind(self, kind: str) -> None:
        if kind not in ("face", "voice"):
            raise ValueError("kind must be face or voice")

    def _selected(self, kind: str) -> tuple[str, dict]:
        settings = self.settings.current
        return getattr(settings, f"{kind}_engine"), getattr(settings, f"{kind}_options")

    def update_settings(self, patch: dict) -> dict:
        with self._settings_lock:
            old = self.settings.current
            for kind in ("face", "voice"):
                if f"{kind}_engine" in patch and self.registry.get(patch[f"{kind}_engine"]).kind != kind:
                    raise ValueError("Engine kind mismatch")
            updated = self.settings.update(patch)
            self.registry.backends = updated.backends
            changed = [kind for kind in ("face", "voice")
                       if (getattr(old, f"{kind}_engine"), getattr(old, f"{kind}_options"), old.backends.get(getattr(old, f"{kind}_engine")))
                       != (getattr(updated, f"{kind}_engine"), getattr(updated, f"{kind}_options"), updated.backends.get(getattr(updated, f"{kind}_engine")))]
        for kind in changed:
            with self._kind_locks[kind]:
                worker = self._supervisors[kind]
                if worker:
                    worker.stop()
                self._supervisors[kind] = None
        if old.voice_enabled != updated.voice_enabled:
            self.reset_backend("voice")
        if self._runtime and hasattr(self._runtime, "on_settings_changed"):
            self._runtime.on_settings_changed(old, updated)
        return updated.to_dict()

    def set_enabled(self, kind: str, enabled: bool) -> None:
        self._kind(kind)
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self.update_settings({f"{kind}_enabled": enabled})

    def select_engine(self, kind: str, engine_id: str, options: dict | None = None) -> None:
        self._kind(kind)
        descriptor = self.registry.get(engine_id)
        if descriptor.kind != kind:
            raise ValueError("Engine kind mismatch")
        if not descriptor.installed:
            raise BackendError(f"{engine_id} is planned and has no configured bridge")
        if options is not None and not isinstance(options, dict):
            raise ValueError("options must be an object")
        self.update_settings({f"{kind}_engine": engine_id, f"{kind}_options": options or {}})

    def _supervisor(self, kind: str) -> BackendSupervisor:
        current = self._supervisors[kind]
        engine_id, options = self._selected(kind)
        cfg = self.settings.current.backends.get(engine_id, {})
        effective_options = {**cfg.get("options", {}), **options}
        if engine_id == "seed-vc":
            effective_options.update(sample_rate=self.settings.current.sample_rate,
                                     chunk_size=self.settings.current.chunk_size)
        if engine_id in ("reswapper", "seed-vc"):
            effective_options.setdefault("gpu", self.settings.current.gpu)
        if current is not None and current.engine_id == engine_id and current.options == effective_options:
            return current
        if current is not None:
            current.stop()
        descriptor = self.registry.get(engine_id)
        if descriptor.kind != kind or not descriptor.installed:
            raise BackendError(f"Engine unavailable: {engine_id}")
        worker = BackendSupervisor(engine_id, kind, self.data_dir, effective_options,
                                   python=cfg.get("python"), module=cfg.get("module"), root=cfg.get("root"),
                                   timeout=cfg.get("processing_timeout", 2.0),
                                   startup_timeout=cfg.get("startup_timeout", 120.0))
        self._supervisors[kind] = worker
        return worker

    def start_backend(self, kind: str) -> dict:
        self._kind(kind)
        with self._kind_locks[kind]:
            worker = self._supervisor(kind)
            worker.start()
            return worker.status()

    def stop_backend(self, kind: str) -> dict:
        self._kind(kind)
        with self._kind_locks[kind]:
            worker = self._supervisors[kind]
            if worker:
                worker.stop()
            return self.backend_status(kind)

    def restart_backend(self, kind: str) -> dict:
        self._kind(kind)
        with self._kind_locks[kind]:
            worker = self._supervisor(kind)
            worker.restart()
            return worker.status()

    def backend_status(self, kind: str | None = None) -> dict:
        if kind is None:
            return {item: self.backend_status(item) for item in ("face", "voice")}
        self._kind(kind)
        worker = self._supervisors[kind]
        if worker:
            return worker.status()
        return {"state": "stopped", "engine_id": self._selected(kind)[0], "pid": None,
                "error": None, "requests": 0, "restarts": 0, "last_latency_ms": None}

    def health_backend(self, kind: str) -> bool:
        self._kind(kind)
        worker = self._supervisors[kind]
        return worker.health() if worker else False

    def reset_backend(self, kind: str) -> None:
        self._kind(kind)
        worker = self._supervisors[kind]
        if worker:
            worker.reset()

    def cancel_backend(self, kind: str) -> None:
        """Interrupt a blocked request so media shutdown can finish promptly."""
        self._kind(kind)
        worker = self._supervisors[kind]
        if worker:
            worker.cancel()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.settings.current.face_enabled:
            return frame
        with self._kind_locks["face"]:
            worker = self._supervisor("face")
            if worker.state == "stopped":
                worker.start()
        return worker.process(frame)

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.settings.current.voice_enabled:
            return audio
        with self._kind_locks["voice"]:
            worker = self._supervisor("voice")
            if worker.state == "stopped":
                worker.start()
        return worker.process(audio)

    def get_state(self) -> dict:
        return {"settings": self.settings.current.to_dict(), "engines": self.registry.list(),
                "backends": self.backend_status(), "runtime": self._runtime.get_state() if self._runtime else {}}

    def list_profiles(self) -> list[str]:
        return self.profiles.list()

    def save_profile(self, name: str) -> None:
        self.profiles.save(name, self.settings.current)

    def load_profile(self, name: str) -> dict:
        return self.update_settings(self.profiles.load(name).to_dict())

    def _delegate(self, method: str, default: Any = None, *args: Any) -> Any:
        if self._runtime is None:
            if method.startswith("start_"):
                raise RuntimeError("Media runtime not attached")
            return default
        return getattr(self._runtime, method)(*args)

    def start_video(self) -> Any:
        return self._delegate("start_video")

    def stop_video(self) -> Any:
        return self._delegate("stop_video")

    def start_audio(self) -> Any:
        return self._delegate("start_audio")

    def stop_audio(self) -> Any:
        return self._delegate("stop_audio")

    def discover_devices(self) -> dict:
        return self._delegate("discover_devices", {"cameras": [], "microphones": [], "outputs": [], "virtual_cameras": []})

    def latest_frame(self) -> np.ndarray | None:
        return self._delegate("latest_frame")

    def telemetry_snapshot(self) -> dict:
        return self._delegate("telemetry_snapshot", {})

    def run_benchmark(self, kind: str, iterations: int = 20) -> dict:
        if kind not in ("face", "voice", "synthetic"):
            raise ValueError("Invalid benchmark kind")
        if type(iterations) is not int or not 1 <= iterations <= 1000:
            raise ValueError("Invalid benchmark iterations")
        return self._delegate("run_benchmark", {}, kind, iterations)

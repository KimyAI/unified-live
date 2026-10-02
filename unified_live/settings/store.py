from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 1_000_000


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("JSON too large")
    name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as out:
            name = out.name
            os.chmod(name, 0o600)
            out.write(raw)
            out.flush()
            os.fsync(out.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def read_json(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON too large")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


@dataclass(frozen=True)
class Settings:
    version: int = 1
    app_name: str = "Unified Live"
    gpu: int = 0
    face_engine: str = "mock-face"
    voice_engine: str = "mock-voice"
    face_enabled: bool = False
    voice_enabled: bool = False
    camera_index: int | None = None
    microphone_index: int | None = None
    audio_output_index: int | None = None
    virtual_camera: bool = False
    virtual_audio: bool = False
    demo: bool = False
    width: int = 640
    height: int = 480
    fps: int = 30
    sample_rate: int = 48000
    chunk_size: int = 960
    auto_sync: bool = True
    video_offset_ms: float = 0.0
    audio_offset_ms: float = 0.0
    performance_profile: str = "balanced"
    advanced: bool = False
    face_options: dict[str, Any] = field(default_factory=dict)
    voice_options: dict[str, Any] = field(default_factory=dict)
    backends: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Settings:
        allowed = {f.name for f in fields(cls)}
        if set(raw) - allowed:
            raise ValueError(f"Unknown setting: {sorted(set(raw) - allowed)[0]}")
        value = cls(**raw)
        value.validate()
        return value

    def validate(self) -> None:
        if type(self.version) is not int or self.version != 1:
            raise ValueError("Unsupported settings version")
        for name in ("app_name", "face_engine", "voice_engine"):
            v = getattr(self, name)
            if not isinstance(v, str) or not v or len(v) > 128:
                raise ValueError(f"Invalid {name}")
        for name in ("face_enabled", "voice_enabled", "virtual_camera", "virtual_audio", "demo", "auto_sync", "advanced"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Invalid {name}")
        for name in ("gpu", "width", "height", "fps", "sample_rate", "chunk_size"):
            v = getattr(self, name)
            if type(v) is not int or v < 0:
                raise ValueError(f"Invalid {name}")
        if not (1 <= self.width <= 8192 and 1 <= self.height <= 8192 and 1 <= self.fps <= 240):
            raise ValueError("Invalid video format")
        if not (8000 <= self.sample_rate <= 384000 and 16 <= self.chunk_size <= 65536):
            raise ValueError("Invalid audio format")
        for name in ("camera_index", "microphone_index", "audio_output_index"):
            v = getattr(self, name)
            if v is not None and (type(v) is not int or v < 0):
                raise ValueError(f"Invalid {name}")
        for name in ("video_offset_ms", "audio_offset_ms"):
            v = getattr(self, name)
            if type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 5000:
                raise ValueError(f"Invalid {name}")
        if self.performance_profile not in {"balanced", "quality", "latency"}:
            raise ValueError("Invalid performance profile")
        for name in ("face_options", "voice_options", "backends"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"Invalid {name}")
        for engine_id, config in self.backends.items():
            if not isinstance(engine_id, str) or not engine_id or not isinstance(config, dict):
                raise ValueError("Invalid backend configuration")
            if set(config) - {"python", "module", "root", "options", "processing_timeout", "startup_timeout"}:
                raise ValueError("Unknown backend configuration field")
            for key in ("python", "module", "root"):
                if key in config and (not isinstance(config[key], str) or not config[key] or len(config[key]) > 1024):
                    raise ValueError(f"Invalid backend {key}")
            if "options" in config and not isinstance(config["options"], dict):
                raise ValueError("Invalid backend options")
            for key in ("processing_timeout", "startup_timeout"):
                if key in config and (type(config[key]) not in (int, float) or not math.isfinite(config[key]) or not 0 < config[key] <= 600):
                    raise ValueError(f"Invalid backend {key}")
        # Also validates that all nested options are JSON and bounded.
        if len(json.dumps(self.to_dict(), allow_nan=False)) > MAX_JSON_BYTES:
            raise ValueError("Settings too large")


class SettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.current = Settings()
        if self.path.exists():
            self.load()

    def load(self) -> Settings:
        self.current = Settings.from_dict(read_json(self.path))
        return self.current

    def save(self) -> None:
        self.current.validate()
        atomic_json(self.path, self.current.to_dict())

    def update(self, patch: dict[str, Any]) -> Settings:
        if not isinstance(patch, dict):
            raise ValueError("Settings patch must be an object")
        updated = Settings.from_dict({**self.current.to_dict(), **patch})
        atomic_json(self.path, updated.to_dict())
        self.current = updated
        return updated

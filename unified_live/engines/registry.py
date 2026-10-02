from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EngineDescriptor:
    id: str
    kind: str
    name: str
    description: str
    installed: bool = False
    available: bool = False
    configured: bool = False
    validated: bool = False
    planned: bool = True
    capabilities: dict[str, Any] = field(default_factory=dict)
    version: str | None = None
    requirements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EngineRegistry:
    def __init__(self, backends: dict[str, dict] | None = None):
        self.backends = backends or {}
        self._entries = {
            "mock-face": EngineDescriptor("mock-face", "face", "Mock color effect", "CPU demonstration, no face swap", True, True, True, True, False, {"cpu": True, "face_swap": False}),
            "mock-voice": EngineDescriptor("mock-voice", "voice", "Mock gain effect", "CPU demonstration, no voice conversion", True, True, True, True, False, {"cpu": True, "voice_conversion": False}),
            "reswapper": EngineDescriptor("reswapper", "face", "ReSwapper", "External bridge; requires local installation"),
            "seed-vc": EngineDescriptor("seed-vc", "voice", "Seed-VC", "External streaming bridge; requires local installation"),
            "rvc": EngineDescriptor("rvc", "voice", "RVC", "External model adapter; planned"),
            "deep-live-cam": EngineDescriptor("deep-live-cam", "face", "Deep-Live-Cam", "External adapter; planned"),
            "w-okada": EngineDescriptor("w-okada", "voice", "w-okada", "External host adapter; planned"),
        }

    def get(self, engine_id: str) -> EngineDescriptor:
        if engine_id not in self._entries:
            raise KeyError(f"Unknown engine: {engine_id}")
        base = self._entries[engine_id]
        if not base.planned:
            return EngineDescriptor(**{**asdict(base), "version": "0.1.0", "requirements": ("Core Python and NumPy",)})
        cfg = self.backends.get(engine_id, {})
        # Merely having a path is configuration, not validation or support.
        configured = isinstance(cfg, dict) and bool(cfg.get("python") and cfg.get("module"))
        installed = configured and Path(cfg["python"]).is_file() and (not cfg.get("root") or Path(cfg["root"]).is_dir())
        requirements = ("Separate Python environment", "External upstream checkout", "Local authorized model assets")
        implemented = base.id in {"reswapper", "seed-vc"}
        capabilities = {
            "reswapper": {"face_swap": True, "resolution": 128, "cuda": True, "experimental": True},
            "seed-vc": {"voice_conversion": True, "stateful": True, "sample_rate": 22050, "cuda": True, "experimental": True},
        }.get(base.id, base.capabilities)
        return EngineDescriptor(base.id, base.kind, base.name, base.description, installed, installed, configured,
                                False, not implemented, capabilities, None, requirements)

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [self.get(id).to_dict() for id, entry in self._entries.items() if kind is None or entry.kind == kind]

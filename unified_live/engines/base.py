from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Engine(ABC):
    kind: str
    capabilities: dict[str, Any] = {}

    def __init__(self) -> None:
        self.running = False
        self.source: str | None = None
        self.model: str | None = None
        self.last_latency_ms: float | None = None

    def get_latency(self) -> float | None:
        """Last measured inference duration in milliseconds, or unavailable."""
        return self.last_latency_ms

    def get_capabilities(self) -> dict[str, Any]:
        return dict(self.capabilities)

    def initialize(self, options: dict[str, Any]) -> None:
        if not isinstance(options, dict):
            raise ValueError("Options must be an object")

    def load_source(self, source: str | None) -> None:
        self.source = source

    def load_model(self, model: str | None) -> None:
        self.model = model

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def reset(self) -> None:
        """Clear streaming history without reloading models; stateless by default."""


class FaceEngine(Engine):
    kind = "face"

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray: ...


class VoiceEngine(Engine):
    kind = "voice"

    @abstractmethod
    def process_audio(self, audio: np.ndarray) -> np.ndarray: ...

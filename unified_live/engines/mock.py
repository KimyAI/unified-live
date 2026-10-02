from __future__ import annotations

import numpy as np

from .base import FaceEngine, VoiceEngine


class MockFaceEngine(FaceEngine):
    """Visible CPU demonstration effect; does not perform face inference."""

    capabilities = {"effect": "color-invert", "cpu": True, "face_swap": False}

    def initialize(self, options: dict) -> None:
        super().initialize(options)
        self.strength = float(options.get("strength", 1.0))
        if not 0 <= self.strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        self.load_source(options.get("source"))
        self.load_model(options.get("model"))

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.running:
            raise RuntimeError("Face engine stopped")
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected HxWx3 uint8 frame")
        return np.ascontiguousarray(np.rint(frame * (1 - self.strength) + (255 - frame) * self.strength).astype(np.uint8))


class MockVoiceEngine(VoiceEngine):
    """CPU demonstration gain; does not perform voice conversion."""

    capabilities = {"effect": "gain", "cpu": True, "voice_conversion": False}

    def initialize(self, options: dict) -> None:
        super().initialize(options)
        self.gain = float(options.get("gain", 0.75))
        if not 0 <= self.gain <= 4:
            raise ValueError("gain must be between 0 and 4")
        self.load_source(options.get("source"))
        self.load_model(options.get("model"))

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.running:
            raise RuntimeError("Voice engine stopped")
        if audio.dtype != np.float32 or audio.ndim not in (1, 2) or (audio.ndim == 2 and audio.shape[1] != 1):
            raise ValueError("Expected mono float32 audio")
        return np.ascontiguousarray(np.clip(audio * self.gain, -1, 1).astype(np.float32))

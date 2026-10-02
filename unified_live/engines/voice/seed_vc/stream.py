"""Pure, sequential streaming window and SOLA logic. No model dependencies."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import FRAME_20MS, MODEL_RATE, SeedVCConfig


class SeedStream:
    def __init__(self, config: SeedVCConfig,
                 infer: Callable[[np.ndarray], np.ndarray],
                 resample_16k: Callable[[np.ndarray], np.ndarray]):
        self.config = config
        self.infer = infer
        self.resample_16k = resample_16k
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self.input_wave = np.zeros(cfg.input_window_samples, dtype=np.float32)
        self.input_16k = np.zeros(320 * cfg.input_window_samples // FRAME_20MS, dtype=np.float32)
        self.sola_buffer = np.zeros(cfg.overlap_samples, dtype=np.float32)
        self.fade_in = np.sin(np.linspace(0, np.pi / 2, cfg.overlap_samples, dtype=np.float32)) ** 2
        self.fade_out = 1 - self.fade_in
        self.processed_blocks = 0
        self.last_sola_offset = 0
        self.is_warming_up = True

    def get_algorithmic_delay_ms(self) -> float:
        """Nominal content slice lag; model phoneme alignment remains unmeasured."""
        cfg = self.config
        return 1000 * (cfg.right_samples + cfg.overlap_samples + cfg.search_samples - self.last_sola_offset) / MODEL_RATE

    def _join(self, generated: np.ndarray) -> np.ndarray:
        cfg = self.config
        expected = cfg.return_frames * FRAME_20MS
        if generated.dtype != np.float32 or generated.ndim != 1 or len(generated) != expected or not np.isfinite(generated).all():
            raise ValueError(f"Seed-VC must return {expected} finite mono float32 samples")
        overlap, search, block = cfg.overlap_samples, cfg.search_samples, cfg.block_samples
        candidates = generated[:overlap + search]
        if np.any(self.sola_buffer):
            correlation = np.correlate(candidates, self.sola_buffer, mode="valid")
            energy = np.convolve(candidates * candidates, np.ones(overlap, dtype=np.float32), mode="valid")
            offset = int(np.argmax(correlation / np.sqrt(energy + 1e-8)))
        else:
            offset = 0
        aligned = generated[offset:]
        output = aligned[:block].copy()
        output[:overlap] = output[:overlap] * self.fade_in + self.sola_buffer * self.fade_out
        self.sola_buffer[:] = aligned[block:block + overlap]
        self.last_sola_offset = offset
        return np.ascontiguousarray(output)

    def process(self, audio: np.ndarray) -> np.ndarray:
        cfg = self.config
        if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1 or len(audio) != cfg.chunk_size:
            raise ValueError(f"Seed-VC input must be mono float32 of {cfg.chunk_size} samples at {MODEL_RATE} Hz")
        if not np.isfinite(audio).all():
            raise ValueError("Seed-VC audio contains non-finite samples")
        block, zc = cfg.block_samples, FRAME_20MS
        block_16k = cfg.block_frames * 320
        self.input_wave[:-block] = self.input_wave[block:].copy()
        self.input_wave[-block:] = audio
        self.input_16k[:-block_16k] = self.input_16k[block_16k:].copy()
        # Keep the extra 20 ms before each appended block, as in the pinned GUI.
        resampled = np.asarray(self.resample_16k(self.input_wave[-block - 2 * zc:]), dtype=np.float32)
        expected = (cfg.block_frames + 2) * 320
        if resampled.ndim != 1 or len(resampled) != expected or not np.isfinite(resampled).all():
            raise ValueError(f"16 kHz resampler must return {expected} samples")
        self.input_16k[-(cfg.block_frames + 1) * 320:] = resampled[320:]
        self.processed_blocks += 1
        self.is_warming_up = self.processed_blocks <= cfg.warmup_blocks
        if self.is_warming_up:
            return np.zeros(cfg.chunk_size, dtype=np.float32)
        result = np.asarray(self.infer(self.input_16k.copy()), dtype=np.float32)
        return self._join(result)

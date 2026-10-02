from __future__ import annotations

import math
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class SyncEngine:
    """Estimates host pipeline time before added delays, then delays the faster stream."""

    def __init__(self, alpha: float = 0.2, max_delay_ms: float = 500.0):
        if not 0 < alpha <= 1 or not math.isfinite(max_delay_ms) or max_delay_ms < 0:
            raise ValueError("Invalid sync parameters")
        self.alpha = alpha
        self.max_delay_ms = max_delay_ms
        self.reset()

    def reset(self) -> None:
        self.video_baseline_ms: float | None = None
        self.audio_baseline_ms: float | None = None

    def _sample(self, current: float | None, sample: float | None) -> float | None:
        if sample is None or not math.isfinite(sample) or sample < 0 or sample > 30000:
            return current
        return sample if current is None else current * (1 - self.alpha) + sample * self.alpha

    def update(self, video_ms: float | None, audio_ms: float | None, auto: bool = True,
               video_offset_ms: float = 0, audio_offset_ms: float = 0) -> dict:
        if not all(math.isfinite(x) and abs(x) <= 5000 for x in (video_offset_ms, audio_offset_ms)):
            raise ValueError("Invalid manual offset")
        self.video_baseline_ms = self._sample(self.video_baseline_ms, video_ms)
        self.audio_baseline_ms = self._sample(self.audio_baseline_ms, audio_ms)
        valid = self.video_baseline_ms is not None and self.audio_baseline_ms is not None
        video = video_offset_ms
        audio = audio_offset_ms
        if auto and valid:
            video += self.audio_baseline_ms - self.video_baseline_ms
        # Signed offsets are normalized; no stream can be played in the past.
        floor = min(video, audio)
        video = min(self.max_delay_ms, max(0.0, video - floor))
        audio = min(self.max_delay_ms, max(0.0, audio - floor))
        return {
            "video_delay_ms": video,
            "audio_delay_ms": audio,
            "video_baseline_ms": self.video_baseline_ms,
            "audio_baseline_ms": self.audio_baseline_ms,
            "valid": valid,
        }


class TimestampBuffer(Generic[T]):
    """Bounded media buffer scheduled from original monotonic capture timestamp."""

    def __init__(self, max_items: int = 8, max_age_ms: float = 1000):
        if max_items < 1 or max_age_ms <= 0:
            raise ValueError("Invalid buffer bounds")
        self.max_items = max_items
        self.max_age_s = max_age_ms / 1000
        self.items: deque[tuple[float, T]] = deque()
        self.dropped = 0

    def clear(self) -> None:
        self.items.clear()

    def push(self, timestamp: float, item: T) -> None:
        if not math.isfinite(timestamp):
            raise ValueError("Invalid timestamp")
        if self.items and timestamp < self.items[-1][0]:
            raise ValueError("Timestamps must be ordered")
        self.items.append((timestamp, item))
        while len(self.items) > self.max_items:
            self.items.popleft()
            self.dropped += 1

    def pop_ready(self, now: float, delay_ms: float = 0) -> T | None:
        if not math.isfinite(now) or not math.isfinite(delay_ms) or delay_ms < 0:
            raise ValueError("Invalid scheduling arguments")
        while self.items and now - self.items[0][0] > self.max_age_s:
            self.items.popleft()
            self.dropped += 1
        if self.items and self.items[0][0] + delay_ms / 1000 <= now:
            return self.items.popleft()[1]
        return None

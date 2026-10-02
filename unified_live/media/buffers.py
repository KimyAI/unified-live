"""Bounded, timestamped media queues with an injectable monotonic clock."""

from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


@dataclass
class Packet:
    captured_at: float
    ready_at: float
    payload: Any
    baseline_ms: float = 0.0


class VideoBuffer:
    def __init__(self, capacity=90):
        self._items = deque()
        self._lock = Lock()
        self.capacity = capacity
        self.dropped = 0

    def push(self, packet):
        with self._lock:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._items.append(packet)

    def pop_due(self, now: float, delay_ms: float):
        with self._lock:
            result = None
            while self._items and self._items[0].ready_at + delay_ms / 1000 <= now:
                if result is not None:
                    self.dropped += 1
                result = self._items.popleft()
            return result

    def clear(self):
        with self._lock:
            self._items.clear()


class AudioBuffer:
    """Preserve sample order across variable output callback sizes.

    Never block waiting for inference. A late packet is dropped; an early packet
    waits. Delay decreases may discard whole stale chunks, increases insert
    silence. Such adjustments are counted and can be audible.
    """

    def __init__(self, sample_rate: int, max_seconds=2.0):
        self.rate = sample_rate
        self.capacity = int(max_seconds * sample_rate)
        self._items = deque()
        self._samples = 0
        self._offset = 0
        self._lock = Lock()
        self.dropped = 0
        self.underruns = 0
        self.last_latency_ms = None

    def push(self, packet):
        packet.payload = np.asarray(packet.payload, dtype=np.float32).reshape(-1)
        with self._lock:
            while self._items and self._samples + len(packet.payload) > self.capacity:
                old = self._items.popleft()
                self._samples -= len(old.payload) - self._offset
                self._offset = 0
                self.dropped += 1
            if len(packet.payload) > self.capacity:
                self.dropped += 1
                return
            self._items.append(packet)
            self._samples += len(packet.payload)

    def read_into(self, output, now, delay_ms):
        output.fill(0)
        written = 0
        # The lock only covers bounded deque/copy operations, never IPC or I/O.
        with self._lock:
            while self._items and written < len(output):
                packet = self._items[0]
                due = packet.ready_at + delay_ms / 1000
                position_time = now + written / self.rate
                if self._offset == 0 and due > position_time:
                    silence = min(len(output) - written, int(np.ceil((due - position_time) * self.rate)))
                    written += silence
                    continue
                # Recover after startup/overload rather than retaining a stale
                # backlog forever. Skip stale samples, bounded to one output
                # callback of scheduling slack, including within a long chunk.
                lateness = position_time - (due + self._offset / self.rate)
                if lateness > len(output) / self.rate + 0.005:
                    skip = min(len(packet.payload) - self._offset, int(lateness * self.rate))
                    self._offset += skip
                    self._samples -= skip
                    self.dropped += 1
                    if self._offset == len(packet.payload):
                        self._items.popleft()
                        self._offset = 0
                    continue
                length = min(len(output) - written, len(packet.payload) - self._offset)
                output[written:written + length] = packet.payload[self._offset:self._offset + length]
                self.last_latency_ms = (position_time - packet.captured_at - self._offset / self.rate) * 1000
                self._offset += length
                self._samples -= length
                written += length
                if self._offset == len(packet.payload):
                    self._items.popleft()
                    self._offset = 0
            if written < len(output):
                self.underruns += 1

    def clear(self):
        with self._lock:
            self._items.clear()
            self._offset = self._samples = 0

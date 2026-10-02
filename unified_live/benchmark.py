"""Measured CPU/IPC or configured backend benchmarks; no reference-score tables."""

import platform
import time
import wave
from pathlib import Path

import numpy as np

from unified_live.engine_manager.supervisor import BackendSupervisor


def benchmark_manager(manager, kind="synthetic", iterations=30, input_path=None):
    if kind == "synthetic":
        return {
            "scenario": "synthetic input, real CPU mock workers and IPC",
            "face": benchmark_manager(manager, "mock-face", iterations),
            "voice": benchmark_manager(manager, "mock-voice", iterations),
        }
    if kind not in ("face", "voice", "mock-face", "mock-voice"):
        raise ValueError("Unknown benchmark kind")
    settings = manager.settings.current
    lane = "face" if kind.endswith("face") else "voice"
    engine_id = kind if kind.startswith("mock-") else getattr(settings, lane + "_engine")
    options = {} if kind.startswith("mock-") else getattr(settings, lane + "_options")
    cfg = settings.backends.get(engine_id, {})
    options = {**cfg.get("options", {}), **options}
    if engine_id == "seed-vc":
        options.update(sample_rate=settings.sample_rate, chunk_size=settings.chunk_size)
    if engine_id in {"reswapper", "seed-vc"}:
        options.setdefault("gpu", settings.gpu)
    if not engine_id.startswith("mock-") and input_path is None:
        raise ValueError("AI benchmarks require --benchmark-input with an authorized target image or PCM16 WAV")
    worker = BackendSupervisor(engine_id, lane, manager.data_dir, options,
                               python=cfg.get("python"), module=cfg.get("module"), root=cfg.get("root"),
                               timeout=cfg.get("processing_timeout", 2.0),
                               startup_timeout=cfg.get("startup_timeout", 120.0))
    rng = np.random.default_rng(0)
    if lane == "face":
        sample = rng.integers(0, 255, (settings.height, settings.width, 3), dtype=np.uint8)
    else:
        sample = rng.uniform(-0.1, 0.1, settings.chunk_size).astype(np.float32)
    samples = [sample]
    if input_path is not None:
        source = Path(input_path)
        if lane == "face":
            import cv2

            target = cv2.imread(str(source))
            if target is None:
                raise ValueError("Cannot decode benchmark target image")
            samples = [cv2.resize(target, (settings.width, settings.height))]
        else:
            with wave.open(str(source), "rb") as wav:
                if wav.getsampwidth() != 2 or wav.getnchannels() not in (1, 2) or wav.getframerate() != settings.sample_rate:
                    raise ValueError("Benchmark WAV must be mono/stereo PCM16 at the configured sample rate")
                audio = np.frombuffer(wav.readframes(min(wav.getnframes(), settings.sample_rate * 60)), dtype="<i2")
                audio = audio.astype(np.float32).reshape(-1, wav.getnchannels()).mean(axis=1) / 32768
            samples = [audio[i:i + settings.chunk_size] for i in range(0, len(audio) - settings.chunk_size + 1, settings.chunk_size)]
            if not samples:
                raise ValueError("Benchmark audio must contain at least one whole block")
    cursor = 0

    def next_sample():
        nonlocal cursor
        value = samples[cursor % len(samples)]
        cursor += 1
        return value

    times = []
    inference = []
    warmup = 0
    try:
        worker.start()
        while warmup < 3 or worker.warming_up:
            if warmup >= 500:
                raise RuntimeError("Backend is still warming up after 500 blocks; no timings reported")
            worker.process(next_sample())
            warmup += 1
        if getattr(worker, "processed_faces", None) == 0:
            raise ValueError("No target face detected in benchmark image; inference benchmark refused")
        for _ in range(iterations):
            begin = time.perf_counter()
            worker.process(next_sample())
            times.append((time.perf_counter() - begin) * 1000)
            if worker.last_latency_ms is not None:
                inference.append(worker.last_latency_ms)
    finally:
        worker.stop()
    result = {
        "engine": engine_id,
        "kind": lane,
        "input": "local target image" if input_path and lane == "face" else ("local PCM16 WAV, cycled in order" if input_path else "synthetic random array"),
        "host": platform.platform(),
        "python": platform.python_version(),
        "iterations": iterations,
        "shape": list(sample.shape),
        "warmup_excluded": warmup,
        "algorithmic_delay_ms": worker.algorithmic_delay_ms,
        "roundtrip_mean_ms": float(np.mean(times)),
        "roundtrip_p50_ms": float(np.percentile(times, 50)),
        "roundtrip_p95_ms": float(np.percentile(times, 95)),
        "inference_mean_ms": float(np.mean(inference)) if inference else None,
        "vram_gb": None,
    }
    if lane == "face":
        result["processing_fps"] = 1000 / np.mean(times)
    else:
        result["audio_block_ms"] = settings.chunk_size / settings.sample_rate * 1000
        result["realtime_factor"] = float(np.mean(times)) / result["audio_block_ms"]
    return result

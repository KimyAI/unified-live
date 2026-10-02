from unified_live.benchmark import benchmark_manager
from unified_live.engine_manager import EngineManager
import wave
import numpy as np
import pytest


def test_measured_mock_benchmark_reports_real_roundtrips(tmp_path):
    report = benchmark_manager(EngineManager(tmp_path), "synthetic", 3)
    for kind in ("face", "voice"):
        result = report[kind]
        assert result["engine"] == "mock-" + kind
        assert result["iterations"] == 3
        assert result["roundtrip_mean_ms"] > 0
        assert result["roundtrip_p95_ms"] >= result["roundtrip_p50_ms"]
        assert result["vram_gb"] is None


def test_ai_benchmark_requires_real_input(tmp_path):
    manager = EngineManager(tmp_path)
    manager.update_settings({"voice_engine": "seed-vc"})
    with pytest.raises(ValueError, match="benchmark-input"):
        benchmark_manager(manager, "voice", 3)


def test_voice_benchmark_uses_negotiated_format_and_excludes_stream_warmup(tmp_path, monkeypatch):
    captured = {}

    class Worker:
        def __init__(self, engine_id, lane, data_dir, options, **kwargs):
            captured.update(options=options, kwargs=kwargs)
            self.warming_up = True
            self.algorithmic_delay_ms = 60
            self.last_latency_ms = 1.0
            self.calls = 0

        def start(self):
            pass

        def process(self, sample):
            self.calls += 1
            self.warming_up = self.calls < 5
            captured["calls"] = self.calls
            assert sample.shape == (3969,)

        def stop(self):
            pass

    monkeypatch.setattr("unified_live.benchmark.BackendSupervisor", Worker)
    manager = EngineManager(tmp_path / "data")
    manager.update_settings({"voice_engine": "seed-vc", "sample_rate": 22050, "chunk_size": 3969,
                             "voice_options": {"cfg": .7}, "backends": {"seed-vc": {"options": {"diffusion_steps": 8}}}})
    path = tmp_path / "input.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(np.zeros(3969 * 2, dtype="<i2").tobytes())
    report = benchmark_manager(manager, "voice", 3, path)
    assert report["warmup_excluded"] == 5
    assert captured["calls"] == 8
    assert captured["options"]["chunk_size"] == 3969
    assert captured["options"]["diffusion_steps"] == 8
    assert captured["options"]["cfg"] == .7

import builtins
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from unified_live.engines.voice.seed_vc.bridge import create_engine
from unified_live.engines.voice.seed_vc.config import SeedVCConfig
from unified_live.engines.voice.seed_vc.stream import SeedStream
from unified_live.engine_manager import BackendError, BackendSupervisor


@pytest.fixture
def options(tmp_path):
    root = tmp_path / "seed"
    (root / "configs").mkdir(parents=True)
    (root / "real-time-gui.py").write_text("# fake source")
    (root / "configs" / "hifigan.yml").write_text("fake: true")
    xlsr = tmp_path / "xlsr"
    xlsr.mkdir()
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (xlsr / name).write_bytes(b"local test asset")
    paths = {}
    for name in ("checkpoint_path", "config_path", "reference_audio", "campplus_checkpoint", "hift_checkpoint"):
        path = tmp_path / name
        path.write_bytes(b"local test asset")
        paths[name] = str(path)
    return {"seed_root": str(root), "xlsr_model_dir": str(xlsr),
            "sample_rate": 22050, "chunk_size": 3969, **paths}


def resample_linear(wave):
    # Deterministic test adapter with the same exact 20 ms frame count.
    size = round(len(wave) / 441 * 320)
    return np.interp(np.linspace(0, len(wave) - 1, size), np.arange(len(wave)), wave).astype(np.float32)


def test_window_progression_warmup_and_exact_output(options):
    cfg = SeedVCConfig.from_options(options)
    windows = []

    def infer(window):
        windows.append(window.copy())
        return np.ones(cfg.return_frames * 441, dtype=np.float32)

    stream = SeedStream(cfg, infer, resample_linear)
    for block in range(cfg.warmup_blocks):
        output = stream.process(np.full(cfg.chunk_size, block / 100, dtype=np.float32))
        assert output.shape == (cfg.chunk_size,)
        assert not output.any()
        assert stream.is_warming_up
    assert windows == []
    output = stream.process(np.ones(cfg.chunk_size, dtype=np.float32))
    assert not stream.is_warming_up
    assert len(windows) == 1
    assert windows[0].shape == (320 * cfg.input_window_samples // 441,)
    assert output.dtype == np.float32 and len(output) == cfg.chunk_size
    assert stream.get_algorithmic_delay_ms() == pytest.approx(80)
    stream.process(np.zeros(cfg.chunk_size, dtype=np.float32))
    preserved = len(windows[0]) - (cfg.block_frames + 1) * 320
    np.testing.assert_array_equal(windows[1][:preserved], windows[0][cfg.block_frames * 320:cfg.block_frames * 320 + preserved])
    assert np.any(windows[1])  # prior speech survives a silent incoming block
    stream.reset()
    assert stream.is_warming_up and stream.processed_blocks == 0
    assert not stream.sola_buffer.any()


def test_sola_recovers_offset_and_keeps_overlap_continuous(options):
    cfg = SeedVCConfig.from_options(options)
    rng = np.random.default_rng(42)
    first = np.cumsum(rng.standard_normal(cfg.return_frames * 441)).astype(np.float32) * .01
    shift = 127
    second = np.zeros_like(first)
    second[shift:shift + cfg.overlap_samples] = first[cfg.block_samples:cfg.block_samples + cfg.overlap_samples]
    second[shift + cfg.overlap_samples:] = .2
    chunks = iter((first, second))
    stream = SeedStream(cfg, lambda _: next(chunks), resample_linear)
    stream.processed_blocks = cfg.warmup_blocks
    out1 = stream.process(np.zeros(cfg.chunk_size, dtype=np.float32))
    out2 = stream.process(np.zeros(cfg.chunk_size, dtype=np.float32))
    assert stream.last_sola_offset == shift
    assert abs(float(out2[0] - out1[-1])) < .1
    assert stream.get_algorithmic_delay_ms() == pytest.approx((cfg.right_samples + cfg.overlap_samples + cfg.search_samples - shift) / 22050 * 1000)


@pytest.mark.parametrize("patch,pattern", [
    ({"sample_rate": 48000}, "sample_rate=22050"),
    ({"chunk_size": 960}, "chunk_size=3969"),
    ({"content_context_left": .4, "context_left": .5}, "content_context_left"),
    ({"context_right": 0}, "context_right"),
    ({"block_time": .17}, "20 ms"),
    ({"diffusion_steps": 0}, "diffusion_steps"),
])
def test_format_and_context_constraints(options, patch, pattern):
    with pytest.raises(ValueError, match=pattern):
        SeedVCConfig.from_options({**options, **patch})


def test_missing_asset_fails_before_ml_import(options, monkeypatch):
    original_import = builtins.__import__
    attempted = []

    def guarded_import(name, *args, **kwargs):
        if name.startswith(("torch", "transformers", "librosa")):
            attempted.append(name)
            raise AssertionError("ML imported before preflight")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    engine = create_engine()
    with pytest.raises(FileNotFoundError, match="CAMPPlus"):
        engine.initialize({**options, "campplus_checkpoint": "/missing/campplus.bin"})
    assert attempted == []
    assert "torch" not in sys.modules or not attempted


def test_input_size_and_nonfinite_rejected(options):
    cfg = SeedVCConfig.from_options(options)
    stream = SeedStream(cfg, lambda _: np.zeros(cfg.return_frames * 441, dtype=np.float32), resample_linear)
    with pytest.raises(ValueError, match="3969"):
        stream.process(np.zeros(960, dtype=np.float32))
    block = np.zeros(cfg.chunk_size, dtype=np.float32)
    block[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        stream.process(block)


def test_sharded_xlsr_requires_all_local_shards(options):
    xlsr = Path(options["xlsr_model_dir"])
    (xlsr / "model.safetensors").unlink()
    (xlsr / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer": "part-1.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="XLS-R shard"):
        SeedVCConfig.from_options(options)
    (xlsr / "part-1.safetensors").write_bytes(b"local shard")
    assert SeedVCConfig.from_options(options).sample_rate == 22050


def test_seed_worker_fails_local_preflight_without_protocol_noise(options, tmp_path):
    worker = BackendSupervisor("seed-vc", "voice", tmp_path / "data", options={
        **options, "checkpoint_path": str(tmp_path / "absent.pth")},
        python=sys.executable, module="unified_live.engines.voice.seed_vc.bridge",
        root=options["seed_root"], startup_timeout=5)
    with pytest.raises(BackendError, match="Seed-VC checkpoint"):
        worker.start()
    assert worker.status()["state"] == "failed"
    assert worker.status()["pid"] is None


def test_upstream_python_and_native_stdout_are_redirected():
    code = ('import os\nfrom unified_live.engines.voice.seed_vc.bridge import _upstream_diagnostics\n'
            'with _upstream_diagnostics():\n'
            ' print("python diagnostic"); os.write(1, b"native diagnostic\\n")\n'
            'print("protocol output")')
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout == "protocol output\n"
    assert "python diagnostic" in result.stderr
    assert "native diagnostic" in result.stderr


def test_toggle_reset_clears_context_without_reloading_model(options):
    cfg = SeedVCConfig.from_options(options)
    engine = create_engine()
    engine.config = cfg
    engine.stream = SeedStream(cfg, lambda _: np.zeros(cfg.return_frames * 441, dtype=np.float32), resample_linear)
    engine.model_set = object()
    engine.reference = np.ones(22050, dtype=np.float32)
    engine.start()
    engine.stream.processed_blocks = cfg.warmup_blocks
    engine.process_audio(np.ones(cfg.chunk_size, dtype=np.float32))
    assert not engine.is_warming_up
    model, reference, stream = engine.model_set, engine.reference, engine.stream
    engine.reset()
    assert engine.is_warming_up
    assert engine.stream is stream and engine.model_set is model and engine.reference is reference
    assert not engine.stream.input_wave.any() and not engine.stream.sola_buffer.any()
    engine.stop()


def test_dynamic_source_and_model_changes_fail_explicitly():
    engine = create_engine()
    with pytest.raises(RuntimeError, match="reference_audio"):
        engine.load_source("new.wav")
    with pytest.raises(RuntimeError, match="checkpoint_path"):
        engine.load_model("new.pth")

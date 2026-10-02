"""Pinned Seed-VC realtime functions, loaded only by the selected voice worker."""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np

from unified_live.engines.base import VoiceEngine

from .config import MODEL_RATE, SeedVCConfig
from .stream import SeedStream


@contextlib.contextmanager
def _upstream_diagnostics():
    """Keep Python and native upstream prints away from protocol stdout."""
    sys.stdout.flush()
    original_stdout = os.dup(sys.stdout.fileno())
    try:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stderr.flush()
        os.dup2(original_stdout, sys.stdout.fileno())
        os.close(original_stdout)


class SeedVCVoiceEngine(VoiceEngine):
    """Stateful local audio conversion; no GUI, VAD, audio devices or auto-download."""

    capabilities = {"streaming": True, "experimental": True, "requires_cuda": True,
                    "sample_rate": MODEL_RATE, "mono": True}

    def __init__(self) -> None:
        super().__init__()
        self.config: SeedVCConfig | None = None
        self.stream: SeedStream | None = None
        self.module: Any = None
        self.model_set: Any = None
        self.device: Any = None
        self.reference: np.ndarray | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None

    def initialize(self, options: dict[str, Any]) -> None:
        super().initialize(options)
        # All local paths and formats are checked before Torch/HF can be imported.
        cfg = SeedVCConfig.from_options(options)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.chdir(cfg.seed_root)  # worker only; upstream opens relative configs/hifigan.yml
        if str(cfg.seed_root) in sys.path:
            sys.path.remove(str(cfg.seed_root))
        sys.path.insert(0, str(cfg.seed_root))
        with _upstream_diagnostics():
            import dotenv
            import librosa
            import torch
            import yaml

            if not torch.cuda.is_available() or cfg.gpu >= torch.cuda.device_count():
                raise RuntimeError(f"Seed-VC requires an available CUDA GPU at index {cfg.gpu}")
            # custom_infer uses default-device CUDA events and synchronize().
            torch.cuda.set_device(cfg.gpu)
            self.device = torch.device(f"cuda:{cfg.gpu}")
            config = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
            try:
                preprocess = config["preprocess_params"]
                tokenizer = config["model_params"]["speech_tokenizer"]
                vocoder = config["model_params"]["vocoder"]
                sr, hop = preprocess["sr"], preprocess["spect_params"]["hop_length"]
            except (KeyError, TypeError) as exc:
                raise ValueError("Unsupported Seed-VC config structure") from exc
            if sr != MODEL_RATE or hop != 256 or tokenizer.get("type") != "xlsr" or vocoder.get("type") != "hifigan":
                raise ValueError("Seed-VC bridge requires the pinned 22050 Hz XLS-R/HiFi-GAN realtime config")
            # from_pretrained receives a local directory, never a Hub repository ID.
            tokenizer["name"] = str(cfg.xlsr_model_dir)
            self._temporary = tempfile.TemporaryDirectory(prefix="unified-live-seed-vc-")
            local_config = Path(self._temporary.name) / "local-config.yml"
            local_config.write_text(yaml.safe_dump(config), encoding="utf-8")
            spec = importlib.util.spec_from_file_location("unified_live_seed_vc_pinned_realtime", cfg.seed_root / "real-time-gui.py")
            if spec is None or spec.loader is None:
                raise RuntimeError("Cannot load pinned Seed-VC realtime source")
            module = importlib.util.module_from_spec(spec)
            original_dotenv = dotenv.load_dotenv
            try:
                dotenv.load_dotenv = lambda *args, **kwargs: False
                spec.loader.exec_module(module)
            finally:
                dotenv.load_dotenv = original_dotenv
            module.device = self.device

            def local_hf_asset(repo: str, filename: str, config_filename: str | None = None) -> str:
                if repo == "funasr/campplus" and filename == "campplus_cn_common.bin" and config_filename is None:
                    return str(cfg.campplus_checkpoint)
                if repo == "FunAudioLLM/CosyVoice-300M" and filename == "hift.pt" and config_filename is None:
                    return str(cfg.hift_checkpoint)
                raise RuntimeError(f"Implicit Hub asset forbidden: {repo}/{filename}")

            module.load_custom_model_from_hf = local_hf_asset
            model_set = module.load_models(Namespace(checkpoint_path=str(cfg.checkpoint_path),
                                                     config_path=str(local_config), fp16=cfg.fp16))
            if model_set[-1].get("sampling_rate") != MODEL_RATE or model_set[-1].get("hop_size") != 256:
                raise ValueError("Seed-VC loaded model has an incompatible audio format")
            reference, _ = librosa.load(str(cfg.reference_audio), sr=MODEL_RATE, mono=True)
            reference = np.ascontiguousarray(reference, dtype=np.float32)
            if len(reference) < MODEL_RATE // 2 or not np.isfinite(reference).all():
                raise ValueError("Reference audio must contain at least 0.5 s of finite mono samples")
            self.module, self.model_set, self.reference = module, model_set, reference

        self.config = cfg
        self.source = str(cfg.reference_audio)
        self.model = str(cfg.checkpoint_path)

        def resample(wave: np.ndarray) -> np.ndarray:
            return librosa.resample(wave, orig_sr=MODEL_RATE, target_sr=16000).astype(np.float32)

        def infer(window_16k: np.ndarray) -> np.ndarray:
            with _upstream_diagnostics():
                tensor = torch.from_numpy(window_16k).to(self.device)
                output = self.module.custom_infer(
                    self.model_set, self.reference, str(cfg.reference_audio), tensor,
                    cfg.block_frames * 320, cfg.ce_frames, cfg.right_frames,
                    cfg.return_frames, cfg.diffusion_steps, cfg.cfg,
                    cfg.max_prompt_length, (cfg.ce_frames - cfg.dit_frames) / 50,
                )
                return output.detach().to("cpu", dtype=torch.float32).numpy().astype(np.float32, copy=False)

        self.stream = SeedStream(cfg, infer, resample)
        # First CUDA execution and reference conditioning belong to init's longer
        # timeout. This does not consume live input or advance the stream clock.
        cold = infer(np.zeros(320 * cfg.input_window_samples // 441, dtype=np.float32))
        if cold.ndim != 1 or len(cold) != cfg.return_frames * 441:
            raise ValueError("Seed-VC cold inference returned an incompatible block")

    def start(self) -> None:
        if self.stream is None:
            raise RuntimeError("Seed-VC is not initialized")
        super().start()

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.running or self.stream is None:
            raise RuntimeError("Seed-VC is stopped")
        return self.stream.process(audio)

    @property
    def is_warming_up(self) -> bool:
        return self.stream is None or self.stream.is_warming_up

    def get_algorithmic_delay_ms(self) -> float:
        return self.stream.get_algorithmic_delay_ms() if self.stream else 0.0

    def reset(self) -> None:
        """Clear live context/SOLA while keeping weights and reference loaded."""
        if self.stream is None:
            raise RuntimeError("Seed-VC is not initialized")
        self.stream.reset()

    def load_source(self, source: str | None) -> None:
        raise RuntimeError("Changing Seed-VC reference requires selecting the engine again with reference_audio")

    def load_model(self, model: str | None) -> None:
        raise RuntimeError("Changing Seed-VC checkpoint requires selecting the engine again with checkpoint_path")

    def stop(self) -> None:
        super().stop()
        if self.stream:
            self.reset()
        if self._temporary:
            self._temporary.cleanup()
            self._temporary = None


def create_engine() -> SeedVCVoiceEngine:
    return SeedVCVoiceEngine()

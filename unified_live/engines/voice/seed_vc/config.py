"""Pure configuration and asset checks; no ML imports or network access."""
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_RATE = 22_050
FRAME_20MS = MODEL_RATE // 50  # 441 samples


def _number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"Invalid {name}: expected {minimum}..{maximum}")
    return float(value)


def _frames(value: Any, name: str, minimum: float, maximum: float) -> int:
    seconds = _number(value, name, minimum, maximum)
    frames = round(seconds * 50)
    if abs(seconds - frames / 50) > 1e-6:
        raise ValueError(f"{name} must be a multiple of 20 ms")
    return frames


def _file(path: str | Path, name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file() or value.stat().st_size == 0:
        raise FileNotFoundError(f"Missing local {name}: {value}")
    return value


def _directory(path: str | Path, name: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"Missing local {name}: {value}")
    return value


def _xlsr_weights(directory: Path) -> None:
    for single in ("model.safetensors", "pytorch_model.bin"):
        path = directory / single
        if path.is_file() and path.stat().st_size > 0:
            return
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / index_name
        if not index.is_file():
            continue
        value = json.loads(index.read_text(encoding="utf-8"))
        mapping = value.get("weight_map") if isinstance(value, dict) else None
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Invalid XLS-R shard index")
        for shard in set(mapping.values()):
            if not isinstance(shard, str) or Path(shard).name != shard:
                raise ValueError("Invalid XLS-R shard path")
            _file(directory / shard, "XLS-R shard")
        return
    raise FileNotFoundError(f"Missing local XLS-R model weights: {directory}")


@dataclass(frozen=True)
class SeedVCConfig:
    seed_root: Path
    checkpoint_path: Path
    config_path: Path
    reference_audio: Path
    campplus_checkpoint: Path
    hift_checkpoint: Path
    xlsr_model_dir: Path
    gpu: int
    fp16: bool
    sample_rate: int
    chunk_size: int
    block_frames: int
    crossfade_frames: int
    ce_frames: int
    dit_frames: int
    right_frames: int
    diffusion_steps: int
    cfg: float
    max_prompt_length: float
    warmup_blocks: int

    @property
    def block_samples(self) -> int:
        return self.block_frames * FRAME_20MS

    @property
    def overlap_samples(self) -> int:
        return min(self.crossfade_frames, 4) * FRAME_20MS

    @property
    def search_samples(self) -> int:
        return FRAME_20MS

    @property
    def right_samples(self) -> int:
        return self.right_frames * FRAME_20MS

    @property
    def input_window_samples(self) -> int:
        return (self.ce_frames + self.crossfade_frames + 1 + self.block_frames + self.right_frames) * FRAME_20MS

    @property
    def return_frames(self) -> int:
        return self.block_frames + self.overlap_samples // FRAME_20MS + 1

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> SeedVCConfig:
        if not isinstance(options, dict):
            raise ValueError("Seed-VC options must be an object")
        required = ("seed_root", "checkpoint_path", "config_path", "reference_audio",
                    "campplus_checkpoint", "hift_checkpoint", "xlsr_model_dir", "sample_rate", "chunk_size")
        missing = [key for key in required if key not in options]
        if missing:
            raise ValueError("Missing Seed-VC option: " + missing[0])
        allowed = set(required) | {"gpu", "fp16", "block_time", "crossfade", "content_context_left",
                                   "context_left", "context_right", "diffusion_steps", "cfg", "max_prompt_length"}
        extra = set(options) - allowed
        if extra:
            raise ValueError("Unknown Seed-VC option: " + sorted(extra)[0])
        block_frames = _frames(options.get("block_time", 0.18), "block_time", .1, 1.0)
        crossfade_frames = _frames(options.get("crossfade", .04), "crossfade", .02, .08)
        ce_frames = _frames(options.get("content_context_left", 2.5), "content_context_left", .2, 10)
        dit_frames = _frames(options.get("context_left", .5), "context_left", 0, 5)
        right_frames = _frames(options.get("context_right", .02), "context_right", .02, .5)
        if ce_frames < dit_frames:
            raise ValueError("content_context_left must be >= context_left")
        if right_frames > block_frames:
            raise ValueError("context_right must fit within one block")
        sample_rate = options["sample_rate"]
        chunk_size = options["chunk_size"]
        if type(sample_rate) is not int or sample_rate != MODEL_RATE:
            raise ValueError("Seed-VC requires sample_rate=22050")
        if type(chunk_size) is not int or chunk_size != block_frames * FRAME_20MS:
            raise ValueError(f"Seed-VC requires chunk_size={block_frames * FRAME_20MS} for the chosen block_time")
        gpu = options.get("gpu", 0)
        if type(gpu) is not int or gpu < 0:
            raise ValueError("gpu must be a nonnegative integer")
        fp16 = options.get("fp16", True)
        if type(fp16) is not bool:
            raise ValueError("fp16 must be boolean")
        steps = options.get("diffusion_steps", 10)
        if type(steps) is not int or not 1 <= steps <= 50:
            raise ValueError("diffusion_steps must be 1..50")
        cfg = _number(options.get("cfg", .7), "cfg", 0, 1)
        prompt = _number(options.get("max_prompt_length", 3), "max_prompt_length", .5, 20)
        root = _directory(options["seed_root"], "Seed-VC checkout")
        _file(root / "real-time-gui.py", "realtime source")
        _file(root / "configs" / "hifigan.yml", "HiFi-GAN config")
        xlsr = _directory(options["xlsr_model_dir"], "XLS-R model")
        _file(xlsr / "config.json", "XLS-R config")
        _file(xlsr / "preprocessor_config.json", "XLS-R feature extractor")
        _xlsr_weights(xlsr)
        warmup = math.ceil((ce_frames + crossfade_frames + 1 + right_frames) / block_frames)
        return cls(root, _file(options["checkpoint_path"], "Seed-VC checkpoint"),
                   _file(options["config_path"], "Seed-VC YAML"),
                   _file(options["reference_audio"], "reference audio"),
                   _file(options["campplus_checkpoint"], "CAMPPlus checkpoint"),
                   _file(options["hift_checkpoint"], "HiFT checkpoint"), xlsr,
                   gpu, fp16, sample_rate, chunk_size, block_frames, crossfade_frames,
                   ce_frames, dit_frames, right_frames, steps, cfg, prompt, warmup)

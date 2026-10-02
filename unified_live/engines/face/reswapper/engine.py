"""Thin adapter around a separately installed, pinned ReSwapper checkout."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np

from unified_live.engines.base import FaceEngine


@dataclass(frozen=True)
class _Runtime:
    torch: Any
    cv2: Any
    onnxruntime: Any
    face_analysis: Any
    model_class: Any
    image: ModuleType
    face_align: ModuleType


_REQUIRED_UPSTREAM = ("StyleTransferModel_128.py", "Image.py", "face_align.py", "emap.npy")
_REQUIRED_FACE_MODELS = ("det_10g.onnx", "w600k_r50.onnx")


def _load_upstream_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Impossible de charger le helper upstream {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_runtime(checkout: Path) -> _Runtime:
    """Import only the three required helpers; never import upstream swap.py."""
    import torch
    import cv2
    import onnxruntime
    from insightface.app import FaceAnalysis

    suffix = hashlib.sha256(str(checkout).encode()).hexdigest()[:12]
    model_module = _load_upstream_module(
        checkout / "StyleTransferModel_128.py", f"_unified_reswapper_model_{suffix}"
    )
    # Image.py loads emap.npy relative to cwd at import time. Import it with cwd
    # rooted at the validated checkout, then restore cwd before inference starts.
    old_cwd = Path.cwd()
    try:
        os.chdir(checkout)
        image_module = _load_upstream_module(checkout / "Image.py", f"_unified_reswapper_image_{suffix}")
    finally:
        os.chdir(old_cwd)
    align_module = _load_upstream_module(
        checkout / "face_align.py", f"_unified_reswapper_align_{suffix}"
    )
    return _Runtime(
        torch, cv2, onnxruntime, FaceAnalysis, model_module.StyleTransferModel, image_module, align_module
    )


class ReSwapperEngine(FaceEngine):
    """Single-face 128px inference using helpers from a local ReSwapper checkout."""

    capabilities = {"face_swap": True, "resolution": 128, "device": "configured"}

    def __init__(self, runtime_loader: Callable[[Path], _Runtime] | None = None) -> None:
        super().__init__()
        self._runtime_loader = runtime_loader or _load_runtime
        self._runtime: _Runtime | None = None
        self._face_app: Any = None
        self._network: Any = None
        self._source_latent: Any = None
        self._device: Any = None
        self._options: dict[str, Any] = {}
        self.processed_faces: int | None = None

    @staticmethod
    def _path_option(options: dict[str, Any], key: str) -> Path:
        raw = options.get(key)
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"L’option '{key}' doit être un chemin local non vide")
        return Path(raw).expanduser().resolve()

    def _preflight(self, options: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
        checkout = self._path_option(options, "reswapper_root")
        source_path = self._path_option(options, "source_image")
        model_path = self._path_option(options, "model_path")
        insightface_root = self._path_option(options, "insightface_root")

        if not checkout.is_dir():
            raise FileNotFoundError(f"Checkout ReSwapper absent: {checkout}")
        for relative in _REQUIRED_UPSTREAM:
            if not (checkout / relative).is_file():
                raise FileNotFoundError(f"Fichier requis absent du checkout ReSwapper: {checkout / relative}")
        if not source_path.is_file():
            raise FileNotFoundError(f"Image source absente: {source_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Poids ReSwapper absents: {model_path}")
        if not insightface_root.is_dir():
            raise FileNotFoundError(f"Racine InsightFace absente: {insightface_root}")

        detector_name = options.get("detector_name", "buffalo_l")
        if not isinstance(detector_name, str) or not detector_name.strip() or Path(detector_name).name != detector_name:
            raise ValueError("detector_name doit être un nom de pack InsightFace simple")
        pack = insightface_root / "models" / detector_name
        missing = [str(pack / name) for name in _REQUIRED_FACE_MODELS if not (pack / name).is_file()]
        if missing:
            joined = ", ".join(missing)
            raise FileNotFoundError(
                "Pack InsightFace local incomplet (détection et reconnaissance requises): " + joined
            )
        return checkout, source_path, model_path, insightface_root

    def initialize(self, options: dict[str, Any]) -> None:
        super().initialize(options)
        resolution = options.get("resolution", 128)
        if resolution != 128:
            raise ValueError("Cette intégration prend en charge uniquement resolution=128")
        det_size = options.get("det_size", 512)
        if not isinstance(det_size, int) or isinstance(det_size, bool) or det_size <= 0:
            raise ValueError("det_size doit être un entier positif")
        gpu = options.get("gpu", 0)
        if not isinstance(gpu, int) or isinstance(gpu, bool) or gpu < 0:
            raise ValueError("gpu doit être un index entier positif ou nul")
        provider = options.get("execution_provider", "CUDAExecutionProvider")
        if not isinstance(provider, str) or provider not in (
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ):
            raise ValueError("execution_provider doit être CUDAExecutionProvider ou CPUExecutionProvider")

        # All filesystem checks happen before importing InsightFace, whose app
        # loader can otherwise fetch a missing model pack automatically.
        checkout, source_path, model_path, insightface_root = self._preflight(options)
        runtime = self._runtime_loader(checkout)
        if provider == "CUDAExecutionProvider":
            device = runtime.torch.device(f"cuda:{gpu}")
            ctx_id = gpu
        else:
            device = runtime.torch.device("cpu")
            ctx_id = -1

        available_providers = runtime.onnxruntime.get_available_providers()
        if provider not in available_providers:
            raise RuntimeError(
                f"Provider ONNX Runtime indisponible: {provider}; disponibles: "
                f"{', '.join(available_providers)}"
            )

        app = runtime.face_analysis(
            name=options.get("detector_name", "buffalo_l"),
            root=str(insightface_root),
            allowed_modules=["detection", "recognition"],
            providers=(
                [(provider, {"device_id": gpu})]
                if provider == "CUDAExecutionProvider"
                else [provider]
            ),
        )
        app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

        source_image = runtime.cv2.imread(str(source_path))
        if source_image is None:
            raise ValueError(f"OpenCV ne peut pas lire l’image source: {source_path}")
        source_faces = app.get(source_image)
        if not source_faces:
            raise ValueError(f"Aucun visage source détecté dans {source_path}")
        latent = runtime.image.getLatent(source_faces[0])
        source_tensor = runtime.torch.from_numpy(np.asarray(latent, dtype=np.float32)).to(device)

        network = runtime.model_class().to(device)
        state = runtime.torch.load(str(model_path), map_location=device, weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Le fichier ReSwapper doit contenir directement un state_dict")
        network.load_state_dict(state, strict=True)
        network.eval()

        self._runtime = runtime
        self._face_app = app
        self._network = network
        self._source_latent = source_tensor
        self._device = device
        self._options = dict(options)
        self.source = str(source_path)
        self.model = str(model_path)

    def start(self) -> None:
        if self._runtime is None or self._network is None or self._face_app is None:
            raise RuntimeError("ReSwapper doit être initialisé avant start()")
        super().start()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.running or self._runtime is None:
            raise RuntimeError("ReSwapper n’est pas démarré")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame doit être un tableau BGR H×W×3")
        if frame.dtype != np.uint8:
            raise ValueError("frame doit être en uint8")

        runtime = self._runtime
        faces = self._face_app.get(frame)
        self.processed_faces = min(1, len(faces))
        if not faces:
            return frame
        aligned, transform = runtime.face_align.norm_crop2(
            frame, faces[0].kps, self._options.get("resolution", 128)
        )
        blob = runtime.image.getBlob(aligned, (128, 128))
        target = runtime.torch.from_numpy(np.asarray(blob, dtype=np.float32)).to(self._device)
        with runtime.torch.inference_mode():
            result = self._network(target, self._source_latent)
        swapped = runtime.image.postprocess_face(result)
        return runtime.image.blend_swapped_image(swapped, frame, transform)

    def stop(self) -> None:
        super().stop()

    def _reload_with(self, key: str, value: str | None) -> None:
        if not self._options:
            raise RuntimeError("ReSwapper doit être initialisé avant de recharger une source ou un modèle")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"ReSwapper ne peut pas recharger un chemin vide: {key}")
        updated = dict(self._options)
        updated[key] = value
        was_running = self.running
        if was_running:
            self.stop()
        try:
            self.initialize(updated)
        except Exception:
            if was_running:
                self.start()
            raise
        if was_running:
            self.start()

    def load_source(self, source: str | None) -> None:
        self._reload_with("source_image", source)

    def load_model(self, model: str | None) -> None:
        self._reload_with("model_path", model)

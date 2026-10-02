from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from unified_live.engines.face.reswapper.engine import ReSwapperEngine, _Runtime
from unified_live.engines.face.reswapper.bridge import create_engine


class _Tensor:
    def __init__(self, value=None):
        self.value = value

    def to(self, device):
        self.device = device
        return self


class _Network:
    def __init__(self, calls):
        self.calls = calls

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state, strict):
        self.calls.append(("load", state, strict))

    def eval(self):
        self.calls.append(("eval",))

    def __call__(self, target, source):
        self.calls.append(("infer", target, source))
        return _Tensor("network-output")


class _FakeRuntime:
    def __init__(self, target_faces=None, available_providers=None):
        self.calls = []
        self.target_faces = target_faces
        self.app_get_count = 0
        self.image = SimpleNamespace(
            getLatent=lambda face: np.ones((1, 512), dtype=np.float32),
            getBlob=lambda image, size: np.ones((1, 3, 128, 128), dtype=np.float32),
            postprocess_face=lambda result: self.calls.append(("postprocess", result)) or "swapped",
            blend_swapped_image=lambda swapped, frame, matrix: self.calls.append(
                ("blend", swapped, matrix)
            ) or np.full_like(frame, 7),
        )
        self.face_align = SimpleNamespace(
            norm_crop2=lambda frame, kps, resolution: (
                np.zeros((128, 128, 3), dtype=np.uint8), "transform"
            )
        )

        class App:
            def prepare(inner_self, **kwargs):
                self.calls.append(("prepare", kwargs))

            def get(inner_self, image):
                self.app_get_count += 1
                if self.app_get_count == 1:
                    return [SimpleNamespace(normed_embedding=np.ones(512))]
                return self.target_faces

        self.app = App()

        class Torch:
            @staticmethod
            def device(name):
                return name

            @staticmethod
            def from_numpy(array):
                return _Tensor(array)

            @staticmethod
            def load(path, map_location, weights_only):
                self.calls.append(("torch-load", path, map_location, weights_only))
                return {"weights": 1}

            @staticmethod
            def inference_mode():
                return nullcontext()

        self.torch = Torch()
        self.onnxruntime = SimpleNamespace(
            get_available_providers=lambda: available_providers or [
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ]
        )

        class FaceAnalysis:
            def __new__(cls, **kwargs):
                self.calls.append(("face-analysis", kwargs))
                return self.app

        self.face_analysis = FaceAnalysis
        self.model_class = lambda: _Network(self.calls)
        self.cv2 = SimpleNamespace(imread=lambda path: np.zeros((16, 16, 3), dtype=np.uint8))

    def as_runtime(self):
        return _Runtime(
            self.torch,
            self.cv2,
            self.onnxruntime,
            self.face_analysis,
            self.model_class,
            self.image,
            self.face_align,
        )


def _assets(tmp_path: Path) -> dict:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("StyleTransferModel_128.py", "Image.py", "face_align.py", "emap.npy"):
        (checkout / name).touch()
    pack = tmp_path / "insightface" / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        (pack / name).touch()
    source = tmp_path / "source.png"
    model = tmp_path / "model.pth"
    source.touch()
    model.touch()
    return {
        "reswapper_root": str(checkout),
        "source_image": str(source),
        "model_path": str(model),
        "insightface_root": str(pack.parents[1]),
        "detector_name": "buffalo_l",
        "det_size": 512,
        "gpu": 0,
        "execution_provider": "CPUExecutionProvider",
        "resolution": 128,
    }


def test_bridge_factory_matches_worker_contract_without_loading_ml_runtime():
    assert isinstance(create_engine(), ReSwapperEngine)


def test_init_preflights_insightface_pack_before_runtime_loader(tmp_path):
    options = _assets(tmp_path)
    (Path(options["insightface_root"]) / "models" / "buffalo_l" / "w600k_r50.onnx").unlink()
    calls = []

    def loader(checkout):
        calls.append(checkout)
        raise AssertionError("runtime import must not happen before asset preflight")

    with pytest.raises(FileNotFoundError, match="détection et reconnaissance"):
        ReSwapperEngine(runtime_loader=loader).initialize(options)
    assert calls == []


def test_process_frame_uses_local_upstream_helpers_and_strict_weights(tmp_path):
    options = _assets(tmp_path)
    fake = _FakeRuntime(target_faces=[SimpleNamespace(kps=np.zeros((5, 2)))])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())
    engine.initialize(options)
    engine.start()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    output = engine.process_frame(frame)

    np.testing.assert_array_equal(output, np.full_like(frame, 7))
    assert ("load", {"weights": 1}, True) in fake.calls
    assert any(call[0] == "torch-load" and call[3] is True for call in fake.calls)
    assert any(call[0] == "infer" for call in fake.calls)
    assert any(call[0] == "blend" and call[2] == "transform" for call in fake.calls)


def test_cuda_provider_is_checked_and_gpu_index_is_forwarded(tmp_path):
    options = _assets(tmp_path)
    options.update(execution_provider="CUDAExecutionProvider", gpu=3)
    fake = _FakeRuntime(target_faces=[SimpleNamespace(kps=np.zeros((5, 2)))])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())

    engine.initialize(options)

    face_analysis_args = next(call[1] for call in fake.calls if call[0] == "face-analysis")
    assert face_analysis_args["providers"] == [
        ("CUDAExecutionProvider", {"device_id": 3})
    ]
    assert next(call[1] for call in fake.calls if call[0] == "prepare")["ctx_id"] == 3


def test_missing_onnx_provider_fails_before_face_analysis_creation(tmp_path):
    options = _assets(tmp_path)
    options["execution_provider"] = "CUDAExecutionProvider"
    fake = _FakeRuntime(available_providers=["CPUExecutionProvider"])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())

    with pytest.raises(RuntimeError, match="Provider ONNX Runtime indisponible"):
        engine.initialize(options)
    assert not any(call[0] == "face-analysis" for call in fake.calls)


def test_frame_without_detected_face_is_returned_unchanged(tmp_path):
    options = _assets(tmp_path)
    fake = _FakeRuntime(target_faces=[])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())
    engine.initialize(options)
    engine.start()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    output = engine.process_frame(frame)

    assert output is frame
    assert not any(call[0] == "infer" for call in fake.calls)


def test_only_128_resolution_is_accepted_before_import(tmp_path):
    options = _assets(tmp_path)
    options["resolution"] = 256
    calls = []
    with pytest.raises(ValueError, match="resolution=128"):
        ReSwapperEngine(runtime_loader=lambda path: calls.append(path)).initialize(options)
    assert calls == []


def test_source_reload_reinitializes_and_preserves_running_state(tmp_path):
    options = _assets(tmp_path)
    fake = _FakeRuntime(target_faces=[SimpleNamespace(kps=np.zeros((5, 2)))])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())
    engine.initialize(options)
    engine.start()
    replacement = tmp_path / "replacement.png"
    replacement.touch()

    engine.load_source(str(replacement))

    assert engine.running
    assert engine.source == str(replacement)
    assert engine._options["source_image"] == str(replacement)


def test_model_reload_reinitializes_and_preserves_running_state(tmp_path):
    options = _assets(tmp_path)
    fake = _FakeRuntime(target_faces=[SimpleNamespace(kps=np.zeros((5, 2)))])
    engine = ReSwapperEngine(runtime_loader=lambda checkout: fake.as_runtime())
    engine.initialize(options)
    engine.start()
    replacement = tmp_path / "replacement.pth"
    replacement.touch()

    engine.load_model(str(replacement))

    assert engine.running
    assert engine.model == str(replacement)
    assert engine._options["model_path"] == str(replacement)

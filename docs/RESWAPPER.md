# ReSwapper face backend

The adapter lives in Unified Live’s isolated worker and imports the helper files from a local ReSwapper checkout. It does not import the upstream command-line `swap.py`: that module constructs `FaceAnalysis` at import time and can select or download models before the application has checked its configuration. The adapter loads only `StyleTransferModel_128.py`, `Image.py`, and `face_align.py` after checking the local checkout and assets. `Image.py` reads `emap.npy` relative to the current directory, so it is imported briefly with the checkout as cwd and cwd is restored immediately.

The implementation was checked against ReSwapper commit [`44474cb8d85d15274ae274e5e0f5f44f0814af51`](https://github.com/somanchiu/ReSwapper/commit/44474cb8d85d15274ae274e5e0f5f44f0814af51). It calls the model and image helpers from that checkout; it does not copy their implementation into Unified Live. The separate `FaceAnalysis` instance is restricted to InsightFace detection and recognition.

## Worker configuration

The worker imports `module`, calls `create_engine()` with no arguments, then calls `initialize(options)`. Configure it like this in the backend settings (replace the example paths and Python executable with local paths):

```json
{
  "backends": {
    "reswapper": {
      "python": "D:\\UnifiedLive\\venvs\\reswapper\\Scripts\\python.exe",
      "module": "unified_live.engines.face.reswapper.bridge",
      "root": "D:\\models\\ReSwapper"
    }
  },
  "face_options": {
    "reswapper_root": "D:\\models\\ReSwapper",
    "source_image": "D:\\models\\faces\\source.png",
    "model_path": "D:\\models\\ReSwapper\\reswapper-1019500.pth",
    "insightface_root": "D:\\models\\insightface",
    "detector_name": "buffalo_l",
    "det_size": 512,
    "gpu": 0,
    "execution_provider": "CUDAExecutionProvider",
    "resolution": 128
  }
}
```

`root` is the upstream checkout used as the worker cwd and import search path; `reswapper_root` points to the same checkout for explicit validation. The application package path is also placed on `PYTHONPATH` by the supervisor, which makes the configured bridge module importable. The selected Python environment must contain Unified Live’s runtime dependencies and the ReSwapper worker dependencies: PyTorch, OpenCV, InsightFace, ONNX Runtime with the selected provider, and scikit-image (used by upstream `face_align.py`). No bridge code is copied into or installed in the upstream checkout.

`insightface_root` is the directory whose `models` child contains `buffalo_l`; the adapter checks for `models/buffalo_l/det_10g.onnx` and `models/buffalo_l/w600k_r50.onnx` before importing InsightFace. Keep the complete locally acquired pack at that path. A missing checkout helper, source image, `.pth`, `emap.npy`, or required ONNX file fails initialization before `FaceAnalysis` is constructed, preventing its missing-pack download path from being reached. The adapter never downloads model weights.

## Behavior and limits

The adapter requires a readable source image with at least one detected face and a local `.pth` containing the model `state_dict`. It loads that state with `torch.load(..., weights_only=True)` and `load_state_dict(..., strict=True)`, then keeps the source latent on the selected device. Each BGR `uint8` frame uses the first detected face, the upstream 128×128 alignment, blob and latent transforms, and the upstream paste-back blend. Frames without a detected face are returned unchanged. Only `resolution=128` is accepted because that is the only checkpoint/alignment combination verified for this adapter.

`execution_provider` may be `CUDAExecutionProvider` or `CPUExecutionProvider`. Startup checks that ONNX Runtime reports the requested provider before constructing InsightFace. CUDA selects `cuda:<gpu>` for PyTorch and passes `device_id=<gpu>` to ONNX Runtime’s InsightFace sessions; CPU selects PyTorch CPU and InsightFace context `-1`. There is no FP16, TensorRT, face enhancer, multi-face selection, or fallback from a failed CUDA provider. The adapter does not measure or promise a frame rate. GPU inference has not been run as part of this implementation; tests use fake runtime dependencies and temporary placeholder files, so they validate preflight and call flow rather than model compatibility or quality.

After initialization, `load_source(path)` and `load_model(path)` rebuild the backend with the changed local file and retain the running/stopped state. Empty paths and calls before initialization fail clearly; clearing either required input is unsupported.

The code license and each model’s terms are separate. Check the upstream repository, the provenance/license of `emap.npy`, the selected ReSwapper weights, and InsightFace model terms for the intended use before distributing or deploying those assets. The adapter source does not grant rights to those assets.

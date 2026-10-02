# Roadmap

Status legend: implemented, tested in CI/local CPU, needs Windows/GPU acceptance,
planned. Never replace a hardware gate with mock benchmark numbers.

## Phase 0 — upstream and design
- [x] Select original modular desktop architecture and process boundaries.
- [x] Write ARCHITECTURE.md before implementation.
- [x] Complete source/weight license audit with pinned upstream revisions.

## Phase 1 — model-free cockpit
- [x] PySide6 dark UI: LIVE, FACE, VOICE, TRAIN, PROFILES, DEVICES, PERFORMANCE,
      SETTINGS and backend management.
- [x] Stable interfaces, registry, EngineManager, private framed IPC, isolated
      workers, health checks, restart, timeout and crash handling.
- [x] Mock face/voice engines; independent camera/audio capture and ON/OFF bypass.
- [x] Camera preview/device selection; optional virtual video/audio output code
      (physical devices remain untested on this headless host).
- [x] JSON settings/profiles; measured telemetry, latency and bounded auto sync.
- [x] Separate logs; real measured benchmark command; Windows setup script
      (script execution still requires Windows).
- [x] Unit/process tests and real Qt offscreen smoke test on Linux.
- [ ] Windows 11 + physical webcam/mic + RTX 4090 acceptance.

Portable Phase 1 implementation is tested. Milestone 1 target-hardware acceptance
is **still open**: no camera, microphone, NVIDIA runtime or virtual driver is
available on this Linux development VM.

Milestone 1 acceptance: `python -m unified_live` opens a usable cockpit; synthetic
demo is available without devices, physical camera/mic work when installed, both
mock effects can toggle independently, profiles persist, telemetry reports actual
values or unavailable, delay queues compensate measured imbalance, and restarting
one backend does not stop the other. Hardware-only checks remain explicit.

## Phase 2 — ReSwapper
- [x] External checkout + independent venv bridge; local checkpoint/source only.
- [x] Face absence/error handling; parameters limited to verified capabilities.
- [x] Contract tests without weights (fake inference dependencies).
- [ ] Real Windows camera/virtual output test with authorized weights.

## Phase 3 — Seed-VC realtime
- [x] Experimental stateful external streaming bridge (context, crossfade/SOLA,
      strict 22050 Hz format, local assets, no implicit downloads).
- [x] Reference voice and diffusion/realtime controls through advanced options.
- [x] Stream continuity/history/reset/IPC tests without model weights.
- [ ] Real microphone/virtual audio acceptance with authorized weights.
- [ ] Combined ReSwapper + Seed-VC MVP on RTX 4090, measured A/V calibration.

Proceed from Phase 1 implementation to these bridges automatically. Do not claim
the GPU MVP is validated until real models and the target hardware are available.

## Phase 4 — RVC
Independent inference adapter, model/index library, f0/pitch/index ratio controls,
streaming buffer validation and real-device comparison.

## Phase 5 — RVC training
Supervised prepare/preprocess/f0/features/train/index jobs, cancel, progress/logs,
VRAM and elapsed time; validated exports enter the voice model library. Training
UI is a clearly marked planned feature until its job runner is implemented.

## Phase 6 — Deep-Live-Cam
Detect pinned external installation, wrap frame processing without launching its
GUI or taking capture ownership, discover actual swap/enhancer support. InSwapper,
HyperSwap and ReSwapper availability must follow that upstream revision.

## Phase 7 — w-okada
Separate host adapter over a verified local protocol; distinguish it from RVC,
validate rate/chunk semantics and selected model licensing.

## Phase 8 — RTX 4090 optimization
Measure first. Evaluate shared memory, GPU transfer costs, bounded pipelining,
FP16, CUDA/ORT providers, TensorRT only when supported, pinned memory and async
work. Compare quality, stability and sync as well as latency. Never invent FPS,
VRAM or backend rankings.

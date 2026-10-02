# Roadmap

Status legend: implemented, tested in CI/local CPU, needs Windows/GPU acceptance,
planned. Never replace a hardware gate with mock benchmark numbers.

## Phase 0 — upstream and design
- [x] Select original modular desktop architecture and process boundaries.
- [x] Write ARCHITECTURE.md before implementation.
- [ ] Complete source/weight license audit with pinned upstream revisions.

## Phase 1 — model-free cockpit
- [ ] PySide6 dark UI: LIVE, FACE, VOICE, TRAIN, PROFILES, DEVICES, PERFORMANCE,
      SETTINGS and backend management.
- [ ] Stable interfaces, registry, EngineManager, private framed IPC, isolated
      workers, health checks, restart, timeout and crash handling.
- [ ] Mock face/voice engines; independent camera/audio capture and ON/OFF bypass.
- [ ] Camera preview/device selection; optional virtual video/audio outputs.
- [ ] JSON settings/profiles; measured telemetry, latency and bounded auto sync.
- [ ] Separate logs; real measured benchmark command; Windows setup script.
- [ ] Unit/process tests and Qt smoke test.
- [ ] Windows 11 + physical webcam/mic + RTX 4090 acceptance.

Milestone 1 acceptance: `python -m unified_live` opens a usable cockpit; synthetic
demo is available without devices, physical camera/mic work when installed, both
mock effects can toggle independently, profiles persist, telemetry reports actual
values or unavailable, delay queues compensate measured imbalance, and restarting
one backend does not stop the other. Hardware-only checks remain explicit.

## Phase 2 — ReSwapper
- [ ] External checkout + independent venv bridge; local checkpoint/source only.
- [ ] Face absence/error handling; parameters limited to verified capabilities.
- [ ] Contract tests without weights, then real Windows camera/virtual output test.

## Phase 3 — Seed-VC realtime
- [ ] Stateful external streaming bridge (context, crossfade, rate negotiation).
- [ ] Reference voice and diffusion/realtime controls.
- [ ] Stream continuity/IPC tests, then real microphone/virtual audio acceptance.
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

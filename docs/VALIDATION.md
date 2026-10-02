# Validation record

This file records observed results, not intended behavior. The local development
host is Linux; Windows/RTX 4090 acceptance is a separate gate.

## Automated validation

Observed on 2026-10-02, Linux / Python 3.13.5:

- Editable package installation succeeds with the declared dependencies.
- 65 unit/integration tests pass, including real subprocess IPC, crash/timeout,
  independent starts, noisy native stdout, profiles/config, sync, media buffers,
  ReSwapper dependency doubles, Seed-VC streaming history/SOLA/reset/preflight,
  and Qt navigation/control tests.
- `python -m unified_live --smoke-seconds 5` opens the real Qt application and
  runs both mock subprocesses simultaneously; `smoke_passed: true`. Preview and
  audio progression are verified. Offscreen rendering is not a physical display test.
- `--benchmark synthetic --iterations 30` executes and reports measured mock
  CPU/IPC timings; no AI quality, RTX or GPU throughput claim follows.
- `--diagnose` correctly reports no camera and unavailable PortAudio/NVML on this
  VM; no virtual driver or physical inference test is claimed.

Local Qt runtime libraries were extracted under `/tmp` for offscreen verification;
the host's system libraries were not modified. The smoke run records audio startup
underruns/dropped stale samples; it is not a glitch-free hardware acceptance test.
The Python 3.11 Windows/Linux CI definition is stored in `docs/ci/tests.yml`.
GitHub rejected creation of `.github/workflows/tests.yml` because the publishing
OAuth connection lacks `workflow` scope. The code was published with the workflow
kept as an inactive template. No GitHub Actions run or Windows test is claimed.

![Qt mock cockpit captured during a real smoke run](images/cockpit-mock.png)

## Hardware acceptance checklist

- [ ] Windows 11 setup.ps1 and clean Python 3.11 installation.
- [ ] Physical camera enumeration, selection and preview.
- [ ] Physical microphone capture, independent of video.
- [ ] Both effects toggle while capture continues.
- [ ] OBS virtual camera visible in a receiving app.
- [ ] Virtual cable receives converted audio without feedback.
- [ ] RTX 4090 NVML name/utilization/VRAM reflect actual device.
- [ ] Real ReSwapper authorized source/checkpoint produces swapped frames.
- [ ] Real Seed-VC reference/model produces continuous converted voice.
- [ ] Combined load fits VRAM and sustains configured FPS/block duration.
- [ ] Clap/loopback calibration of actual output A/V delay.
- [ ] Kill either backend and verify the other stream continues.
- [ ] Load a complete profile and verify output/device/source settings.

## What the numbers mean

Mock benchmark numbers measure this host's CPU/IPC path. They are never RTX 4090,
ReSwapper or Seed-VC scores. Synthetic face input is not a valid likeness test.
Host timestamps omit unknown camera sensor and downstream consumer buffering.
No measured quality ranking or claim of universal best engine is made.

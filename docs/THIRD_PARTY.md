# Third-party notices and distribution policy

Unified Live contains original integration code under AGPL-3.0-only. No upstream
engine tree or AI checkpoint is included. The full upstream license review,
revisions and primary links are in [UPSTREAM_AUDIT.md](UPSTREAM_AUDIT.md).

Core installation pulls dependencies with their own license notices: PySide6/Qt
(LGPL/GPL/commercial terms depending on component), NumPy (BSD), OpenCV (Apache-2.0),
sounddevice (MIT, PortAudio separately), psutil (BSD), NVIDIA's Python NVML bindings
and pyvirtualcam (GPL). Review the installed distributions before creating an
installer or binary redistribution. This repository currently ships source only.

ReSwapper and Deep-Live-Cam code: AGPL-3.0. Seed-VC: GPL-3.0. RVC and w-okada
principal code: MIT with dependencies/embedded notices that require separate
review. DoppleDanger is conceptual inspiration only. No DoppleDanger code is copied.

OBS, VB-CABLE and VoiceMeeter are external installations. Their names describe
compatibility, not affiliation or redistribution permission. InsightFace pretrained
models have separate restrictions, including noncommercial research conditions
for its public model zoo. No automatic download is enabled by the core.

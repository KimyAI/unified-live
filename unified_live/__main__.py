"""Run the desktop, diagnostics, or an actual measured benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from unified_live.engine_manager import EngineManager
from unified_live.logging_setup import configure_logging


def default_data_dir():
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "UnifiedLive"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "unified-live"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Local modular face and voice cockpit")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--demo", action="store_true", help="Synthetic sources, no physical capture")
    parser.add_argument("--benchmark", choices=("synthetic", "face", "voice"))
    parser.add_argument("--benchmark-input", type=Path, help="Authorized target image or PCM16 WAV for AI benchmarks")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--diagnose", action="store_true", help="Report dependencies/devices without starting capture")
    parser.add_argument("--smoke-seconds", type=float, default=0, help="Timed synthetic UI smoke test")
    parser.add_argument("--screenshot", type=Path, help="Save the real Qt window before a timed smoke test closes")
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.iterations > 1000:
        parser.error("--iterations must be between 1 and 1000")
    configure_logging(args.data_dir)
    manager = EngineManager(args.data_dir)
    if not manager.settings.path.exists():
        manager.settings.save()
    if args.benchmark:
        from unified_live.benchmark import benchmark_manager

        print(json.dumps(benchmark_manager(manager, args.benchmark, args.iterations, args.benchmark_input), indent=2))
        return 0
    if args.demo or args.smoke_seconds:
        manager.update_settings({"demo": True, "virtual_audio": False, "virtual_camera": False})
    from unified_live.media.runtime import MediaRuntime

    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    if args.diagnose:
        try:
            print(json.dumps({"devices": manager.discover_devices(), "telemetry": manager.telemetry_snapshot(), "engines": manager.registry.list()}, indent=2))
        finally:
            runtime.close()
        return 0
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from unified_live.ui import MainWindow

    app = QApplication(sys.argv[:1])
    app.setApplicationName(manager.settings.current.app_name)
    window = MainWindow(manager)
    window.show()
    smoke_result = {"ok": True}
    if args.smoke_seconds:
        manager.set_enabled("face", True)
        manager.set_enabled("voice", True)
        manager.start_video()
        manager.start_audio()

        def finish_smoke():
            state = manager.get_state()
            smoke_result["ok"] = bool(manager.latest_frame() is not None and
                                      state["runtime"]["metrics"]["audio_total_ms"] is not None and
                                      state["backends"]["face"]["state"] == "running" and
                                      state["backends"]["voice"]["state"] == "running")
            print(json.dumps({"smoke_passed": smoke_result["ok"], "runtime": state["runtime"], "backends": state["backends"]}, indent=2))
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                window.grab().save(str(args.screenshot))
            app.quit()

        QTimer.singleShot(max(1000, int(args.smoke_seconds * 1000)), finish_smoke)
    try:
        result = app.exec()
    finally:
        runtime.close()
        manager.stop_backend("face")
        manager.stop_backend("voice")
    return result if smoke_result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

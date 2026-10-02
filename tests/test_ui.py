import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from unified_live.ui import MainWindow


class FakeManager:
    def __init__(self):
        self.settings = {
            "face_engine": "face.mock",
            "voice_engine": "voice.mock",
            "face_enabled": False,
            "voice_enabled": False,
            "face_options": {},
            "voice_options": {"pitch_shift": 2},
            "camera_index": None,
            "microphone_index": None,
            "audio_output_index": None,
            "virtual_camera": False,
            "virtual_audio": False,
            "auto_sync": True,
            "video_offset_ms": 12,
            "audio_offset_ms": -4,
            "performance_profile": "balanced",
            "width": 1280,
            "height": 720,
            "fps": 30,
            "sample_rate": 48000,
            "chunk_size": 512,
            "demo": False,
            "advanced": False,
            "backends": {"voice.other": {"options": {"reference_audio": "local.wav"}}},
        }
        self.state = {
            "settings": self.settings,
            "engines": [
                {"id": "face.mock", "kind": "face", "name": "Mock Face", "installed": True, "configured": True, "validated": True, "planned": False, "capabilities": ["swap"]},
                {"id": "voice.mock", "kind": "voice", "name": "Mock Voice", "installed": True, "configured": True, "validated": True, "planned": False, "capabilities": ["streaming"]},
                {"id": "voice.other", "kind": "voice", "name": "Other Voice", "installed": True, "configured": True, "validated": True, "planned": False, "capabilities": []},
                {"id": "seed-vc", "kind": "voice", "name": "Seed-VC", "installed": True, "configured": True, "validated": True, "planned": False, "capabilities": []},
            ],
            "backends": {
                "face": {"state": "stopped", "engine_id": "face.mock", "last_latency_ms": 4.5},
                "voice": {"state": "stopped", "engine_id": "voice.mock"},
            },
            "runtime": {"video_running": False, "audio_running": False, "logs": ["ready"], "metrics": {}, "sync": {"valid": True, "video_delay_ms": 2.0}},
        }
        self.enabled = []
        self.profiles = []
        self.loaded = []
        self.settings_thread = None
        self.selected_engines = []
        self.cancelled_backends = []

    def get_state(self):
        return self.state

    def update_settings(self, patch):
        self.settings_thread = threading.get_ident()
        self.settings.update(patch)
        return self.settings

    def set_enabled(self, kind, enabled):
        self.enabled.append((kind, enabled))
        self.settings[f"{kind}_enabled"] = enabled

    def start_video(self):
        self.state["runtime"]["video_running"] = True

    def stop_video(self):
        self.state["runtime"]["video_running"] = False

    def start_audio(self):
        self.state["runtime"]["audio_running"] = True

    def stop_audio(self):
        self.state["runtime"]["audio_running"] = False

    def select_engine(self, kind, engine_id, options=None):
        self.selected_engines.append((kind, engine_id, dict(options or {})))
        self.settings[f"{kind}_engine"] = engine_id
        self.settings[f"{kind}_options"] = options or {}

    def cancel_backend(self, kind):
        self.cancelled_backends.append(kind)

    def restart_backend(self, kind):
        return kind

    def backend_status(self):
        return self.state["backends"]

    def save_profile(self, name):
        self.profiles.append(name)

    def load_profile(self, name):
        self.loaded.append(name)
        return self.settings

    def list_profiles(self):
        return list(self.profiles)

    def discover_devices(self):
        return {
            "cameras": [{"id": 0, "name": "Camera 0"}],
            "microphones": [{"id": 1, "name": "Mic 1"}],
            "outputs": [{"id": 2, "name": "Output 2"}],
        }

    def latest_frame(self):
        return None

    def telemetry_snapshot(self):
        return {"metrics": {"fps_input": 28.2, "face_inference_ms": 4.5}, "gpu_name": "Test GPU", "cpu_percent": 11.0}


@pytest.fixture
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def wait_idle(app, window):
    for _ in range(100):
        app.processEvents()
        if not window._busy and not window._task_queue and not window._workers:
            return
        app.processEvents()
        from PySide6.QtTest import QTest
        QTest.qWait(10)
    assert not window._busy


def test_navigation_snapshot_and_profiles(app):
    manager = FakeManager()
    window = MainWindow(manager)
    window.show()
    assert window.nav.count() == 9
    assert window.stack.count() == 9
    assert window.title.text() == "LIVE"
    window.nav.setCurrentRow(6)
    assert window.title.text() == "PERFORMANCE"
    assert window.perf_values["fps_input"].text() == "28.2 FPS"
    assert window.perf_values["gpu_name"].text() == "Test GPU"

    window.profile_name.setText("Studio")
    window._save_profile()
    wait_idle(app, window)
    assert manager.profiles == ["Studio"]
    assert window.profile_list.count() == 1
    window.profile_list.setCurrentRow(0)
    window._load_profile()
    wait_idle(app, window)
    assert manager.loaded == ["Studio"]
    window.close()


def test_effect_toggles_and_controls_use_manager(app):
    manager = FakeManager()
    window = MainWindow(manager)
    window.face_toggle.setChecked(True)
    window.voice_toggle.setChecked(True)
    wait_idle(app, window)
    assert ("face", True) in manager.enabled
    assert ("voice", True) in manager.enabled
    assert not window.pitch.isEnabled()

    main_thread = threading.get_ident()
    window.width_spin.setValue(1920)
    wait_idle(app, window)
    assert manager.settings_thread != main_thread
    assert manager.settings["width"] == 1920
    window.profile_combo.setCurrentText("LOW LATENCY")
    wait_idle(app, window)
    assert manager.settings["performance_profile"] == "latency"
    assert manager.settings["fps"] == 60
    assert manager.settings["chunk_size"] == 256

    window._discover_devices()
    wait_idle(app, window)
    assert window.camera_combo.findText("Camera 0") >= 0
    window.nav.setCurrentRow(7)
    assert window.logs.toPlainText() == "ready"
    window.close()


def test_slow_settings_update_keeps_qt_event_loop_responsive(app):
    class SlowManager(FakeManager):
        def update_settings(self, patch):
            time.sleep(0.12)
            return super().update_settings(patch)

    manager = SlowManager()
    window = MainWindow(manager)
    heartbeat = {"ticks": 0}
    timer = QTimer(window)
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.__setitem__("ticks", heartbeat["ticks"] + 1))
    timer.start()
    main_thread = threading.get_ident()

    window._setting("width", 1600)
    wait_idle(app, window)

    assert manager.settings_thread != main_thread
    assert heartbeat["ticks"] > 0
    assert manager.settings["width"] == 1600
    timer.stop()
    window.close()


def test_engine_change_uses_target_defaults_and_seed_checkpoint(app):
    manager = FakeManager()
    window = MainWindow(manager)
    other_index = window.voice_engine.findData("voice.other")
    window.voice_engine.setCurrentIndex(other_index)
    wait_idle(app, window)
    assert manager.selected_engines[-1] == ("voice", "voice.other", {"reference_audio": "local.wav"})

    manager.settings["voice_engine"] = "seed-vc"
    manager.settings["voice_options"] = {"checkpoint_path": "/models/seed.pt"}
    window._refresh_state()
    assert window._voice_model_option_key() == "checkpoint_path"
    assert window.voice_model_path.text() == "/models/seed.pt"
    window._sync_path_option("checkpoint_path", "/models/new-seed.pt")
    assert '"checkpoint_path": "/models/new-seed.pt"' in window.voice_options.toPlainText()
    window.close()


def test_close_waits_asynchronously_for_active_worker(app):
    manager = FakeManager()
    window = MainWindow(manager)
    window.show()
    window._async("slow action", lambda: time.sleep(0.2))
    window.close()

    assert window._closing
    assert window.isVisible()
    assert manager.cancelled_backends == ["face", "voice"]
    for _ in range(40):
        app.processEvents()
        if not window.isVisible():
            break
        from PySide6.QtTest import QTest
        QTest.qWait(10)

    assert not window.isVisible()
    assert window._thread_pool.activeThreadCount() == 0
    assert not window._workers

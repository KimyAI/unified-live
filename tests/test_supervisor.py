import sys
import os
import signal
import threading
import time

import numpy as np
import pytest

from unified_live.engine_manager import BackendError, BackendSupervisor, EngineManager


def test_mock_subprocess_isolated_and_independent(tmp_path):
    manager = EngineManager(tmp_path)
    manager.set_enabled("face", True)
    manager.set_enabled("voice", True)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    audio = np.ones(16, dtype=np.float32)
    assert manager.process_frame(frame)[0, 0, 0] == 255
    assert manager.process_audio(audio)[0] == pytest.approx(.75)
    face = manager.backend_status("face")
    voice = manager.backend_status("voice")
    assert face["pid"] != voice["pid"]
    manager.restart_backend("face")
    assert manager.backend_status("voice")["pid"] == voice["pid"]
    manager.stop_backend("face")
    manager.stop_backend("voice")


@pytest.mark.parametrize("behavior,reason", [("crash", "Worker"), ("timeout", "timeout")])
def test_external_worker_crash_and_timeout_are_visible(tmp_path, behavior, reason):
    source = '''import os, time\nfrom unified_live.engines.base import FaceEngine\nclass Bridge(FaceEngine):\n    def process_frame(self, frame):\n        if BEHAVIOR == "crash": os._exit(21)\n        time.sleep(1)\n        return frame\ndef create_engine(): return Bridge()\n'''.replace("BEHAVIOR", repr(behavior))
    (tmp_path / "bridge.py").write_text(source)
    worker = BackendSupervisor("reswapper", "face", tmp_path, python=sys.executable,
                               module="bridge", root=str(tmp_path), timeout=2)
    try:
        worker.start()
        worker.timeout = .2
        with pytest.raises(BackendError, match=reason):
            worker.process(np.zeros((2, 2, 3), dtype=np.uint8))
        assert worker.status()["state"] == "failed"
        assert worker.status()["pid"] is None
    finally:
        worker.stop()


def test_status_does_not_wait_for_inference(tmp_path):
    (tmp_path / "slow.py").write_text('''import time\nfrom unified_live.engines.base import FaceEngine\nclass Bridge(FaceEngine):\n    def process_frame(self, frame):\n        time.sleep(.5)\n        return frame\ndef create_engine(): return Bridge()\n''')
    worker = BackendSupervisor("reswapper", "face", tmp_path, python=sys.executable,
                               module="slow", root=str(tmp_path), timeout=2)
    worker.start()
    result = []
    thread = threading.Thread(target=lambda: result.append(worker.process(np.zeros((2, 2, 3), dtype=np.uint8))))
    try:
        thread.start()
        time.sleep(.1)
        start = time.monotonic()
        assert worker.status()["state"] == "running"
        assert time.monotonic() - start < .1
        thread.join(timeout=2)
        assert len(result) == 1
    finally:
        worker.stop()


@pytest.mark.skipif(os.name == "nt", reason="SIGSTOP is POSIX-only")
def test_pipe_write_timeout_when_worker_stops_reading(tmp_path):
    worker = BackendSupervisor("mock-face", "face", tmp_path, timeout=.2)
    worker.start()
    try:
        os.kill(worker.status()["pid"], signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(BackendError, match="timeout"):
            worker.process(np.zeros((512, 512, 3), dtype=np.uint8))
        assert time.monotonic() - started < 2
        assert worker.status()["state"] == "failed"
    finally:
        worker.stop()


def test_slow_face_start_does_not_block_voice(tmp_path):
    (tmp_path / "startup.py").write_text('''import time\nfrom unified_live.engines.base import FaceEngine\nclass Bridge(FaceEngine):\n    def initialize(self, options):\n        time.sleep(.6)\n    def process_frame(self, frame): return frame\ndef create_engine(): return Bridge()\n''')
    manager = EngineManager(tmp_path / "data")
    manager.update_settings({"backends": {"reswapper": {"python": sys.executable, "module": "startup", "root": str(tmp_path)}}})
    manager.select_engine("face", "reswapper")
    manager.set_enabled("voice", True)
    manager.start_backend("voice")  # time isolation, not Python/NumPy cold import
    thread = threading.Thread(target=lambda: manager.start_backend("face"))
    try:
        thread.start()
        time.sleep(.1)
        start = time.monotonic()
        output = manager.process_audio(np.ones(16, dtype=np.float32))
        assert output[0] == pytest.approx(.75)
        assert time.monotonic() - start < .5
        thread.join(timeout=3)
        assert not thread.is_alive()
    finally:
        manager.stop_backend("face")
        manager.stop_backend("voice")


def test_backend_configuration_change_replaces_supervisor(tmp_path):
    manager = EngineManager(tmp_path / "data")
    manager.start_backend("face")
    old = manager._supervisors["face"]
    manager.update_settings({"face_options": {"strength": .25}})
    assert manager.backend_status("face")["state"] == "stopped"
    manager.start_backend("face")
    assert manager._supervisors["face"] is not old
    manager.stop_backend("face")


def test_noisy_native_and_python_stdout_cannot_corrupt_protocol(tmp_path):
    (tmp_path / "noisy.py").write_text('''import os
from unified_live.engines.base import FaceEngine
print("upstream import log", flush=True)
os.write(1, b"native library import log\\n")
class Bridge(FaceEngine):
    def process_frame(self, frame):
        print("per frame log", flush=True)
        os.write(1, b"native inference log\\n")
        return frame
def create_engine(): return Bridge()
''')
    worker = BackendSupervisor("reswapper", "face", tmp_path, python=sys.executable,
                               module="noisy", root=str(tmp_path))
    try:
        worker.start()
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        np.testing.assert_equal(worker.process(frame), frame)
        assert worker.health()
    finally:
        worker.stop()

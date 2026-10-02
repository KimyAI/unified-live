import time
import threading
import sys

import numpy as np

from unified_live.engine_manager import EngineManager
from unified_live.media.buffers import AudioBuffer, Packet, VideoBuffer
from unified_live.media.runtime import MediaRuntime


def test_video_delay_releases_and_drops_only_due():
    buffer = VideoBuffer(capacity=2)
    buffer.push(Packet(0, 1, "one"))
    buffer.push(Packet(0, 1.1, "two"))
    assert buffer.pop_due(1.04, 50) is None
    assert buffer.pop_due(1.06, 50).payload == "one"
    assert buffer.pop_due(1.16, 50).payload == "two"
    assert buffer.dropped == 0


def test_audio_keeps_order_across_callback_sizes():
    buffer = AudioBuffer(1000)
    buffer.push(Packet(1, 1, np.arange(10, dtype=np.float32)))
    out = np.zeros(4, dtype=np.float32)
    buffer.read_into(out, 1.01, 10)
    np.testing.assert_equal(out, [0, 1, 2, 3])
    buffer.read_into(out, 1.014, 10)
    np.testing.assert_equal(out, [4, 5, 6, 7])
    buffer.read_into(out, 1.018, 10)
    np.testing.assert_equal(out, [8, 9, 0, 0])
    assert buffer.underruns == 1


def test_audio_delay_silence_and_bounded_backlog():
    buffer = AudioBuffer(1000, max_seconds=0.01)
    buffer.push(Packet(0, 1, np.ones(10, dtype=np.float32)))
    buffer.push(Packet(0, 1.01, np.full(10, 2, dtype=np.float32)))
    assert buffer.dropped == 1
    out = np.ones(10, dtype=np.float32)
    buffer.read_into(out, 1, 50)
    assert not out.any()
    buffer.read_into(out, 1.06, 50)
    np.testing.assert_equal(out, np.full(10, 2))


def test_audio_recovers_old_startup_backlog_instead_of_accumulating_delay():
    buffer = AudioBuffer(1000)
    for i in range(10):
        buffer.push(Packet(i * .02, i * .02, np.full(20, i, dtype=np.float32)))
    out = np.zeros(10, dtype=np.float32)
    buffer.read_into(out, .18, 0)
    assert buffer.last_latency_ms < 30
    assert buffer.dropped > 0
    assert np.all(out >= 8)


def wait_until(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition timed out")


def test_both_pipelines_toggle_profile_and_stop_independently(tmp_path):
    manager = EngineManager(tmp_path)
    manager.update_settings({"demo": True, "face_enabled": True, "voice_enabled": True})
    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    try:
        manager.start_video()
        manager.start_audio()
        wait_until(lambda: runtime.latest_frame() is not None and runtime.get_state()["metrics"]["audio_total_ms"] is not None)
        assert runtime.get_state()["sync"]["valid"]
        manager.save_profile("Both")
        manager.set_enabled("face", False)
        before = manager.backend_status("voice")["requests"]
        wait_until(lambda: manager.backend_status("voice")["requests"] > before)
        manager.stop_video()
        assert runtime.get_state()["audio_running"]
        manager.load_profile("Both")
        assert manager.settings.current.face_enabled
        manager.start_video()
        wait_until(lambda: runtime.get_state()["video_running"])
    finally:
        runtime.close()
        manager.stop_backend("face")
        manager.stop_backend("voice")


def test_sync_samples_only_changed_pipeline_and_exposes_telemetry(tmp_path):
    manager = EngineManager(tmp_path)
    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    try:
        runtime._video_valid = runtime._audio_valid = True
        runtime._video_baseline, runtime._audio_baseline = 20, 90
        runtime._update_sync(video_sample=True, audio_sample=True)
        runtime._audio_baseline = 50  # no new audio sample has been reported
        runtime._video_baseline = 30
        runtime._update_sync(video_sample=True)
        assert runtime.get_state()["sync"]["audio_baseline_ms"] == 90
        assert runtime.telemetry_snapshot()["sync"]["valid"]
        runtime._video_queue.push(Packet(1, 1, np.zeros((2, 2, 3), dtype=np.uint8)))
        manager.update_settings({"face_enabled": True})
        assert not runtime._video_queue._items
        assert not runtime.get_state()["sync"]["valid"]
    finally:
        runtime.close()


def test_video_stop_does_not_hold_audio_control(tmp_path):
    manager = EngineManager(tmp_path)
    manager.update_settings({"demo": True})
    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    entered = threading.Event()

    def slow_capture():
        entered.set()
        time.sleep(.7)

    runtime._video_capture = slow_capture
    runtime.start_video()
    assert entered.wait(1)
    stopper = threading.Thread(target=runtime.stop_video)
    try:
        stopper.start()
        time.sleep(.05)
        start = time.monotonic()
        runtime.start_audio()
        assert time.monotonic() - start < .5
        assert runtime.get_state()["audio_running"]
        stopper.join(timeout=2)
        assert not stopper.is_alive()
    finally:
        runtime.close()


def test_profile_change_drops_inflight_audio(tmp_path):
    manager = EngineManager(tmp_path)
    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    entered = threading.Event()
    release = threading.Event()

    def slow_audio(audio):
        entered.set()
        release.wait(2)
        return audio

    manager.process_audio = slow_audio
    thread = threading.Thread(target=runtime._audio_process)
    try:
        thread.start()
        runtime._enqueue_audio(np.ones(16, dtype=np.float32), time.monotonic(), 0)
        assert entered.wait(1)
        manager.update_settings({"voice_options": {"source": "new"}})
        release.set()
        wait_until(lambda: runtime._audio_input.empty())
        time.sleep(.05)
        assert not runtime._audio_queue._items
    finally:
        runtime._audio_stop.set()
        release.set()
        thread.join(timeout=2)
        runtime.close()


def test_video_stop_cancels_slow_backend_start_and_can_restart(tmp_path):
    (tmp_path / "slow_init.py").write_text('''import time\nfrom unified_live.engines.base import FaceEngine\nclass Bridge(FaceEngine):\n    def initialize(self, options): time.sleep(5)\n    def process_frame(self, frame): return frame\ndef create_engine(): return Bridge()\n''')
    manager = EngineManager(tmp_path / "data")
    manager.update_settings({"demo": True, "face_enabled": True,
                             "backends": {"reswapper": {"python": sys.executable, "module": "slow_init", "root": str(tmp_path)}}})
    manager.select_engine("face", "reswapper")
    runtime = MediaRuntime(manager)
    manager.attach_runtime(runtime)
    try:
        runtime.start_video()
        wait_until(lambda: manager.backend_status("face")["pid"] is not None)
        start = time.monotonic()
        runtime.stop_video()
        assert time.monotonic() - start < 3
        assert manager.backend_status("face")["state"] == "stopped"
        manager.set_enabled("face", False)
        runtime.start_video()
        wait_until(lambda: runtime.latest_frame() is not None)
    finally:
        runtime.close()
        manager.stop_backend("face")

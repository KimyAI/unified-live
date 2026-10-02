"""Independent capture/inference/presentation workers for the desktop host."""

from __future__ import annotations

import logging
import platform
import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from unified_live.media.buffers import AudioBuffer, Packet, VideoBuffer
from unified_live.output import VirtualCamera, open_audio_output
from unified_live.sync import SyncEngine
from unified_live.telemetry import Telemetry


class MediaRuntime:
    def __init__(self, manager):
        self.manager = manager
        self._lock = threading.RLock()
        self._video_control = threading.RLock()
        self._audio_control = threading.RLock()
        self._video_stop = threading.Event()
        self._audio_stop = threading.Event()
        self._video_threads = []
        self._audio_threads = []
        self._input_stream = self._output_stream = None
        self._frame = None
        self._logs = deque(maxlen=100)
        self._video_queue = VideoBuffer()
        self._audio_queue = AudioBuffer(manager.settings.current.sample_rate)
        self._audio_input = queue.Queue(maxsize=6)
        self._sync = SyncEngine()
        self._sync_state = {}
        self._video_valid = self._audio_valid = False
        self._video_running = self._audio_running = False
        self._video_error = self._audio_error = None
        self._metrics = dict.fromkeys([
            "camera_capture_ms", "face_inference_ms", "video_output_ms",
            "microphone_capture_ms", "voice_inference_ms", "audio_output_ms",
            "video_total_ms", "audio_total_ms", "fps_input", "fps_output",
        ])
        self._metrics.update(dropped_frames=0, audio_underruns=0, audio_dropped_chunks=0)
        self._video_baseline = self._audio_baseline = None
        self._timing_generation = 0
        self._counts = {"input": deque(maxlen=120), "output": deque(maxlen=120)}
        self.telemetry = Telemetry(manager.settings.current.gpu)

    def _log(self, kind, message, error=False):
        logging.getLogger("unified_live." + kind).log(
            logging.ERROR if error else logging.INFO, message
        )
        with self._lock:
            self._logs.append(time.strftime("%H:%M:%S") + " " + kind + ": " + message)

    def _error(self, kind, exc):
        message = str(exc)
        attr = "_video_error" if kind == "video" else "_audio_error"
        if getattr(self, attr) != message:
            self._log(kind, message, error=True)
        setattr(self, attr, message)

    def _count_fps(self, key):
        times = self._counts[key]
        times.append(time.monotonic())
        if len(times) > 1:
            self._metrics["fps_" + key] = (len(times) - 1) / max(1e-9, times[-1] - times[0])

    def _update_sync(self, video_sample=False, audio_sample=False):
        settings = self.manager.settings.current
        with self._lock:
            self._sync_state = self._sync.update(
                self._video_baseline if video_sample and self._video_valid else None,
                self._audio_baseline if audio_sample and self._audio_valid else None,
                auto=settings.auto_sync and self._video_valid and self._audio_valid,
                video_offset_ms=settings.video_offset_ms,
                audio_offset_ms=settings.audio_offset_ms,
            )

    def _clear_timing(self):
        with self._lock:
            self._timing_generation += 1
            self._video_queue.clear()
            self._audio_queue.clear()
            while True:
                try:
                    self._audio_input.get_nowait()
                except queue.Empty:
                    break
            self._video_baseline = self._audio_baseline = None
            self._video_valid = self._audio_valid = False
            self._sync.reset()
        self._update_sync()

    def _delay(self, kind):
        with self._lock:
            return self._sync_state.get(kind + "_delay_ms", 0.0)

    def get_state(self):
        with self._lock:
            metrics = self._metrics.copy()
            metrics["dropped_frames"] = self._video_queue.dropped
            metrics["audio_underruns"] = self._audio_queue.underruns
            metrics["audio_dropped_chunks"] = self._audio_queue.dropped
            return {
                "video_running": self._video_running, "audio_running": self._audio_running,
                "video_error": self._video_error, "audio_error": self._audio_error,
                "demo": self.manager.settings.current.demo,
                "metrics": metrics, "sync": self._sync_state.copy(), "logs": list(self._logs),
            }

    def latest_frame(self):
        with self._lock:
            return self._frame  # immutable after publication; Qt copies before display

    def telemetry_snapshot(self):
        state = self.get_state()
        return {**self.telemetry.snapshot(), **state["metrics"], "sync": state["sync"]}

    def discover_devices(self):
        devices = {"cameras": [], "microphones": [], "outputs": [], "virtual_cameras": [], "errors": []}
        # OpenCV cannot enumerate friendly Windows names portably. Probe only on
        # an explicit refresh, outside Qt's event thread, with a small fixed range.
        if self._video_running:
            index = self.manager.settings.current.camera_index
            devices["cameras"].append({"id": index, "name": f"Camera {index} (active)"})
        else:
            for index in range(6):
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                try:
                    if cap.isOpened():
                        devices["cameras"].append({"id": index, "name": f"Camera {index}"})
                finally:
                    cap.release()
        try:
            import sounddevice as sd

            apis = sd.query_hostapis()
            for index, device in enumerate(sd.query_devices()):
                name = f"{device['name']} [{apis[device['hostapi']]['name']}]"
                if device["max_input_channels"]:
                    devices["microphones"].append({"id": index, "name": name})
                if device["max_output_channels"]:
                    devices["outputs"].append({"id": index, "name": name})
        except Exception as exc:
            devices["errors"].append(str(exc))
        # Driver presence is only confirmed by opening the virtual camera.
        devices["virtual_cameras"] = [{"id": "auto", "name": "pyvirtualcam auto (OBS on Windows; verified on start)"}]
        return devices

    def start_video(self):
        with self._video_control:
            if self._video_running:
                return
            if self._video_threads:
                self.stop_video()
            self._video_stop.clear()
            self._video_error = None
            self._video_valid = False
            self._video_queue = VideoBuffer()
            self._frame = None
            self._counts = {"input": deque(maxlen=120), "output": deque(maxlen=120)}
            self._video_running = True
            self._video_threads = [
                threading.Thread(target=self._video_capture, name="video-capture", daemon=True),
                threading.Thread(target=self._video_present, name="video-output", daemon=True),
            ]
            for thread in self._video_threads:
                thread.start()
            self._log("video", "Pipeline started")

    def _video_capture(self):
        settings = self.manager.settings.current
        cap = None
        try:
            if not settings.demo:
                cap = cv2.VideoCapture(
                    settings.camera_index if settings.camera_index is not None else 0,
                    cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY,
                )
                if not cap.isOpened():
                    raise RuntimeError("Camera could not be opened. Select a device or enable Demo mode.")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
                cap.set(cv2.CAP_PROP_FPS, settings.fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            tick = 0
            while not self._video_stop.is_set():
                generation = self._timing_generation
                begin = time.monotonic()
                if settings.demo:
                    frame = self._demo_frame(settings.width, settings.height, tick)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError("Camera disconnected or returned an empty frame")
                    frame = cv2.resize(frame, (settings.width, settings.height))
                captured = time.monotonic()
                if self._video_stop.is_set():
                    break
                capture_ms = (captured - begin) * 1000
                self._count_fps("input")
                try:
                    transformed = self.manager.process_frame(frame)
                except Exception as exc:
                    self._error("video", exc)
                    transformed = frame
                ready = time.monotonic()
                process_ms = (ready - captured) * 1000
                with self._lock:
                    if generation != self._timing_generation:
                        continue
                    self._metrics.update(camera_capture_ms=capture_ms, face_inference_ms=process_ms)
                    output = self._metrics["video_output_ms"] or 0
                    self._video_baseline = capture_ms + process_ms + output
                    self._video_valid = True
                    self._update_sync(video_sample=True)
                    self._video_queue.push(Packet(begin, ready, transformed, self._video_baseline))
                tick += 1
                self._video_stop.wait(max(0, 1 / settings.fps - (time.monotonic() - begin)))
        except Exception as exc:
            self._error("video", exc)
            self._video_stop.set()
        finally:
            if cap:
                cap.release()
            self._video_running = self._video_valid = False
            self._video_baseline = None
            self._update_sync()

    @staticmethod
    def _demo_frame(width, height, tick):
        frame = np.full((height, width, 3), (30, 26, 22), dtype=np.uint8)
        x = int((np.sin(tick / 30) + 1) * (width - 120) / 2) + 60
        cv2.circle(frame, (x, height // 2), min(55, height // 4), (105, 220, 165), -1)
        cv2.putText(frame, "SYNTHETIC INPUT / NO CAMERA", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)
        cv2.putText(frame, f"Frame {tick}", (18, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        return frame

    def _video_present(self):
        camera = None
        try:
            while not self._video_stop.is_set():
                settings = self.manager.settings.current
                packet = self._video_queue.pop_due(time.monotonic(), self._delay("video"))
                if packet is None:
                    self._video_stop.wait(0.003)
                    continue
                start = time.monotonic()
                if settings.virtual_camera and camera is None:
                    try:
                        height, width = packet.payload.shape[:2]
                        camera = VirtualCamera(width, height, settings.fps)
                        self._log("video", "Virtual camera opened: " + camera.name)
                    except Exception as exc:
                        self._error("video", exc)
                        # A failed output doesn't interrupt preview/capture.
                        self.manager.update_settings({"virtual_camera": False})
                if not settings.virtual_camera and camera:
                    camera.close()
                    camera = None
                if camera:
                    try:
                        camera.send(packet.payload)
                    except Exception as exc:
                        self._error("video", exc)
                        camera.close()
                        camera = None
                        self.manager.update_settings({"virtual_camera": False})
                with self._lock:
                    self._frame = packet.payload
                    self._metrics["video_output_ms"] = (time.monotonic() - start) * 1000
                    self._metrics["video_total_ms"] = (time.monotonic() - packet.captured_at) * 1000
                    self._count_fps("output")
        except Exception as exc:
            self._error("video", exc)
            self._video_stop.set()
        finally:
            if camera:
                camera.close()

    def stop_video(self):
        with self._video_control:
            self._video_stop.set()
            for thread in self._video_threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=.5)
            cancelled = any(t.is_alive() for t in self._video_threads)
            if cancelled:
                self.manager.cancel_backend("face")
                for thread in self._video_threads:
                    if thread is not threading.current_thread():
                        thread.join(timeout=3.5)
            if any(t.is_alive() for t in self._video_threads):
                raise RuntimeError("Video worker has not stopped; close the busy camera/backend before restarting")
            if cancelled:
                self.manager.stop_backend("face")
            self._video_threads = []
            self._video_running = self._video_valid = False
            self._video_queue.clear()
            self._sync.reset()
            self._update_sync()
            self._log("video", "Pipeline stopped")

    def start_audio(self):
        with self._audio_control:
            if self._audio_running:
                return
            if self._audio_threads or self._input_stream or self._output_stream:
                self.stop_audio()
            self.manager.reset_backend("voice")
            settings = self.manager.settings.current
            self._audio_stop.clear()
            self._audio_error = None
            self._audio_valid = False
            self._audio_queue = AudioBuffer(settings.sample_rate)
            self._audio_input = queue.Queue(maxsize=6)
            self._audio_running = True
            self._audio_threads = [threading.Thread(target=self._audio_process, name="audio-inference", daemon=True)]
            try:
                if settings.demo:
                    self._audio_threads.append(threading.Thread(target=self._audio_demo, name="audio-demo", daemon=True))
                else:
                    import sounddevice as sd

                    sd.check_input_settings(device=settings.microphone_index, channels=1, dtype="float32", samplerate=settings.sample_rate)
                    self._input_stream = sd.InputStream(
                        device=settings.microphone_index, channels=1, dtype="float32",
                        samplerate=settings.sample_rate, blocksize=settings.chunk_size,
                        latency="low", callback=self._capture_audio,
                    )
                    self._input_stream.start()
                if settings.virtual_audio:
                    self._output_stream = open_audio_output(
                        settings.audio_output_index, settings.sample_rate, min(settings.chunk_size, 480), self._present_audio
                    )
                    self._output_stream.start()
                else:
                    self._audio_threads.append(threading.Thread(target=self._audio_drain, name="audio-monitor", daemon=True))
                for thread in self._audio_threads:
                    thread.start()
                self._log("audio", "Pipeline started" + (" (silent monitor)" if not settings.virtual_audio else ""))
            except Exception:
                self._audio_running = False
                self._audio_stop.set()
                for stream in (self._input_stream, self._output_stream):
                    if stream:
                        stream.close()
                self._input_stream = self._output_stream = None
                self._audio_threads = []
                raise

    def _enqueue_audio(self, audio, captured_at, capture_ms):
        packet = Packet(captured_at, time.monotonic(), audio, capture_ms)
        packet.generation = self._timing_generation
        try:
            self._audio_input.put_nowait(packet)
        except queue.Full:
            try:
                self._audio_input.get_nowait()
            except queue.Empty:
                pass
            self._audio_queue.dropped += 1
            try:
                self._audio_input.put_nowait(packet)
            except queue.Full:
                self._audio_queue.dropped += 1

    def _capture_audio(self, indata, frames, timing, status):
        now = time.monotonic()
        # Translate PortAudio's clock to monotonic through age, never mix epochs.
        capture_ms = max(0.0, timing.currentTime - timing.inputBufferAdcTime) * 1000
        if status.input_overflow:
            self._audio_queue.dropped += 1
        self._enqueue_audio(indata[:, 0].copy(), now - capture_ms / 1000, capture_ms)

    def _audio_demo(self):
        settings = self.manager.settings.current
        tick = 0
        deadline = time.monotonic()
        while not self._audio_stop.is_set():
            captured = time.monotonic()
            indices = np.arange(settings.chunk_size) + tick * settings.chunk_size
            audio = (np.sin(indices * (2 * np.pi * 220 / settings.sample_rate)) * 0.08).astype(np.float32)
            self._enqueue_audio(audio, captured, 0.0)
            tick += 1
            deadline += settings.chunk_size / settings.sample_rate
            self._audio_stop.wait(max(0, deadline - time.monotonic()))

    def _audio_process(self):
        try:
            while not self._audio_stop.is_set():
                try:
                    packet = self._audio_input.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self._audio_stop.is_set():
                    break
                if packet.generation != self._timing_generation:
                    continue
                start = time.monotonic()
                algorithmic_ms = 0.0
                warming_up = False
                try:
                    converted = self.manager.process_audio(packet.payload)
                    if len(converted) != len(packet.payload):
                        raise RuntimeError("Voice engine must preserve negotiated block length/sample rate")
                    if self.manager.settings.current.voice_enabled:
                        backend = self.manager.backend_status("voice")
                        algorithmic_ms = backend.get("algorithmic_delay_ms", 0.0)
                        warming_up = backend.get("warming_up", False)
                except Exception as exc:
                    self._error("audio", exc)
                    converted = packet.payload
                ready = time.monotonic()
                with self._lock:
                    if packet.generation != self._timing_generation:
                        continue
                    self._metrics["microphone_capture_ms"] = packet.baseline_ms
                    self._metrics["voice_inference_ms"] = (ready - start) * 1000
                    self._metrics["voice_algorithmic_delay_ms"] = algorithmic_ms
                    self._metrics["voice_warming_up"] = warming_up
                    # Include processing queue wait, exclude sync/output wait.
                    output_ms = self._metrics["audio_output_ms"] or 0
                    content_time = packet.captured_at - algorithmic_ms / 1000
                    self._audio_baseline = (ready - content_time) * 1000 + output_ms
                    self._audio_valid = not warming_up
                    self._update_sync(audio_sample=True)
                    self._audio_queue.push(Packet(content_time, ready, converted, self._audio_baseline))
        except Exception as exc:
            self._error("audio", exc)
            self._audio_stop.set()
            self._audio_running = self._audio_valid = False
            self._audio_baseline = None

    def _present_audio(self, outdata, frames, timing, status):
        now = time.monotonic()
        self._audio_queue.read_into(outdata[:, 0], now, self._delay("audio"))
        if status.output_underflow:
            self._audio_queue.underruns += 1
        # Device output latency estimate, from the callback's own clock.
        output_ms = max(0.0, timing.outputBufferDacTime - timing.currentTime) * 1000
        self._metrics["audio_output_ms"] = output_ms
        if self._audio_queue.last_latency_ms is not None:
            self._metrics["audio_total_ms"] = self._audio_queue.last_latency_ms + output_ms

    def _audio_drain(self):
        settings = self.manager.settings.current
        sink = np.zeros(min(settings.chunk_size, 480), dtype=np.float32)
        deadline = time.monotonic()
        while not self._audio_stop.is_set():
            self._audio_queue.read_into(sink, time.monotonic(), self._delay("audio"))
            self._metrics["audio_output_ms"] = 0.0
            self._metrics["audio_total_ms"] = self._audio_queue.last_latency_ms
            deadline += len(sink) / settings.sample_rate
            self._audio_stop.wait(max(0, deadline - time.monotonic()))

    def stop_audio(self):
        with self._audio_control:
            self._audio_stop.set()
            for stream in (self._input_stream, self._output_stream):
                if stream:
                    stream.abort()
                    stream.close()
            self._input_stream = self._output_stream = None
            for thread in self._audio_threads:
                if thread is not threading.current_thread():
                    thread.join(timeout=.5)
            cancelled = any(t.is_alive() for t in self._audio_threads)
            if cancelled:
                self.manager.cancel_backend("voice")
                for thread in self._audio_threads:
                    if thread is not threading.current_thread():
                        thread.join(timeout=3.5)
            if any(t.is_alive() for t in self._audio_threads):
                raise RuntimeError("Audio worker has not stopped; restart the busy backend first")
            if cancelled:
                self.manager.stop_backend("voice")
            self._audio_threads = []
            self._audio_running = self._audio_valid = False
            self._audio_queue.clear()
            self._sync.reset()
            self._update_sync()
            self._log("audio", "Pipeline stopped")

    def on_settings_changed(self, before, after):
        """Reconfigure only affected pipelines; called outside Qt's UI thread."""
        # Manager passes dictionaries or dataclasses; keep the boundary simple.
        old = before.to_dict() if hasattr(before, "to_dict") else before
        new = after.to_dict() if hasattr(after, "to_dict") else after
        video_keys = ("camera_index", "width", "height", "fps", "demo")
        audio_keys = ("microphone_index", "audio_output_index", "sample_rate", "chunk_size", "virtual_audio", "demo")
        timing_keys = (*video_keys, *audio_keys, "virtual_camera", "face_engine", "voice_engine",
                       "face_options", "voice_options", "face_enabled", "voice_enabled",
                       "video_offset_ms", "audio_offset_ms", "auto_sync", "backends", "gpu")
        if any(old.get(k) != new.get(k) for k in timing_keys):
            self._clear_timing()
        if any(old.get(k) != new.get(k) for k in video_keys) and self._video_running:
            self.stop_video()
            self.start_video()
        if any(old.get(k) != new.get(k) for k in audio_keys) and self._audio_running:
            self.stop_audio()
            self.start_audio()
        self.telemetry.gpu = new.get("gpu", 0)
        self._update_sync()

    def run_benchmark(self, kind, iterations=30):
        from unified_live.benchmark import benchmark_manager

        if self._video_running or self._audio_running:
            raise RuntimeError("Stop live pipelines before benchmarking to avoid contention")
        return benchmark_manager(self.manager, kind, iterations)

    def close(self):
        self.stop_video()
        self.stop_audio()
        self.telemetry.close()

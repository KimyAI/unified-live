"""Modern, manager-only desktop UI for Unified Live."""

from __future__ import annotations

import json
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


ACCENT = "#42d6c5"
MUTED = "#9aa8b8"


def _nested(mapping: Any, key: str, default: Any = None) -> Any:
    """Read dotted keys while also accepting flat settings dictionaries."""
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return mapping[key]
    current: Any = mapping
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


class _WorkerSignals(QObject):
    finished = Signal(object, object, object)


class _TaskWorker(QRunnable):

    def __init__(self, operation: Callable[[], Any]):
        super().__init__()
        self.operation = operation
        self.signals = _WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self, self.operation(), None)
        except Exception as exc:  # surfaced in the UI, never swallowed
            self.signals.finished.emit(self, None, f"{exc}\n{traceback.format_exc()}")


class MainWindow(QMainWindow):
    """The UI depends exclusively on the EngineManager façade."""

    PAGES = [
        ("LIVE", "◉"),
        ("FACE", "◌"),
        ("VOICE", "♫"),
        ("TRAIN", "▤"),
        ("PROFILES", "▣"),
        ("DEVICES", "⌁"),
        ("PERFORMANCE", "⌁"),
        ("SETTINGS", "⚙"),
        ("BACKENDS", "⬡"),
    ]

    def __init__(self, manager: Any):
        super().__init__()
        self.manager = manager
        self._workers: dict[_TaskWorker, tuple[str, Callable[[Any], None] | None]] = {}
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._busy = False
        self._closing = False
        self._close_retry_scheduled = False
        self._task_queue: list[tuple[str, Callable[[], Any], Callable[[Any], None] | None]] = []
        self._last_state: dict[str, Any] = {}
        self._device_snapshot: dict[str, Any] = {}
        self._pending_path_options: dict[str, dict[str, str]] = {"face": {}, "voice": {}}
        self._benchmark_dialog: QDialog | None = None
        self.setWindowTitle("Unified Live")
        self.resize(1280, 820)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(self._stylesheet())
        self._build_shell()
        self._build_pages()
        self.nav.currentRowChanged.connect(self._navigate)
        self.nav.setCurrentRow(0)
        self._refresh_state()
        self._frame_timer = QTimer(self)
        self._frame_timer.timeout.connect(self._refresh_frame)
        self._frame_timer.start(40)
        self._state_timer = QTimer(self)
        self._state_timer.timeout.connect(self._refresh_state)
        self._state_timer.start(1000)

    @staticmethod
    def _stylesheet() -> str:
        return f"""
        QWidget {{ background: #11171e; color: #e8edf2; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
        QLabel, QCheckBox {{ background: transparent; }}
        QWidget#liveSyncPanel, QWidget#liveMetricPanel {{ background: transparent; }}
        QMainWindow {{ background: #0d1218; }}
        QFrame#sidebar {{ background: #0c1117; border-right: 1px solid #26323d; }}
        QLabel#brand {{ color: #f1f7fa; font-size: 18px; font-weight: 700; padding: 7px 4px 18px; }}
        QLabel#brandDot {{ color: {ACCENT}; font-size: 24px; }}
        QListWidget#nav {{ background: transparent; border: 0; outline: 0; }}
        QListWidget#nav::item {{ padding: 12px 12px; margin: 2px 0; border-radius: 8px; color: #aab6c2; }}
        QListWidget#nav::item:selected {{ color: #eafffb; background: #17312f; border-left: 3px solid {ACCENT}; }}
        QLabel#pageTitle {{ color: #f0f5f8; font-size: 24px; font-weight: 650; }}
        QLabel#muted {{ color: {MUTED}; }}
        QLabel#sectionTitle {{ color: #dfe8ed; font-size: 15px; font-weight: 650; }}
        QFrame.card {{ background: #171f28; border: 1px solid #28343f; border-radius: 12px; }}
        QPushButton {{ background: #25313c; border: 1px solid #364450; border-radius: 7px; padding: 9px 14px; color: #eaf0f4; }}
        QPushButton:hover {{ background: #2c3b47; border-color: #4b606d; }}
        QPushButton[primary="true"] {{ background: #137d75; border-color: #168d83; color: white; font-weight: 650; }}
        QPushButton[primary="true"]:hover {{ background: #19958a; }}
        QPushButton:disabled {{ color: #65717a; background: #1b232b; }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{ background: #0f151b; border: 1px solid #34414c; border-radius: 6px; padding: 7px; selection-background-color: #1a8279; }}
        QComboBox::drop-down {{ border: 0; width: 24px; }}
        QCheckBox {{ spacing: 9px; }}
        QCheckBox::indicator {{ width: 17px; height: 17px; }}
        QTableWidget {{ background: #121920; alternate-background-color: #171f27; border: 1px solid #2a3742; gridline-color: #27343e; border-radius: 8px; }}
        QHeaderView::section {{ background: #1b252e; color: #aebbc4; padding: 8px; border: 0; }}
        QScrollArea {{ border: 0; }}
        """

    def _build_shell(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 18, 14, 16)
        brand_row = QHBoxLayout()
        dot = QLabel("●")
        dot.setObjectName("brandDot")
        brand = QLabel("UNIFIED LIVE")
        brand.setObjectName("brand")
        self.brand_label = brand
        brand_row.addWidget(dot)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        side.addLayout(brand_row)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for name, icon in self.PAGES:
            item = QListWidgetItem(f"{icon}     {name}")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.nav.addItem(item)
        side.addWidget(self.nav, 1)
        self.sidebar_status = QLabel("●  Manager connecting")
        self.sidebar_status.setObjectName("muted")
        side.addWidget(self.sidebar_status)
        layout.addWidget(sidebar)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(28, 23, 28, 22)
        main_layout.setSpacing(16)
        header = QHBoxLayout()
        self.title = QLabel("LIVE")
        self.title.setObjectName("pageTitle")
        header.addWidget(self.title)
        header.addStretch()
        self.top_status = QLabel("●  Idle")
        self.top_status.setObjectName("muted")
        header.addWidget(self.top_status)
        main_layout.addLayout(header)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)
        layout.addWidget(main, 1)
        self.setCentralWidget(root)

    def _page(self, title: str, subtitle: str = "") -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(14)
        if subtitle:
            label = QLabel(subtitle)
            label.setObjectName("muted")
            outer.addWidget(label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
        return page, outer

    def _card(self, parent: QVBoxLayout | QHBoxLayout, title: str | None = None) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setProperty("class", "card")
        frame.setStyleSheet("QFrame[class='card'] { background:#171f28; border:1px solid #28343f; border-radius:12px; }")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(11)
        if title:
            label = QLabel(title)
            label.setObjectName("sectionTitle")
            layout.addWidget(label)
        parent.addWidget(frame)
        return frame, layout

    def _build_pages(self) -> None:
        self._build_live()
        self._build_face()
        self._build_voice()
        self._build_train()
        self._build_profiles()
        self._build_devices()
        self._build_performance()
        self._build_settings()
        self._build_backends()

    def _build_live(self) -> None:
        page, outer = self._page("LIVE", "Independent capture and effect controls. Camera starts off.")
        row = QHBoxLayout()
        preview_card, preview = self._card(row, "LIVE PREVIEW")
        self.preview = QLabel("Camera is off\n\nStart video to preview")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(500, 320)
        self.preview.setStyleSheet("background:#090d11; border-radius:8px; color:#6f7e88; font-size:15px;")
        preview.addWidget(self.preview, 1)
        self.preview_meta = QLabel("No frame available")
        self.preview_meta.setObjectName("muted")
        preview.addWidget(self.preview_meta)
        ctl_card, ctl = self._card(row, "SESSION")
        self.video_button = QPushButton("Start video")
        self.video_button.setProperty("primary", True)
        self.video_button.clicked.connect(self._toggle_video)
        ctl.addWidget(self.video_button)
        self.audio_button = QPushButton("Start audio")
        self.audio_button.clicked.connect(self._toggle_audio)
        ctl.addWidget(self.audio_button)
        engine_row = QHBoxLayout()
        face_setup = QPushButton("Face setup")
        face_setup.clicked.connect(lambda: self.nav.setCurrentRow(1))
        voice_setup = QPushButton("Voice setup")
        voice_setup.clicked.connect(lambda: self.nav.setCurrentRow(2))
        engine_row.addWidget(face_setup)
        engine_row.addWidget(voice_setup)
        ctl.addLayout(engine_row)
        ctl.addSpacing(8)
        self.face_toggle = QCheckBox("Face effect")
        self.face_toggle.setObjectName("liveFaceToggle")
        self.face_toggle.toggled.connect(lambda value: self._async("Set face effect", lambda: self.manager.set_enabled("face", value)))
        ctl.addWidget(self.face_toggle)
        self.voice_toggle = QCheckBox("Voice effect")
        self.voice_toggle.setObjectName("liveVoiceToggle")
        self.voice_toggle.toggled.connect(lambda value: self._async("Set voice effect", lambda: self.manager.set_enabled("voice", value)))
        ctl.addWidget(self.voice_toggle)
        ctl.addStretch()
        self.live_summary = QLabel("Video and audio pipelines are independent.")
        self.live_summary.setWordWrap(True)
        self.live_summary.setObjectName("muted")
        ctl.addWidget(self.live_summary)
        row.addWidget(preview_card, 3)
        row.addWidget(ctl_card, 1)
        outer.addLayout(row)
        bottom = QHBoxLayout()
        self.live_engine = QLabel("Face engine: —\nVoice engine: —")
        self.live_engine.setObjectName("muted")
        bottom.addWidget(self.live_engine)
        bottom.addStretch()
        outer.addLayout(bottom)
        live_card, live_layout = self._card(outer, "AUTO SYNC & LIVE METRICS")
        details = QHBoxLayout()
        live_layout.addLayout(details)
        self._live_sync_panel = QWidget()
        self._live_sync_panel.setObjectName("liveSyncPanel")
        self._live_sync_panel.setMaximumWidth(450)
        sync_layout = QVBoxLayout(self._live_sync_panel)
        sync_layout.setContentsMargins(0, 0, 0, 0)
        details.addWidget(self._live_sync_panel)
        self.live_auto_sync = QCheckBox("Auto-align video and audio")
        self.live_auto_sync.toggled.connect(self._set_auto_sync)
        sync_layout.addWidget(self.live_auto_sync)
        self.live_video_offset = self._double_spin(-2000, 2000, " ms")
        self.live_audio_offset = self._double_spin(-2000, 2000, " ms")
        self.live_video_offset.setMaximumWidth(155)
        self.live_audio_offset.setMaximumWidth(155)
        self.live_video_offset.valueChanged.connect(lambda value: self._setting("video_offset_ms", value))
        self.live_audio_offset.valueChanged.connect(lambda value: self._setting("audio_offset_ms", value))
        offsets = QHBoxLayout()
        offsets.addWidget(QLabel("Video"))
        offsets.addWidget(self.live_video_offset)
        offsets.addSpacing(14)
        offsets.addWidget(QLabel("Audio"))
        offsets.addWidget(self.live_audio_offset)
        offsets.addStretch(1)
        sync_layout.addLayout(offsets)
        self.live_sync_measure = QLabel("Measured delays: Indisponible")
        self.live_sync_measure.setObjectName("muted")
        self.live_sync_measure.setWordWrap(True)
        self.live_sync_measure.setMinimumWidth(0)
        sync_layout.addWidget(self.live_sync_measure)
        metric_panel = QWidget()
        metric_panel.setObjectName("liveMetricPanel")
        metrics_layout = QVBoxLayout(metric_panel)
        metrics_layout.setContentsMargins(0, 0, 0, 0)
        metrics_layout.setSpacing(5)
        metrics_heading = QLabel("LIVE METRICS")
        metrics_heading.setObjectName("sectionTitle")
        metrics_layout.addWidget(metrics_heading)
        self.live_metrics: dict[str, QLabel] = {}
        for key in ("fps", "gpu", "video", "audio"):
            label = QLabel("Indisponible")
            label.setObjectName("muted")
            label.setMinimumWidth(0)
            label.setWordWrap(True)
            self.live_metrics[key] = label
            metrics_layout.addWidget(label)
        details.addWidget(metric_panel, 1)
        self._live_details_layout = details

    def _build_face(self) -> None:
        page, outer = self._page("FACE", "Choose an installed adapter and edit its supported options.")
        card, form = self._card(outer, "FACE ENGINE")
        self.face_engine = QComboBox()
        self.face_engine.setObjectName("faceEngine")
        self.face_engine.currentIndexChanged.connect(lambda: self._select_engine("face", self.face_engine))
        form.addLayout(self._form_row("Engine", self.face_engine))
        self.face_status = QLabel("Backend status unavailable")
        self.face_status.setObjectName("muted")
        form.addLayout(self._form_row("Status", self.face_status))
        self.face_source = QLineEdit()
        self.face_source.setObjectName("faceSource")
        self._add_path_picker(form, "Source image", self.face_source, "source_image")
        self.face_model_path = QLineEdit()
        self.face_model_path.setObjectName("faceModelPath")
        self._add_path_picker(form, "Model path", self.face_model_path, "model_path")
        self.face_options = QPlainTextEdit()
        self.face_options.setObjectName("faceOptions")
        self.face_options.setPlaceholderText('{"strength": 1.0}')
        self.face_options.setMaximumHeight(220)
        form.addWidget(QLabel("Advanced options (JSON)"))
        form.addWidget(self.face_options)
        apply_face = QPushButton("Apply face options")
        apply_face.clicked.connect(lambda: self._apply_options("face", self.face_options))
        form.addWidget(apply_face)
        form.addStretch()

    def _build_voice(self) -> None:
        page, outer = self._page("VOICE", "Select the voice model and pitch controls exposed by the manager.")
        card, form = self._card(outer, "VOICE ENGINE")
        self.voice_engine = QComboBox()
        self.voice_engine.setObjectName("voiceEngine")
        self.voice_engine.currentIndexChanged.connect(lambda: self._select_engine("voice", self.voice_engine))
        form.addLayout(self._form_row("Engine / model", self.voice_engine))
        self.voice_status = QLabel("Backend status unavailable")
        self.voice_status.setObjectName("muted")
        form.addLayout(self._form_row("Status", self.voice_status))
        self.voice_reference = QLineEdit()
        self.voice_reference.setObjectName("voiceReference")
        self._add_path_picker(form, "Reference audio", self.voice_reference, "reference_audio")
        self.voice_model_path = QLineEdit()
        self.voice_model_path.setObjectName("voiceModelPath")
        self._add_path_picker(form, "Model path", self.voice_model_path, "voice_model_path")
        self.pitch = QSpinBox()
        self.pitch.setRange(-24, 24)
        self.pitch.setSuffix(" semitones")
        self.pitch.setObjectName("pitch")
        self.pitch.valueChanged.connect(self._change_pitch)
        form.addLayout(self._form_row("Pitch (backend capability required)", self.pitch))
        self.voice_options = QPlainTextEdit()
        self.voice_options.setObjectName("voiceOptions")
        self.voice_options.setMaximumHeight(190)
        form.addWidget(QLabel("Advanced options (JSON)"))
        form.addWidget(self.voice_options)
        apply_voice = QPushButton("Apply voice options")
        apply_voice.clicked.connect(lambda: self._apply_options("voice", self.voice_options))
        form.addWidget(apply_voice)
        form.addStretch()

    def _build_train(self) -> None:
        page, outer = self._page("TRAIN", "Training jobs are not implemented in this phase.")
        card, form = self._card(outer, "VOICE MODEL TRAINING · PHASE 5 — PLANNED")
        fields = [("Dataset", "Select a local dataset when training is available"), ("Model", "—"), ("Sample rate", "—"), ("Epochs", "—"), ("Batch size", "—"), ("GPU", "—"), ("F0 method", "—")]
        self.train_values: dict[str, QLabel] = {}
        for field, initial in fields:
            label = QLabel(initial)
            label.setObjectName("muted")
            self.train_values[field] = label
            form.addLayout(self._form_row(field, label))
        note = QLabel("Dataset preparation, feature extraction, training and export will appear here after the supervised job runner is implemented.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        form.addWidget(note)
        for text in ("Prepare dataset", "Extract features", "Start training", "Cancel job"):
            button = QPushButton(text)
            button.setEnabled(False)
            form.addWidget(button)
        form.addStretch()

    def _build_profiles(self) -> None:
        page, outer = self._page("PROFILES", "Save and restore the current engine, device and synchronization settings.")
        card, form = self._card(outer, "PROFILE LIBRARY")
        row = QHBoxLayout()
        self.profile_name = QLineEdit()
        self.profile_name.setObjectName("profileName")
        self.profile_name.setPlaceholderText("Profile name")
        row.addWidget(self.profile_name, 1)
        save = QPushButton("Save profile")
        save.setProperty("primary", True)
        save.clicked.connect(self._save_profile)
        row.addWidget(save)
        form.addLayout(row)
        self.profile_list = QListWidget()
        self.profile_list.setObjectName("profileList")
        form.addWidget(self.profile_list, 1)
        load = QPushButton("Load selected profile")
        load.clicked.connect(self._load_profile)
        form.addWidget(load)
        form.addStretch()

    def _build_devices(self) -> None:
        page, outer = self._page("DEVICES", "Select physical inputs and optional virtual outputs. No device is assumed present.")
        card, form = self._card(outer, "CAPTURE AND OUTPUT")
        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("cameraDevice")
        self.camera_combo.currentIndexChanged.connect(lambda: self._device_selected("camera_index", self.camera_combo))
        self.microphone_combo = QComboBox()
        self.microphone_combo.setObjectName("microphoneDevice")
        self.microphone_combo.currentIndexChanged.connect(lambda: self._device_selected("microphone_index", self.microphone_combo))
        self.output_combo = QComboBox()
        self.output_combo.setObjectName("audioOutputDevice")
        self.output_combo.currentIndexChanged.connect(lambda: self._device_selected("audio_output_index", self.output_combo))
        form.addLayout(self._form_row("Camera", self.camera_combo))
        form.addLayout(self._form_row("Microphone", self.microphone_combo))
        form.addLayout(self._form_row("Audio output", self.output_combo))
        self.virtual_camera = QCheckBox("Enable virtual camera output")
        self.virtual_camera.toggled.connect(lambda value: self._setting("virtual_camera", value))
        self.virtual_audio = QCheckBox("Enable virtual audio output")
        self.virtual_audio.toggled.connect(lambda value: self._setting("virtual_audio", value))
        form.addWidget(self.virtual_camera)
        form.addWidget(self.virtual_audio)
        refresh = QPushButton("Refresh devices")
        refresh.clicked.connect(self._discover_devices)
        form.addWidget(refresh)
        form.addStretch()

    def _build_performance(self) -> None:
        page, outer = self._page("PERFORMANCE", "Measurements come from active pipelines and host telemetry.")
        card, form = self._card(outer, "RUNTIME TELEMETRY")
        self.perf_values: dict[str, QLabel] = {}
        for title, key in [("Input FPS", "fps_input"), ("Output FPS", "fps_output"), ("Camera capture", "camera_capture_ms"), ("Face inference", "face_inference_ms"), ("Video output", "video_output_ms"), ("Video total", "video_total_ms"), ("Dropped frames", "dropped_frames"), ("Microphone capture", "microphone_capture_ms"), ("Voice inference", "voice_inference_ms"), ("Audio output", "audio_output_ms"), ("Audio total", "audio_total_ms"), ("Audio underruns", "audio_underruns"), ("GPU", "gpu_name"), ("GPU utilization", "gpu_utilization"), ("VRAM used / total", "vram"), ("CPU utilization", "cpu_percent"), ("RAM utilization", "ram_percent"), ("RAM used", "ram_used_gb"), ("Video delay", "video_delay_ms"), ("Audio delay", "audio_delay_ms"), ("Auto sync samples valid", "sync_valid")]:
            label = QLabel("Indisponible")
            label.setObjectName("muted")
            self.perf_values[key] = label
            form.addLayout(self._form_row(title, label))
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("performanceProfile")
        self.profile_combo.addItems(["QUALITY", "BALANCED", "LOW LATENCY"])
        self.profile_combo.currentTextChanged.connect(self._apply_performance_profile)
        form.addLayout(self._form_row("Processing profile", self.profile_combo))
        self.auto_sync = QCheckBox("Auto Sync measured video and audio")
        self.auto_sync.setObjectName("autoSync")
        self.auto_sync.toggled.connect(self._set_auto_sync)
        form.addWidget(self.auto_sync)
        offsets = QHBoxLayout()
        self.video_offset = self._double_spin(-2000, 2000, " ms")
        self.audio_offset = self._double_spin(-2000, 2000, " ms")
        self.video_offset.valueChanged.connect(lambda value: self._setting("video_offset_ms", value))
        self.audio_offset.valueChanged.connect(lambda value: self._setting("audio_offset_ms", value))
        offsets.addWidget(QLabel("Video offset"))
        offsets.addWidget(self.video_offset)
        offsets.addSpacing(14)
        offsets.addWidget(QLabel("Audio offset"))
        offsets.addWidget(self.audio_offset)
        form.addLayout(offsets)
        benchmark = QPushButton("Run measured benchmark")
        benchmark.setObjectName("runBenchmark")
        benchmark.clicked.connect(self._run_benchmark)
        form.addWidget(benchmark)
        form.addStretch()

    def _build_settings(self) -> None:
        page, outer = self._page("SETTINGS", "These controls map to persisted manager settings.")
        card, form = self._card(outer, "STREAM SETTINGS")
        self.width_spin = self._spin(160, 7680)
        self.height_spin = self._spin(120, 4320)
        self.fps_spin = self._spin(1, 240)
        self.sample_rate_spin = self._spin(8000, 192000, " Hz")
        self.chunk_spin = self._spin(64, 8192, " samples")
        for name, widget, key in [("Width", self.width_spin, "width"), ("Height", self.height_spin, "height"), ("Frame rate", self.fps_spin, "fps"), ("Sample rate", self.sample_rate_spin, "sample_rate"), ("Audio chunk", self.chunk_spin, "chunk_size")]:
            form.addLayout(self._form_row(name, widget))
            widget.valueChanged.connect(lambda value, setting=key: self._setting(setting, value))
        self.demo_check = QCheckBox("Synthetic demo source")
        self.demo_check.setObjectName("demoMode")
        self.demo_check.toggled.connect(lambda value: self._setting("demo", value))
        form.addWidget(self.demo_check)
        self.advanced_check = QCheckBox("Advanced mode")
        self.advanced_check.toggled.connect(lambda value: self._setting("advanced", value))
        form.addWidget(self.advanced_check)
        backend_card, backend_form = self._card(outer, "EXTERNAL BACKEND ENVIRONMENTS")
        self.backend_settings = QPlainTextEdit()
        self.backend_settings.setObjectName("backendSettings")
        self.backend_settings.setPlaceholderText('{"face": {"python": "...", "root": "..."}}')
        self.backend_settings.setMaximumHeight(150)
        backend_form.addWidget(QLabel("Python executable and checkout paths for isolated workers"))
        backend_form.addWidget(self.backend_settings)
        apply_backends = QPushButton("Apply backend paths")
        apply_backends.clicked.connect(self._apply_backend_settings)
        backend_form.addWidget(apply_backends)
        self.logs = QPlainTextEdit()
        self.logs.setObjectName("developerLogs")
        self.logs.setReadOnly(True)
        self.logs.setMaximumBlockCount(300)
        outer.addWidget(QLabel("DEVELOPER LOGS"))
        outer.addWidget(self.logs, 1)
        form.addStretch()

    def _build_backends(self) -> None:
        page, outer = self._page("BACKENDS", "Installed and available adapters are reported by the registry.")
        card, form = self._card(outer, "BACKEND REGISTRY")
        self.backend_table = QTableWidget(0, 8)
        self.backend_table.setObjectName("backendTable")
        self.backend_table.setHorizontalHeaderLabels(["Backend", "Kind", "Installed", "Available", "Configured", "Version", "Capabilities", "State"])
        self.backend_table.horizontalHeader().setStretchLastSection(True)
        self.backend_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.backend_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        form.addWidget(self.backend_table, 1)
        row = QHBoxLayout()
        self.restart_kind = QComboBox()
        self.restart_kind.addItems(["face", "voice"])
        row.addWidget(self.restart_kind)
        restart = QPushButton("Restart backend")
        restart.clicked.connect(self._restart_backend)
        row.addWidget(restart)
        row.addStretch()
        form.addLayout(row)
        self.backend_docs = QLabel(
            'Install backends separately. Unified Live does not download engines. '
            '<a href="https://github.com/somanchiu/ReSwapper">ReSwapper</a> · '
            '<a href="https://github.com/Plachtaa/seed-vc">Seed-VC</a> · '
            '<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI">RVC</a> · '
            '<a href="https://github.com/w-okada/voice-changer">w-okada</a>'
        )
        self.backend_docs.setWordWrap(True)
        self.backend_docs.setOpenExternalLinks(True)
        self.backend_docs.setObjectName("muted")
        form.addWidget(self.backend_docs)

    @staticmethod
    def _form_row(label: str, widget: QWidget) -> QHBoxLayout:
        row = QHBoxLayout()
        name = QLabel(label)
        name.setMinimumWidth(150)
        row.addWidget(name)
        row.addWidget(widget, 1)
        return row

    def _add_path_picker(self, layout: QVBoxLayout, label: str, field: QLineEdit, option_key: str) -> None:
        row = QHBoxLayout()
        title = QLabel(label)
        title.setMinimumWidth(150)
        row.addWidget(title)
        field.setReadOnly(True)
        row.addWidget(field, 1)
        button = QPushButton("Browse…")
        button.clicked.connect(lambda _checked=False: self._browse_option(field, option_key))
        row.addWidget(button)
        layout.addLayout(row)

    def _browse_option(self, field: QLineEdit, key: str) -> None:
        if key == "voice_model_path":
            key = self._voice_model_option_key()
        filters = {"source_image": "Images (*.png *.jpg *.jpeg *.webp)", "reference_audio": "Audio (*.wav *.mp3 *.flac *.ogg)", "model_path": "Models (*.onnx *.pth *.pt *.index);;All files (*)"}
        path, _ = QFileDialog.getOpenFileName(self, f"Select {key.replace('_', ' ')}", field.text(), filters.get(key, "All files (*)"))
        if not path:
            return
        field.setText(path)
        self._sync_path_option(key, path)

    def _voice_model_option_key(self) -> str:
        settings = self._last_state.get("settings", {})
        engine_id = str(settings.get("voice_engine", "")).lower().replace("_", "-")
        descriptor = next((item for item in self._last_state.get("engines", []) if isinstance(item, dict) and item.get("id") == settings.get("voice_engine")), {})
        engine_name = str(descriptor.get("name", "")).lower() if isinstance(descriptor, dict) else ""
        return "checkpoint_path" if "seed" in engine_id or "seed" in engine_name else "model_path"

    def _sync_path_option(self, key: str, value: str) -> None:
        kind = "face" if key == "source_image" or self.stack.currentIndex() == 1 else "voice"
        options = self._last_state.get("settings", {}).get(f"{kind}_options", {})
        options = dict(options) if isinstance(options, dict) else {}
        options[key] = value
        self._pending_path_options[kind][key] = value
        editor = self.face_options if kind == "face" else self.voice_options
        editor.setPlainText(json.dumps(options, indent=2, ensure_ascii=False))

    @staticmethod
    def _spin(minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
        widget = QSpinBox()
        widget.setRange(minimum, maximum)
        widget.setSuffix(suffix)
        return widget

    @staticmethod
    def _double_spin(minimum: float, maximum: float, suffix: str = "") -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(1)
        widget.setSuffix(suffix)
        return widget

    def _navigate(self, row: int) -> None:
        if row < 0:
            return
        page = self.nav.item(row).data(Qt.ItemDataRole.UserRole)
        self.title.setText(page)
        self.stack.setCurrentIndex(row)
        if page == "PROFILES":
            self._refresh_profiles()
        if page == "BACKENDS":
            self._render_backends(self._last_state)

    def _async(self, label: str, operation: Callable[[], Any], callback: Callable[[Any], None] | None = None) -> None:
        if self._busy:
            self._task_queue.append((label, operation, callback))
            return
        self._busy = True
        self.sidebar_status.setText(f"●  {label}…")
        worker = _TaskWorker(operation)
        worker.signals.finished.connect(self._worker_finished, Qt.ConnectionType.QueuedConnection)
        self._workers[worker] = (label, callback)
        self._thread_pool.start(worker)

    @Slot(object, object, object)
    def _worker_finished(self, worker: Any, result: Any, error: Any) -> None:
        context = self._workers.get(worker)
        if context is None:
            return
        label, callback = context
        self.sidebar_status.setText("●  Ready" if not error else "●  Action failed")
        if self._closing:
            pass
        elif error:
            self._show_error(label, str(error))
        elif callback:
            callback(result)
        if not self._closing:
            self._refresh_state()
        self._workers.pop(worker, None)
        self._busy = False
        if not self._closing:
            QTimer.singleShot(0, self._run_next_task)

    def _run_next_task(self) -> None:
        if self._busy or not self._task_queue:
            return
        label, operation, callback = self._task_queue.pop(0)
        self._async(label, operation, callback)

    def _show_error(self, title: str, detail: str) -> None:
        self.sidebar_status.setText("●  Action failed")
        self.logs.appendPlainText(f"[{title}] {detail.splitlines()[0]}")
        QMessageBox.critical(self, title, detail.splitlines()[0])

    def _refresh_state(self) -> None:
        try:
            state = self.manager.get_state() or {}
        except Exception as exc:
            self.sidebar_status.setText("●  Manager unavailable")
            self.top_status.setText(f"●  {exc}")
            return
        self._last_state = state
        settings = state.get("settings", {}) or {}
        app_name = settings.get("app_name")
        if app_name:
            self.setWindowTitle(str(app_name))
            self.brand_label.setText(str(app_name).upper())
        backends = state.get("backends", {}) or {}
        video = backends.get("video", {}) or {}
        audio = backends.get("audio", {}) or {}
        face = backends.get("face", {}) or {}
        voice = backends.get("voice", {}) or {}
        runtime = state.get("runtime", {}) or {}
        self.sidebar_status.setText("●  Manager ready")
        video_running = bool(runtime.get("video_running", video.get("state") == "running"))
        audio_running = bool(runtime.get("audio_running", audio.get("state") == "running"))
        running = video_running or audio_running
        self.top_status.setText("●  Live" if running else "●  Idle")
        self.video_button.setText("Stop video" if video_running else "Start video")
        self.audio_button.setText("Stop audio" if audio_running else "Start audio")
        for widget, value in ((self.face_toggle, settings.get("face_enabled", False)), (self.voice_toggle, settings.get("voice_enabled", False)), (self.virtual_camera, settings.get("virtual_camera", False)), (self.virtual_audio, settings.get("virtual_audio", False)), (self.auto_sync, settings.get("auto_sync", False)), (self.demo_check, settings.get("demo", False)), (self.advanced_check, settings.get("advanced", False))):
            widget.blockSignals(True)
            widget.setChecked(bool(value))
            widget.blockSignals(False)
        self.live_auto_sync.blockSignals(True)
        self.live_auto_sync.setChecked(bool(settings.get("auto_sync", False)))
        self.live_auto_sync.blockSignals(False)
        for widget, key in ((self.video_offset, "video_offset_ms"), (self.audio_offset, "audio_offset_ms"), (self.width_spin, "width"), (self.height_spin, "height"), (self.fps_spin, "fps"), (self.sample_rate_spin, "sample_rate"), (self.chunk_spin, "chunk_size")):
            widget.blockSignals(True)
            raw_value = settings.get(key, 0) or 0
            widget.setValue(float(raw_value) if isinstance(widget, QDoubleSpinBox) else int(raw_value))
            widget.blockSignals(False)
        for widget, key in ((self.live_video_offset, "video_offset_ms"), (self.live_audio_offset, "audio_offset_ms")):
            widget.blockSignals(True)
            widget.setValue(float(settings.get(key, 0) or 0))
            widget.blockSignals(False)
        self._sync_engines(state.get("engines", []), settings)
        voice_options = settings.get("voice_options", {})
        self.pitch.blockSignals(True)
        self.pitch.setValue(int(voice_options.get("pitch_shift", 0)) if isinstance(voice_options, dict) else 0)
        self.pitch.blockSignals(False)
        voice_descriptor = next((item for item in state.get("engines", []) if isinstance(item, dict) and item.get("id") == settings.get("voice_engine")), {})
        capabilities = voice_descriptor.get("capabilities", []) if isinstance(voice_descriptor, dict) else []
        supports_pitch = ("pitch" in capabilities or "pitch_shift" in capabilities) if isinstance(capabilities, list) else False
        self.pitch.setEnabled(supports_pitch)
        self.pitch.setToolTip("This backend declares pitch control support." if supports_pitch else "The selected backend does not declare pitch support.")
        face_options = settings.get("face_options", {}) if isinstance(settings.get("face_options"), dict) else {}
        voice_options = voice_options if isinstance(voice_options, dict) else {}
        face_options.update(self._pending_path_options["face"])
        voice_options.update(self._pending_path_options["voice"])
        voice_model_key = self._voice_model_option_key()
        for field, value in ((self.face_source, face_options.get("source_image", "")), (self.face_model_path, face_options.get("model_path", "")), (self.voice_reference, voice_options.get("reference_audio", "")), (self.voice_model_path, voice_options.get(voice_model_key, ""))):
            if not field.hasFocus():
                field.setText(str(value))
        self.face_status.setText(self._status_text(face))
        self.voice_status.setText(self._status_text(voice))
        self.train_values["Model"].setText(str(settings.get("voice_engine") or "Indisponible"))
        self.train_values["Sample rate"].setText(f"{settings.get('sample_rate')} Hz" if settings.get("sample_rate") else "Indisponible")
        self.live_engine.setText(f"Face engine: {settings.get('face_engine', '—')}  ·  {self._status_text(face)}\nVoice engine: {settings.get('voice_engine', '—')}  ·  {self._status_text(voice)}")
        errors = [runtime.get(key) for key in ("video_error", "audio_error") if runtime.get(key)]
        self.live_summary.setText(" · ".join(map(str, errors)) if errors else ("Synthetic demo source enabled." if runtime.get("demo") else "Video and audio pipelines run independently."))
        selected_profile = {"quality": "QUALITY", "balanced": "BALANCED", "latency": "LOW LATENCY", "low_latency": "LOW LATENCY"}.get(str(settings.get("performance_profile", "balanced")).lower(), "BALANCED")
        index = self.profile_combo.findText(selected_profile)
        if index >= 0:
            self.profile_combo.blockSignals(True)
            self.profile_combo.setCurrentIndex(index)
            self.profile_combo.blockSignals(False)
        for kind, widget in (("face", self.face_options), ("voice", self.voice_options)):
            options = settings.get(f"{kind}_options", {})
            options = dict(options) if isinstance(options, dict) else {}
            options.update(self._pending_path_options[kind])
            text = json.dumps(options, indent=2, ensure_ascii=False)
            if not widget.hasFocus() and widget.toPlainText() != text:
                widget.setPlainText(text)
        backend_options = settings.get("backends", {})
        backend_text = json.dumps(backend_options, indent=2, ensure_ascii=False) if isinstance(backend_options, dict) else "{}"
        if not self.backend_settings.hasFocus() and self.backend_settings.toPlainText() != backend_text:
            self.backend_settings.setPlainText(backend_text)
        self._render_backends(state)
        self._render_devices(settings)
        self._render_logs(runtime)
        try:
            telemetry = self.manager.telemetry_snapshot() or {}
            self._render_telemetry(telemetry)
        except Exception:
            self._render_telemetry({})

    @staticmethod
    def _status_text(status: dict[str, Any]) -> str:
        state = status.get("state", "unavailable")
        engine = status.get("engine_id")
        latency = status.get("last_latency_ms")
        parts = [str(state)]
        if status.get("warming_up"):
            parts.append("warming up (silent)")
        if engine:
            parts.append(str(engine))
        if isinstance(latency, (int, float)):
            parts.append(f"{latency:.1f} ms")
        algorithmic_delay = status.get("algorithmic_delay_ms")
        if isinstance(algorithmic_delay, (int, float)):
            parts.append(f"{algorithmic_delay:.1f} ms algorithmic delay")
        error = status.get("error")
        if error:
            parts.append(str(error))
        return " · ".join(parts)

    def _sync_engines(self, engines: Any, settings: dict[str, Any]) -> None:
        engines = engines if isinstance(engines, list) else []
        for kind, combo in (("face", self.face_engine), ("voice", self.voice_engine)):
            current = str(settings.get(f"{kind}_engine", ""))
            matches = [item for item in engines if isinstance(item, dict) and item.get("kind") == kind]
            for target in [combo]:
                target.blockSignals(True)
                target.clear()
                target.addItem("No engine selected", "")
                for item in matches:
                    label = str(item.get("name", item.get("id", "Unknown")))
                    status = " · planned" if item.get("planned") else (" · not installed" if not item.get("installed") else "")
                    target.addItem(label + status, item.get("id"))
                    index = target.count() - 1
                    target.setItemData(index, item, Qt.ItemDataRole.ToolTipRole)
                index = target.findData(current)
                target.setCurrentIndex(index if index >= 0 else 0)
                target.blockSignals(False)

    def _render_backends(self, state: dict[str, Any]) -> None:
        descriptors = state.get("engines", []) if isinstance(state, dict) else []
        statuses = state.get("backends", {}) if isinstance(state, dict) else {}
        descriptors = descriptors if isinstance(descriptors, list) else []
        self.backend_table.setRowCount(len(descriptors))
        for row, descriptor in enumerate(descriptors):
            if not isinstance(descriptor, dict):
                continue
            status = statuses.get(descriptor.get("kind"), {}) if isinstance(statuses, dict) else {}
            values = [descriptor.get("name", descriptor.get("id", "—")), descriptor.get("kind", "—"), self._yesno(descriptor.get("installed")), self._yesno(descriptor.get("available")), self._yesno(descriptor.get("configured", descriptor.get("validated", False))), str(descriptor.get("version", "—")), ", ".join(map(str, descriptor.get("capabilities", []))) or "—", status.get("state", "stopped")]
            for column, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                self.backend_table.setItem(row, column, cell)

    @staticmethod
    def _yesno(value: Any) -> str:
        return "Unknown" if value is None else ("Yes" if value else "No")

    def _render_devices(self, settings: dict[str, Any]) -> None:
        runtime = self._last_state.get("runtime", {}) or {}
        devices = runtime.get("devices", {}) if isinstance(runtime, dict) else {}
        devices = devices if isinstance(devices, dict) else {}
        devices = self._device_snapshot or devices
        self._set_device_combo(self.camera_combo, devices.get("cameras", []), settings.get("camera_index"), "No camera device data")
        self._set_device_combo(self.microphone_combo, devices.get("microphones", []), settings.get("microphone_index"), "No microphone device data")
        self._set_device_combo(self.output_combo, devices.get("outputs", devices.get("audio_outputs", [])), settings.get("audio_output_index"), "No audio output data")

    def _set_device_combo(self, combo: QComboBox, entries: Any, selected: Any, empty: str) -> None:
        if combo.hasFocus():
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(empty, None)
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    label = item.get("name", str(item.get("index", "Device")))
                    value = item.get("index", item.get("id"))
                else:
                    label, value = str(item), item
                combo.addItem(str(label), value)
        index = combo.findData(selected)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _render_logs(self, runtime: dict[str, Any]) -> None:
        logs = runtime.get("logs", []) if isinstance(runtime, dict) else []
        text = "\n".join(map(str, logs[-250:])) if isinstance(logs, list) else str(logs or "")
        if text and self.logs.toPlainText() != text:
            self.logs.setPlainText(text)

    def _render_telemetry(self, telemetry: dict[str, Any]) -> None:
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        runtime = self._last_state.get("runtime", {}) if isinstance(self._last_state, dict) else {}
        runtime_metrics = runtime.get("metrics", {}) if isinstance(runtime, dict) else {}
        snapshot_metrics = telemetry.get("metrics", {}) if isinstance(telemetry.get("metrics"), dict) else {}
        source = {**runtime_metrics, **snapshot_metrics, **telemetry}
        sync = telemetry.get("sync", runtime.get("sync", {}) if isinstance(runtime, dict) else {})
        sync = sync if isinstance(sync, dict) else {}
        mapping = {
            "fps_input": "fps_input", "fps_output": "fps_output", "camera_capture_ms": "camera_capture_ms", "face_inference_ms": "face_inference_ms", "video_output_ms": "video_output_ms", "video_total_ms": "video_total_ms", "dropped_frames": "dropped_frames", "microphone_capture_ms": "microphone_capture_ms", "voice_inference_ms": "voice_inference_ms", "audio_output_ms": "audio_output_ms", "audio_total_ms": "audio_total_ms", "audio_underruns": "audio_underruns", "gpu_name": "gpu_name", "gpu_utilization": "gpu_utilization", "cpu_percent": "cpu_percent", "ram_percent": "ram_percent", "ram_used_gb": "ram_used_gb",
        }
        for key, metric in mapping.items():
            self.perf_values[key].setText(self._format_metric(source.get(metric), key))
        used, total = source.get("vram_used_gb"), source.get("vram_total_gb")
        self.perf_values["vram"].setText(f"{used:.1f} / {total:.1f} GB" if isinstance(used, (int, float)) and isinstance(total, (int, float)) else "Indisponible")
        for key, sync_key in (("video_delay_ms", "video_delay_ms"), ("audio_delay_ms", "audio_delay_ms"), ("sync_valid", "valid")):
            self.perf_values[key].setText(self._format_metric(sync.get(sync_key), key))
        gpu_name = telemetry.get("gpu_name")
        self.train_values["GPU"].setText(str(gpu_name) if gpu_name else "Indisponible")
        fps_in, fps_out = source.get("fps_input"), source.get("fps_output")
        self.live_metrics["fps"].setText(f"{self._format_metric(fps_in, 'fps_input')} / {self._format_metric(fps_out, 'fps_output')}")
        self.live_metrics["gpu"].setText(f"GPU {gpu_name or 'Indisponible'} · VRAM {self.perf_values['vram'].text()} · utilization {self._format_metric(source.get('gpu_utilization'), 'gpu_utilization')}")
        self.live_metrics["video"].setText(f"Video total {self._format_metric(source.get('video_total_ms'), 'video_total_ms')} · face inference {self._format_metric(source.get('face_inference_ms'), 'face_inference_ms')}")
        self.live_metrics["audio"].setText(f"Audio total {self._format_metric(source.get('audio_total_ms'), 'audio_total_ms')} · voice inference {self._format_metric(source.get('voice_inference_ms'), 'voice_inference_ms')}")
        self.live_sync_measure.setText(f"Measured delays: video {self.perf_values['video_delay_ms'].text()} · audio {self.perf_values['audio_delay_ms'].text()} · valid {self.perf_values['sync_valid'].text()}")

    @staticmethod
    def _format_metric(value: Any, key: str) -> str:
        if value is None:
            return "Indisponible"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        suffix = " ms" if key.endswith("_ms") else (" FPS" if key.startswith("fps_") else ("%" if key.endswith("_percent") or key == "gpu_utilization" else (" GB" if key.endswith("_gb") else "")))
        if isinstance(value, (int, float)):
            return f"{value:.1f}{suffix}"
        return str(value)

    def _refresh_frame(self) -> None:
        runtime = self._last_state.get("runtime", {}) if isinstance(self._last_state, dict) else {}
        if isinstance(runtime, dict) and not runtime.get("video_running", False):
            self.preview.clear()
            self.preview.setText("Camera is off\n\nStart video to preview")
            self.preview_meta.setText("No active video frame")
            return
        try:
            frame = self.manager.latest_frame()
        except Exception:
            frame = None
        if frame is None:
            return
        try:
            height, width, channels = frame.shape
            if channels == 3:
                image = QImage(frame.data, width, height, int(frame.strides[0]), QImage.Format.Format_BGR888).copy()
            elif channels == 4:
                image = QImage(frame.data, width, height, int(frame.strides[0]), QImage.Format.Format_RGBA8888).copy()
            else:
                return
            self.preview.setPixmap(QPixmap.fromImage(image).scaled(self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.preview_meta.setText(f"{width} × {height} · latest frame")
        except (AttributeError, TypeError, ValueError):
            return

    def _toggle_video(self) -> None:
        action = self.manager.stop_video if "Stop" in self.video_button.text() else self.manager.start_video
        self._async("Video pipeline", action)

    def _toggle_audio(self) -> None:
        action = self.manager.stop_audio if "Stop" in self.audio_button.text() else self.manager.start_audio
        self._async("Audio pipeline", action)

    def _setting(self, key: str, value: Any) -> None:
        self._async(f"Update {key}", lambda: self.manager.update_settings({key: value}))

    def _set_auto_sync(self, enabled: bool) -> None:
        self._setting("auto_sync", enabled)

    def _apply_performance_profile(self, display: str) -> None:
        modes = {
            "QUALITY": ("quality", 1920, 1080, 30, 1024),
            "BALANCED": ("balanced", 1280, 720, 30, 512),
            "LOW LATENCY": ("latency", 1280, 720, 60, 256),
        }
        profile, width, height, fps, chunk = modes.get(display, modes["BALANCED"])
        patch = {"performance_profile": profile, "width": width, "height": height, "fps": fps, "chunk_size": chunk}
        self._async("Apply performance profile", lambda: self.manager.update_settings(patch))

    def _select_engine(self, kind: str, combo: QComboBox) -> None:
        engine_id = combo.currentData()
        if not engine_id:
            return
        settings = self._last_state.get("settings", {}) or {}
        current_id = settings.get(f"{kind}_engine")
        if engine_id == current_id:
            opts = settings.get(f"{kind}_options", {})
            opts = dict(opts) if isinstance(opts, dict) else {}
            opts.update(self._pending_path_options[kind])
        else:
            backend_registry = settings.get("backends", {})
            backend_config = backend_registry.get(engine_id, {}) if isinstance(backend_registry, dict) else {}
            opts = backend_config.get("options", {}) if isinstance(backend_config, dict) else {}
            opts = dict(opts) if isinstance(opts, dict) else {}
        callback = None if engine_id == current_id else lambda _: self._pending_path_options[kind].clear()
        self._async(f"Select {kind} engine", lambda: self.manager.select_engine(kind, engine_id, options=opts), callback)

    def _apply_options(self, kind: str, editor: QPlainTextEdit) -> None:
        try:
            options = json.loads(editor.toPlainText() or "{}")
            if not isinstance(options, dict):
                raise ValueError("Options must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_error("Invalid options", str(exc))
            return
        engine_id = self._last_state.get("settings", {}).get(f"{kind}_engine")
        if not engine_id:
            self._setting(f"{kind}_options", options)
            self._pending_path_options[kind].clear()
            return
        self._async(f"Apply {kind} options", lambda: self.manager.select_engine(kind, engine_id, options=options), lambda _: self._pending_path_options[kind].clear())

    def _apply_backend_settings(self) -> None:
        try:
            options = json.loads(self.backend_settings.toPlainText() or "{}")
            if not isinstance(options, dict):
                raise ValueError("Backend settings must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            self._show_error("Invalid backend settings", str(exc))
            return
        self._setting("backends", options)

    def _restart_backend(self) -> None:
        kind = self.restart_kind.currentText()
        self._async(f"Restart {kind} backend", lambda: self.manager.restart_backend(kind))

    def _discover_devices(self) -> None:
        def refresh(result: Any) -> None:
            self._device_snapshot = result if isinstance(result, dict) else {}
            self._render_devices(self._last_state.get("settings", {}))
        self._async("Discover devices", self.manager.discover_devices, refresh)

    def _change_pitch(self, value: int) -> None:
        if not self.pitch.isEnabled():
            return
        settings = self._last_state.get("settings", {}) or {}
        options = settings.get("voice_options", {})
        options = dict(options) if isinstance(options, dict) else {}
        options["pitch_shift"] = value
        engine_id = settings.get("voice_engine")
        self._async("Set voice pitch", lambda: self.manager.select_engine("voice", engine_id, options=options))

    def _device_selected(self, key: str, combo: QComboBox) -> None:
        value = combo.currentData()
        if value is not None:
            self._setting(key, value)

    def _run_benchmark(self) -> None:
        benchmark = getattr(self.manager, "run_benchmark", None)
        if not callable(benchmark):
            self._show_error("Benchmark unavailable", "The manager does not expose a benchmark runner.")
            return
        self._async("Measured benchmark", lambda: benchmark(kind="synthetic", iterations=30), self._show_benchmark)

    def _show_benchmark(self, result: Any) -> None:
        if not isinstance(result, dict):
            self._show_error("Benchmark", "The manager returned no measured result.")
            return
        self.logs.appendPlainText("[benchmark] " + json.dumps(result, ensure_ascii=False))
        self._render_telemetry(result)
        if self._benchmark_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("Measured benchmark")
            dialog.setMinimumSize(560, 380)
            layout = QVBoxLayout(dialog)
            output = QPlainTextEdit()
            output.setObjectName("benchmarkResult")
            output.setReadOnly(True)
            layout.addWidget(output, 1)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.close)
            buttons.clicked.connect(dialog.close)
            layout.addWidget(buttons)
            self._benchmark_dialog = dialog
            self._benchmark_output = output
        self._benchmark_output.setPlainText(json.dumps(result, indent=2, ensure_ascii=False))
        self._benchmark_dialog.show()
        self._benchmark_dialog.raise_()

    def _refresh_profiles(self) -> None:
        try:
            selected = self.profile_list.currentItem().text() if self.profile_list.currentItem() else None
            names = self.manager.list_profiles() or []
            self.profile_list.clear()
            for name in names:
                self.profile_list.addItem(str(name))
            if selected:
                items = self.profile_list.findItems(selected, Qt.MatchFlag.MatchExactly)
                if items:
                    self.profile_list.setCurrentItem(items[0])
        except Exception as exc:
            self._show_error("List profiles", str(exc))

    def _save_profile(self) -> None:
        name = self.profile_name.text().strip()
        if not name:
            self._show_error("Save profile", "Enter a profile name.")
            return
        self._async("Save profile", lambda: self.manager.save_profile(name), lambda _: self._refresh_profiles())

    def _load_profile(self) -> None:
        item = self.profile_list.currentItem()
        if not item:
            return
        self._async("Load profile", lambda: self.manager.load_profile(item.text()), lambda _: self._refresh_state())

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if not self._closing:
            self._closing = True
            self._task_queue.clear()
            self._frame_timer.stop()
            self._state_timer.stop()
            self.setEnabled(False)
            cancel_backend = getattr(self.manager, "cancel_backend", None)
            if callable(cancel_backend):
                for kind in ("face", "voice"):
                    try:
                        cancel_backend(kind)
                    except Exception:
                        pass
        if self._thread_pool.activeThreadCount() or self._workers:
            event.ignore()
            if not self._close_retry_scheduled:
                self._close_retry_scheduled = True
                QTimer.singleShot(100, self._retry_close)
            return
        event.accept()

    def _retry_close(self) -> None:
        self._close_retry_scheduled = False
        self.close()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not hasattr(self, "_live_details_layout"):
            return
        if self.width() < 1160:
            self._live_details_layout.setDirection(QBoxLayout.Direction.TopToBottom)
            self._live_sync_panel.setMaximumWidth(16777215)
        else:
            self._live_details_layout.setDirection(QBoxLayout.Direction.LeftToRight)
            self._live_sync_panel.setMaximumWidth(450)

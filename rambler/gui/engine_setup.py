"""First-run dialog that installs prerequisites, builds whisper.cpp and downloads models."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QGroupBox, QHBoxLayout, QLabel,
                               QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QVBoxLayout)

from .. import setup_engine as se
from ..config import Config
from .workers import JobContext, Worker


class EngineSetupDialog(QDialog):
    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set up the transcription engine")
        self.resize(720, 620)
        self.cfg = cfg
        self.worker: Worker | None = None
        self.result_cfg: Config | None = None

        st = se.status(cfg)
        intro = QLabel(
            "Rambler runs Whisper locally. This installs an engine for your hardware and downloads the "
            "models. It takes a few minutes and a few GB, and only needs doing once." +
            (f"<br><br>Currently: faster-whisper {'installed' if st.faster_whisper else 'not installed'}, "
             f"whisper-cli {'found' if st.whisper_cli else 'missing'}." if (st.whisper_cli or st.faster_whisper) else ""))
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)

        # backend
        gb = QGroupBox("Engine")
        gl = QVBoxLayout(gb)
        self.backends = QButtonGroup(self)
        nvidia = se.has_nvidia()
        nvcc = se.find_nvcc()
        opts = [
            ("faster-whisper", "faster-whisper on " + ("your NVIDIA GPU" if nvidia else "CPU") +
             " · pip install, nothing to compile (recommended)", True),
            ("cuda", "whisper.cpp + CUDA" + ("" if nvcc else f" · installs {se.cuda_toolkit_package()}, ~3 GB"),
             nvidia),
            ("vulkan", "whisper.cpp + Vulkan (AMD/Intel/NVIDIA)", True),
            ("cpu", "whisper.cpp on CPU", True),
        ]
        default = se.default_backend()
        for key, label, enabled in opts:
            rb = QRadioButton(label)
            rb.setProperty("key", key)
            rb.setEnabled(enabled)
            rb.setProperty("allowed", enabled)
            rb.setChecked(key == default)
            self.backends.addButton(rb)
            gl.addWidget(rb)
        if not nvidia:
            gl.addWidget(QLabel("No NVIDIA driver detected."))

        # models
        mb = QGroupBox("Models")
        ml = QVBoxLayout(mb)
        self.model_boxes: dict[str, QCheckBox] = {}
        for name, (size, desc) in se.MODELS.items():
            cb = QCheckBox(f"{name}  ·  {size}  ·  {desc}")
            cb.setChecked(name == "large-v3")
            if any(m == f"ggml-{name}.bin" for m in st.models):
                cb.setText(cb.text() + "  ·  installed")
            self.model_boxes[name] = cb
            ml.addWidget(cb)

        self.apt = QCheckBox()
        self.apt.setChecked(True)
        self._refresh_apt()
        self.backends.buttonToggled.connect(lambda *_: self._refresh_apt())

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar_all = QProgressBar()
        self.step = QLabel("")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(3000)

        self.btn_start = QPushButton("Set up")
        self.btn_start.setDefault(True)
        self.btn_close = QPushButton("Close")
        self.btn_start.clicked.connect(self.start)
        self.btn_close.clicked.connect(self._close_or_cancel)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_start)

        lay = QVBoxLayout(self)
        lay.addWidget(intro)
        top = QHBoxLayout()
        top.addWidget(gb, 1)
        top.addWidget(mb, 1)
        lay.addLayout(top)
        lay.addWidget(self.apt)
        lay.addWidget(self.step)
        lay.addWidget(self.bar)
        lay.addWidget(self.bar_all)
        lay.addWidget(self.log, 1)
        lay.addLayout(row)

    def backend(self) -> str:
        return self.backends.checkedButton().property("key")

    def _refresh_apt(self):
        pkgs = se.missing_packages(self.backend())
        if pkgs:
            self.apt.setText("Install missing system packages (asks for your password): " + " ".join(pkgs))
            self.apt.setEnabled(True)
        else:
            self.apt.setText("System packages: all present")
            self.apt.setEnabled(False)

    def start(self):
        models = [n for n, cb in self.model_boxes.items() if cb.isChecked()]
        backend = self.backend()
        install_apt = self.apt.isEnabled() and self.apt.isChecked()
        cfg = self.cfg
        for w in [self.apt, self.btn_start, *self.model_boxes.values(), *self.backends.buttons()]:
            w.setEnabled(False)
        self.btn_close.setText("Cancel")

        def job(ctx: JobContext):
            return se.run_setup(cfg, backend, models, ctx.log, ctx.progress, ctx.cancel,
                                install_apt=install_apt, gui=True, overall=ctx.overall)

        self.worker = w = Worker("Engine setup", job, self)
        w.log.connect(self.log.appendPlainText)
        w.progress.connect(lambda f, t: (self.bar.setValue(int(f * 1000)), self.bar.setFormat(t or "%p%")))
        w.overall.connect(lambda d, t, s: (self.bar_all.setRange(0, t), self.bar_all.setValue(d),
                                           self.step.setText(f"<b>Step {min(d + 1, t)} of {t}:</b> {s}")))
        w.succeeded.connect(self._done)
        w.failed.connect(self._failed)
        w.start()

    def _done(self, cfg: Config):
        self.worker = None
        self.result_cfg = cfg
        from ..engines import make_engine
        self.step.setText("<b>Done.</b> " + make_engine(cfg).available()[1])
        self.btn_close.setText("Close")
        self.btn_start.setEnabled(False)

    def _failed(self, err: str):
        self.worker = None
        self.step.setText("<b style='color:#c62828'>Setup stopped.</b> Fix the problem shown in the log and "
                          "press Set up again; finished steps and partial downloads are kept.")
        self.log.appendPlainText(err)
        self.btn_close.setText("Close")
        self.btn_start.setEnabled(True)
        for w in [self.apt, *self.model_boxes.values()]:
            w.setEnabled(True)
        for b in self.backends.buttons():
            b.setEnabled(bool(b.property("allowed")))
        self._refresh_apt()

    def _close_or_cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.step.setText("Cancelling…")
            return
        self.accept() if self.result_cfg else self.reject()

    def closeEvent(self, e):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        super().closeEvent(e)

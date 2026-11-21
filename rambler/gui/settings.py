"""Settings dialog."""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton, QSpinBox,
                               QTabWidget, QVBoxLayout, QWidget)

from ..config import Config
from ..engines import list_ggml_models, make_engine

LANGS = [("auto", "Auto-detect"), ("en", "English"), ("de", "German"), ("nl", "Dutch"), ("fr", "French"),
         ("es", "Spanish"), ("it", "Italian"), ("hu", "Hungarian"), ("pt", "Portuguese"), ("ja", "Japanese")]


def autostart_path():
    from pathlib import Path
    import os
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "io.github.akh13.Rambler.desktop"


def set_autostart(on: bool) -> None:
    import shutil
    import sys
    from pathlib import Path
    p = autostart_path()
    if not on:
        p.unlink(missing_ok=True)
        return
    exe = Path(sys.executable).parent / "rambler"
    cmd = str(exe) if exe.exists() else (shutil.which("rambler") or f"{sys.executable} -m rambler")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("[Desktop Entry]\nType=Application\nName=Rambler\nIcon=audio-input-microphone\n"
                 f"Exec={cmd} --minimized\nX-GNOME-Autostart-enabled=true\nNoDisplay=false\n")


class FolderList(QWidget):
    def __init__(self, title: str, hint: str, dirs: list[str], parent=None):
        super().__init__(parent)
        self.list = QListWidget()
        self.list.addItems(dirs)
        self.list.setMinimumHeight(110)
        add, rem, up, down = (QPushButton(t) for t in ("Add…", "Remove", "Move up", "Move down"))
        add.clicked.connect(self._add)
        rem.clicked.connect(lambda: [self.list.takeItem(self.list.row(i)) for i in self.list.selectedItems()])
        up.clicked.connect(lambda: self._move(-1))
        down.clicked.connect(lambda: self._move(1))
        btns = QVBoxLayout()
        for b in (add, rem, up, down):
            btns.addWidget(b)
        btns.addStretch(1)
        row = QHBoxLayout()
        row.addWidget(self.list, 1)
        row.addLayout(btns)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        t = QLabel(f"<b>{title}</b>")
        h = QLabel(hint)
        h.setWordWrap(True)
        h.setStyleSheet("color: palette(mid);")
        lay.addWidget(t)
        lay.addWidget(h)
        lay.addLayout(row)

    def _add(self):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", "/media")
        if d and d not in self.values():
            self.list.addItem(d)

    def _move(self, delta):
        r = self.list.currentRow()
        if r < 0 or not (0 <= r + delta < self.list.count()):
            return
        it = self.list.takeItem(r)
        self.list.insertItem(r + delta, it)
        self.list.setCurrentRow(r + delta)

    def values(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]


class SettingsDialog(QDialog):
    def __init__(self, cfg: Config, parent=None, first_run=False):
        super().__init__(parent)
        self.setWindowTitle("Rambler settings" if not first_run else "Welcome to Rambler")
        self.resize(680, 520)
        self.cfg = replace(cfg)
        tabs = QTabWidget()

        # Folders
        fw = QWidget()
        fl = QVBoxLayout(fw)
        if first_run:
            w = QLabel("Pick where your recordings live and where transcripts should go. Add every drive you "
                       "want kept in step; drives that aren't plugged in are skipped until they are.")
            w.setWordWrap(True)
            fl.addWidget(w)
        self.audio = FolderList("Recording folders", "MP3/WAV/etc. The first one is primary: dropped files land "
                                "there. All of them are kept in sync.", cfg.audio_dirs)
        self.trans = FolderList("Transcript folders", "Transcripts, SRTs and metadata. New transcripts are written "
                                "to the first connected one, then synced to the rest.", cfg.transcript_dirs)
        fl.addWidget(self.audio)
        fl.addWidget(self.trans)
        tabs.addTab(fw, "Folders")

        # Transcription
        tw = QWidget()
        tf = QFormLayout(tw)
        self.engine = QComboBox()
        self.engine.addItem("Auto (whichever is set up)", "auto")
        self.engine.addItem("faster-whisper", "faster-whisper")
        self.engine.addItem("whisper.cpp (whisper-cli)", "whisper.cpp")
        self.engine.setCurrentIndex(max(0, self.engine.findData(cfg.engine)))
        self.bin = self._path_row(cfg.whisper_cpp_bin, "auto-detect", file=True)
        self.models_dir = self._path_row(cfg.models_dir, "")
        self.model = QComboBox()
        self.model.setEditable(True)
        self.lang = QComboBox()
        for code, label in LANGS:
            self.lang.addItem(f"{label} ({code})", code)
        i = self.lang.findData(cfg.language)
        if i < 0:
            self.lang.addItem(cfg.language, cfg.language)
            i = self.lang.count() - 1
        self.lang.setCurrentIndex(i)
        self.lang.setEditable(False)
        self.threads = QSpinBox()
        self.threads.setRange(1, 64)
        self.threads.setValue(cfg.threads)
        self.prompt = QLineEdit(cfg.initial_prompt)
        self.prompt.setPlaceholderText("Optional: names and words whisper should spell right")
        self.hotwords = QLineEdit(cfg.hotwords)
        self.hotwords.setPlaceholderText("faster-whisper: names and terms, comma-separated")
        self.batch = QSpinBox()
        self.batch.setRange(1, 32)
        self.batch.setValue(cfg.fw_batch)
        self.batch.setToolTip("faster-whisper batch size; halves itself if the GPU runs out of memory")
        self.layout_combo = QComboBox()
        self.layout_combo.addItem("Same subfolders as the recordings (2026/261114_0001.txt)", "mirror")
        self.layout_combo.addItem("All in one folder (261114_0001.txt)", "flat")
        self.layout_combo.setCurrentIndex(max(0, self.layout_combo.findData(cfg.transcript_layout)))
        self.post = QLineEdit(cfg.post_transcribe_cmd)
        self.post.setPlaceholderText("optional, e.g. ./my-script.sh  (runs in the transcript folder's parent)")
        self.extra = QLineEdit(cfg.extra_args)
        self.extra.setPlaceholderText("e.g. --beam-size 5 --max-len 60 --split-on-word")
        self.srt = QCheckBox("Generate SRT subtitles by default")
        self.srt.setChecked(cfg.generate_srt)
        self.vad = QCheckBox("Skip silence with Silero VAD when its model is installed")
        self.vad.setChecked(cfg.vad)
        self.beside = QCheckBox("Also put a copy of the SRT next to the recording (VLC picks it up)")
        self.beside.setChecked(cfg.srt_beside_audio)
        self.engine_status = QLabel()
        self.engine_status.setWordWrap(True)
        tf.addRow("Engine", self.engine)
        tf.addRow("whisper-cli", self.bin[0])
        tf.addRow("Models folder", self.models_dir[0])
        tf.addRow("Model", self.model)
        tf.addRow("Language", self.lang)
        tf.addRow("CPU threads", self.threads)
        tf.addRow("Initial prompt", self.prompt)
        tf.addRow("Hotwords", self.hotwords)
        tf.addRow("Batch size", self.batch)
        tf.addRow("whisper.cpp arguments", self.extra)
        tf.addRow("Transcript layout", self.layout_combo)
        tf.addRow("After transcribing, run", self.post)
        tf.addRow("", self.srt)
        tf.addRow("", self.vad)
        tf.addRow("", self.beside)
        setup_btn = QPushButton("Set up engine…  (build whisper.cpp, download models)")
        setup_btn.clicked.connect(self._engine_setup)
        tf.addRow("", self.engine_status)
        tf.addRow("", setup_btn)
        tabs.addTab(tw, "Transcription")
        self._fill_models(cfg.model)
        self.engine.currentIndexChanged.connect(lambda: self._fill_models(self.model.currentData() or "auto"))
        self.models_dir[1].editingFinished.connect(lambda: self._fill_models(self.model.currentData() or "auto"))

        # Sync
        sw = QWidget()
        sf = QFormLayout(sw)
        self.sync_audio = QCheckBox("Keep recording folders identical")
        self.sync_audio.setChecked(cfg.sync_audio)
        self.sync_trans = QCheckBox("Keep transcript folders identical")
        self.sync_trans.setChecked(cfg.sync_transcripts)
        self.policy = QComboBox()
        self.policy.addItem("Newest copy wins (size breaks ties)", "newer")
        self.policy.addItem("Largest copy wins (date breaks ties)", "larger")
        self.policy.setCurrentIndex(0 if cfg.conflict_policy != "larger" else 1)
        self.keep = QCheckBox("Move replaced copies into .rambler-replaced/ instead of overwriting")
        self.keep.setChecked(cfg.keep_replaced)
        self.confirm = QCheckBox("Show the copy plan before syncing")
        self.confirm.setChecked(cfg.confirm_sync_plan)
        note = QLabel("Sync only ever adds or updates files. It never deletes anything.")
        note.setStyleSheet("color: palette(mid);")
        sf.addRow("", self.sync_audio)
        sf.addRow("", self.sync_trans)
        sf.addRow("When copies differ", self.policy)
        sf.addRow("", self.keep)
        sf.addRow("", self.confirm)
        sf.addRow("", note)
        tabs.addTab(sw, "Sync")

        # Automation
        aw = QWidget()
        af = QFormLayout(aw)
        self.auto_launch = QCheckBox("When Rambler opens, transcribe anything new and sync")
        self.auto_launch.setChecked(cfg.auto_on_launch)
        self.auto_drive = QCheckBox("When a configured drive is plugged in, transcribe anything new and sync")
        self.auto_drive.setChecked(cfg.auto_on_drive_connect)
        self.incoming = QLineEdit(cfg.incoming_subdir)
        self.incoming.setPlaceholderText("(top of the primary recording folder)")
        af.addRow("", self.auto_launch)
        af.addRow("", self.auto_drive)
        af.addRow("Dropped files go to subfolder", self.incoming)
        tabs.addTab(aw, "Automation")

        # Recorders
        rw = QWidget()
        rl = QVBoxLayout(rw)
        hint = QLabel("Field recorders Rambler imports from when their card shows up (e.g. a TASCAM Portacapture X8 "
                      "switched to SD CARD READER over USB). Recordings are copied, never moved or changed, into "
                      "the primary recording folder, then transcribed and synced. Plug the recorder in while "
                      "it's in card reader mode and use \"Add from connected card\" to pin it by UUID.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: palette(mid);")
        rl.addWidget(hint)
        self.rec_data: list[dict] = [dict(r) for r in cfg.recorders]
        self.rec_list = QListWidget()
        self.rec_list.setMaximumWidth(200)
        form_w = QWidget()
        rf = QFormLayout(form_w)
        self.r_name = QLineEdit()
        self.r_label = QLineEdit()
        self.r_uuid = QLineEdit()
        self.r_uuid.setPlaceholderText("optional; matched before the label")
        self.r_card = QLineEdit()
        self.r_card.setPlaceholderText("(whole card)")
        self.r_dest = QLineEdit()
        self.r_dest.setPlaceholderText("{year} = year from the file name, e.g. 261121_0002 -> 2026")
        self.r_excl = QLineEdit()
        self.r_auto = QCheckBox("Import automatically when the card appears")
        self.r_unmount = QCheckBox("Unmount the card when done (safe to switch the recorder back)")
        self.r_enabled = QCheckBox("Enabled")
        rf.addRow("Name", self.r_name)
        rf.addRow("Volume label", self.r_label)
        rf.addRow("Filesystem UUID", self.r_uuid)
        rf.addRow("Folder on card", self.r_card)
        rf.addRow("Import into subfolder", self.r_dest)
        rf.addRow("Skip (globs, comma-sep.)", self.r_excl)
        rf.addRow("", self.r_auto)
        rf.addRow("", self.r_unmount)
        rf.addRow("", self.r_enabled)
        btn_add = QPushButton("Add from connected card…")
        btn_new = QPushButton("Add blank")
        btn_del = QPushButton("Remove")
        btn_add.clicked.connect(self._rec_add_from_mount)
        btn_new.clicked.connect(lambda: self._rec_append({"name": "Recorder", "label": "", "card_subdir": "",
                                                          "exclude": []}))
        btn_del.clicked.connect(self._rec_remove)
        left = QVBoxLayout()
        left.addWidget(self.rec_list, 1)
        for b_ in (btn_add, btn_new, btn_del):
            left.addWidget(b_)
        row = QHBoxLayout()
        row.addLayout(left)
        row.addWidget(form_w, 1)
        rl.addLayout(row, 1)
        self._rec_row = -1
        self._rec_form = form_w
        self.rec_list.currentRowChanged.connect(self._rec_select)
        for r in self.rec_data:
            self.rec_list.addItem(r.get("name", "Recorder"))
        if self.rec_data:
            self.rec_list.setCurrentRow(0)
        else:
            form_w.setEnabled(False)
        tabs.addTab(rw, "Recorders")

        self.autostart = QCheckBox("Start Rambler minimized when I log in (so cards and drives are caught)")
        self.autostart.setChecked(autostart_path().exists())
        af.addRow("", self.autostart)

        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        lay.addWidget(bb)

    # ---------------------------------------------------------------- recorders
    def _rec_fields(self):
        return {"name": self.r_name, "label": self.r_label, "uuid": self.r_uuid, "card_subdir": self.r_card,
                "dest_subdir": self.r_dest}

    def _rec_store(self):
        i = self._rec_row
        if not (0 <= i < len(self.rec_data)):
            return
        d = self.rec_data[i]
        for k, w in self._rec_fields().items():
            d[k] = w.text().strip()
        d["exclude"] = [x.strip() for x in self.r_excl.text().split(",") if x.strip()]
        d["auto_import"] = self.r_auto.isChecked()
        d["unmount_after"] = self.r_unmount.isChecked()
        d["enabled"] = self.r_enabled.isChecked()
        self.rec_list.item(i).setText(d["name"] or "Recorder")

    def _rec_select(self, i):
        self._rec_store()
        self._rec_row = i
        if not (0 <= i < len(self.rec_data)):
            self._rec_form.setEnabled(False)
            return
        self._rec_form.setEnabled(True)
        from ..recorders import Recorder
        r = Recorder.from_dict(self.rec_data[i])
        for k, w in self._rec_fields().items():
            w.setText(getattr(r, k))
        self.r_excl.setText(", ".join(r.exclude))
        self.r_auto.setChecked(r.auto_import)
        self.r_unmount.setChecked(r.unmount_after)
        self.r_enabled.setChecked(r.enabled)

    def _rec_append(self, d):
        from ..recorders import Recorder
        self._rec_store()
        self.rec_data.append(Recorder.from_dict(d).to_dict())
        self.rec_list.addItem(self.rec_data[-1]["name"])
        self.rec_list.setCurrentRow(len(self.rec_data) - 1)

    def _rec_remove(self):
        i = self.rec_list.currentRow()
        if 0 <= i < len(self.rec_data):
            self._rec_row = -1
            self.rec_data.pop(i)
            self.rec_list.takeItem(i)

    def _rec_add_from_mount(self):
        from PySide6.QtGui import QCursor
        from PySide6.QtWidgets import QMenu
        from ..recorders import list_mounts
        mounts = list_mounts()
        m = QMenu(self)
        if not mounts:
            a = m.addAction("No removable volumes mounted. Put the recorder in card reader mode first.")
            a.setEnabled(False)
        for mt in mounts:
            label = mt.label or mt.target.name
            m.addAction(f"{label}  ·  {mt.target}  ·  {mt.fstype}  ·  UUID {mt.uuid or '?'}",
                        lambda mt=mt, label=label: self._rec_append({
                            "name": "TASCAM X8" if label == "X8" else label, "label": mt.label, "uuid": mt.uuid,
                            "card_subdir": "SOUND" if label == "X8" else "",
                            "dest_subdir": "{year}"}))
        m.exec(QCursor.pos())

    def _engine_setup(self):
        from .engine_setup import EngineSetupDialog
        dlg = EngineSetupDialog(self.collect(), self)
        if dlg.exec() == QDialog.Accepted and dlg.result_cfg:
            self.cfg = dlg.result_cfg
            self.bin[1].setText(self.cfg.whisper_cpp_bin)
        self._fill_models(self.model.currentData() or "auto")

    def _path_row(self, value: str, placeholder: str, file=False):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        e = QLineEdit(value)
        e.setPlaceholderText(placeholder)
        b = QPushButton("Browse…")

        def pick():
            if file:
                p, _ = QFileDialog.getOpenFileName(self, "Choose whisper-cli")
            else:
                p = QFileDialog.getExistingDirectory(self, "Choose folder", e.text())
            if p:
                e.setText(p)
                e.editingFinished.emit()
        b.clicked.connect(pick)
        h.addWidget(e, 1)
        h.addWidget(b)
        return w, e

    def _fill_models(self, current: str):
        tmp = self.collect()
        self.model.blockSignals(True)
        self.model.clear()
        self.model.addItem("Best available", "auto")
        from ..engines import FasterWhisperEngine
        if tmp.engine == "faster-whisper" or (tmp.engine == "auto" and FasterWhisperEngine(tmp).available()[0]):
            for m in ("large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small"):
                self.model.addItem(m, m)
        else:
            for p in list_ggml_models(tmp):
                self.model.addItem(p.name.removeprefix("ggml-").removesuffix(".bin"), str(p))
        i = self.model.findData(current)
        if i < 0 and current and current != "auto":
            self.model.addItem(current, current)
            i = self.model.count() - 1
        self.model.setCurrentIndex(max(0, i))
        self.model.blockSignals(False)
        ok, msg = make_engine(self.collect()).available()
        self.engine_status.setText(("✔ " if ok else "⚠ ") + msg)
        self.engine_status.setStyleSheet("color: #2e7d32;" if ok else "color: #c62828;")

    def collect(self) -> Config:
        c = replace(self.cfg)
        c.audio_dirs = self.audio.values()
        c.transcript_dirs = self.trans.values()
        c.engine = self.engine.currentData()
        c.whisper_cpp_bin = self.bin[1].text().strip()
        c.models_dir = self.models_dir[1].text().strip() or c.models_dir
        if hasattr(self, "model") and self.model.count():
            data = self.model.currentData()
            c.model = data if data else (self.model.currentText().strip() or "auto")
        if hasattr(self, "lang"):
            c.language = self.lang.currentData()
            c.threads = self.threads.value()
            c.initial_prompt = self.prompt.text().strip()
            c.extra_args = self.extra.text().strip()
            c.generate_srt = self.srt.isChecked()
            c.hotwords = self.hotwords.text().strip()
            c.fw_batch = self.batch.value()
            c.transcript_layout = self.layout_combo.currentData()
            c.post_transcribe_cmd = self.post.text().strip()
            c.srt_beside_audio = self.beside.isChecked()
            c.vad = self.vad.isChecked()
        if hasattr(self, "sync_audio"):
            c.sync_audio = self.sync_audio.isChecked()
            c.sync_transcripts = self.sync_trans.isChecked()
            c.conflict_policy = self.policy.currentData()
            c.keep_replaced = self.keep.isChecked()
            c.confirm_sync_plan = self.confirm.isChecked()
        if hasattr(self, "rec_data"):
            self._rec_store()
            c.recorders = [dict(d) for d in self.rec_data]
        if hasattr(self, "autostart"):
            c.start_minimized = self.autostart.isChecked()
        if hasattr(self, "auto_launch"):
            c.auto_on_launch = self.auto_launch.isChecked()
            c.auto_on_drive_connect = self.auto_drive.isChecked()
            c.incoming_subdir = self.incoming.text().strip()
        return c

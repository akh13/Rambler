"""Main window: library, job queue, player, drag-and-drop."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath

from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt, QTimer, QUrl)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QIcon
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy, QSplitter, QStyle, QTableView,
                               QToolBar, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..config import Config
from ..engines import Cancelled, EngineError, make_engine
from ..library import Recording, scan
from ..pipeline import expand_drops, import_files, run_post_hook, transcribe_one
from ..syncer import Plan, build_plan, execute, human
from .player import PlayerPanel
from .settings import SettingsDialog, set_autostart
from .. import recorders as R
from .workers import JobContext, Worker

COLS = ["Status", "Recording", "Folder", "Size", "Modified", "Copies", "Subtitles"]
STATUS_COLORS = {
    "To be transcribed": "#e08a00", "Transcribed (no SRT)": "#3b82c4", "Transcribed": "#2e7d32",
    "Failed": "#c62828", "Queued": "#7a7a7a", "Too short": "#9e9e9e", "Track (MIX transcribed)": "#6d8f71",
    "Track (MIX pending)": "#9e9e9e",
}


class LibraryModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recs: list[Recording] = []
        self.overlay: dict[str, str] = {}      # rel -> transient status (Queued, Transcribing 40%, Failed)
        self.n_audio_dirs = 0

    def set_recs(self, recs, n_audio_dirs):
        self.beginResetModel()
        self.recs = recs
        self.n_audio_dirs = n_audio_dirs
        self.overlay = {k: v for k, v in self.overlay.items() if v.startswith("Failed")}
        self.endResetModel()

    def set_overlay(self, key: str, state: str):
        if state:
            self.overlay[key] = state
        else:
            self.overlay.pop(key, None)
        for row, r in enumerate(self.recs):
            if r.rel.as_posix() == key:
                self.dataChanged.emit(self.index(row, 0), self.index(row, len(COLS) - 1))
                break

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.recs)

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLS[section]
        return None

    def status_of(self, r: Recording) -> str:
        return self.overlay.get(r.rel.as_posix()) or r.status()

    def data(self, index, role=Qt.DisplayRole):
        r = self.recs[index.row()]
        c = index.column()
        if role == Qt.UserRole:
            return r
        if role == Qt.DisplayRole:
            if c == 0:
                return self.status_of(r)
            if c == 1:
                return r.name
            if c == 2:
                return str(r.rel.parent) if str(r.rel.parent) != "." else ""
            if c == 3:
                return human(r.size)
            if c == 4:
                return time.strftime("%Y-%m-%d %H:%M", time.localtime(r.mtime))
            if c == 5:
                return f"{len(r.locations)}/{self.n_audio_dirs}"
            if c == 6:
                return r.srt.suffix.lstrip(".").upper() if r.srt else ""
        if role == Qt.UserRole + 1:   # sort keys
            return {0: self.status_of(r), 1: r.name.lower(), 2: str(r.rel.parent).lower(), 3: r.size,
                    4: r.mtime, 5: len(r.locations), 6: bool(r.srt)}[c]
        if role == Qt.ForegroundRole and c == 0:
            s = self.status_of(r)
            key = "Failed" if s.startswith("Failed") else s if s in STATUS_COLORS else None
            if s.startswith("Transcribing") or s.startswith("Decoding"):
                return QBrush(QColor("#8e44ad"))
            if key:
                return QBrush(QColor(STATUS_COLORS[key]))
        if role == Qt.ForegroundRole and c == 5 and len(r.locations) < self.n_audio_dirs:
            return QBrush(QColor("#e08a00"))
        if role == Qt.ToolTipRole:
            locs = "\n".join(str(l) for l in r.locations)
            return f"{r.rel}\nOn: \n{locs}\nTranscript: {r.txt or '—'}\nSubtitles: {r.srt or '—'}"
        if role == Qt.TextAlignmentRole and c in (3, 5):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class FilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.mode = "all"
        self.setSortRole(Qt.UserRole + 1)

    def set_filter(self, text, mode):
        self.text, self.mode = text.lower(), mode
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        r: Recording = self.sourceModel().recs[row]
        if self.text and self.text not in r.rel.as_posix().lower():
            return False
        if self.mode == "pending":
            return r.pending
        if self.mode == "nosrt":
            return not r.srt
        if self.mode == "missing":
            return len(r.locations) < self.sourceModel().n_audio_dirs
        return True


class DropOverlay(QLabel):
    def __init__(self, parent):
        super().__init__("Drop recordings to add them to your library", parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background: rgba(30,90,160,0.82); color: white; font-size: 22px; "
                           "border: 3px dashed white; border-radius: 12px;")
        self.hide()


class MainWindow(QMainWindow):
    def __init__(self, open_files: list[Path] | None = None):
        super().__init__()
        self.open_files = open_files or []
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon.fromTheme("audio-input-microphone"))
        self.resize(1320, 820)
        self.setAcceptDrops(True)
        self.cfg = Config.load()
        self.worker: Worker | None = None
        self.queue_after: list = []          # follow-up steps to run when the current job ends
        self._last_avail: tuple = ()

        # ---- toolbar
        st = self.style()
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        self.act_scan = QAction(st.standardIcon(QStyle.SP_BrowserReload), "Rescan", self)
        self.act_trans = QAction(st.standardIcon(QStyle.SP_MediaPlay), "Transcribe new", self)
        self.act_trans_sel = QAction("Transcribe selected", self)
        self.act_sync = QAction(st.standardIcon(QStyle.SP_DriveHDIcon), "Sync", self)
        self.act_add = QAction(st.standardIcon(QStyle.SP_FileDialogNewFolder), "Add files…", self)
        self.act_settings = QAction(st.standardIcon(QStyle.SP_FileDialogDetailedView), "Settings", self)
        self.act_setup = QAction(st.standardIcon(QStyle.SP_MessageBoxWarning), "Set up engine…", self)
        self.act_setup.setToolTip("Build whisper.cpp for your GPU and download models")
        self.act_import = QAction(st.standardIcon(QStyle.SP_DriveCDIcon), "Import from recorder", self)
        self.act_import.setVisible(False)
        self.act_trans.setToolTip("Transcribe every recording that has no transcript yet")
        self.act_sync.setToolTip("Make every connected folder hold the same recordings and transcripts")
        self.chk_srt = QCheckBox("Make SRT subtitles")
        self.chk_srt.setChecked(self.cfg.generate_srt)
        for a in (self.act_scan, self.act_add, self.act_import):
            tb.addAction(a)
        tb.addSeparator()
        tb.addAction(self.act_trans)
        tb.addAction(self.act_trans_sel)
        tb.addWidget(self.chk_srt)
        tb.addSeparator()
        tb.addAction(self.act_sync)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.engine_label = QLabel()
        self.engine_label.setContentsMargins(0, 0, 10, 0)
        tb.addWidget(self.engine_label)
        tb.addAction(self.act_setup)
        tb.addAction(self.act_settings)

        # ---- drive chips
        self.drive_bar = QHBoxLayout()
        self.drive_bar.setContentsMargins(8, 4, 8, 0)
        drive_w = QWidget()
        drive_w.setLayout(self.drive_bar)

        # ---- library
        self.model = LibraryModel(self)
        self.proxy = FilterProxy(self)
        self.proxy.setSourceModel(self.model)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(4, Qt.DescendingOrder)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for col, w in ((0, 150), (2, 120), (3, 75), (4, 130), (5, 60), (6, 70)):
            self.table.setColumnWidth(col, w)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        self.table.selectionModel().currentRowChanged.connect(self._on_current)
        self.table.doubleClicked.connect(lambda i: self._on_current(i, None, autoplay=True))

        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by name or folder…")
        self.search.setClearButtonEnabled(True)
        self.show_mode = QComboBox()
        for label, mode in (("All recordings", "all"), ("To be transcribed", "pending"),
                            ("Missing subtitles", "nosrt"), ("Not on every drive", "missing")):
            self.show_mode.addItem(label, mode)
        self.search.textChanged.connect(self._apply_filter)
        self.show_mode.currentIndexChanged.connect(self._apply_filter)
        self.counts = QLabel()

        lib = QWidget()
        ll = QVBoxLayout(lib)
        ll.setContentsMargins(8, 4, 4, 4)
        fr = QHBoxLayout()
        fr.addWidget(self.search, 1)
        fr.addWidget(self.show_mode)
        ll.addLayout(fr)
        ll.addWidget(self.table, 1)
        ll.addWidget(self.counts)

        self.player = PlayerPanel()
        self.player.message.connect(self.log)
        pw = QWidget()
        pl = QVBoxLayout(pw)
        pl.setContentsMargins(4, 4, 8, 4)
        pl.addWidget(self.player)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(lib)
        split.addWidget(pw)
        split.setSizes([760, 560])

        # ---- job area
        self.job_title = QLabel("Idle")
        self.job_title.setStyleSheet("font-weight: 600;")
        self.job_detail = QLabel("")
        self.bar_item = QProgressBar()
        self.bar_item.setFormat("%p%")
        self.bar_all = QProgressBar()
        self.bar_all.setFormat("%v / %m")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_log = QPushButton("Log")
        self.btn_log.setCheckable(True)
        self.logview = QPlainTextEdit()
        self.logview.setReadOnly(True)
        self.logview.setMaximumBlockCount(5000)
        self.logview.setMaximumHeight(160)
        self.logview.hide()
        self.btn_log.toggled.connect(self.logview.setVisible)
        job = QWidget()
        jl = QVBoxLayout(job)
        jl.setContentsMargins(8, 2, 8, 6)
        r1 = QHBoxLayout()
        r1.addWidget(self.job_title)
        r1.addWidget(self.job_detail, 1)
        r1.addWidget(self.btn_log)
        r1.addWidget(self.btn_cancel)
        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Current"))
        r2.addWidget(self.bar_item, 2)
        r2.addWidget(QLabel("Overall"))
        r2.addWidget(self.bar_all, 1)
        jl.addLayout(r1)
        jl.addLayout(r2)
        jl.addWidget(self.logview)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
        cl.addWidget(drive_w)
        cl.addWidget(split, 1)
        cl.addWidget(job)
        self.setCentralWidget(central)
        self.overlay = DropOverlay(central)

        self.player.install_shortcuts(self)
        self.act_scan.triggered.connect(lambda: self.start_scan())
        self.act_trans.triggered.connect(lambda: self.transcribe([r for r in self.model.recs if r.pending]))
        self.act_trans_sel.triggered.connect(lambda: self.transcribe(self.selected(), force=True))
        self.act_sync.triggered.connect(self.sync_clicked)
        self.act_add.triggered.connect(self._add_files)
        self.act_settings.triggered.connect(self.open_settings)
        self.act_setup.triggered.connect(self.open_engine_setup)
        self.act_import.triggered.connect(lambda: [self.import_from_recorder(r, m) for r, m in self._recorders_now[:1]])
        self._recorders_now: list = []
        self._handled_recorders: set[str] = set()
        self.chk_srt.toggled.connect(self._srt_toggled)

        self.drive_timer = QTimer(self)
        self.drive_timer.timeout.connect(self._poll_drives)
        self.drive_timer.start(4000)
        self._refresh_drive_bar()
        self._refresh_engine()
        QTimer.singleShot(0, self._startup)

    # ================================================================ startup / settings
    def _startup(self):
        if not self.cfg.audio_dirs or not self.cfg.transcript_dirs:
            self.open_settings(first_run=True)
        if not make_engine(self.cfg).available()[0] and not self.cfg.engine_setup_declined:
            self.open_engine_setup()
            if not make_engine(self.cfg).available()[0]:
                self.cfg.engine_setup_declined = True     # don't nag every launch; toolbar button stays
                self.cfg.save()
        if self.open_files:
            files, self.open_files = self.open_files, []
            self.start_scan(then=lambda _: self.add_paths(files))
        elif self.cfg.auto_on_launch:
            self.run_auto("Opened Rambler")
        else:
            self.start_scan()

    def open_settings(self, first_run=False):
        if self.busy():
            QMessageBox.information(self, APP_NAME, "Settings can be changed once the current job finishes.")
            return
        dlg = SettingsDialog(self.cfg, self, first_run=first_run)
        if dlg.exec() == QDialog.Accepted:
            self.cfg = dlg.collect()
            self.cfg.save()
            try:
                set_autostart(self.cfg.start_minimized)
            except OSError as e:
                self.log(f"Could not update autostart: {e}")
            self.chk_srt.setChecked(self.cfg.generate_srt)
            self._refresh_drive_bar()
            self._refresh_engine()
            if not first_run:
                self.start_scan()

    def _srt_toggled(self, on):
        self.cfg.generate_srt = on
        self.cfg.save()

    def _refresh_engine(self):
        ok, msg = make_engine(self.cfg).available()
        self.engine_label.setText(("● " if ok else "⚠ ") + msg)
        self.engine_label.setStyleSheet("color: #2e7d32;" if ok else "color: #c62828;")
        self.engine_label.setToolTip(msg)
        self.act_setup.setVisible(not ok)
        return ok

    def open_engine_setup(self):
        if self.busy():
            return
        from .engine_setup import EngineSetupDialog
        dlg = EngineSetupDialog(self.cfg, self)
        if dlg.exec() == QDialog.Accepted and dlg.result_cfg:
            self.cfg = dlg.result_cfg
            self.cfg.save()
        self._refresh_engine()

    # ================================================================ drives
    def _availability(self):
        return tuple(Config.expand(d).is_dir() for d in self.cfg.audio_dirs + self.cfg.transcript_dirs)

    def _refresh_drive_bar(self):
        while self.drive_bar.count():
            w = self.drive_bar.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._last_avail = self._availability()
        for kind, dirs in (("🎙", self.cfg.audio_dirs), ("📝", self.cfg.transcript_dirs)):
            for i, d in enumerate(dirs):
                ok = Config.expand(d).is_dir()
                parts = Config.expand(d).parts
                if len(parts) > 3 and parts[1] in ("media", "run") and "media" in parts[:3]:
                    i_lbl = parts.index("media") + 2            # /media/<user>/<LABEL>/...
                    name = "/".join(parts[i_lbl:i_lbl + 1] + parts[-1:]) if len(parts) > i_lbl + 1 else parts[-1]
                else:
                    name = "/".join(parts[-2:])
                lab = QLabel(f"{kind} {name}{' ★' if i == 0 else ''}")
                lab.setToolTip(f"{d}\n{'Connected' if ok else 'Not connected'}{' · primary' if i == 0 else ''}")
                lab.setStyleSheet(
                    "padding: 2px 8px; border-radius: 9px; " +
                    ("background: #e3f1e5; color: #1b5e20;" if ok else "background: #eee; color: #888; "
                     "text-decoration: line-through;"))
                self.drive_bar.addWidget(lab)
        self.drive_bar.addStretch(1)

    def _poll_drives(self):
        self._poll_recorders()
        now = self._availability()
        if now == self._last_avail:
            return
        appeared = any(n and not o for n, o in zip(now, self._last_avail)) if len(now) == len(self._last_avail) else True
        self._refresh_drive_bar()
        if self.busy():
            return
        if appeared and self.cfg.auto_on_drive_connect:
            self.run_auto("Drive connected")
        else:
            self.start_scan()

    # ================================================================ recorders
    def _poll_recorders(self):
        conns = R.connected(self.cfg)
        self._recorders_now = conns
        present = {f"{r.name}|{m.source}|{m.target}" for r, m in conns}
        self._handled_recorders &= present
        if conns:
            r, m = conns[0]
            self.act_import.setText(f"Import from {r.name}")
            self.act_import.setToolTip(f"{m.target} ({m.label or 'no label'})")
        self.act_import.setVisible(bool(conns))
        for r, m in conns:
            key = f"{r.name}|{m.source}|{m.target}"
            if key in self._handled_recorders or not r.auto_import:
                continue
            if self.busy():
                return          # try again on the next poll
            self._handled_recorders.add(key)
            self.log(f"{r.name} connected at {m.target}")
            self.import_from_recorder(r, m)
            return

    def import_from_recorder(self, rec: "R.Recorder", mount: "R.Mount"):
        if self.busy():
            return
        with R.pipeline_lock() as got:
            if not got:
                self.log("`rambler watch` is already handling this; skipping.")
                return
        cfg = self.cfg
        self._handled_recorders.add(f"{rec.name}|{mount.source}|{mount.target}")
        if self.isMinimized():
            self._notify(APP_NAME, f"{rec.name} connected, importing new recordings…")

        def job(ctx: JobContext):
            return R.import_recorder(cfg, rec, mount, ctx.log,
                                     lambda d, t, s: (ctx.progress(d / t if t else 1, s),
                                                      ctx.overall(int(d / 1_048_576), max(1, int(t / 1_048_576)), s)),
                                     ctx.cancel)

        def finish_card(n_imported):
            def unmount_job(ctx: JobContext):
                ctx.progress(0, "Flushing and unmounting card")
                return R.unmount(mount, ctx.log)

            def unmounted(ok):
                self._notify(APP_NAME, f"{rec.name}: {n_imported} new recording(s) imported, transcribed and synced."
                             + (" Card unmounted, safe to switch the recorder back." if ok else ""))
            if rec.unmount_after:
                self.run_job(f"Unmounting {rec.name}", unmount_job, unmounted)
            else:
                unmounted(False)

        def imported(rels):
            self.log(f"{rec.name}: {len(rels)} new recording(s)")

            def after_scan(recs):
                if rels:
                    self._reselect(rels[0])
                pending = [r for r in recs if r.pending]
                after = lambda: self.start_sync(confirm=False, then=lambda: finish_card(len(rels)))
                if pending and make_engine(self.cfg).available()[0]:
                    self.transcribe(pending, want_srt=self.cfg.generate_srt, then=after)
                else:
                    after()
            self.start_scan(then=after_scan)
        self.run_job(f"Importing from {rec.name}", job, imported)

    # ================================================================ job plumbing
    def busy(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def log(self, text: str):
        self.logview.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _set_actions(self, enabled: bool):
        for a in (self.act_scan, self.act_trans, self.act_trans_sel, self.act_sync, self.act_add, self.act_settings,
                  self.act_setup, self.act_import):
            a.setEnabled(enabled)
        self.btn_cancel.setEnabled(not enabled)

    def run_job(self, title: str, fn, on_done=None, on_fail=None):
        w = Worker(title, fn, self)
        self.worker = w
        self.job_title.setText(title)
        self.job_detail.setText("")
        self.bar_item.setValue(0)
        self.bar_all.setRange(0, 1)
        self.bar_all.setValue(0)
        self._set_actions(False)
        w.progress.connect(lambda f, t: (self.bar_item.setRange(0, 1000), self.bar_item.setValue(int(f * 1000)),
                                         self.job_detail.setText(t)))
        w.overall.connect(lambda d, t, s: (self.bar_all.setRange(0, max(1, t)), self.bar_all.setValue(d),
                                           self.job_detail.setText(s) if s else None))
        w.log.connect(self.log)
        w.item_state.connect(self.model.set_overlay)

        def finish(ok, payload):
            self.worker = None
            w.deleteLater()
            self._set_actions(True)
            self.job_title.setText("Idle")
            self.job_detail.setText("")
            if ok:
                if on_done:
                    on_done(payload)
            else:
                self.log(f"{title} failed: {payload}")
                if on_fail:
                    on_fail(payload)
                else:
                    QMessageBox.warning(self, APP_NAME, f"{title} failed:\n{payload}")
                self.queue_after.clear()
            if not self.busy() and self.queue_after:
                nxt = self.queue_after.pop(0)
                nxt()
        w.succeeded.connect(lambda p: finish(True, p))
        w.failed.connect(lambda e: finish(False, e))
        self.log(title)
        w.start()

    def _cancel(self):
        if self.worker:
            self.worker.cancel()
            self.queue_after.clear()
            self.job_detail.setText("Cancelling…")

    def selected(self) -> list[Recording]:
        rows = {self.proxy.mapToSource(i).row() for i in self.table.selectionModel().selectedRows()}
        return [self.model.recs[r] for r in sorted(rows)]

    # ================================================================ scan
    def start_scan(self, then=None):
        cfg = self.cfg

        def job(ctx: JobContext):
            ctx.progress(0, "Reading folders")
            return scan(cfg)

        def done(recs):
            cur = self._current_rec()
            self.model.set_recs(recs, len(self.cfg.audio_roots()))
            self._update_counts()
            if cur:
                self._reselect(cur.rel)
            if then:
                then(recs)
        self.run_job("Scanning", job, done)

    def _update_counts(self):
        recs = self.model.recs
        pending = sum(1 for r in recs if r.pending)
        nosrt = sum(1 for r in recs if r.transcribed and not r.srt)
        missing = sum(1 for r in recs if len(r.locations) < len(self.cfg.audio_roots()))
        self.counts.setText(f"{len(recs)} recordings · {pending} to be transcribed · {nosrt} without subtitles"
                            f" · {missing} not on every connected drive")
        self.act_trans.setText(f"Transcribe new ({pending})" if pending else "Transcribe new")

    def _apply_filter(self):
        self.proxy.set_filter(self.search.text(), self.show_mode.currentData())

    # ================================================================ transcription
    def transcribe(self, recs: list[Recording], want_srt: bool | None = None, force=False, then=None):
        if not recs:
            if not then:
                self.statusBar().showMessage("Nothing to transcribe.", 5000)
            else:
                then()
            return
        if not self._refresh_engine():
            QMessageBox.warning(self, APP_NAME, make_engine(self.cfg).available()[1])
            return
        if not self.cfg.transcript_roots():
            QMessageBox.warning(self, APP_NAME, "None of the transcript folders are connected.")
            return
        if force and any(r.transcribed for r in recs):
            if QMessageBox.question(self, APP_NAME, "Some selected recordings already have transcripts. "
                                    "Transcribe them again and replace them?") != QMessageBox.Yes:
                recs = [r for r in recs if not r.transcribed]
                if not recs:
                    return
        want = self.chk_srt.isChecked() if want_srt is None else want_srt
        cfg = self.cfg
        engine = make_engine(cfg)
        for r in recs:
            self.model.set_overlay(r.rel.as_posix(), "Queued")

        def job(ctx: JobContext):
            try:
                res = transcribe_all(ctx)
            finally:
                engine.close()
            if res[0] and cfg.post_transcribe_cmd.strip() and not ctx.cancel.is_set():
                ctx.progress(0, "Running post-transcribe command")
                run_post_hook(cfg, ctx.log, ctx.cancel)
            return res

        def transcribe_all(ctx: JobContext):
            ok = failed = 0
            for i, r in enumerate(recs):
                if ctx.cancel.is_set():
                    break
                key = r.rel.as_posix()
                ctx.overall(i, len(recs), r.name)
                last = [0.0]

                def prog(f, msg, key=key):
                    ctx.progress(f, f"{r.name}: {msg}")
                    if f - last[0] >= 0.01 or f >= 1:
                        last[0] = f
                        ctx.item_state(key, f"{msg} {int(f * 100)}%" if msg != "Done" else "")
                try:
                    if not r.source.exists():
                        raise EngineError("recording is on a drive that is not connected")
                    t0 = time.time()
                    transcribe_one(cfg, engine, r, want, prog, ctx.cancel)
                    ctx.item_state(key, "")
                    ctx.log(f"✔ {r.rel} ({time.time() - t0:.0f}s, {engine.model_label()})")
                    ok += 1
                except Cancelled:
                    ctx.item_state(key, "")
                    break
                except Exception as e:  # keep going with the rest of the queue
                    failed += 1
                    ctx.item_state(key, "Failed")
                    ctx.log(f"✘ {r.rel}: {e}")
            ctx.overall(len(recs), len(recs), "")
            return ok, failed, ctx.cancel.is_set()

        def done(res):
            ok, failed, cancelled = res
            for r in recs:
                if self.model.overlay.get(r.rel.as_posix()) != "Failed":
                    self.model.set_overlay(r.rel.as_posix(), "")
            msg = f"Transcribed {ok}" + (f", {failed} failed (see Log)" if failed else "") + \
                  (" · cancelled" if cancelled else "")
            self.statusBar().showMessage(msg, 15000)
            self.log(msg)
            if failed:
                self.btn_log.setChecked(True)
            self.start_scan(then=lambda _: self._after_transcribe(ok, then, cancelled))
        self.run_job(f"Transcribing {len(recs)} recording(s)" + (" + SRT" if want else ""), job, done)

    def _after_transcribe(self, ok, then, cancelled):
        if cancelled:
            return
        if then:
            then()
        elif ok and self._sync_needed_quick():
            if QMessageBox.question(self, APP_NAME, f"Transcribed {ok} recording(s). Sync the folders now?") \
                    == QMessageBox.Yes:
                self.start_sync()

    def _sync_needed_quick(self) -> bool:
        return (self.cfg.sync_transcripts and len(self.cfg.transcript_roots()) > 1) or \
               (self.cfg.sync_audio and len(self.cfg.audio_roots()) > 1)

    # ================================================================ sync
    def sync_clicked(self):
        recs = self.model.recs
        pending = [r for r in recs if r.pending]
        nosrt = [r for r in recs if r.transcribed and not r.srt]
        if pending or nosrt:
            eng = make_engine(self.cfg)
            ok, _ = eng.available()
            if ok:
                box = QMessageBox(self)
                box.setWindowTitle(APP_NAME)
                box.setIcon(QMessageBox.Question)
                parts = []
                if pending:
                    parts.append(f"{len(pending)} recording(s) have no transcript")
                if nosrt:
                    parts.append(f"{len(nosrt)} transcript(s) have no SRT subtitles")
                box.setText(" and ".join(parts) + ".")
                box.setInformativeText(f"Generate transcripts and SRTs with {eng.model_label()} first, "
                                       "then sync?")
                b_all = box.addButton("Transcribe + SRT, then sync", QMessageBox.AcceptRole)
                b_new = box.addButton("Only new ones, then sync", QMessageBox.AcceptRole) if nosrt and pending else None
                b_sync = box.addButton("Just sync", QMessageBox.DestructiveRole)
                box.addButton(QMessageBox.Cancel)
                box.setDefaultButton(b_all)
                box.exec()
                clicked = box.clickedButton()
                if clicked == b_all:
                    self.transcribe(pending + nosrt, want_srt=True, then=self.start_sync)
                    return
                if b_new is not None and clicked == b_new:
                    self.transcribe(pending, want_srt=True, then=self.start_sync)
                    return
                if clicked != b_sync:
                    return
        self.start_sync()

    def start_sync(self, confirm: bool | None = None, then=None):
        cfg = self.cfg
        confirm = cfg.confirm_sync_plan if confirm is None else confirm

        def plan_job(ctx: JobContext):
            ctx.progress(0, "Comparing folders")
            return build_plan(cfg)

        def planned(plan: Plan):
            self.log(plan.summary())
            if not plan.actions:
                self.statusBar().showMessage("Everything is already in sync.", 8000)
                if then:
                    then()
                return
            if confirm or plan.space_problems:
                if not self._confirm_plan(plan):
                    return
            self._run_sync(plan, then)
        self.run_job("Planning sync", plan_job, planned)

    def _confirm_plan(self, plan: Plan) -> bool:
        dlg = QDialog(self)
        dlg.setWindowTitle("Sync plan")
        dlg.resize(760, 480)
        lay = QVBoxLayout(dlg)
        head = QLabel(plan.summary().replace("\n", "<br>"))
        head.setTextFormat(Qt.RichText)
        lay.addWidget(head)
        lst = QPlainTextEdit()
        lst.setReadOnly(True)
        lst.setPlainText("\n".join(
            f"{'REPLACE' if a.replace else 'copy   '}  {a.src.rel}   {a.src.root} → {a.dst_root}  ({human(a.bytes)})"
            for a in plan.actions[:5000]))
        lay.addWidget(lst, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Start sync")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        return dlg.exec() == QDialog.Accepted

    def _run_sync(self, plan: Plan, then=None):
        cfg = self.cfg

        def job(ctx: JobContext):
            t0 = time.time()

            def prog(done, total, label):
                ctx.progress(done / total if total else 1, label)
                el = max(0.001, time.time() - t0)
                ctx.overall(int(done / 1_048_576), max(1, int(total / 1_048_576)),
                            f"{label}  ·  {human(done / el)}/s")
            return execute(plan, cfg, prog, ctx.cancel)

        def done(errors):
            self.bar_all.setFormat("%p%")
            if errors:
                self.log("\n".join(errors))
                QMessageBox.warning(self, APP_NAME, f"Sync finished with {len(errors)} error(s). See Log.")
                self.btn_log.setChecked(True)
            else:
                self.statusBar().showMessage(f"Synced {len(plan.actions)} file(s), {human(plan.total_bytes)}.", 15000)
            self.start_scan(then=(lambda _: then()) if then else None)
        self.bar_all.setFormat("%v / %m MB")
        self.run_job(f"Syncing {len(plan.actions)} file(s)", job, done)

    # ================================================================ automation
    def run_auto(self, why: str):
        self.log(f"Auto run: {why}")

        def after_scan(recs):
            pending = [r for r in recs if r.pending]
            ok, _ = make_engine(self.cfg).available()
            nxt = lambda: self.start_sync(confirm=False, then=lambda: self._notify("Rambler", "Transcribe and sync finished."))
            if pending and ok:
                self.transcribe(pending, want_srt=self.cfg.generate_srt, then=nxt)
            else:
                nxt()
        self.start_scan(then=after_scan)

    def _notify(self, title, body):
        self.statusBar().showMessage(body, 15000)
        if shutil.which("notify-send") and not self.isActiveWindow():
            subprocess.Popen(["notify-send", "-a", APP_NAME, "-i", "audio-input-microphone", title, body])

    # ================================================================ drag and drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(u.isLocalFile() for u in e.mimeData().urls()):
            e.acceptProposedAction()
            self.overlay.setGeometry(self.centralWidget().rect().adjusted(20, 20, -20, -20))
            self.overlay.raise_()
            self.overlay.show()

    def dragLeaveEvent(self, e):
        self.overlay.hide()

    def dropEvent(self, e):
        self.overlay.hide()
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        e.acceptProposedAction()
        QTimer.singleShot(0, lambda: self.add_paths(paths))

    def _add_files(self):
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Add recordings", str(Path.home()),
                                                "Audio (*.mp3 *.wav *.flac *.m4a *.ogg *.opus *.aac *.aiff *.mp4 *.webm)")
        if files:
            self.add_paths([Path(f) for f in files])

    def add_paths(self, paths: list[Path]):
        if self.busy():
            QMessageBox.information(self, APP_NAME, "Rambler is busy; drop the files again when the current job ends.")
            return
        items = expand_drops(paths)
        if not items:
            QMessageBox.information(self, APP_NAME, "No audio files in what was dropped.")
            return
        roots = self.cfg.audio_roots()
        if not roots:
            QMessageBox.warning(self, APP_NAME, "No recording folder is connected.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Add recordings")
        lay = QVBoxLayout(dlg)
        total = sum(src.stat().st_size for src, _ in items)
        dest = roots[0] / self.cfg.incoming_subdir if self.cfg.incoming_subdir else roots[0]
        lay.addWidget(QLabel(f"Add {len(items)} recording(s) ({human(total)}) to\n{dest}"))
        eng = make_engine(self.cfg)
        eng_ok, _ = eng.available()
        c_tr = QCheckBox("Transcribe them now")
        c_tr.setChecked(eng_ok)
        c_tr.setEnabled(eng_ok)
        c_srt = QCheckBox(f"Generate SRT subtitles with the best available model ({eng.model_label()})")
        c_srt.setChecked(eng_ok and self.chk_srt.isChecked())
        c_srt.setEnabled(eng_ok)
        c_tr.toggled.connect(c_srt.setEnabled)
        n_other = len(self.cfg.audio_dirs) - 1
        c_sync = QCheckBox(f"Then sync to the other {n_other} recording folder(s) and transcript folders")
        c_sync.setChecked(True)
        for c in (c_tr, c_srt, c_sync):
            lay.addWidget(c)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Add")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return
        do_tr, do_srt, do_sync = c_tr.isChecked(), c_srt.isChecked(), c_sync.isChecked()
        cfg = self.cfg

        def job(ctx: JobContext):
            return import_files(cfg, items, ctx.cancel, lambda d, t, s: (ctx.overall(d, t, s), ctx.progress(d / max(1, t), s)))

        def imported(rels: list[PurePosixPath]):
            self.log(f"Imported {len(rels)} file(s) into {dest}")
            wanted = {r.as_posix() for r in rels}

            def after_scan(recs):
                new = [r for r in recs if r.rel.as_posix() in wanted]
                if new:
                    self._reselect(new[0].rel)
                sync = (lambda: self.start_sync(confirm=False)) if do_sync else None
                todo = [r for r in new if r.pending or (do_srt and not r.srt)]
                if do_tr and todo:
                    self.transcribe(todo, want_srt=do_srt, then=sync or (lambda: None))
                elif sync:
                    sync()
            self.start_scan(then=after_scan)
        self.run_job(f"Adding {len(items)} recording(s)", job, imported)

    # ================================================================ player / selection
    def _current_rec(self) -> Recording | None:
        idx = self.table.currentIndex()
        if not idx.isValid():
            return None
        return self.model.recs[self.proxy.mapToSource(idx).row()]

    def _reselect(self, rel: PurePosixPath):
        for row, r in enumerate(self.model.recs):
            if r.rel == rel:
                idx = self.proxy.mapFromSource(self.model.index(row, 0))
                if idx.isValid():
                    self.table.selectionModel().blockSignals(True)
                    self.table.setCurrentIndex(idx)
                    self.table.selectionModel().blockSignals(False)
                    self.table.scrollTo(idx)
                    # refresh transcript panel if it changed, without interrupting playback of the same file
                    if self.player.screen.title.text() != str(r.rel) or (r.srt and not self.player.cues):
                        if self.player.player.playbackState() != QMediaPlayer.PlayingState:
                            self._load_player(r)
                return

    def _on_current(self, cur, prev, autoplay=False):
        if not cur.isValid():
            return
        self._load_player(self.model.recs[self.proxy.mapToSource(cur).row()], autoplay)

    def _load_player(self, r: Recording, autoplay=False):
        src = next((loc / r.rel for loc in r.locations if (loc / r.rel).exists()), None)
        self.player.load(src, str(r.rel), r.srt, r.txt, autoplay)

    def _context_menu(self, pos):
        r = self._current_rec()
        if not r:
            return
        m = QMenu(self)
        m.addAction("Play", lambda: self._load_player(r, True))
        m.addAction("Transcribe selected", lambda: self.transcribe(self.selected(), force=True))
        m.addAction("Transcribe selected (with SRT)", lambda: self.transcribe(self.selected(), want_srt=True, force=True))
        m.addSeparator()
        m.addAction("Open containing folder", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(r.source.parent))))
        if r.txt:
            m.addAction("Open transcript", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(r.txt))))
        if r.srt and shutil.which("vlc"):
            m.addAction("Open in VLC", lambda: subprocess.Popen(["vlc", str(r.source), "--sub-file", str(r.srt)]))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def closeEvent(self, e):
        if self.busy():
            if QMessageBox.question(self, APP_NAME, "A job is still running. Cancel it and quit?") != QMessageBox.Yes:
                e.ignore()
                return
            self.worker.cancel()
            self.worker.wait(10000)
        e.accept()

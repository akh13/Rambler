# SPDX-License-Identifier: GPL-3.0-or-later
"""VLC-style player: waveform + live visualizer behind word-highlighted subtitles, and an editable
transcript list with confidence marks and speaker tags."""
from __future__ import annotations

import html
import math
from array import array
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QKeySequence, QPainter, QPainterPath, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QListWidgetItem, QMenu, QMessageBox,
                               QPlainTextEdit, QSlider, QSplitter, QStackedWidget, QStyle, QToolButton,
                               QVBoxLayout, QWidget)

from .. import peaks as peaksmod
from ..transcript import Transcript
from . import theme

try:
    from PySide6.QtMultimedia import QAudioBufferOutput
except ImportError:          # Qt < 6.8
    QAudioBufferOutput = None
try:
    import numpy as np
except ImportError:
    np = None

BARS = 48


def fmt_time(ms: int) -> str:
    s = max(0, int(ms) // 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class PeaksWorker(QThread):
    done = Signal(str, bytes)

    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            data = peaksmod.compute(self.path)
        except Exception:
            data = b""
        self.done.emit(str(self.path), data)


class Screen(QFrame):
    """The dark 'video' area: waveform and level bars painted behind the title and subtitle."""
    seekRequested = Signal(float)   # 0..1

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setCursor(Qt.PointingHandCursor)
        self.peaks = b""
        self.frac = 0.0
        self.levels = [0.0] * BARS
        self.show_wave = True
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 18)
        top = QHBoxLayout()
        self.title = QLabel("No recording selected")
        self.title.setStyleSheet("color: #9aa0a6; font-size: 13px; background: transparent;")
        self.title.setWordWrap(True)
        self.clock = QLabel("")
        self.clock.setStyleSheet("color: #9aa0a6; font-size: 13px; background: transparent;")
        top.addWidget(self.title, 1)
        top.addWidget(self.clock)
        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #7d838c; font-size: 12px; font-style: italic; background: transparent;")
        self.summary.hide()
        self.sub = QLabel("")
        self.sub.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.sub.setWordWrap(True)
        self.sub.setTextFormat(Qt.RichText)
        f = QFont()
        f.setPointSize(17)
        f.setWeight(QFont.DemiBold)
        self.sub.setFont(f)
        self.sub.setStyleSheet("color: #ffffff; background: rgba(0,0,0,0.6); padding: 6px 12px; border-radius: 4px;")
        self.sub.setAttribute(Qt.WA_TransparentForMouseEvents)
        shadow = QGraphicsDropShadowEffect(self.sub)
        shadow.setBlurRadius(6)
        shadow.setOffset(1, 1)
        shadow.setColor(QColor(0, 0, 0))
        self.sub.setGraphicsEffect(shadow)
        self.sub.hide()
        lay.addLayout(top)
        lay.addWidget(self.summary)
        lay.addStretch(1)
        lay.addWidget(self.sub)

    def set_sub(self, html_text: str):
        self.sub.setText(html_text)
        self.sub.setVisible(bool(html_text))

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton and self.width() > 0:
            self.seekRequested.emit(max(0.0, min(1.0, e.position().x() / self.width())))

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(r, 6, 6)
        p.fillPath(path, QColor("#0b0b0d"))
        p.setClipPath(path)
        w, h = r.width(), r.height()
        mid = h * 0.52
        # waveform of the whole recording, played part brighter
        if self.show_wave and self.peaks:
            n = len(self.peaks)
            step = 3.0
            cols = max(1, int(w / step))
            amp = h * 0.30
            play_x = self.frac * w
            for i in range(cols):
                lo = int(i * n / cols)
                hi = max(lo + 1, int((i + 1) * n / cols))
                v = max(self.peaks[lo:hi]) / 255.0
                bh = max(1.0, v * amp)
                x = i * step
                p.fillRect(QRectF(x, mid - bh, step - 1, bh * 2),
                           QColor(110, 165, 255, 95) if x < play_x else QColor(255, 255, 255, 34))
            p.fillRect(QRectF(play_x - 1, mid - amp - 6, 2, amp * 2 + 12), QColor(140, 190, 255, 200))
        # live level bars along the bottom
        if any(v > 0.01 for v in self.levels):
            bw = w / BARS
            top = h * 0.42
            for i, v in enumerate(self.levels):
                bh = v * top
                p.fillRect(QRectF(i * bw + 1, h - bh, bw - 2, bh), QColor(120, 180, 255, int(40 + 70 * v)))
        p.end()


class PlayerPanel(QWidget):
    message = Signal(str)
    edited = Signal(object, list)        # Transcript, new words (hotword candidates)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.9)
        self.transcript: Transcript | None = None
        self.current = -1
        self.current_word = -1
        self._seeking = False
        self._source: Path | None = None
        self._peaks_jobs: list[PeaksWorker] = []
        self.show_confidence = True
        self.live = True
        self.speaker_source = lambda: []          # set by the main window: known speaker names
        self._latest = None                       # latest audio buffer for the visualizer
        self._buf_out = None
        if QAudioBufferOutput is not None:
            self._buf_out = QAudioBufferOutput(self)
            self.player.setAudioBufferOutput(self._buf_out)
            self._buf_out.audioBufferReceived.connect(self._on_buffer)

        self.screen = Screen()
        self.screen.seekRequested.connect(self._seek_frac)

        st = self.style()
        self.btn_back = self._tool(st.standardIcon(QStyle.SP_MediaSeekBackward), "Back 10 s (←)")
        self.btn_play = self._tool(st.standardIcon(QStyle.SP_MediaPlay), "Play / pause (Space)")
        self.btn_stop = self._tool(st.standardIcon(QStyle.SP_MediaStop), "Stop")
        self.btn_fwd = self._tool(st.standardIcon(QStyle.SP_MediaSeekForward), "Forward 10 s (→)")
        self.btn_play.setIconSize(self.btn_play.iconSize() * 1.4)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.time = QLabel("0:00 / 0:00")
        self.time.setMinimumWidth(110)
        self.speed = QComboBox()
        for s in ("0.75×", "1×", "1.25×", "1.5×", "1.75×", "2×"):
            self.speed.addItem(s, float(s[:-1]))
        self.speed.setCurrentIndex(1)
        self.speed.setToolTip("Playback speed")
        self.vol = QSlider(Qt.Horizontal)
        self.vol.setRange(0, 100)
        self.vol.setValue(90)
        self.vol.setFixedWidth(80)
        self.vol.setToolTip("Volume")

        controls = QHBoxLayout()
        for w in (self.btn_back, self.btn_play, self.btn_stop, self.btn_fwd):
            controls.addWidget(w)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time)
        controls.addWidget(self.speed)
        vicon = QLabel()
        vicon.setPixmap(st.standardIcon(QStyle.SP_MediaVolume).pixmap(16, 16))
        controls.addWidget(vicon)
        controls.addWidget(self.vol)

        self.cue_list = QListWidget()
        self.cue_list.setAlternatingRowColors(True)
        self.cue_list.setWordWrap(True)
        self.cue_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.cue_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.cue_list.customContextMenuRequested.connect(self._cue_menu)
        self.cue_list.itemClicked.connect(self._cue_clicked)
        self.cue_list.itemDoubleClicked.connect(lambda it: self.edit_line(self.cue_list.row(it)))
        self.plain = QPlainTextEdit()
        self.plain.setReadOnly(True)
        self.empty = QLabel("No transcript yet.")
        self.empty.setAlignment(Qt.AlignCenter)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.cue_list)
        self.stack.addWidget(self.plain)
        self.stack.addWidget(self.empty)
        self.stack.setCurrentWidget(self.empty)
        self.transcript_label = QLabel("Transcript")
        self.transcript_label.setStyleSheet("font-weight: 600;")
        self.transcript_hint = QLabel("")
        self.transcript_hint.setWordWrap(True)

        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.transcript_label)
        ll.addWidget(self.transcript_hint)
        ll.addWidget(self.stack)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.screen, 1)
        tl.addLayout(controls)

        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(lower)
        split.setSizes([330, 300])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        self.btn_play.clicked.connect(self.toggle)
        self.btn_stop.clicked.connect(self.player.stop)
        self.btn_back.clicked.connect(lambda: self.skip(-10_000))
        self.btn_fwd.clicked.connect(lambda: self.skip(10_000))
        self.slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.slider.sliderReleased.connect(self._slider_released)
        self.slider.sliderMoved.connect(lambda v: self._update_time(v))
        self.speed.currentIndexChanged.connect(lambda: self.player.setPlaybackRate(self.speed.currentData()))
        self.vol.valueChanged.connect(lambda v: self.audio.setVolume(v / 100))
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(self._on_duration)
        self.player.playbackStateChanged.connect(self._on_state)
        self.player.errorOccurred.connect(lambda e, s: self.message.emit(f"Playback error: {s}"))

        self.viz_timer = QTimer(self)
        self.viz_timer.setInterval(33)
        self.viz_timer.timeout.connect(self._viz_tick)
        self.set_enabled(False)
        self.restyle()

    def _tool(self, icon, tip):
        b = QToolButton()
        b.setIcon(icon)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        return b

    def restyle(self):
        """Re-apply theme colours (called after a light/dark switch)."""
        self.empty.setStyleSheet(f"color: {theme.c('muted')};")
        self.transcript_hint.setStyleSheet(f"color: {theme.c('help')}; font-size: 11px;")
        if self.transcript:
            self._fill_cues()

    def install_shortcuts(self, window: QWidget):
        for key, fn in ((Qt.Key_Space, self.toggle), (Qt.Key_Left, lambda: self.skip(-5000)),
                        (Qt.Key_Right, lambda: self.skip(5000))):
            sc = QShortcut(QKeySequence(key), window)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(fn)

    def set_enabled(self, on: bool):
        for w in (self.btn_back, self.btn_play, self.btn_stop, self.btn_fwd, self.slider):
            w.setEnabled(on)

    # ------------------------------------------------------------ loading
    def load(self, audio: Path | None, title: str, transcript: Transcript | None, summary: str = "",
             autoplay: bool = False, keep_position: bool = False):
        pos = self.player.position() if keep_position and audio == self._source else 0
        same_audio = audio == self._source and audio is not None
        if not same_audio:
            self.player.stop()
        self.transcript = transcript
        self.current = self.current_word = -1
        self.screen.set_sub("")
        self.screen.title.setText(title)
        self.screen.summary.setText(summary)
        self.screen.summary.setVisible(bool(summary))
        self.screen.setToolTip(summary)
        if audio is None or not audio.exists():
            self.player.setSource(QUrl())
            self._source = None
            self.set_enabled(False)
            self.screen.clock.setText("not connected")
            self.screen.peaks = b""
        elif not same_audio:
            self._source = audio
            self.player.setSource(QUrl.fromLocalFile(str(audio)))
            self.set_enabled(True)
            self.screen.clock.setText("0:00")
            self.screen.peaks = peaksmod.cached(audio) or b""
            self.screen.frac = 0.0
            if not self.screen.peaks and self.screen.show_wave:
                job = PeaksWorker(audio, self)
                job.done.connect(self._peaks_ready)
                job.finished.connect(lambda j=job: self._peaks_jobs.remove(j) if j in self._peaks_jobs else None)
                self._peaks_jobs.append(job)
                job.start()
        self.screen.update()
        self._fill_cues()
        if pos:
            self.player.setPosition(pos)
        if autoplay and self._source is not None:
            self.player.play()

    def _peaks_ready(self, path: str, data: bytes):
        if self._source is not None and str(self._source) == path:
            self.screen.peaks = data
            self.screen.update()

    def _fill_cues(self):
        t = self.transcript
        self.cue_list.clear()
        if t and t.segments:
            unc = QColor(theme.c("uncertain"))
            for s in t.segments:
                prefix = f"[{s.speaker}] " if s.speaker else ""
                mark = " ✎" if s.edited else ""
                it = QListWidgetItem(f"{fmt_time(int(s.start * 1000))}   {prefix}{s.text}{mark}")
                it.setData(Qt.UserRole, s.start)
                tips = []
                if self.show_confidence and s.shaky and not s.edited:
                    it.setForeground(unc)
                    words = ", ".join(w.w.strip() for w in s.uncertain_words[:8])
                    tips.append(f"Whisper wasn't sure here{': ' + words if words else ''}.")
                if s.speaker:
                    tips.append(f"Speaker: {s.speaker}")
                tips.append("Double-click to correct · right-click for speaker")
                it.setToolTip("\n".join(tips))
                self.cue_list.addItem(it)
            self.stack.setCurrentWidget(self.cue_list)
            name = (t.json_path or t.srt_path).name if (t.json_path or t.srt_path) else ""
            self.transcript_label.setText(f"Transcript · {name}")
            shaky = sum(1 for s in t.segments if s.shaky and not s.edited)
            hint = "Click a line to jump there. Double-click to correct it."
            if self.show_confidence and shaky:
                hint += f" {shaky} line(s) in orange are ones Whisper wasn't sure about."
            self.transcript_hint.setText(hint)
        elif t and t.plain_text:
            self.plain.setPlainText(t.plain_text)
            self.stack.setCurrentWidget(self.plain)
            self.transcript_label.setText(f"Transcript · {t.txt_path.name if t.txt_path else ''}")
            self.transcript_hint.setText("Plain text only (no timings), so it can't follow along or be edited here.")
        else:
            self.stack.setCurrentWidget(self.empty)
            self.transcript_label.setText("Transcript")
            self.transcript_hint.setText("")
        self.current = -1

    # ------------------------------------------------------------ editing
    def _selected_rows(self) -> list[int]:
        return sorted(self.cue_list.row(i) for i in self.cue_list.selectedItems())

    def _cue_menu(self, pos):
        if not (self.transcript and self.transcript.segments):
            return
        item = self.cue_list.itemAt(pos)
        if item is not None and not item.isSelected():
            self.cue_list.setCurrentItem(item)
        rows = self._selected_rows()
        if not rows:
            return
        m = QMenu(self)
        m.addAction("Correct this line…", lambda: self.edit_line(rows[0])).setEnabled(len(rows) == 1)
        m.addAction(f"Set speaker for {len(rows)} line(s)…", lambda: self.set_speaker(rows))
        if any(self.transcript.segments[r].speaker for r in rows):
            m.addAction("Clear speaker", lambda: self._apply_speaker(rows, ""))
        m.addSeparator()
        m.addAction("Copy text", lambda: QGuiApplication.clipboard().setText(
            " ".join(self.transcript.segments[r].text for r in rows)))
        m.exec(self.cue_list.viewport().mapToGlobal(pos))

    def edit_line(self, row: int):
        t = self.transcript
        if not (t and t.editable and 0 <= row < len(t.segments)):
            return
        seg = t.segments[row]
        text, ok = QInputDialog.getMultiLineText(self, "Correct line", f"{fmt_time(int(seg.start * 1000))}", seg.text)
        if not ok or " ".join(text.split()) == seg.text:
            return
        new_words = t.edit(row, text)
        self._save(new_words)

    def set_speaker(self, rows: list[int]):
        known = sorted(set(self.speaker_source()) | set(self.transcript.speakers()))
        cur = self.transcript.segments[rows[0]].speaker
        name, ok = QInputDialog.getItem(self, "Speaker", "Who's speaking?", known or [""],
                                        known.index(cur) if cur in known else 0, True)
        if ok and name.strip():
            self._apply_speaker(rows, name.strip())

    def _apply_speaker(self, rows, name):
        self.transcript.set_speaker(rows, name)
        self._save([])

    def _save(self, new_words):
        try:
            self.transcript.save()
        except (OSError, RuntimeError) as e:
            QMessageBox.warning(self, "Rambler", f"Couldn't save the transcript:\n{e}")
            return
        row = self.cue_list.currentRow()
        self._fill_cues()
        if row >= 0:
            self.cue_list.setCurrentRow(row)
        self.edited.emit(self.transcript, new_words)

    # ------------------------------------------------------------ transport
    def toggle(self):
        if not self.btn_play.isEnabled():
            return
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def skip(self, ms: int):
        if self.btn_play.isEnabled():
            self.player.setPosition(max(0, min(self.player.duration(), self.player.position() + ms)))

    def seek_seconds(self, t: float, play: bool = True):
        if self._source is None:
            return
        self.player.setPosition(int(t * 1000) + 1)
        if play and self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.play()

    def _seek_frac(self, f: float):
        if self.player.duration() > 0:
            self.player.setPosition(int(f * self.player.duration()))

    def _slider_released(self):
        self._seeking = False
        self.player.setPosition(self.slider.value())

    def _cue_clicked(self, item: QListWidgetItem):
        if len(self.cue_list.selectedItems()) > 1:
            return
        self.seek_seconds(float(item.data(Qt.UserRole)))

    def _on_duration(self, d: int):
        self.slider.setRange(0, d)
        self._update_time(self.player.position())

    def _on_state(self, s):
        playing = s == QMediaPlayer.PlayingState
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))
        if playing and self.live and self._buf_out is not None:
            self.viz_timer.start()
        elif not playing:
            self._latest = None

    def _update_time(self, pos: int):
        self.time.setText(f"{fmt_time(pos)} / {fmt_time(self.player.duration())}")
        self.screen.clock.setText(fmt_time(pos))
        d = self.player.duration()
        self.screen.frac = pos / d if d else 0.0
        self.screen.update()

    def _on_position(self, pos: int):
        if not self._seeking:
            self.slider.setValue(pos)
        self._update_time(pos)
        t = self.transcript
        if not (t and t.segments):
            return
        sec = pos / 1000
        i = t.index_at(sec)
        wi = t.segments[i].word_at(sec) if i >= 0 else -1
        if i != self.current or wi != self.current_word:
            changed_line = i != self.current
            self.current, self.current_word = i, wi
            self.screen.set_sub(self._sub_html(t.segments[i], wi) if i >= 0 else "")
            if changed_line and i >= 0 and len(self.cue_list.selectedItems()) <= 1:
                self.cue_list.setCurrentRow(i)
                self.cue_list.scrollToItem(self.cue_list.item(i), QListWidget.PositionAtCenter)

    def _sub_html(self, seg, wi: int) -> str:
        spk = (f"<span style='color:#b39ddb; font-weight:600'>{html.escape(seg.speaker)}:</span> "
               if seg.speaker else "")
        if not seg.words:
            return spk + html.escape(seg.text)
        # long segments (journal ones run ~70 words) show as a moving window of ~14 words
        size = 14
        start = 0
        if len(seg.words) > size + 4:
            start = (max(wi, 0) // size) * size
            if len(seg.words) - start < 5:          # don't strand a few words on their own
                start = max(0, len(seg.words) - size)
        out = ["…"] if start else []
        end = start + size if len(seg.words) > size + 4 else len(seg.words)
        for k, w in list(enumerate(seg.words))[start:end]:
            txt = html.escape(w.w.strip())
            style = []
            if k == wi:
                style.append("color:#8ec5ff")
            elif k > wi >= 0:
                style.append("color:#d8d8d8")
            if self.show_confidence and w.uncertain and not seg.edited:
                style.append("text-decoration: underline; color:#ffb74d" if k != wi else "text-decoration: underline")
            out.append(f"<span style='{';'.join(style)}'>{txt}</span>" if style else txt)
        if end < len(seg.words):
            out.append("…")
        return spk + " ".join(out)

    # ------------------------------------------------------------ live visualizer
    def _on_buffer(self, buf):
        if self.live:
            self._latest = buf

    def _viz_tick(self):
        lv = self.screen.levels
        buf, self._latest = self._latest, None
        target = None
        if buf is not None:
            target = self._levels_from(buf)
        for i in range(BARS):
            v = target[i] if target else 0.0
            lv[i] = v if v > lv[i] else lv[i] * 0.86     # fast attack, slow fall
        if self.player.playbackState() != QMediaPlayer.PlayingState and max(lv) < 0.01:
            self.screen.levels = [0.0] * BARS
            self.viz_timer.stop()
        self.screen.update()

    @staticmethod
    def _levels_from(buf) -> list[float] | None:
        fmt = buf.format()
        ch = max(1, fmt.channelCount())
        raw = bytes(buf.constData())
        sf = fmt.sampleFormat()
        try:
            if np is not None:
                dt = {fmt.SampleFormat.Float: np.float32, fmt.SampleFormat.Int16: np.int16,
                      fmt.SampleFormat.Int32: np.int32, fmt.SampleFormat.UInt8: np.uint8}.get(sf)
                if dt is None:
                    return None
                a = np.frombuffer(raw, dtype=dt).astype(np.float32)
                if sf == fmt.SampleFormat.Int16:
                    a /= 32768.0
                elif sf == fmt.SampleFormat.Int32:
                    a /= 2147483648.0
                elif sf == fmt.SampleFormat.UInt8:
                    a = (a - 128.0) / 128.0
                mono = a[: len(a) - len(a) % ch].reshape(-1, ch).mean(axis=1)[-2048:]
                if len(mono) < 64:
                    return None
                spec = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
                # log-spaced bands from ~60 Hz to ~12 kHz
                rate = fmt.sampleRate() or 48000
                freqs = np.fft.rfftfreq(len(mono), 1 / rate)
                edges = np.geomspace(60, min(12000, rate / 2), BARS + 1)
                out = []
                for lo, hi in zip(edges[:-1], edges[1:]):
                    band = spec[(freqs >= lo) & (freqs < hi)]
                    out.append(float(band.max()) if band.size else 0.0)
                return [max(0.0, min(1.0, (20 * math.log10(v + 1e-9) + 10) / 50)) for v in out]
            # no numpy: loudness envelope across the buffer instead of a spectrum
            if sf != fmt.SampleFormat.Int16 and sf != fmt.SampleFormat.Float:
                return None
            a = array("f" if sf == fmt.SampleFormat.Float else "h")
            a.frombytes(raw[: len(raw) - len(raw) % a.itemsize])
            scale = 1.0 if sf == fmt.SampleFormat.Float else 32768.0
            n = max(1, len(a) // BARS)
            return [min(1.0, 3 * math.sqrt(sum((x / scale) ** 2 for x in a[i * n:(i + 1) * n:4]) / max(1, n / 4)))
                    for i in range(BARS)]
        except (ValueError, TypeError):
            return None

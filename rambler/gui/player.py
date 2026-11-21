"""VLC-style audio player with subtitle overlay and a clickable cue list."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QComboBox, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel,
                               QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSlider,
                               QSplitter, QStackedWidget, QStyle, QToolButton, QVBoxLayout, QWidget)

from .. import srt as srtmod


def fmt_time(ms: int) -> str:
    s = max(0, ms // 1000)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class Screen(QFrame):
    """The black 'video' area: title on top, big subtitle line at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("screen")
        self.setStyleSheet("#screen { background: #0b0b0d; border-radius: 6px; }")
        self.setMinimumHeight(220)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 14, 18, 18)
        self.title = QLabel("No recording selected")
        self.title.setStyleSheet("color: #9aa0a6; font-size: 13px;")
        self.title.setWordWrap(True)
        self.clock = QLabel("")
        self.clock.setAlignment(Qt.AlignCenter)
        self.clock.setStyleSheet("color: #3a3d42; font-size: 44px; font-weight: 300;")
        self.sub = QLabel("")
        self.sub.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        self.sub.setWordWrap(True)
        f = QFont()
        f.setPointSize(17)
        f.setWeight(QFont.DemiBold)
        self.sub.setFont(f)
        self.sub.setStyleSheet("color: #ffffff; background: rgba(0,0,0,0.55); padding: 6px 12px; border-radius: 4px;")
        shadow = QGraphicsDropShadowEffect(self.sub)
        shadow.setBlurRadius(6)
        shadow.setOffset(1, 1)
        shadow.setColor(QColor(0, 0, 0))
        self.sub.setGraphicsEffect(shadow)
        self.sub.hide()
        lay.addWidget(self.title)
        lay.addStretch(1)
        lay.addWidget(self.clock)
        lay.addStretch(1)
        lay.addWidget(self.sub)

    def set_sub(self, text: str):
        self.sub.setText(text)
        self.sub.setVisible(bool(text))


class PlayerPanel(QWidget):
    message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.9)
        self.cues: list[srtmod.Cue] = []
        self.index = srtmod.CueIndex([])
        self.current_cue = -1
        self._seeking = False

        self.screen = Screen()

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
        self.plain = QPlainTextEdit()
        self.plain.setReadOnly(True)
        self.empty = QLabel("No transcript yet.")
        self.empty.setAlignment(Qt.AlignCenter)
        self.empty.setStyleSheet("color: palette(mid);")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.cue_list)
        self.stack.addWidget(self.plain)
        self.stack.addWidget(self.empty)
        self.stack.setCurrentWidget(self.empty)
        self.transcript_label = QLabel("Transcript")
        self.transcript_label.setStyleSheet("font-weight: 600;")

        lower = QWidget()
        ll = QVBoxLayout(lower)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(self.transcript_label)
        ll.addWidget(self.stack)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.screen, 1)
        tl.addLayout(controls)

        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(lower)
        split.setSizes([320, 300])
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
        self.cue_list.itemClicked.connect(self._cue_clicked)
        self.set_enabled(False)

    def _tool(self, icon, tip):
        b = QToolButton()
        b.setIcon(icon)
        b.setToolTip(tip)
        b.setAutoRaise(True)
        return b

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
    def load(self, audio: Path | None, title: str, subs: Path | None, txt: Path | None, autoplay=False):
        self.player.stop()
        self.cues = []
        self.current_cue = -1
        self.screen.set_sub("")
        self.screen.title.setText(title)
        self.cue_list.clear()
        if audio is None or not audio.exists():
            self.player.setSource(QUrl())
            self.set_enabled(False)
            self.screen.clock.setText("not connected")
        else:
            self.player.setSource(QUrl.fromLocalFile(str(audio)))
            self.set_enabled(True)
            self.screen.clock.setText("0:00")
        if subs and subs.exists():
            try:
                self.cues = srtmod.load(subs)
            except OSError as e:
                self.message.emit(f"Could not read subtitles: {e}")
        self.index = srtmod.CueIndex(self.cues)
        if self.cues:
            for c in self.cues:
                it = QListWidgetItem(f"{fmt_time(int(c.start * 1000))}   {c.text}")
                it.setData(Qt.UserRole, c.start)
                self.cue_list.addItem(it)
            self.stack.setCurrentWidget(self.cue_list)
            self.transcript_label.setText(f"Subtitles · {subs.name} · click a line to jump")
        elif txt and txt.exists():
            self.plain.setPlainText(txt.read_text(encoding="utf-8", errors="replace"))
            self.stack.setCurrentWidget(self.plain)
            self.transcript_label.setText(f"Transcript · {txt.name} (no SRT, so no timing)")
        else:
            self.stack.setCurrentWidget(self.empty)
            self.transcript_label.setText("Transcript")
        if autoplay and audio is not None:
            self.player.play()

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

    def _slider_released(self):
        self._seeking = False
        self.player.setPosition(self.slider.value())

    def _cue_clicked(self, item: QListWidgetItem):
        self.player.setPosition(int(item.data(Qt.UserRole) * 1000) + 1)
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            self.player.play()

    def _on_duration(self, d: int):
        self.slider.setRange(0, d)
        self._update_time(self.player.position())

    def _on_state(self, s):
        icon = QStyle.SP_MediaPause if s == QMediaPlayer.PlayingState else QStyle.SP_MediaPlay
        self.btn_play.setIcon(self.style().standardIcon(icon))

    def _update_time(self, pos: int):
        self.time.setText(f"{fmt_time(pos)} / {fmt_time(self.player.duration())}")
        self.screen.clock.setText(fmt_time(pos))

    def _on_position(self, pos: int):
        if not self._seeking:
            self.slider.setValue(pos)
        self._update_time(pos)
        i = self.index.at(pos / 1000)
        if i != self.current_cue:
            self.current_cue = i
            self.screen.set_sub(self.cues[i].text if i >= 0 else "")
            if i >= 0:
                self.cue_list.setCurrentRow(i)
                self.cue_list.scrollToItem(self.cue_list.item(i), QListWidget.PositionAtCenter)

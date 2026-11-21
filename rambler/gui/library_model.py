# SPDX-License-Identifier: GPL-3.0-or-later
"""The library as a tree: each multitrack take's mix is a row with its tracks nested under it (or a flat
list when grouping is off), plus tag chips and filtering."""
from __future__ import annotations

import time

from PySide6.QtCore import QAbstractItemModel, QModelIndex, QRectF, QSize, QSortFilterProxyModel, Qt
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from ..library import Recording
from ..syncer import human
from ..tags import TagStore
from . import theme

COLS = ["Status", "Recording", "Tags", "Folder", "Size", "Modified", "Copies", "Subtitles", "Speakers", "Summary"]
C_STATUS, C_NAME, C_TAGS, C_FOLDER, C_SIZE, C_MOD, C_COPIES, C_SUBS, C_SPK, C_SUMM = range(len(COLS))
STATUS_COLORS = {
    "To be transcribed": "pending", "Transcribed (no SRT)": "nosrt", "Transcribed": "done",
    "Failed": "failed", "Queued": "muted", "Too short": "muted", "Track (mix transcribed)": "muted",
    "Track (mix pending)": "muted",
}
RecRole = Qt.UserRole
SortRole = Qt.UserRole + 1
TagsRole = Qt.UserRole + 2


def _dot(color: str, size: int = 10) -> QIcon:
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pm)


class LibraryModel(QAbstractItemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recs: list[Recording] = []                  # every recording, flat
        self.top: list[Recording] = []                   # rows at the top level
        self.kids: dict[int, list[Recording]] = {}       # top row -> nested track rows
        self.grouped = True
        self.overlay: dict[str, str] = {}                # rel -> transient status (Queued, 40%, Failed)
        self.n_audio_dirs = 0
        self.info: dict[str, tuple[str, str]] = {}       # rel -> (summary, speakers), from the search index
        self.tags = TagStore()
        self._dots: dict[str, QIcon] = {}

    # ------------------------------------------------------------------ structure
    def _rebuild(self):
        self.top, self.kids = [], {}
        if not self.grouped:
            self.top = list(self.recs)
            return
        children: dict[int, list[Recording]] = {}
        for r in self.recs:
            if r.mix is not None:
                children.setdefault(id(r.mix), []).append(r)
        for r in self.recs:
            if r.mix is None:
                self.top.append(r)
                if id(r) in children:
                    self.kids[len(self.top) - 1] = sorted(children[id(r)], key=lambda x: x.name.lower())
        # tracks whose main isn't in the list (shouldn't happen) still show
        shown = {id(r) for r in self.top} | {id(k) for ks in self.kids.values() for k in ks}
        self.top += [r for r in self.recs if id(r) not in shown]

    def set_recs(self, recs, n_audio_dirs):
        self.beginResetModel()
        self.recs = recs
        self.n_audio_dirs = n_audio_dirs
        # A failure stays visible across rescans, but only while the recording still has no transcript.
        done = {r.rel.as_posix() for r in recs if r.transcribed}
        self.overlay = {k: v for k, v in self.overlay.items() if v.startswith("Failed") and k not in done}
        self._rebuild()
        self.endResetModel()

    def set_grouped(self, on: bool):
        if on != self.grouped:
            self.beginResetModel()
            self.grouped = on
            self._rebuild()
            self.endResetModel()

    def set_tags(self, store: TagStore):
        self.tags = store
        self._dots.clear()
        self.refresh()

    def set_info(self, info):
        self.info = info
        self.refresh()

    def refresh(self):
        """Repaint everything (tags, info, theme) without losing expansion or selection."""
        if self.top:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.top) - 1, len(COLS) - 1))
            for row, ks in self.kids.items():
                parent = self.index(row, 0)
                self.dataChanged.emit(self.index(0, 0, parent), self.index(len(ks) - 1, len(COLS) - 1, parent))

    def set_overlay(self, key: str, state: str):
        if state:
            self.overlay[key] = state
        else:
            self.overlay.pop(key, None)
        idx = self.index_of(key)
        if idx.isValid():
            self.dataChanged.emit(idx, idx.siblingAtColumn(len(COLS) - 1))

    # ------------------------------------------------------------------ lookups
    def rec_at(self, index: QModelIndex) -> Recording | None:
        if not index.isValid():
            return None
        pid = index.internalId()
        if pid == 0:
            return self.top[index.row()] if index.row() < len(self.top) else None
        ks = self.kids.get(pid - 1, [])
        return ks[index.row()] if index.row() < len(ks) else None

    def index_of(self, rel) -> QModelIndex:
        key = rel if isinstance(rel, str) else rel.as_posix()
        for row, r in enumerate(self.top):
            if r.rel.as_posix() == key:
                return self.index(row, 0)
        for row, ks in self.kids.items():
            for k, r in enumerate(ks):
                if r.rel.as_posix() == key:
                    return self.index(k, 0, self.index(row, 0))
        return QModelIndex()

    # ------------------------------------------------------------------ Qt model API
    def index(self, row, column, parent=QModelIndex()):
        if column < 0 or column >= len(COLS) or row < 0:
            return QModelIndex()
        if not parent.isValid():
            return self.createIndex(row, column, 0) if row < len(self.top) else QModelIndex()
        if parent.internalId() == 0 and row < len(self.kids.get(parent.row(), [])):
            return self.createIndex(row, column, parent.row() + 1)
        return QModelIndex()

    def parent(self, index=QModelIndex()):
        if not index.isValid() or index.internalId() == 0:
            return QModelIndex()
        return self.createIndex(index.internalId() - 1, 0, 0)

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return len(self.top)
        if parent.internalId() == 0 and parent.column() == 0:
            return len(self.kids.get(parent.row(), []))
        return 0

    def columnCount(self, parent=QModelIndex()):
        return len(COLS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLS[section]
        return None

    def status_of(self, r: Recording) -> str:
        ov = self.overlay.get(r.rel.as_posix())
        if ov and ov.startswith("Failed") and r.transcribed:
            return r.status()
        return ov or r.status()

    def data(self, index, role=Qt.DisplayRole):
        r = self.rec_at(index)
        if r is None:
            return None
        c = index.column()
        rel = r.rel.as_posix()
        if role == RecRole:
            return r
        if role == TagsRole:
            return self.tags.of(rel)
        if role == Qt.DisplayRole:
            if c == C_STATUS:
                s = self.status_of(r)
                n = len(self.kids.get(index.row(), [])) if index.internalId() == 0 and self.grouped else 0
                return f"{s}  ·  {n + 1} files" if n and not s.startswith(("Queued", "Transcribing", "Decoding")) \
                    else s
            if c == C_NAME:
                return r.name
            if c == C_FOLDER:
                return str(r.rel.parent) if str(r.rel.parent) != "." else ""
            if c == C_SIZE:
                return human(r.size)
            if c == C_MOD:
                return time.strftime("%Y-%m-%d %H:%M", time.localtime(r.mtime))
            if c == C_COPIES:
                return f"{len(r.locations)}/{self.n_audio_dirs}"
            if c == C_SUBS:
                return r.srt.suffix.lstrip(".").upper() if r.srt else ""
            if c == C_TAGS:
                return ", ".join(self.tags.of(rel))
            if c == C_SPK:
                return self.info.get(rel, ("", ""))[1]
            if c == C_SUMM:
                return self.info.get(rel, ("", ""))[0].replace("\n", " ")
        if role == SortRole:
            inf = self.info.get(rel, ("", ""))
            return {C_STATUS: self.status_of(r), C_NAME: r.name.lower(), C_FOLDER: str(r.rel.parent).lower(),
                    C_SIZE: r.size, C_MOD: r.mtime, C_COPIES: len(r.locations), C_SUBS: bool(r.srt),
                    C_TAGS: ", ".join(self.tags.of(rel)).lower() or "~", C_SPK: inf[1].lower(),
                    C_SUMM: inf[0].lower()}[c]
        if role == Qt.DecorationRole and c == C_NAME:
            ts = self.tags.of(rel)
            if ts:
                col = self.tags.color(ts[0])
                if col not in self._dots:
                    self._dots[col] = _dot(col)
                return self._dots[col]
        if role == Qt.ForegroundRole and c == C_STATUS:
            s = self.status_of(r)
            key = "Failed" if s.startswith("Failed") else s if s in STATUS_COLORS else None
            if s.startswith(("Transcribing", "Decoding", "Loading")):
                return QBrush(QColor(theme.c("working")))
            if key:
                return QBrush(QColor(theme.c(STATUS_COLORS[key])))
        if role == Qt.ForegroundRole and c == C_COPIES and len(r.locations) < self.n_audio_dirs:
            return QBrush(QColor(theme.c("pending")))
        if role == Qt.ToolTipRole and c == C_SUMM:
            return self.info.get(rel, ("", ""))[0] or None
        if role == Qt.ToolTipRole:
            locs = "\n".join(str(l) for l in r.locations)
            tip = f"{r.rel}\nOn:\n{locs}\nTranscript: {r.txt or '—'}\nSubtitles: {r.srt or '—'}"
            if r.mix is not None:
                tip += f"\nPart of the take mixed in {r.mix.name}"
            return tip
        if role == Qt.TextAlignmentRole and c in (C_SIZE, C_COPIES):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


class FilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = ""
        self.mode = "all"
        self.tag = ""            # "" = any, "\0untagged" = no tags, else a tag name
        self.setSortRole(SortRole)
        self.setRecursiveFilteringEnabled(True)   # a matching track keeps its take visible

    def set_filter(self, text, mode, tag=""):
        self.text, self.mode, self.tag = text.lower(), mode, tag
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        src: LibraryModel = self.sourceModel()
        r = src.rec_at(src.index(row, 0, parent))
        if r is None:
            return False
        rel = r.rel.as_posix()
        tags = src.tags.of(rel)
        if r.mix is not None:                       # a take's tags cover its tracks too
            tags = list(dict.fromkeys(tags + src.tags.of(r.mix.rel.as_posix())))
        if self.tag == "\0untagged" and tags:
            return False
        if self.tag and self.tag != "\0untagged" and self.tag not in tags:
            return False
        if self.text:
            summ, spk = src.info.get(rel, ("", ""))
            hay = " ".join((rel, spk, summ, " ".join(tags))).lower()
            if self.text not in hay:
                return False
        if self.mode == "pending":
            return r.pending
        if self.mode == "nosrt":
            return not r.srt
        if self.mode == "missing":
            return len(r.locations) < src.n_audio_dirs
        return True


class TagChips(QStyledItemDelegate):
    """Paints a recording's tags as small coloured chips."""

    def __init__(self, model: LibraryModel, parent=None):
        super().__init__(parent)
        self.model = model

    def paint(self, painter, option, index):
        tags = index.data(TagsRole) or []
        self.initStyleOption(option, index)
        option.text = ""
        option.widget.style().drawControl(QStyle.CE_ItemViewItem, option, painter, option.widget)
        if not tags:
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        f = option.font
        f.setPointSizeF(max(7.0, f.pointSizeF() * 0.85))
        painter.setFont(f)
        fm = QFontMetrics(f)
        x = option.rect.x() + 4
        h = fm.height() + 2
        y = option.rect.y() + (option.rect.height() - h) / 2
        for t in tags:
            w = fm.horizontalAdvance(t) + 12
            if x + w > option.rect.right() - 2:
                painter.setPen(option.palette.text().color())
                painter.drawText(QRectF(x, y, 16, h), Qt.AlignVCenter, "…")
                break
            col = QColor(self.model.tags.color(t))
            painter.setPen(Qt.NoPen)
            painter.setBrush(col)
            painter.drawRoundedRect(QRectF(x, y, w, h), h / 2, h / 2)
            painter.setPen(QColor("#ffffff") if col.lightness() < 170 else QColor("#111111"))
            painter.drawText(QRectF(x, y, w, h), Qt.AlignCenter, t)
            x += w + 4
        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(s.width(), max(s.height(), QFontMetrics(option.font).height() + 6))

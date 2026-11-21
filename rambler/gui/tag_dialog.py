# SPDX-License-Identifier: GPL-3.0-or-later
"""Create, rename, recolour and delete tags."""
from __future__ import annotations

import copy

from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QColorDialog, QDialog, QDialogButtonBox, QHBoxLayout, QInputDialog, QLabel,
                               QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout)

from ..tags import PALETTE, TagStore
from . import theme
from .library_model import _dot


class TagDialog(QDialog):
    def __init__(self, store: TagStore, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tags")
        self.resize(420, 380)
        self.result_store = copy.deepcopy(store)
        hint = QLabel("Tags help you sort recordings, say work from personal. Right-click recordings in the "
                      "library to tag them. Tags are saved alongside your transcripts and sync to every drive.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.c('help')};")
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _: self._recolor())
        add, ren, col, rm = (QPushButton(t) for t in ("New…", "Rename…", "Colour…", "Delete"))
        add.clicked.connect(self._add)
        ren.clicked.connect(self._rename)
        col.clicked.connect(self._recolor)
        rm.clicked.connect(self._delete)
        side = QVBoxLayout()
        for b in (add, ren, col, rm):
            side.addWidget(b)
        side.addStretch(1)
        row = QHBoxLayout()
        row.addWidget(self.list, 1)
        row.addLayout(side)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addLayout(row, 1)
        lay.addWidget(bb)
        self._fill()

    def _fill(self, select: str | None = None):
        self.list.clear()
        st = self.result_store
        for t in st.names():
            n = len(st.recordings_with(t))
            it = QListWidgetItem(_dot(st.color(t), 14), f"{t}    ({n} recording{'s' if n != 1 else ''})")
            it.setData(256, t)
            self.list.addItem(it)
            if t == select:
                self.list.setCurrentItem(it)
        if not st.names():
            self.list.addItem("No tags yet. Click New… to make one.")

    def _current(self) -> str | None:
        it = self.list.currentItem()
        return it.data(256) if it else None

    def _add(self):
        name, ok = QInputDialog.getText(self, "New tag", "Tag name:")
        if ok and name.strip():
            self._fill(self.result_store.create(name))

    def _rename(self):
        t = self._current()
        if not t:
            return
        name, ok = QInputDialog.getText(self, "Rename tag", "New name:", text=t)
        if ok and name.strip() and name.strip() != t:
            if name.strip() in self.result_store.colors and QMessageBox.question(
                    self, "Tags", f"'{name.strip()}' already exists. Merge '{t}' into it?") != QMessageBox.Yes:
                return
            self.result_store.rename(t, name)
            self._fill(" ".join(name.split()))

    def _recolor(self):
        t = self._current()
        if not t:
            return
        dlg = QColorDialog(QColor(self.result_store.color(t)), self)
        for i, c in enumerate(PALETTE):
            QColorDialog.setCustomColor(i, QColor(c))
        if dlg.exec() == QDialog.Accepted:
            self.result_store.recolor(t, dlg.selectedColor().name())
            self._fill(t)

    def _delete(self):
        t = self._current()
        if not t:
            return
        n = len(self.result_store.recordings_with(t))
        if n and QMessageBox.question(self, "Tags", f"Remove '{t}' from {n} recording(s) and delete it?") \
                != QMessageBox.Yes:
            return
        self.result_store.delete(t)
        self._fill()

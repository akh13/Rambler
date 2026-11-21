# SPDX-License-Identifier: GPL-3.0-or-later
"""Light / dark / follow-the-system themes, plus the handful of app colours that aren't in a QPalette."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

_current = {"dark": False}

COLORS = {
    #                       light       dark
    "pending":           ("#c77700", "#ffb74d"),
    "nosrt":             ("#2f6fb0", "#64b5f6"),
    "done":              ("#2e7d32", "#81c784"),
    "failed":            ("#c62828", "#ef9a9a"),
    "muted":             ("#808080", "#8a8f98"),
    "working":           ("#7b3fa0", "#ce93d8"),
    "chip_ok_bg":        ("#e3f1e5", "#1f3a24"),
    "chip_ok_fg":        ("#1b5e20", "#a5d6a7"),
    "chip_off_bg":       ("#eeeeee", "#2b2d31"),
    "chip_off_fg":       ("#888888", "#7d828a"),
    "chip_busy_bg":      ("#e3edf9", "#1d2f45"),
    "chip_busy_fg":      ("#0d47a1", "#90caf9"),
    "chip_err_bg":       ("#fdecea", "#4a1f1f"),
    "chip_err_fg":       ("#b71c1c", "#ef9a9a"),
    "help":              ("#6b6f76", "#9aa0a6"),
    "uncertain":         ("#d17a00", "#ffb74d"),
    "speaker":           ("#6a4fb3", "#b39ddb"),
    "edited":            ("#2e7d32", "#81c784"),
}


def is_dark() -> bool:
    return _current["dark"]


def c(name: str) -> str:
    light, dark = COLORS[name]
    return dark if _current["dark"] else light


def system_is_dark() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except AttributeError:
        return QApplication.palette().window().color().lightness() < 128


def _dark_palette() -> QPalette:
    p = QPalette()
    base, alt, win, text = QColor("#1e1f22"), QColor("#26282c"), QColor("#2b2d31"), QColor("#e6e6e6")
    hl = QColor("#3d7bd9")
    for group in (QPalette.Active, QPalette.Inactive):
        p.setColor(group, QPalette.Window, win)
        p.setColor(group, QPalette.WindowText, text)
        p.setColor(group, QPalette.Base, base)
        p.setColor(group, QPalette.AlternateBase, alt)
        p.setColor(group, QPalette.ToolTipBase, QColor("#3a3d42"))
        p.setColor(group, QPalette.ToolTipText, text)
        p.setColor(group, QPalette.PlaceholderText, QColor("#7d828a"))
        p.setColor(group, QPalette.Text, text)
        p.setColor(group, QPalette.Button, QColor("#33363b"))
        p.setColor(group, QPalette.ButtonText, text)
        p.setColor(group, QPalette.BrightText, QColor("#ff6b6b"))
        p.setColor(group, QPalette.Highlight, hl)
        p.setColor(group, QPalette.HighlightedText, QColor("#ffffff"))
        p.setColor(group, QPalette.Link, QColor("#8ab4f8"))
        p.setColor(group, QPalette.Mid, QColor("#55585e"))
        p.setColor(group, QPalette.Dark, QColor("#18191b"))
        p.setColor(group, QPalette.Light, QColor("#45484e"))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor("#6b6f76"))
    p.setColor(QPalette.Disabled, QPalette.Base, base)
    p.setColor(QPalette.Disabled, QPalette.Window, win)
    p.setColor(QPalette.Disabled, QPalette.Button, QColor("#2e3035"))
    return p


def apply(app: QApplication, mode: str) -> bool:
    """mode: system | light | dark. Returns whether the result is dark."""
    app.setStyle("Fusion")
    dark = system_is_dark() if mode == "system" else mode == "dark"
    _current["dark"] = dark
    app.setPalette(_dark_palette() if dark else app.style().standardPalette())
    # widgets with their own style sheets cache text colours; re-polish so they follow the new palette
    st = app.style()
    for w in app.allWidgets():
        if w.styleSheet():
            w.setPalette(app.palette())
            st.unpolish(w)
            st.polish(w)
            w.update()
    return dark

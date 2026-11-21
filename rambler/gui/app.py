import sys

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from .. import APP_ID, APP_NAME, __version__


def run(files: list[str] | None = None, minimized: bool = False) -> int:
    QGuiApplication.setDesktopFileName(APP_ID)
    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    from ..config import Config
    from . import theme
    theme.apply(app, Config.load().theme)
    from .main_window import MainWindow
    from pathlib import Path
    w = MainWindow(open_files=[Path(f) for f in (files or [])])
    w.showMinimized() if minimized else w.show()
    return app.exec()

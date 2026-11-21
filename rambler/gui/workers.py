"""Background job runner so the UI never blocks on disk or whisper."""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class JobContext:
    def __init__(self, worker: "Worker"):
        self._w = worker
        self.cancel = threading.Event()

    def progress(self, frac: float, text: str = "") -> None:
        self._w.progress.emit(float(max(0.0, min(1.0, frac))), text)

    def overall(self, done: int, total: int, text: str = "") -> None:
        self._w.overall.emit(done, total, text)

    def log(self, text: str) -> None:
        self._w.log.emit(text)

    def item_state(self, key: str, state: str) -> None:
        self._w.item_state.emit(key, state)


class Worker(QThread):
    progress = Signal(float, str)
    overall = Signal(int, int, str)
    log = Signal(str)
    item_state = Signal(str, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, title: str, fn: Callable[[JobContext], Any], parent=None):
        super().__init__(parent)
        self.title = title
        self.fn = fn
        self.ctx = JobContext(self)

    def cancel(self) -> None:
        self.ctx.cancel.set()

    def run(self) -> None:
        try:
            result = self.fn(self.ctx)
        except Exception as e:  # surfaced in the UI log
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"{type(e).__name__}: {e}")
            return
        self.succeeded.emit(result)

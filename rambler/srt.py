"""Minimal SRT / WebVTT parsing and writing."""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")


@dataclass
class Cue:
    start: float
    end: float
    text: str


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        raise ValueError(s)
    h, mi, se, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def parse(text: str) -> list[Cue]:
    cues: list[Cue] = []
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("﻿")
    for block in re.split(r"\n\s*\n", text):
        lines = [l for l in block.strip().split("\n")]
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        a, b = lines[idx].split("-->", 1)
        try:
            start, end = _ts(a), _ts(b)
        except ValueError:
            continue
        body = "\n".join(lines[idx + 1:]).strip()
        body = re.sub(r"<[^>]+>", "", body)  # strip vtt/srt styling tags
        cues.append(Cue(start, end, body))
    cues.sort(key=lambda c: c.start)
    return cues


def load(path: Path) -> list[Cue]:
    return parse(Path(path).read_text(encoding="utf-8", errors="replace"))


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def dumps(cues: list[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(f"{i}\n{fmt_ts(c.start)} --> {fmt_ts(c.end)}\n{c.text.strip()}\n")
    return "\n".join(out)


class CueIndex:
    """Fast lookup of the cue active at a given time."""

    def __init__(self, cues: list[Cue]):
        self.cues = cues
        self._starts = [c.start for c in cues]

    def at(self, t: float) -> int:
        i = bisect.bisect_right(self._starts, t) - 1
        if 0 <= i < len(self.cues) and self.cues[i].start <= t < self.cues[i].end:
            return i
        return -1

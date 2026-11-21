# SPDX-License-Identifier: GPL-3.0-or-later
"""Coloured tags on recordings (Work, Personal, ...).

Stored as rambler-tags.json at the top of every connected transcript folder, so tags travel with the
transcripts and sync to every drive. Recordings without a transcript can be tagged too: entries are keyed
by the recording's path inside the recording folders.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Config

FILE = "rambler-tags.json"
PALETTE = ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444", "#a855f7", "#14b8a6", "#ec4899", "#64748b"]


class TagStore:
    def __init__(self):
        self.colors: dict[str, str] = {}          # tag -> colour, in creation order
        self.by_rec: dict[str, list[str]] = {}    # recording rel path -> tags

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, cfg: Config) -> "TagStore":
        """Read the newest copy across the connected transcript folders (they're normally identical)."""
        t = cls()
        newest = None
        for root in cfg.transcript_roots():
            p = root / FILE
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if newest is None or m > newest[0]:
                newest = (m, p)
        if newest:
            try:
                data = json.loads(newest[1].read_text(encoding="utf-8"))
                t.colors = {str(k): str(v) for k, v in (data.get("tags") or {}).items()}
                t.by_rec = {str(k): [x for x in v if x in t.colors]
                            for k, v in (data.get("recordings") or {}).items() if v}
            except (OSError, json.JSONDecodeError, AttributeError):
                pass
        return t

    def save(self, cfg: Config) -> list[Path]:
        data = {"version": 1, "tags": self.colors,
                "recordings": {k: v for k, v in sorted(self.by_rec.items()) if v}}
        text = json.dumps(data, indent=1, ensure_ascii=False)
        written = []
        for root in cfg.transcript_roots():
            p = root / FILE
            tmp = p.with_name(p.name + ".rambler-part")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, p)
            written.append(p)
        if not written:
            raise RuntimeError("No transcript folder is connected, so tags can't be saved.")
        return written

    # ------------------------------------------------------------------ queries
    def names(self) -> list[str]:
        return list(self.colors)

    def color(self, tag: str) -> str:
        return self.colors.get(tag, "#888888")

    def of(self, rel: str) -> list[str]:
        return [t for t in self.by_rec.get(rel, []) if t in self.colors]

    def recordings_with(self, tag: str) -> set[str]:
        return {rel for rel, ts in self.by_rec.items() if tag in ts}

    def next_color(self) -> str:
        used = set(self.colors.values())
        return next((c for c in PALETTE if c not in used), PALETTE[len(self.colors) % len(PALETTE)])

    # ------------------------------------------------------------------ edits
    def create(self, tag: str, color: str | None = None) -> str:
        tag = " ".join(tag.split())
        if not tag:
            raise ValueError("A tag needs a name.")
        if tag not in self.colors:
            self.colors[tag] = color or self.next_color()
        return tag

    def assign(self, rels: list[str], tag: str, on: bool = True) -> None:
        for rel in rels:
            cur = self.by_rec.setdefault(rel, [])
            if on and tag not in cur:
                cur.append(tag)
            elif not on and tag in cur:
                cur.remove(tag)
            if not cur:
                self.by_rec.pop(rel, None)

    def rename(self, old: str, new: str) -> None:
        new = " ".join(new.split())
        if not new or old not in self.colors or new == old:
            return
        if new in self.colors:                       # merging into an existing tag
            self.colors.pop(old)
        else:
            self.colors = {(new if k == old else k): v for k, v in self.colors.items()}
        for rel, ts in self.by_rec.items():
            self.by_rec[rel] = list(dict.fromkeys(new if t == old else t for t in ts))

    def recolor(self, tag: str, color: str) -> None:
        if tag in self.colors:
            self.colors[tag] = color

    def delete(self, tag: str) -> None:
        self.colors.pop(tag, None)
        for rel in list(self.by_rec):
            self.by_rec[rel] = [t for t in self.by_rec[rel] if t != tag]
            if not self.by_rec[rel]:
                del self.by_rec[rel]

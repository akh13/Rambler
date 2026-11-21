# SPDX-License-Identifier: GPL-3.0-or-later
"""Full-text search over every transcript, kept in a small SQLite FTS5 index in ~/.cache/rambler.

The index is incremental: a transcript is re-read only when its file changes, so after the first run
updating it is quick even for thousands of recordings.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .transcript import Transcript, read_summary, summary_path

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rambler"
DB_PATH = CACHE_DIR / "search.db"
SCHEMA_VERSION = 1


@dataclass
class Hit:
    rel: str
    start: float
    end: float
    speaker: str
    snippet: str          # HTML, matches wrapped in <b>


class SearchIndex:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        ver = self.db.execute("PRAGMA user_version").fetchone()[0]
        if ver != SCHEMA_VERSION:
            self.db.executescript("""
                DROP TABLE IF EXISTS docs; DROP TABLE IF EXISTS segs;
                CREATE TABLE docs(rel TEXT PRIMARY KEY, sig TEXT, summary TEXT, speakers TEXT);
                CREATE VIRTUAL TABLE segs USING fts5(text, speaker, rel UNINDEXED, start UNINDEXED,
                                                     "end" UNINDEXED, tokenize='unicode61 remove_diacritics 2');
            """)
            self.db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self.db.commit()

    # ------------------------------------------------------------------ indexing
    @staticmethod
    def _sig(rec) -> str:
        parts = []
        for p in (rec.meta, rec.srt, rec.txt, summary_path(rec)):
            try:
                st = p.stat() if p else None
            except OSError:
                st = None
            parts.append(f"{st.st_size}:{int(st.st_mtime)}" if st else "-")
        return "|".join(parts)

    def update(self, recs: Iterable, progress: Callable[[int, int], None] | None = None,
               cancel: threading.Event | None = None) -> int:
        """Bring the index in line with these recordings. Returns how many were (re)indexed."""
        recs = [r for r in recs if r.txt or r.srt or r.meta]
        with self._lock:
            have = dict(self.db.execute("SELECT rel, sig FROM docs"))
        wanted = {r.rel.as_posix(): r for r in recs}
        stale = [rel for rel in have if rel not in wanted]
        todo = [(rel, r, sig) for rel, r in wanted.items() if have.get(rel) != (sig := self._sig(r))]
        for i, (rel, r, sig) in enumerate(todo):
            if cancel is not None and cancel.is_set():
                break
            self._index_one(rel, r, sig, commit=(i % 50 == 0))
            if progress:
                progress(i + 1, len(todo))
        with self._lock:
            for rel in stale:
                self.db.execute("DELETE FROM segs WHERE rel = ?", (rel,))
                self.db.execute("DELETE FROM docs WHERE rel = ?", (rel,))
            self.db.commit()
        return len(todo)

    def _index_one(self, rel: str, r, sig: str, commit: bool = True) -> None:
        t = Transcript.for_recording(r)
        rows = [(seg.text, seg.speaker, rel, seg.start, seg.end) for seg in t.segments if seg.text.strip()]
        if not rows and t.plain_text.strip():
            rows = [(t.plain_text, "", rel, 0.0, 0.0)]
        with self._lock:
            self.db.execute("DELETE FROM segs WHERE rel = ?", (rel,))
            self.db.executemany('INSERT INTO segs(text, speaker, rel, start, "end") VALUES (?,?,?,?,?)', rows)
            self.db.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?)",
                            (rel, sig, read_summary(r), ", ".join(t.speakers())))
            if commit:
                self.db.commit()

    def reindex(self, recs) -> None:
        """Re-index just these recordings (after an edit), leaving everything else alone."""
        for r in recs:
            self._index_one(r.rel.as_posix(), r, self._sig(r))

    # ------------------------------------------------------------------ querying
    @staticmethod
    def to_fts(query: str) -> str:
        """User text -> FTS5 query: quoted phrases kept, other words ANDed, last word prefix-matched."""
        query = query.strip()
        if not query:
            return ""
        parts = re.findall(r'"([^"]+)"|(\S+)', query)
        terms = []
        for i, (phrase, word) in enumerate(parts):
            if phrase:
                terms.append('"' + phrase.replace('"', "") + '"')
            else:
                w = re.sub(r'["*^():]', "", word)
                if not w:
                    continue
                last = i == len(parts) - 1
                terms.append(f'"{w}"*' if last else f'"{w}"')
        return " ".join(terms)

    def search(self, query: str, speaker: str = "", limit: int = 500) -> list[Hit]:
        fts = self.to_fts(query)
        if not fts and not speaker:
            return []
        clauses, args = [], []
        if fts:
            clauses.append("segs MATCH ?")
            args.append(f"text : ({fts})")
        if speaker:
            clauses.append("speaker = ?")
            args.append(speaker)
        sql = ('SELECT rel, start, "end", speaker, '
               + ("snippet(segs, 0, '<b>', '</b>', '…', 18)" if fts else "substr(text, 1, 160)")
               + " FROM segs WHERE " + " AND ".join(clauses)
               + (" ORDER BY rank" if fts else " ORDER BY rel DESC, start") + " LIMIT ?")
        args.append(limit)
        with self._lock:
            try:
                rows = self.db.execute(sql, args).fetchall()
            except sqlite3.OperationalError:
                return []
        return [Hit(r[0], float(r[1]), float(r[2]), r[3] or "", r[4]) for r in rows]

    def doc_info(self) -> dict[str, tuple[str, str]]:
        """rel -> (summary, speakers) for every indexed recording."""
        with self._lock:
            return {rel: (summ or "", spk or "") for rel, summ, spk in
                    self.db.execute("SELECT rel, summary, speakers FROM docs")}

    def all_speakers(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self.db.execute(
                "SELECT DISTINCT speaker FROM segs WHERE speaker != '' ORDER BY speaker")]

    def close(self):
        with self._lock:
            self.db.close()

# SPDX-License-Identifier: GPL-3.0-or-later
"""A recording's transcript as editable segments (with word timings, confidence and speakers).

Loads from the whisper JSON when there is one, otherwise from the SRT, otherwise from the plain text.
Saving rewrites the .json, .srt and .txt together so they never disagree.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import srt as srtmod

LOW_WORD_P = 0.5          # word probability below this is "Whisper wasn't sure"
LOW_SEG_LOGPROB = -0.6    # segment average log-probability below this is a shaky segment
LOW_SEG_FRACTION = 0.10   # ...or when more than this share of its words are unsure
# (on ~13k real journal segments these flag about 3-4% of lines; "any unsure word" flagged 78%)


@dataclass
class Word:
    w: str
    s: float
    e: float
    p: float = 1.0

    @property
    def uncertain(self) -> bool:
        return self.p < LOW_WORD_P


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    logprob: float | None = None
    speaker: str = ""
    edited: bool = False
    extra: dict = field(default_factory=dict)      # any other keys in the JSON, preserved on save

    @property
    def uncertain_words(self) -> list[Word]:
        return [w for w in self.words if w.uncertain]

    @property
    def shaky(self) -> bool:
        if self.logprob is not None and self.logprob < LOW_SEG_LOGPROB:
            return True
        return bool(self.words) and len(self.uncertain_words) / len(self.words) > LOW_SEG_FRACTION

    def word_at(self, t: float) -> int:
        for i, w in enumerate(self.words):
            if w.s <= t < w.e:
                return i
        return -1


_SEG_KEYS = {"start", "end", "text", "words", "logprob", "speaker", "edited"}


class Transcript:
    def __init__(self, segments: list[Segment], json_path: Path | None = None, srt_path: Path | None = None,
                 txt_path: Path | None = None, meta: dict | None = None, source: str = "none"):
        self.segments = segments
        self.json_path, self.srt_path, self.txt_path = json_path, srt_path, txt_path
        self.meta = meta or {}
        self.source = source            # "json" | "srt" | "txt" | "none"
        self.plain_text = ""            # only for txt-only transcripts

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, json_path: Path | None, srt_path: Path | None, txt_path: Path | None) -> "Transcript":
        if json_path and json_path.exists():
            try:
                meta = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
                segs = [_seg_from_json(x) for x in meta.get("segments") or []]
                if segs or not (srt_path and srt_path.exists()):
                    return cls(segs, json_path, srt_path, txt_path, meta, "json")
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        if srt_path and srt_path.exists():
            try:
                cues = srtmod.load(srt_path)
                segs = []
                for c in cues:
                    m = re.match(r"^\[([^\]]{1,40})\]\s*(.*)$", c.text, re.S)   # "[Speaker] text"
                    segs.append(Segment(c.start, c.end, m.group(2), speaker=m.group(1)) if m
                                else Segment(c.start, c.end, c.text))
                return cls(segs, json_path, srt_path, txt_path, {}, "srt")
            except OSError:
                pass
        t = cls([], json_path, srt_path, txt_path, {}, "none")
        if txt_path and txt_path.exists():
            t.source = "txt"
            t.plain_text = txt_path.read_text(encoding="utf-8", errors="replace")
        return t

    @classmethod
    def for_recording(cls, rec) -> "Transcript":
        return cls.load(rec.meta, rec.srt if rec.srt and rec.srt.suffix.lower() == ".srt" else None, rec.txt)

    @property
    def timed(self) -> bool:
        return bool(self.segments)

    @property
    def editable(self) -> bool:
        return self.source in ("json", "srt")

    def index_at(self, t: float) -> int:
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.segments[mid]
            if t < s.start:
                hi = mid - 1
            elif t >= s.end:
                lo = mid + 1
            else:
                return mid
        return -1

    def speakers(self) -> list[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})

    def text(self) -> str:
        if not self.segments:
            return self.plain_text
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    # ------------------------------------------------------------------ editing
    def edit(self, i: int, new_text: str) -> list[str]:
        """Change one segment's text. Returns words that are new to the segment (hotword candidates)."""
        seg = self.segments[i]
        old_tokens = seg.text.split()
        new_text = " ".join(new_text.split())
        new_tokens = new_text.split()
        if seg.words and len(new_tokens) == len(seg.words):
            # same word count: keep each word's timing, mark changed words as checked by a person
            for w, tok in zip(seg.words, new_tokens):
                if _bare(w.w) != _bare(tok):
                    w.w, w.p = " " + tok, 1.0
        elif seg.words:
            # re-split: spread the new words evenly across the segment so highlighting still roughly works
            span = (seg.end - seg.start) / max(1, len(new_tokens))
            seg.words = [Word(" " + tok, round(seg.start + k * span, 2), round(seg.start + (k + 1) * span, 2), 1.0)
                         for k, tok in enumerate(new_tokens)]
        seg.text = new_text
        seg.edited = True
        old = {_bare(t) for t in old_tokens}
        return [_bare(t, keep_case=True) for t in new_tokens if _bare(t) and _bare(t) not in old]

    def set_speaker(self, indices: list[int], name: str) -> None:
        for i in indices:
            self.segments[i].speaker = name.strip()

    # ------------------------------------------------------------------ saving
    def save(self) -> list[Path]:
        """Rewrite .json (if it exists or speakers/edits need somewhere to live), .srt and .txt."""
        written = []
        base = self._base()
        if base is None:
            raise RuntimeError("Nowhere to save this transcript.")
        jpath = self.json_path or base.with_name(base.name + ".json")
        meta = dict(self.meta)
        meta["segments"] = [_seg_to_json(s) for s in self.segments]
        meta["text"] = self.text()
        speakers = self.speakers()
        if speakers:
            meta["speakers"] = speakers
        else:
            meta.pop("speakers", None)
        _atomic(jpath, json.dumps(meta, indent=1, ensure_ascii=False))
        written.append(jpath)
        self.json_path, self.meta = jpath, meta
        spath = self.srt_path or base.with_name(base.name + ".srt")
        _atomic(spath, srtmod.dumps([srtmod.Cue(s.start, s.end, _label(s)) for s in self.segments]))
        written.append(spath)
        self.srt_path = spath
        tpath = self.txt_path or base.with_name(base.name + ".txt")
        _atomic(tpath, self.text() + "\n")
        written.append(tpath)
        self.txt_path = tpath
        return written

    def _base(self) -> Path | None:
        for p in (self.json_path, self.srt_path, self.txt_path):
            if p:
                return p.with_suffix("")
        return None


def summary_path(rec) -> Path | None:
    """<stem>.summary next to the recording's transcript, if any tool has written one."""
    for p in (rec.txt, rec.meta, rec.srt):
        if p:
            cand = p.with_suffix(".summary")
            if cand.exists():
                return cand
    return None


def read_summary(rec) -> str:
    p = summary_path(rec)
    if not p:
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


# ------------------------------------------------------------------ helpers

def _label(s: Segment) -> str:
    return f"[{s.speaker}] {s.text}" if s.speaker else s.text


def _bare(tok: str, keep_case: bool = False) -> str:
    t = re.sub(r"^[^\w]+|[^\w]+$", "", tok.strip())
    return t if keep_case else t.lower()


def _seg_from_json(x: dict) -> Segment:
    words = [Word(w.get("w", w.get("word", "")), float(w.get("s", w.get("start", 0))),
                  float(w.get("e", w.get("end", 0))), float(w.get("p", w.get("probability", 1.0))))
             for w in x.get("words") or []]
    return Segment(float(x.get("start", 0)), float(x.get("end", 0)), str(x.get("text", "")).strip(), words,
                   x.get("logprob"), x.get("speaker") or "", bool(x.get("edited")),
                   {k: v for k, v in x.items() if k not in _SEG_KEYS})


def _seg_to_json(s: Segment) -> dict:
    d = {"start": s.start, "end": s.end, "text": s.text}
    if s.logprob is not None:
        d["logprob"] = s.logprob
    d.update(s.extra)
    if s.words:
        d["words"] = [{"w": w.w, "s": w.s, "e": w.e, "p": w.p} for w in s.words]
    if s.speaker:
        d["speaker"] = s.speaker
    if s.edited:
        d["edited"] = True
    return d


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".rambler-part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

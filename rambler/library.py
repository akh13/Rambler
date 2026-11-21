"""Scan audio + transcript folders and work out what still needs transcribing."""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .config import AUDIO_EXTS, Config

AUDIO_EXT_SET = {e.lower() for e in AUDIO_EXTS}

SKIP_DIRS = {".rambler-replaced", ".Trash-1000", ".Trashes", "lost+found", "$RECYCLE.BIN",
             "System Volume Information", ".git", "__pycache__"}


def walk_files(root: Path):
    """Yield (relative posix path, os.stat_result) for non-hidden files under root."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith(".rambler-part"):
                continue
            full = os.path.join(dirpath, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            rel = PurePosixPath(Path(full).relative_to(root).as_posix())
            yield rel, st


TRACK_RE = re.compile(r"^(?P<take>.+)_(?P<part>MIX|\d+(?:-\d+)?)$", re.I)
MIN_SECONDS = 2.0   # shorter clips are accidental presses; whisper only hallucinates on them


def take_stem(rel: PurePosixPath) -> str:
    """Name a recording's outputs are filed under.

    TP-7 style exports are a folder named like a file (2024-01-01_120000_000.wav/track_0.wav); those are
    filed under the folder's stem, so transcripts get a sensible name.
    """
    if rel.name.lower().startswith("track_") and Path(rel.parent.name).suffix.lower() in AUDIO_EXT_SET:
        return Path(rel.parent.name).stem
    return rel.stem


def _norm(name: str) -> str:
    return name.lower().replace(" ", "_")


def audio_seconds(path: Path, size: int) -> float | None:
    """Cheap duration from a WAV header (or an MP3 size estimate); None if unknown."""
    if path.suffix.lower() == ".mp3":
        return size / (128_000 / 8)
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"RIFF":
                return None
            f.read(8)
            byte_rate = data = None
            while True:
                h = f.read(8)
                if len(h) < 8:
                    break
                cid, sz = struct.unpack("<4sI", h)
                if cid == b"fmt ":
                    byte_rate = struct.unpack("<I", f.read(sz)[8:12])[0]
                elif cid == b"data":
                    data = sz
                    break
                else:
                    f.seek(sz + (sz & 1), 1)
            return data / byte_rate if (data and byte_rate) else None
    except (OSError, struct.error):
        return None


@dataclass
class Recording:
    rel: PurePosixPath            # e.g. 2026/261114_0001.wav
    size: int
    mtime: float
    locations: list[Path] = field(default_factory=list)   # audio roots holding a copy
    txt: Path | None = None
    srt: Path | None = None
    meta: Path | None = None
    mix: "Recording | None" = None   # set on multitrack track files: the take's _MIX file
    too_short: bool = False

    @property
    def name(self) -> str:
        return self.rel.name

    @property
    def stem(self) -> str:
        return take_stem(self.rel)

    @property
    def source(self) -> Path:
        """Best local copy to read/play from."""
        for loc in self.locations:
            if (loc / self.rel).exists():
                return loc / self.rel
        return self.locations[0] / self.rel

    @property
    def transcribed(self) -> bool:
        return self.txt is not None

    @property
    def pending(self) -> bool:
        """Needs transcribing: no transcript, and not a multitrack track or an accidental blip."""
        return self.txt is None and self.mix is None and not self.too_short

    def status(self) -> str:
        if self.txt:
            return "Transcribed" if self.srt else "Transcribed (no SRT)"
        if self.mix is not None:
            return "Track (MIX transcribed)" if self.mix.txt else "Track (MIX pending)"
        if self.too_short:
            return "Too short"
        return "To be transcribed"


def transcript_base(cfg: Config, rel: PurePosixPath) -> PurePosixPath:
    """Relative path (without extension) for a recording's outputs inside a transcript root."""
    stem = take_stem(rel)
    if cfg.transcript_layout == "flat":
        return PurePosixPath(stem)
    parent = rel.parent.parent if stem != rel.stem else rel.parent
    return parent / stem


class TranscriptIndex:
    """Finds a recording's transcript: mirrored path first, then its name anywhere in the transcript tree
    (flat layouts), but only when that name is unambiguous. Matching ignores case and treats spaces as
    underscores, so "221214 0194.txt" counts for 221214_0194.wav."""

    def __init__(self, roots: list[Path]):
        self.by_rel: dict[str, Path] = {}
        self.by_name: dict[str, dict[str, Path]] = {}
        for root in roots:
            for rel, _ in walk_files(root):
                p = root / rel
                self.by_rel.setdefault(_norm(rel.as_posix()), p)
                self.by_name.setdefault(_norm(rel.name), {}).setdefault(_norm(rel.as_posix()), p)

    def find(self, rec_rel: PurePosixPath, ext: str) -> Path | None:
        stem = take_stem(rec_rel)
        parent = rec_rel.parent.parent if stem != rec_rel.stem else rec_rel.parent
        for c in (parent / (stem + ext), rec_rel.parent / (rec_rel.name + ext)):
            hit = self.by_rel.get(_norm(c.as_posix()))
            if hit:
                return hit
        for name in (stem + ext, rec_rel.name + ext):
            hits = self.by_name.get(_norm(name), {})
            if len(hits) == 1:
                return next(iter(hits.values()))
        return None


def scan(cfg: Config, audio_exts: list[str] | None = None) -> list[Recording]:
    exts = {e.lower() for e in (audio_exts or AUDIO_EXTS)}
    recs: dict[str, Recording] = {}
    for root in cfg.audio_roots():
        for rel, st in walk_files(root):
            if rel.suffix.lower() not in exts:
                continue
            key = rel.as_posix()
            r = recs.get(key)
            if r is None:
                recs[key] = Recording(rel, st.st_size, st.st_mtime, [root])
            else:
                r.locations.append(root)
                if st.st_mtime > r.mtime:
                    r.size, r.mtime = st.st_size, st.st_mtime

    tindex = TranscriptIndex(cfg.transcript_roots())
    for r in recs.values():
        r.txt = tindex.find(r.rel, ".txt")
        r.srt = tindex.find(r.rel, ".srt") or tindex.find(r.rel, ".vtt")
        r.meta = tindex.find(r.rel, ".json") or tindex.find(r.rel, ".rambler.json")
        if r.srt is None:
            for loc in r.locations:
                for ext in (".srt", ".vtt"):
                    beside = (loc / r.rel).with_suffix(ext)
                    if beside.exists():
                        r.srt = beside
                        break
                if r.srt:
                    break
        if r.txt is None and r.size < 2_000_000:
            secs = audio_seconds(r.source, r.size)
            r.too_short = secs is not None and secs < MIN_SECONDS

    # Multitrack takes (TASCAM X8 etc.): <take>_MIX plus <take>_1-2, <take>_5 ... Only the MIX is transcribed.
    mixes = {}
    for r in recs.values():
        m = TRACK_RE.match(r.rel.stem)
        if m and m.group("part").upper() == "MIX":
            mixes[(r.rel.parent, m.group("take").lower())] = r
    for r in recs.values():
        m = TRACK_RE.match(r.rel.stem)
        if m and m.group("part").upper() != "MIX":
            mix = mixes.get((r.rel.parent, m.group("take").lower()))
            if mix is not None:
                r.mix = mix
    return sorted(recs.values(), key=lambda r: r.rel.as_posix().lower())

"""Transcribe recordings and write outputs; import dropped files. GUI-free so the CLI can share it."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Callable

from . import __version__
from .config import AUDIO_EXTS, Config
from .engines import Engine, Result
from .library import Recording, transcript_base
from .syncer import copy_file


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".rambler-part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_outputs(cfg: Config, rec: Recording, res: Result) -> list[Path]:
    """Write .txt/.srt/.rambler.json into the primary transcript dir (mirroring the audio subfolders)."""
    roots = cfg.transcript_roots()
    if not roots:
        raise RuntimeError("No transcript folder is connected.")
    root = roots[0]
    base = root / transcript_base(cfg, rec.rel)
    written = []
    txt = base.with_name(base.name + ".txt")
    _atomic_write(txt, res.txt)
    written.append(txt)
    if res.srt is not None:   # empty when there was no speech; still written so it isn't redone
        srt = base.with_name(base.name + ".srt")
        _atomic_write(srt, res.srt)
        written.append(srt)
        if cfg.srt_beside_audio:
            for loc in rec.locations:
                beside = (loc / rec.rel).with_suffix(".srt")
                if loc.is_dir():
                    _atomic_write(beside, res.srt)
    # Common whisper JSON shape (audio, duration_s, model, segments, text),
    # so existing tools (summarisers, indexers) keep working; Rambler's own fields are added alongside.
    meta = {
        "audio": rec.rel.as_posix(),
        "duration_s": round(res.duration, 1) if res.duration else None,
        "model": res.model,
        **({"batch": res.extra["batch"]} if res.extra.get("batch") else {}),
        "duplicates": [],
        "segments": res.segments,
        "text": " ".join(x.get("text", "") for x in res.segments).strip() if res.segments else res.txt.strip(),
        "language": res.language,
        "engine": res.engine,
        "source_size": rec.size,
        "source_mtime": rec.mtime,
        "transcribed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rambler_version": __version__,
    }
    mpath = base.with_name(base.name + ".json")
    _atomic_write(mpath, json.dumps(meta, indent=1))
    written.append(mpath)
    rec.txt = txt
    rec.srt = written[1] if res.srt is not None else rec.srt
    rec.meta = mpath
    return written


def transcribe_one(cfg: Config, engine: Engine, rec: Recording, want_srt: bool,
                   progress: Callable[[float, str], None], cancel: threading.Event) -> list[Path]:
    res = engine.transcribe(rec.source, want_srt, progress, cancel)
    return write_outputs(cfg, rec, res)


def unique_dest(dst: Path) -> Path:
    if not dst.exists():
        return dst
    i = 2
    while True:
        cand = dst.with_name(f"{dst.stem} ({i}){dst.suffix}")
        if not cand.exists():
            return cand
        i += 1


def expand_drops(paths: list[Path]) -> list[tuple[Path, PurePosixPath]]:
    """Dropped files/folders -> (source file, relative path to place it at). Folders keep their name."""
    exts = {e.lower() for e in AUDIO_EXTS}
    out = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in exts and not f.name.startswith("."):
                    out.append((f, PurePosixPath(p.name) / f.relative_to(p).as_posix()))
        elif p.is_file() and p.suffix.lower() in exts:
            out.append((p, PurePosixPath(p.name)))
    return out


def import_files(cfg: Config, items: list[tuple[Path, PurePosixPath]],
                 cancel: threading.Event | None = None,
                 progress: Callable[[int, int, str], None] | None = None) -> list[PurePosixPath]:
    """Copy dropped audio into the primary audio dir. Identical files already present are skipped.
    Returns the relative paths (within the audio roots) of the imported recordings."""
    roots = cfg.audio_roots()
    if not roots:
        raise RuntimeError("No audio folder is connected.")
    root = roots[0]
    sub = PurePosixPath(cfg.incoming_subdir.strip("/")) if cfg.incoming_subdir.strip("/") else PurePosixPath()
    out = []
    for i, (src, rel) in enumerate(items):
        if cancel is not None and cancel.is_set():
            break
        dst = root / sub / rel
        if progress:
            progress(i, len(items), src.name)
        try:
            if dst.exists():
                if dst.stat().st_size == src.stat().st_size:
                    out.append((sub / rel))
                    continue
                dst = unique_dest(dst)
            if src.resolve() != dst.resolve():
                copy_file(src, dst, cancel)
        except InterruptedError:
            break
        out.append(PurePosixPath(dst.relative_to(root).as_posix()))
    if progress:
        progress(len(items), len(items), "Imported")
    return out


def run_post_hook(cfg: Config, log: Callable[[str], None], cancel: threading.Event | None = None) -> int:
    """Run the user's post-transcribe command (e.g. a summariser). Output goes to the log."""
    import subprocess
    cmd = cfg.post_transcribe_cmd.strip()
    if not cmd:
        return 0
    cwd = cfg.transcript_roots()[0].parent if cfg.transcript_roots() else Path.home()
    log(f"$ {cmd}   (in {cwd})")
    proc = subprocess.Popen(["bash", "-lc", cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    for line in proc.stdout:  # type: ignore[union-attr]
        log(line.rstrip())
        if cancel is not None and cancel.is_set():
            proc.terminate()
            break
    rc = proc.wait()
    if rc:
        log(f"post-transcribe command exited with {rc}")
    return rc

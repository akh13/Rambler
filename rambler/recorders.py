"""Field recorders (e.g. a TASCAM Portacapture X8 in SD CARD READER mode) as import sources.

A recorder is matched by filesystem UUID or volume label when it mounts. New recordings are copied
(never moved, never modified) into the primary recording folder, remembered in a ledger so they aren't
imported twice, and the card can be unmounted afterwards so it's safe to switch the recorder back.
"""
from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from .config import AUDIO_EXTS, DATA_DIR, Config
from .library import walk_files
from .pipeline import unique_dest
from .syncer import copy_file

LEDGER = DATA_DIR / "imported.json"
LOCK = DATA_DIR / "pipeline.lock"


@dataclass
class Recorder:
    name: str = "TASCAM X8"
    label: str = "X8"               # volume label the recorder gives its card
    uuid: str = ""                  # filesystem UUID; matched first if set (changes when the card is formatted)
    card_subdir: str = "SOUND"      # only look inside this folder on the card ("" = whole card)
    # Where imports land inside the primary recording folder. {year} comes from the file name
    # (261121_0002 -> 2026, 2026-11-21_... -> 2026) or, failing that, the file's date. {recorder} = name.
    dest_subdir: str = "{year}"
    keep_card_folders: bool = False  # recreate the card's subfolders under dest_subdir
    exclude: list[str] = field(default_factory=lambda: ["UTILITY/*", "SOUNDPAD/*"])
    auto_import: bool = True
    unmount_after: bool = True
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "Recorder":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Mount:
    target: Path
    source: str
    uuid: str
    label: str
    fstype: str


def list_mounts() -> list[Mount]:
    """Removable-looking mounts (under /media or /run/media)."""
    try:
        out = subprocess.run(["findmnt", "-J", "-l", "-o", "TARGET,SOURCE,UUID,LABEL,FSTYPE"],
                             capture_output=True, text=True, timeout=10).stdout
        rows = json.loads(out).get("filesystems", [])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    mounts = []
    for r in rows:
        t = r.get("target") or ""
        if t.startswith("/media/") or t.startswith("/run/media/"):
            mounts.append(Mount(Path(t), r.get("source") or "", r.get("uuid") or "", r.get("label") or "",
                                r.get("fstype") or ""))
    return mounts


def recorders(cfg: Config) -> list[Recorder]:
    return [Recorder.from_dict(d) for d in cfg.recorders]


def match(rec: Recorder, mounts: list[Mount]) -> Mount | None:
    if not rec.enabled:
        return None
    if rec.uuid:
        for m in mounts:
            if m.uuid and m.uuid.lower() == rec.uuid.lower():
                return m
    if rec.label:
        for m in mounts:
            if m.label == rec.label or (not m.label and m.target.name == rec.label):
                return m
    return None


def connected(cfg: Config, mounts: list[Mount] | None = None) -> list[tuple[Recorder, Mount]]:
    mounts = list_mounts() if mounts is None else mounts
    out = []
    for r in recorders(cfg):
        m = match(r, mounts)
        if m:
            out.append((r, m))
    return out


# ------------------------------------------------------------------ ledger

def _load_ledger() -> dict:
    try:
        return json.loads(LEDGER.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_ledger(d: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    os.replace(tmp, LEDGER)


def _key(rel: PurePosixPath, st: os.stat_result) -> str:
    return f"{rel.as_posix()}|{st.st_size}|{int(st.st_mtime)}"


def year_of(name: str, mtime: float) -> str:
    m = re.match(r"^(\d{2})(\d{2})(\d{2})[_ ]", name)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"20{m.group(1)}"
    m = re.match(r"^((?:19|20)\d{2})-\d{2}-\d{2}", name)
    if m:
        return m.group(1)
    return time.strftime("%Y", time.localtime(mtime))


def dest_rel(rec: Recorder, rel: PurePosixPath, st: os.stat_result) -> PurePosixPath:
    sub = rec.dest_subdir.replace("{year}", year_of(rel.name, st.st_mtime)).replace("{recorder}", rec.name)
    base = PurePosixPath(sub.strip("/")) if sub.strip("/") else PurePosixPath()
    return base / (rel if rec.keep_card_folders else rel.name)


def library_index(cfg: Config) -> set[tuple[str, int]]:
    """(lowercased file name, size) for every recording already in any connected recording folder."""
    out = set()
    exts = {e.lower() for e in AUDIO_EXTS}
    for root in cfg.audio_roots():
        for rel, st in walk_files(root):
            if rel.suffix.lower() in exts:
                out.add((rel.name.lower(), st.st_size))
    return out


def pending(rec: Recorder, mount: Mount, cfg: Config | None = None) -> list[tuple[Path, PurePosixPath, os.stat_result]]:
    """Recordings on the card that aren't in the ledger and (if cfg given) aren't already in the library
    under the same name and size, wherever they were filed."""
    root = mount.target / rec.card_subdir if rec.card_subdir else mount.target
    if not root.is_dir():
        root = mount.target          # card without the expected folder: look everywhere
    if not root.is_dir():
        return []
    seen = set(_load_ledger().get(rec.name, []))
    have = library_index(cfg) if cfg is not None else set()
    exts = {e.lower() for e in AUDIO_EXTS}
    out = []
    for rel, st in walk_files(root):
        if rel.suffix.lower() not in exts:
            continue
        if any(fnmatch.fnmatch(rel.as_posix(), pat) for pat in rec.exclude):
            continue
        if _key(rel, st) in seen or (rel.name.lower(), st.st_size) in have:
            continue
        out.append((root / rel, rel, st))
    return out


def import_recorder(cfg: Config, rec: Recorder, mount: Mount, log: Callable[[str], None],
                    progress: Callable[[int, int, str], None] | None = None,
                    cancel: threading.Event | None = None) -> list[PurePosixPath]:
    """Copy new recordings into the primary recording folder. Returns their paths relative to the audio roots."""
    roots = cfg.audio_roots()
    if not roots:
        raise RuntimeError("No recording folder is connected to import into.")
    dest_root = roots[0]
    items = pending(rec, mount, cfg)
    total = sum(st.st_size for _, _, st in items)
    free = shutil.disk_usage(dest_root).free
    if total > free:
        raise RuntimeError(f"{dest_root} has {free / 1e9:.1f} GB free; {total / 1e9:.1f} GB needed.")
    ledger = _load_ledger()
    seen = ledger.setdefault(rec.name, [])
    imported: list[PurePosixPath] = []
    done = 0
    for src, rel, st in items:
        if cancel is not None and cancel.is_set():
            break
        target = dest_rel(rec, rel, st)
        dst = dest_root / target
        if dst.exists() and dst.stat().st_size == st.st_size:
            log(f"already have {target}")
        else:
            if dst.exists():
                dst = unique_dest(dst)   # same name, different recording (file counter was reset)
            def on_bytes(n, _rel=rel):
                nonlocal done
                done += n
                if progress:
                    progress(done, total, f"{rec.name}: {_rel}")
            try:
                copy_file(src, dst, cancel, on_bytes)
            except InterruptedError:
                break
            log(f"imported {rel} → {dst.relative_to(dest_root)}")
        imported.append(PurePosixPath(dst.relative_to(dest_root).as_posix()))
        seen.append(_key(rel, st))
        _save_ledger(ledger)           # after every file, so a yanked cable loses nothing
    return imported


def unmount(mount: Mount, log: Callable[[str], None]) -> bool:
    if not shutil.which("udisksctl") or not mount.source.startswith("/dev/"):
        log("udisksctl not available; unmount the card yourself before switching the recorder back.")
        return False
    subprocess.run(["sync"], timeout=120)
    r = subprocess.run(["udisksctl", "unmount", "-b", mount.source], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        log(f"Could not unmount {mount.target}: {r.stderr.strip()}")
        return False
    log(f"Unmounted {mount.target}. Safe to switch the recorder out of card reader mode.")
    return True


# ------------------------------------------------------------------ cross-process lock (GUI vs `rambler watch`)

@contextmanager
def pipeline_lock():
    """Yields True if this process got the lock, False if another Rambler is mid-run."""
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "w")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()

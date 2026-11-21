"""Additive multi-folder sync. Never deletes; replaced files are kept aside unless disabled."""
from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable

from .config import Config
from .library import walk_files

MTIME_SLACK = 2.0          # FAT/exFAT store mtimes at 2 s resolution
CHUNK = 4 * 1024 * 1024


@dataclass
class FileInfo:
    root: Path
    rel: PurePosixPath
    size: int
    mtime: float

    @property
    def path(self) -> Path:
        return self.root / self.rel


@dataclass
class Action:
    group: str                  # "audio" or "transcripts"
    src: FileInfo
    dst_root: Path
    replace: FileInfo | None    # existing file at destination that will be replaced

    @property
    def dst(self) -> Path:
        return self.dst_root / self.src.rel

    @property
    def bytes(self) -> int:
        return self.src.size


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    skipped_roots: list[str] = field(default_factory=list)       # configured but not mounted
    space_problems: list[str] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes for a in self.actions)

    def per_destination(self) -> dict[Path, tuple[int, int]]:
        out: dict[Path, tuple[int, int]] = {}
        for a in self.actions:
            n, b = out.get(a.dst_root, (0, 0))
            out[a.dst_root] = (n + 1, b + a.bytes)
        return out

    def summary(self) -> str:
        if not self.actions:
            return "Everything is already in sync."
        lines = [f"{len(self.actions)} file(s), {human(self.total_bytes)} to copy:"]
        for root, (n, b) in self.per_destination().items():
            lines.append(f"  → {root}: {n} file(s), {human(b)}")
        reps = sum(1 for a in self.actions if a.replace)
        if reps:
            lines.append(f"  {reps} older/smaller copies will be replaced")
        for s in self.skipped_roots:
            lines.append(f"  (not connected, skipped: {s})")
        for s in self.space_problems:
            lines.append(f"  ⚠ {s}")
        return "\n".join(lines)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _quick_hash(p: Path) -> bytes:
    import hashlib
    h = hashlib.blake2b(digest_size=16)
    size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(size - (1 << 20))
            h.update(f.read(1 << 20))
    h.update(str(size).encode())
    return h.digest()


def _same(a: FileInfo, b: FileInfo) -> bool:
    if a.size != b.size:
        return False
    if abs(a.mtime - b.mtime) <= MTIME_SLACK:
        return True
    try:  # same size, different mtime: probably copied without preserving times
        return _quick_hash(a.path) == _quick_hash(b.path)
    except OSError:
        return False


def _winner(copies: list[FileInfo], policy: str) -> FileInfo:
    if policy == "larger":
        return max(copies, key=lambda c: (c.size, c.mtime))
    return max(copies, key=lambda c: (round(c.mtime / MTIME_SLACK), c.size))


def plan_group(group: str, roots: list[Path], exts: set[str] | None, policy: str) -> list[Action]:
    if len(roots) < 2:
        return []
    inventory: dict[str, list[FileInfo]] = {}
    for root in roots:
        for rel, st in walk_files(root):
            if exts is not None and rel.suffix.lower() not in exts and not rel.name.lower().endswith(".rambler.json"):
                continue
            inventory.setdefault(rel.as_posix(), []).append(FileInfo(root, rel, st.st_size, st.st_mtime))
    actions = []
    for rel, copies in sorted(inventory.items()):
        best = _winner(copies, policy)
        have = {c.root: c for c in copies}
        for root in roots:
            cur = have.get(root)
            if cur is None:
                actions.append(Action(group, best, root, None))
            elif cur is not best and not _same(cur, best):
                if (policy == "larger" and cur.size < best.size) or \
                   (policy != "larger" and (best.mtime - cur.mtime > MTIME_SLACK or
                                            (abs(best.mtime - cur.mtime) <= MTIME_SLACK and best.size > cur.size))):
                    actions.append(Action(group, best, root, cur))
    return actions


def build_plan(cfg: Config) -> Plan:
    plan = Plan()
    if cfg.sync_audio:
        roots = cfg.audio_roots()
        plan.skipped_roots += [d for d in cfg.audio_dirs if Config.expand(d) not in roots]
        plan.actions += plan_group("audio", roots, {e.lower() for e in cfg.audio_sync_exts}, cfg.conflict_policy)
    if cfg.sync_transcripts:
        roots = cfg.transcript_roots()
        plan.skipped_roots += [d for d in cfg.transcript_dirs if Config.expand(d) not in roots]
        plan.actions += plan_group("transcripts", roots, None, cfg.conflict_policy)
    for root, (_, need) in plan.per_destination().items():
        try:
            free = shutil.disk_usage(root).free
        except OSError:
            continue
        if need > free:
            plan.space_problems.append(f"{root} needs {human(need)} but has {human(free)} free")
    return plan


ProgressFn = Callable[[int, int, str], None]   # (bytes_done, bytes_total, current file)


def copy_file(src: Path, dst: Path, cancel: threading.Event | None = None,
              on_bytes: Callable[[int], None] | None = None) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_name(dst.name + ".rambler-part")
    try:
        with open(src, "rb") as fi, open(part, "wb") as fo:
            while True:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError()
                buf = fi.read(CHUNK)
                if not buf:
                    break
                fo.write(buf)
                if on_bytes:
                    on_bytes(len(buf))
            fo.flush()
            os.fsync(fo.fileno())
        shutil.copystat(src, part)
        os.replace(part, dst)
    except BaseException:
        try:
            part.unlink()
        except OSError:
            pass
        raise


def execute(plan: Plan, cfg: Config, progress: ProgressFn | None = None,
            cancel: threading.Event | None = None) -> list[str]:
    """Run a plan. Returns a list of error strings (empty on full success)."""
    errors: list[str] = []
    total = plan.total_bytes
    done = 0
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for a in plan.actions:
        if cancel is not None and cancel.is_set():
            break
        if not a.dst_root.is_dir():
            errors.append(f"{a.dst_root} disappeared (drive unplugged?)")
            continue
        label = f"{a.src.rel} → {a.dst_root}"
        base = done

        def on_bytes(n, _label=label):
            nonlocal done
            done += n
            if progress:
                progress(done, total, _label)
        try:
            if a.replace and cfg.keep_replaced and a.dst.exists():
                keep = a.dst_root / ".rambler-replaced" / stamp / a.src.rel
                keep.parent.mkdir(parents=True, exist_ok=True)
                os.replace(a.dst, keep)
            copy_file(a.src.path, a.dst, cancel, on_bytes)
        except InterruptedError:
            break
        except OSError as e:
            errors.append(f"{label}: {e}")
            done = base + a.bytes
    if progress:
        progress(total, total, "Sync finished")
    return errors

# SPDX-License-Identifier: GPL-3.0-or-later
"""Waveform overview of a recording: one 0-255 peak per bucket, cached in ~/.cache/rambler/peaks."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import threading
from array import array
from pathlib import Path

CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rambler" / "peaks"
RATE = 4000            # decode rate; plenty for an overview
PER_SECOND = 25        # buckets per second of audio


def _key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()


def cached(path: Path) -> bytes | None:
    try:
        return (CACHE / (_key(path) + ".bin")).read_bytes()
    except OSError:
        return None


def compute(path: Path, cancel: threading.Event | None = None) -> bytes:
    """Decode with ffmpeg to 4 kHz mono and keep the loudest sample per 1/25 s, normalised to 0-255."""
    hit = cached(path)
    if hit is not None:
        return hit
    exe = shutil.which("ffmpeg")
    if not exe:
        return b""
    proc = subprocess.Popen([exe, "-nostdin", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(RATE),
                             "-f", "s16le", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    per = RATE // PER_SECOND
    peaks = array("H")
    carry = b""
    try:
        while True:
            if cancel is not None and cancel.is_set():
                proc.kill()
                return b""
            chunk = proc.stdout.read(per * 2 * 1000)  # type: ignore[union-attr]
            if not chunk:
                break
            chunk = carry + chunk
            usable = len(chunk) - len(chunk) % (per * 2)
            carry = chunk[usable:]
            a = array("h")
            a.frombytes(chunk[:usable])
            for i in range(0, len(a), per):
                block = a[i:i + per]
                peaks.append(max(max(block), -min(block), 0))
    finally:
        proc.wait()
    if not peaks:
        return b""
    top = max(peaks) or 1
    out = bytes(min(255, v * 255 // top) for v in peaks)
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / (_key(path) + ".bin")).write_bytes(out)
    except OSError:
        pass
    return out

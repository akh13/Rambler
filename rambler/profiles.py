# SPDX-License-Identifier: GPL-3.0-or-later
"""What popular field recorders put on their cards, so Rambler can recognise a card and know where the
recordings are. Matching is by what's on the card (marker files and folders), then the volume label, then
the brand the recorder stamps into its WAV headers, so formatting a card or renaming it doesn't matter.

Sources: the manufacturers' reference manuals (folder and file structure sections).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class Profile:
    id: str
    name: str                                  # shown to the user
    markers: tuple[str, ...] = ()              # any of these existing at the card root identifies it (case-insensitive)
    labels: tuple[str, ...] = ()               # volume label regexes
    brands: tuple[bytes, ...] = ()             # strings recorders write into WAV headers (bext/INFO)
    folders: tuple[str, ...] = ()              # where recordings live, in order of preference ("" = whole card)
    exclude: tuple[str, ...] = ()              # globs (relative to the card) never imported
    note: str = ""


PROFILES: tuple[Profile, ...] = (
    Profile("tascam", "TASCAM recorder",
            markers=("dr-1.sys",), labels=(r"^X[468]$", r"^DR-?\d+", r"^TASCAM"),
            brands=(b"TASCAM", b"TEAC"),
            folders=("SOUND", "MUSIC"), exclude=("UTILITY/*", "SOUNDPAD/*"),
            note="Portacapture X4/X6/X8, DR-05X/07X/40X/08/10 and friends: recordings in SOUND or MUSIC."),
    Profile("zoom", "Zoom recorder",
            markers=("STEREO/FOLDER01", "MULTI/FOLDER01", "4CH/FOLDER01", "FOLDER01/ZOOM0001"),
            labels=(r"^H\d[A-Z]*_SD$", r"^ZOOM", r"^F\d+", r"^H\d"),
            brands=(b"ZOOM",),
            folders=("",), exclude=("TRASH/*", "Trash/*", "FALSE TAKE/*"),
            note="H1n/H2n/H4n/H5/H6/H8, the essential series and F-series: everything on the card, "
                 "with _BU backups and tracks grouped under their stereo mix."),
    Profile("sony", "Sony recorder",
            markers=("PRIVATE/SONY", "REC_FILE"), labels=(r"^IC RECORDER", r"^PCM", r"^MEMORY CARD"),
            brands=(b"SONY", b"Sony"),
            folders=("PRIVATE/SONY/REC_FILE", "REC_FILE"),
            note="ICD-UX/TX and PCM-D/A series: recordings in REC_FILE."),
    Profile("olympus", "OM System / Olympus recorder",
            markers=("RECORDER",), labels=(r"^LS-?P", r"^OLYMPUS", r"^OM"),
            brands=(b"OLYMPUS", b"OM Digital", b"OM SYSTEM"),
            folders=("RECORDER",), exclude=("Music/*", "MUSIC/*", "Podcast/*", "Audible/*"),
            note="LS-P and DM series: recordings in the Recorder folder (Folder A-E and friends)."),
)

BY_ID = {p.id: p for p in PROFILES}

# Labels that can only be a recorder's card, so Rambler may mount them before anyone opens them
# (needed on desktops like COSMIC that only mount drives when you open them).
RECORDER_LABELS = (r"^X[468]$", r"^H\d[A-Z]*_SD$", r"^DR-?\d{2}[A-Z]*$", r"^LS-?P\d$")


def looks_like_recorder_label(label: str) -> bool:
    return any(re.match(rx, label or "", re.I) for rx in RECORDER_LABELS)
AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".wma", ".ogg"}


def ci_path(root: Path, rel: str) -> Path | None:
    """Resolve rel under root ignoring case (FAT/exFAT cards are case-insensitive, Linux isn't always)."""
    cur = root
    for part in [p for p in rel.split("/") if p]:
        nxt = cur / part
        if not nxt.exists():
            try:
                hit = next((e for e in os.listdir(cur) if e.lower() == part.lower()), None)
            except OSError:
                return None
            if hit is None:
                return None
            nxt = cur / hit
        cur = nxt
    return cur


def _brand_in(path: Path) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read(65536)
    except OSError:
        return b""


def _sample_audio(root: Path, limit: int = 12, depth: int = 3) -> list[Path]:
    out = []
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        if len(Path(dirpath).parts) - base_depth >= depth:
            dirnames[:] = []
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("System Volume Information",)]
        for f in filenames:
            if Path(f).suffix.lower() == ".wav":
                out.append(Path(dirpath) / f)
                if len(out) >= limit:
                    return out
    return out


def identify(root: Path, label: str = "") -> Profile | None:
    """Best guess at which recorder wrote this card, or None if it doesn't look like a recorder card."""
    for p in PROFILES:
        if any(ci_path(root, m) is not None for m in p.markers):
            return p
    # Zoom project folders anywhere in the first two levels (FOLDERxx/ZOOMnnnn)
    try:
        for d in os.listdir(root):
            if re.fullmatch(r"FOLDER\d\d", d, re.I) and any(
                    re.fullmatch(r"ZOOM\d{4}(\.WAV)?", e, re.I) for e in os.listdir(root / d)):
                return BY_ID["zoom"]
    except OSError:
        pass
    for p in PROFILES:
        if label and any(re.search(rx, label, re.I) for rx in p.labels):
            return p
    # Last resort: the brand recorders write into their WAV headers
    for wav in _sample_audio(root):
        head = _brand_in(wav)
        for p in PROFILES:
            if any(b in head for b in p.brands):
                return p
    return None


def folders_for(profile: Profile, root: Path) -> list[str]:
    """The profile's recording folders that exist on this card, spelled as they are on the card."""
    out = []
    for f in profile.folders:
        if f == "":
            return [""]
        hit = ci_path(root, f)
        if hit is not None and hit.is_dir():
            out.append(hit.relative_to(root).as_posix())
    return out


# ------------------------------------------------------------------ naming on import

DATED = re.compile(r"^(\d{6}[_ -]|\d{8}[_ -]|(?:19|20)\d{2}-\d{2}-\d{2}|[A-Z]?\d{6}_\d{3})", re.I)

# <take>_<part>: multitrack/backup/mixdown suffixes used by TASCAM and Zoom
PART_RE = re.compile(r"^(?P<take>.+?)[_-](?P<part>MIX|ST\d{3}|TrLR_BU|LR_BU|TrLR|LR|BU|MN|VM|Tr\d+(?:_\d+)?|"
                     r"\d+(?:-\d+)?)$", re.I)
MAIN_PRIORITY = ("MIX", "ST", "LR", "TRLR")      # which file of a take gets transcribed


def take_of(stem: str) -> tuple[str, str]:
    m = PART_RE.match(stem)
    return (m.group("take"), m.group("part")) if m else (stem, "")


def main_rank(part: str) -> int | None:
    p = part.upper()
    if p.startswith("ST") and p[2:].isdigit():
        p = "ST"
    return MAIN_PRIORITY.index(p) if p in MAIN_PRIORITY else None


def is_dated(name: str) -> bool:
    return bool(DATED.match(name))


def dated_name(name: str, when: float) -> str:
    """ZOOM0001_LR.WAV -> 261121-143000_ZOOM0001_LR.WAV; names that already carry a date are kept."""
    if is_dated(name):
        return name
    return time.strftime("%y%m%d-%H%M%S_", time.localtime(when)) + name


def take_times(items: list[tuple[PurePosixPath, float]]) -> dict[tuple[str, str], float]:
    """Earliest mtime per (folder, take) so every file of one take gets the same date prefix."""
    out: dict[tuple[str, str], float] = {}
    for rel, mtime in items:
        k = (rel.parent.as_posix(), take_of(rel.stem)[0].lower())
        out[k] = min(out.get(k, mtime), mtime)
    return out

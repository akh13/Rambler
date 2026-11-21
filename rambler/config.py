"""Settings, stored as JSON in ~/.config/rambler/config.json."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "rambler"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "rambler"
CONFIG_PATH = CONFIG_DIR / "config.json"

AUDIO_EXTS = [".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif", ".mp4", ".webm"]
SUBTITLE_EXTS = [".srt", ".vtt"]


@dataclass
class Config:
    # First entry in each list is the primary. Order matters only for where new files land.
    audio_dirs: list[str] = field(default_factory=list)
    transcript_dirs: list[str] = field(default_factory=list)

    sync_audio: bool = True
    sync_transcripts: bool = True
    # "newer": newest mtime wins (ties broken by size). "larger": biggest file wins (ties broken by mtime).
    conflict_policy: str = "newer"
    # Files that would be overwritten are moved into <root>/.rambler-replaced/<timestamp>/ first.
    keep_replaced: bool = True
    # Extensions synced between audio dirs. Transcript dirs sync every non-hidden file.
    audio_sync_exts: list[str] = field(default_factory=lambda: AUDIO_EXTS + SUBTITLE_EXTS)

    engine: str = "auto"                 # "auto", "faster-whisper" or "whisper.cpp"
    whisper_cpp_bin: str = ""            # blank = auto-detect
    models_dir: str = str(DATA_DIR / "models")
    model: str = "auto"                  # "auto" = best available, or a model file / name
    language: str = "auto"
    threads: int = max(1, min(8, (os.cpu_count() or 4)))
    initial_prompt: str = ""
    hotwords: str = ""                   # faster-whisper: names/terms to bias towards, comma-separated
    fw_model: str = "large-v3"
    fw_device: str = "auto"              # auto | cuda | cpu
    fw_compute: str = "auto"             # auto -> float16 on GPU, int8 on CPU
    fw_batch: int = 8                    # halves automatically on VRAM pressure (8 suits an 8 GB card)
    fw_beam: int = 5
    filter_hallucinations: bool = True   # drop short "thanks for watching"-style segments
    vad_min_silence_ms: int = 800
    vad_speech_pad_ms: int = 200
    post_transcribe_cmd: str = ""        # run after each transcription batch, e.g. a summariser script
    vad: bool = True                     # use Silero VAD if a ggml-silero model is present (cuts hallucinated text in silences)
    extra_args: str = ""                 # appended to the whisper-cli command line

    generate_srt: bool = True
    transcript_naming: str = "stem"      # legacy; outputs are always <stem>.*
    # "mirror": transcripts/2026/261114_0001.txt (same subfolders as the recordings)
    # "flat":   transcripts/261114_0001.txt (everything in one folder)
    transcript_layout: str = "mirror"
    srt_beside_audio: bool = False       # also drop the .srt next to the recording (VLC auto-loads it)

    incoming_subdir: str = ""            # where drag-and-dropped files land inside the primary audio dir
    auto_on_launch: bool = False         # transcribe pending + sync as soon as the app opens
    auto_on_drive_connect: bool = False  # same, whenever a configured folder appears (drive plugged in)
    confirm_sync_plan: bool = True
    engine_setup_declined: bool = False
    # Field recorders to import from when their card mounts (see recorders.py). Ships with a TASCAM X8 entry.
    recorders: list[dict] = field(default_factory=lambda: [{
        "name": "TASCAM X8", "label": "X8", "uuid": "", "card_subdir": "SOUND", "dest_subdir": "{year}",
        "keep_card_folders": False, "exclude": ["UTILITY/*", "SOUNDPAD/*"], "auto_import": True,
        "unmount_after": True, "enabled": True}])
    start_minimized: bool = False

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        if not path.exists():
            return cls()
        try:
            raw = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)

    @staticmethod
    def expand(p: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(p)))

    def available(self, dirs: list[str]) -> list[Path]:
        """Configured dirs that currently exist (i.e. the drive is mounted)."""
        out = []
        for d in dirs:
            p = self.expand(d)
            if p.is_dir():
                out.append(p)
        return out

    def audio_roots(self) -> list[Path]:
        return self.available(self.audio_dirs)

    def transcript_roots(self) -> list[Path]:
        return self.available(self.transcript_dirs)

"""Transcription engines: whisper.cpp (whisper-cli) and, optionally, faster-whisper."""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import srt as srtmod
from .config import DATA_DIR, Config

ProgressFn = Callable[[float, str], None]   # (0..1, message)


class Cancelled(Exception):
    pass


class EngineError(RuntimeError):
    pass


@dataclass
class Result:
    txt: str
    srt: str | None
    language: str | None
    model: str
    engine: str
    duration: float | None
    segments: list = field(default_factory=list)   # [{start, end, text, ...}]
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers

def probe_duration(path: Path) -> float | None:
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip())
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def to_wav16k(src: Path, dst: Path, cancel: threading.Event) -> None:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise EngineError("ffmpeg not found. Install it with: sudo apt install ffmpeg")
    proc = subprocess.Popen(
        [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    while True:
        try:
            _, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if cancel.is_set():
                proc.kill()
                raise Cancelled()
    if proc.returncode != 0:
        raise EngineError(f"ffmpeg could not decode {src.name}: {err.strip()[-500:]}")


# ---------------------------------------------------------------- model discovery

_MODEL_RANK = [
    (r"large-v3(?!-turbo)", 100), (r"large-v3-turbo", 95), (r"large-v2", 90), (r"large(-v1)?\b", 85),
    (r"medium", 70), (r"small", 50), (r"base", 30), (r"tiny", 10),
]


def model_rank(name: str, language: str = "auto") -> float:
    n = name.lower()
    score = 0.0
    for pat, val in _MODEL_RANK:
        if re.search(pat, n):
            score = val
            break
    if ".en" in n:
        score += 2 if language in ("en", "english") else -40
    if re.search(r"q[2-8](_[01k])?", n):
        score -= 1      # quantised: same model, slightly lower fidelity
    if "-tdrz" in n or "distil" in n:
        score -= 5
    return score


def model_search_dirs(cfg: Config) -> list[Path]:
    dirs = [Config.expand(cfg.models_dir), DATA_DIR / "models", DATA_DIR / "whisper.cpp" / "models",
            Path.home() / "whisper.cpp" / "models", Path.home() / ".cache" / "whisper"]
    seen, out = set(), []
    for d in dirs:
        if d not in seen and d.is_dir():
            seen.add(d)
            out.append(d)
    return out


def list_ggml_models(cfg: Config) -> list[Path]:
    found = []
    for d in model_search_dirs(cfg):
        found += [p for p in d.glob("ggml-*.bin") if "silero" not in p.name and "for-tests" not in p.name]
    return sorted(found, key=lambda p: -model_rank(p.name, cfg.language))


def find_vad_model(cfg: Config) -> Path | None:
    for d in model_search_dirs(cfg):
        hits = sorted(d.glob("ggml-silero-*.bin"))
        if hits:
            return hits[-1]
    return None


def find_whisper_cpp(cfg: Config) -> Path | None:
    if cfg.whisper_cpp_bin:
        p = Config.expand(cfg.whisper_cpp_bin)
        return p if p.exists() else None
    for name in ("whisper-cli", "whisper-cpp", "whisper.cpp"):
        w = shutil.which(name)
        if w:
            return Path(w)
    for p in (DATA_DIR / "whisper.cpp" / "build" / "bin" / "whisper-cli",
              Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"):
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- engines

class Engine:
    name = "engine"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def model_label(self) -> str:
        raise NotImplementedError

    def transcribe(self, audio: Path, want_srt: bool, progress: ProgressFn, cancel: threading.Event) -> Result:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources (worker process, VRAM) after a batch."""


class WhisperCppEngine(Engine):
    name = "whisper.cpp"
    _PROG = re.compile(r"progress\s*=\s*(\d+)%")

    def model_path(self) -> Path | None:
        m = self.cfg.model
        if m and m != "auto":
            p = Config.expand(m)
            if p.exists():
                return p
            for d in model_search_dirs(self.cfg):
                for cand in (d / m, d / f"ggml-{m}.bin"):
                    if cand.exists():
                        return cand
            return None
        models = list_ggml_models(self.cfg)
        return models[0] if models else None

    def model_label(self) -> str:
        p = self.model_path()
        return p.name.removeprefix("ggml-").removesuffix(".bin") if p else "no model"

    def available(self) -> tuple[bool, str]:
        exe = find_whisper_cpp(self.cfg)
        if not exe:
            return False, "whisper.cpp is not set up yet. Use Set up engine (or run: rambler setup)."
        if not self.model_path():
            return False, f"No whisper model found in {self.cfg.models_dir}. Use Set up engine (or run: rambler setup)."
        if not shutil.which("ffmpeg"):
            return False, "ffmpeg not found (sudo apt install ffmpeg)."
        return True, f"whisper.cpp · {self.model_label()}"

    def transcribe(self, audio, want_srt, progress, cancel):
        exe = find_whisper_cpp(self.cfg)
        model = self.model_path()
        if not exe or not model:
            raise EngineError(self.available()[1])
        duration = probe_duration(audio)
        with tempfile.TemporaryDirectory(prefix="rambler-") as td:
            td = Path(td)
            wav = td / "input.wav"
            progress(0.0, "Decoding audio")
            to_wav16k(audio, wav, cancel)
            outbase = td / "out"
            cmd = [str(exe), "-m", str(model), "-f", str(wav), "-of", str(outbase), "-otxt", "-oj",
                   "-t", str(self.cfg.threads), "-l", self.cfg.language or "auto", "-pp"]
            if want_srt:
                cmd.append("-osrt")
            if self.cfg.initial_prompt:
                cmd += ["--prompt", self.cfg.initial_prompt]
            vad = find_vad_model(self.cfg) if self.cfg.vad else None
            if vad and "--vad" not in self.cfg.extra_args:
                cmd += ["--vad", "-vm", str(vad)]
            if self.cfg.extra_args:
                cmd += shlex.split(self.cfg.extra_args)
            progress(0.02, "Transcribing")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    bufsize=1, errors="replace")
            tail: list[str] = []
            lang = None
            stop = threading.Event()

            def killer():
                while not stop.wait(0.3):
                    if cancel.is_set():
                        proc.terminate()
                        return
            threading.Thread(target=killer, daemon=True).start()
            try:
                for line in proc.stdout:  # type: ignore[union-attr]
                    tail = (tail + [line.rstrip()])[-40:]
                    m = self._PROG.search(line)
                    if m:
                        progress(0.02 + 0.97 * int(m.group(1)) / 100, "Transcribing")
                    lm = re.search(r"auto-detected language:\s*(\w+)", line)
                    if lm:
                        lang = lm.group(1)
                proc.wait()
            finally:
                stop.set()
            if cancel.is_set():
                raise Cancelled()
            if proc.returncode != 0:
                raise EngineError(f"whisper-cli exited with {proc.returncode}:\n" + "\n".join(tail[-12:]))
            txt_path = Path(str(outbase) + ".txt")
            if not txt_path.exists():
                raise EngineError("whisper-cli produced no transcript:\n" + "\n".join(tail[-12:]))
            txt = txt_path.read_text(encoding="utf-8", errors="replace")
            srt_text = None
            if want_srt:
                srt_text = ""
            if want_srt and Path(str(outbase) + ".srt").exists():
                srt_text = Path(str(outbase) + ".srt").read_text(encoding="utf-8", errors="replace")
            jpath = Path(str(outbase) + ".json")
            segments = []
            if jpath.exists():
                try:
                    j = json.loads(jpath.read_text(errors="replace"))
                    lang = lang or j.get("result", {}).get("language")
                    for t in j.get("transcription", []):
                        off = t.get("offsets", {})
                        segments.append({"start": round(off.get("from", 0) / 1000, 2),
                                         "end": round(off.get("to", 0) / 1000, 2), "text": t.get("text", "").strip()})
                except (json.JSONDecodeError, AttributeError):
                    pass
        progress(1.0, "Done")
        return Result(_clean_txt(txt), srt_text, lang or (None if self.cfg.language == "auto" else self.cfg.language),
                      self.model_label(), self.name, duration, segments)


def nvidia_lib_dirs() -> list[str]:
    """lib folders of the pip nvidia-* wheels (cuBLAS, cuDNN) in this Python environment."""
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    dirs = []
    for base in (spec.submodule_search_locations or []) if spec else []:
        for d in sorted(Path(base).glob("*/lib")):
            dirs.append(str(d))
    return dirs


class FasterWhisperEngine(Engine):
    """faster-whisper (CTranslate2) in a worker process; CUDA via pip wheels, no toolkit needed."""
    name = "faster-whisper"

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.proc: subprocess.Popen | None = None
        self.device = ""

    def _model_name(self) -> str:
        m = self.cfg.model
        if not m or m == "auto" or m.endswith(".bin"):
            return self.cfg.fw_model or "large-v3"
        return m

    def model_label(self) -> str:
        return self._model_name()

    def available(self):
        import importlib.util
        if importlib.util.find_spec("faster_whisper") is None:
            return False, "faster-whisper is not installed yet. Use Set up engine (or run: rambler setup)."
        if not shutil.which("ffmpeg"):
            return False, "ffmpeg not found (sudo apt install ffmpeg)."
        gpu = "GPU" if nvidia_lib_dirs() else "CPU"
        return True, f"faster-whisper · {self._model_name()} · {gpu}"

    # ---- worker plumbing
    def _env(self) -> dict:
        env = dict(os.environ)
        libs = nvidia_lib_dirs()
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
        pkg_parent = str(Path(__file__).resolve().parent.parent)
        env["PYTHONPATH"] = os.pathsep.join([pkg_parent, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        return env

    def _send(self, **kw):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(kw) + "\n")
        self.proc.stdin.flush()

    def _read(self, cancel: threading.Event, on_event) -> dict:
        """Read events until result/ready/error. Kills the worker on cancel."""
        assert self.proc and self.proc.stdout
        stop = threading.Event()
        proc = self.proc

        def killer():
            while not stop.wait(0.3):
                if cancel.is_set():
                    proc.kill()
                    return
        threading.Thread(target=killer, daemon=True).start()
        try:
            for line in proc.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("event")
                if kind in ("result", "ready"):
                    return ev
                if kind == "error":
                    raise EngineError(ev.get("msg", "worker error"))
                on_event(ev)
        finally:
            stop.set()
        self.proc = None
        if cancel.is_set():
            raise Cancelled()
        try:
            err = (DATA_DIR / "fw_worker.log").read_text(errors="replace")[-1500:]
        except OSError:
            err = ""
        raise EngineError("faster-whisper worker exited unexpectedly:\n" + err)

    def _ensure_worker(self, progress, cancel):
        if self.proc and self.proc.poll() is None:
            return
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._log = open(DATA_DIR / "fw_worker.log", "w")   # a file, not a pipe: model downloads are chatty
        self.proc = subprocess.Popen([sys.executable, "-m", "rambler.fw_worker"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._log, text=True, bufsize=1,
                                     env=self._env())
        c = self.cfg
        progress(0.0, f"Loading {self._model_name()} (first use downloads it)")
        self._send(cmd="load", model=self._model_name(), device=c.fw_device, compute=c.fw_compute,
                   threads=c.threads, language=c.language, beam=c.fw_beam, batch=c.fw_batch,
                   hotwords=c.hotwords, prompt=c.initial_prompt, filter_hallucinations=c.filter_hallucinations,
                   vad_min_silence_ms=c.vad_min_silence_ms, vad_speech_pad_ms=c.vad_speech_pad_ms)
        ev = self._read(cancel, lambda e: None)
        self.device = ev.get("device", "")

    def transcribe(self, audio, want_srt, progress, cancel):
        c = self.cfg
        self._ensure_worker(progress, cancel)
        with tempfile.TemporaryDirectory(prefix="rambler-") as td:
            wav = Path(td) / "input.wav"
            progress(0.0, "Decoding audio")
            to_wav16k(audio, wav, cancel)
            self._send(cmd="transcribe", path=str(wav))
            ev = self._read(cancel, lambda e: progress(e["p"], "Transcribing") if e.get("event") == "progress" else None)
        segs = ev.get("segments", [])
        text = " ".join(x["text"] for x in segs).strip()
        srt_text = srtmod.dumps([srtmod.Cue(x["start"], x["end"], x["text"]) for x in segs]) if want_srt else None
        progress(1.0, "Done")
        return Result(text + "\n", srt_text, ev.get("language"), self._model_name(), self.name,
                      ev.get("duration"), segs, {"batch": ev.get("batch"), "device": self.device})

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send(cmd="quit")
                self.proc.wait(5)
            except Exception:
                self.proc.kill()
        self.proc = None


def _clean_txt(txt: str) -> str:
    lines = [l.strip() for l in txt.replace("\r\n", "\n").split("\n")]
    return "\n".join(l for l in lines if l) + "\n"


def make_engine(cfg: Config) -> Engine:
    if cfg.engine == "faster-whisper":
        return FasterWhisperEngine(cfg)
    if cfg.engine == "whisper.cpp":
        return WhisperCppEngine(cfg)
    # auto: whichever is set up, preferring faster-whisper
    fw = FasterWhisperEngine(cfg)
    if fw.available()[0]:
        return fw
    wc = WhisperCppEngine(cfg)
    if wc.available()[0]:
        return wc
    return fw

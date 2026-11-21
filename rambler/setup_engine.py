"""Fetch and build the transcription toolchain: apt prerequisites, whisper.cpp, models.

Shared by `rambler setup` and the GUI's first-run dialog. Nothing here needs root except the optional
apt step, which goes through pkexec (GUI) or sudo (CLI).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import DATA_DIR, Config

WHISPER_REPO = "https://github.com/ggml-org/whisper.cpp.git"
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin"
VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-{name}.bin"
VAD_MODEL = "silero-v5.1.2"
SRC = DATA_DIR / "whisper.cpp"
FW_SIZES = {"large-v3": 3.1e9, "large-v3-turbo": 1.6e9, "medium": 1.5e9, "small": 0.48e9, "base.en": 0.15e9}
MODELS = {   # name: (approx size, description)
    "large-v3": ("3.1 GB", "Best accuracy"),
    "large-v3-turbo": ("1.6 GB", "Nearly as good, much faster"),
    "medium": ("1.5 GB", "Smaller GPUs"),
    "small": ("466 MB", "CPU-friendly"),
    "base.en": ("142 MB", "Tiny, English only, for testing"),
}

Log = Callable[[str], None]
Progress = Callable[[float, str], None]


class SetupError(RuntimeError):
    pass


# ------------------------------------------------------------------ detection

def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None or Path("/proc/driver/nvidia/version").exists()


def find_nvcc() -> str | None:
    for p in (shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", "/usr/lib/cuda/bin/nvcc"):
        if p and Path(p).exists():
            return p
    return None


def is_pop_os() -> bool:
    try:
        return "pop" in Path("/etc/os-release").read_text().lower()
    except OSError:
        return False


BACKENDS = ("faster-whisper", "cuda", "vulkan", "cpu")   # first = faster-whisper via pip, rest = whisper.cpp builds


def default_backend() -> str:
    if has_nvidia():
        return "faster-whisper"      # pip wheels bring their own CUDA libs: nothing to compile
    if shutil.which("vulkaninfo"):
        return "vulkan"              # AMD/Intel GPUs: whisper.cpp over Vulkan
    return "faster-whisper"          # CPU, int8


def cuda_toolkit_package() -> str:
    return "system76-cuda-latest" if is_pop_os() else "nvidia-cuda-toolkit"


def _dpkg_installed(pkg: str) -> bool:
    r = subprocess.run(["dpkg-query", "-W", "-f=${Status}", pkg], capture_output=True, text=True)
    return "install ok installed" in r.stdout


def missing_packages(backend: str) -> list[str]:
    """apt packages needed for this backend and to run Rambler."""
    need = {"ffmpeg": "ffmpeg"} if backend == "faster-whisper" else \
        {"git": "git", "cmake": "cmake", "g++": "build-essential", "ffmpeg": "ffmpeg"}
    missing = [pkg for exe, pkg in need.items() if not shutil.which(exe)]
    if not shutil.which("dpkg-query"):
        return missing
    if not _dpkg_installed("libxcb-cursor0"):
        missing.append("libxcb-cursor0")
    if backend == "cuda" and not find_nvcc():
        missing.append(cuda_toolkit_package())
    if backend == "vulkan":
        if not _dpkg_installed("libvulkan-dev"):
            missing.append("libvulkan-dev")
        if not shutil.which("glslc"):
            missing.append("glslc")
    return missing


@dataclass
class Status:
    whisper_cli: Path | None
    models: list[str] = field(default_factory=list)
    vad: bool = False

    faster_whisper: bool = False

    @property
    def ready(self) -> bool:
        return self.faster_whisper or (self.whisper_cli is not None and bool(self.models))


def status(cfg: Config) -> Status:
    import importlib.util
    from .engines import find_vad_model, find_whisper_cpp, list_ggml_models
    return Status(find_whisper_cpp(cfg), [p.name for p in list_ggml_models(cfg)], find_vad_model(cfg) is not None,
                  importlib.util.find_spec("faster_whisper") is not None)


# ------------------------------------------------------------------ steps

def _run(cmd: list[str], log: Log, cancel: threading.Event, on_line: Callable[[str], None] | None = None,
         env: dict | None = None, cwd: Path | None = None) -> None:
    log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace", env=env, cwd=cwd)
    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        if on_line:
            on_line(line)
        if cancel.is_set():
            proc.terminate()
            proc.wait()
            raise SetupError("Cancelled")
    if proc.wait() != 0:
        raise SetupError(f"{cmd[0]} failed ({proc.returncode}):\n" + "\n".join(tail[-15:]))


def install_packages(pkgs: list[str], log: Log, cancel: threading.Event, gui: bool) -> None:
    """One privilege prompt: apt-get update + install. Failure is fatal only if build tools are still missing."""
    if not pkgs:
        return
    script = "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y " + " ".join(pkgs)
    if os.geteuid() == 0:
        cmd = ["sh", "-c", script]
    elif gui and shutil.which("pkexec"):
        cmd = ["pkexec", "sh", "-c", script]
    else:
        cmd = ["sudo", "sh", "-c", script]
    try:
        _run(cmd, log, cancel, on_line=log)
    except SetupError as e:
        if cancel.is_set():
            raise
        still = [x for x in ("git", "cmake", "g++", "ffmpeg") if not shutil.which(x)]
        if still:
            raise SetupError(f"Package install failed and {', '.join(still)} still missing.\n{e}")
        log(f"Package install failed, continuing because the build tools are present:\n{e}")


def fetch_source(log: Log, cancel: threading.Event, ref: str = "") -> None:
    if (SRC / ".git").exists():
        _run(["git", "-C", str(SRC), "fetch", "--tags", "--quiet"], log, cancel)
        if not ref:
            _run(["git", "-C", str(SRC), "pull", "--ff-only", "--quiet"], log, cancel)
    else:
        SRC.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", "--quiet", WHISPER_REPO, str(SRC)], log, cancel)
    if ref:
        _run(["git", "-C", str(SRC), "checkout", "--quiet", ref], log, cancel)


def build(backend: str, log: Log, progress: Progress, cancel: threading.Event) -> Path:
    env = dict(os.environ)
    flags = ["-DCMAKE_BUILD_TYPE=Release", "-DWHISPER_BUILD_TESTS=OFF", "-DWHISPER_BUILD_EXAMPLES=ON"]
    if backend == "cuda":
        nvcc = find_nvcc()
        if not nvcc:
            raise SetupError(f"CUDA toolkit (nvcc) not found. Install {cuda_toolkit_package()} or pick Vulkan.")
        env["PATH"] = str(Path(nvcc).parent) + os.pathsep + env.get("PATH", "")
        flags += ["-DGGML_CUDA=1", f"-DCMAKE_CUDA_ARCHITECTURES={os.environ.get('CUDA_ARCH', 'native')}"]
    elif backend == "vulkan":
        flags += ["-DGGML_VULKAN=1"]
    builddir = SRC / "build"
    cache = builddir / "CMakeCache.txt"
    if cache.exists():  # backend changed? start clean
        txt = cache.read_text(errors="replace")
        if ("GGML_CUDA:BOOL=ON" in txt) != (backend == "cuda") or ("GGML_VULKAN:BOOL=ON" in txt) != (backend == "vulkan"):
            shutil.rmtree(builddir)
    progress(0.0, "Configuring whisper.cpp")
    _run(["cmake", "-S", str(SRC), "-B", str(builddir), *flags], log, cancel, env=env)
    pct = re.compile(r"^\[\s*(\d+)%\]")

    def on_line(line):
        m = pct.match(line)
        if m:
            progress(int(m.group(1)) / 100, f"Building whisper.cpp ({backend})")
        elif "error" in line.lower():
            log(line)
    _run(["cmake", "--build", str(builddir), "-j", str(os.cpu_count() or 4), "--config", "Release",
          "--target", "whisper-cli"], log, cancel, on_line=on_line, env=env)
    exe = builddir / "bin" / "whisper-cli"
    if not exe.exists():
        raise SetupError("Build finished but whisper-cli is missing.")
    progress(1.0, "Built whisper-cli")
    return exe


def download(url: str, dest: Path, log: Log, progress: Progress, cancel: threading.Event, label: str) -> None:
    """Resumable download to dest (atomic rename on completion)."""
    if dest.exists():
        log(f"{dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "rambler-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    log(f"Downloading {url}")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0  # server ignored the range: start over
        total = have + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if have else "wb") as f:
            done = have
            while True:
                if cancel.is_set():
                    raise SetupError("Cancelled")
                buf = r.read(1 << 20)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    progress(done / total, f"{label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB")
    if total and part.stat().st_size < total:
        raise SetupError(f"Download of {dest.name} was cut short; run setup again to resume.")
    os.replace(part, dest)


def pip_install(log: Log, progress: Progress, cancel: threading.Event) -> None:
    import sys
    pkgs = ["faster-whisper>=1.1"]
    if has_nvidia():
        pkgs += ["nvidia-cublas-cu12", "nvidia-cudnn-cu12"]
    progress(0.0, "Installing " + ", ".join(pkgs))
    try:
        _run([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs], log, cancel, on_line=log)
    except SetupError as e:
        if "externally-managed" in str(e):
            raise SetupError("This Python won't take pip packages. Install Rambler with ./install.sh "
                             "(it makes its own venv) and run setup from there.")
        raise
    progress(1.0, "Installed faster-whisper")


def _dir_size(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def fw_download(cfg: Config, model: str, log: Log, progress: Progress, cancel: threading.Event) -> None:
    """Pre-fetch a faster-whisper model so the first transcription doesn't stall on a 3 GB download."""
    import json as _json
    import sys
    import time
    from .engines import FasterWhisperEngine
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    start = _dir_size(hub) if hub.exists() else 0
    want = FW_SIZES.get(model, 1e9)
    env = FasterWhisperEngine(cfg)._env()
    proc = subprocess.Popen([sys.executable, "-m", "rambler.fw_worker"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
    proc.stdin.write(_json.dumps({"cmd": "download", "model": model}) + "\n")
    proc.stdin.close()
    result: dict = {}

    def reader():
        for line in proc.stdout:
            try:
                result.update(_json.loads(line))
            except ValueError:
                pass
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    while proc.poll() is None:
        if cancel.is_set():
            proc.kill()
            raise SetupError("Cancelled")
        got = (_dir_size(hub) - start) if hub.exists() else 0
        progress(min(0.99, got / want), f"{model}: {got / 1e9:.2f} / ~{want / 1e9:.1f} GB")
        time.sleep(1)
    t.join(5)
    if result.get("event") == "error":
        raise SetupError(f"Model download failed: {result.get('msg')}")
    log(f"{model} ready at {result.get('path', '?')}")
    progress(1.0, f"{model} ready")


def run_setup(cfg: Config, backend: str, models: list[str], log: Log, progress: Progress,
              cancel: threading.Event, install_apt: bool = True, gui: bool = False, ref: str = "",
              overall: Callable[[int, int, str], None] | None = None) -> Config:
    """Do everything. Returns cfg updated to point at the new engine (caller saves it)."""
    fw = backend == "faster-whisper"
    if fw:
        steps = (["packages"] if install_apt else []) + ["pip"] + [f"fwmodel:{m}" for m in models]
    else:
        steps = (["packages"] if install_apt else []) + ["source", "build"] + [f"model:{m}" for m in models] + ["vad"]
    exe = None
    models_dir = Config.expand(cfg.models_dir)

    def step(i, label):
        if overall:
            overall(i, len(steps), label)
        log(f"== {label}")

    for i, s in enumerate(steps):
        if s == "packages":
            pkgs = missing_packages(backend)
            step(i, "Installing packages: " + (" ".join(pkgs) if pkgs else "none needed"))
            install_packages(pkgs, log, cancel, gui)
        elif s == "pip":
            step(i, "Installing faster-whisper and CUDA libraries (pip)")
            pip_install(log, progress, cancel)
        elif s.startswith("fwmodel:"):
            name = s.split(":", 1)[1]
            step(i, f"Downloading model {name}")
            fw_download(cfg, name, log, progress, cancel)
        elif s == "source":
            step(i, "Fetching whisper.cpp")
            progress(0, "Fetching whisper.cpp")
            fetch_source(log, cancel, ref)
        elif s == "build":
            step(i, f"Building whisper.cpp ({backend})")
            exe = build(backend, log, progress, cancel)
        elif s.startswith("model:"):
            name = s.split(":", 1)[1]
            step(i, f"Downloading model {name}")
            download(MODEL_URL.format(name=name), models_dir / f"ggml-{name}.bin", log, progress, cancel, name)
        elif s == "vad":
            step(i, "Downloading Silero VAD")
            try:
                download(VAD_URL.format(name=VAD_MODEL), models_dir / f"ggml-{VAD_MODEL}.bin", log, progress,
                         cancel, "Silero VAD")
            except (OSError, SetupError) as e:
                if cancel.is_set():
                    raise
                log(f"VAD download failed (optional): {e}")
    if overall:
        overall(len(steps), len(steps), "Engine ready")
    if fw:
        cfg.engine = "faster-whisper"
        if models:
            cfg.fw_model = "large-v3" if "large-v3" in models else models[0]
        return cfg
    log(f"whisper-cli: {exe}")
    if cfg.whisper_cpp_bin and not Config.expand(cfg.whisper_cpp_bin).exists():
        cfg.whisper_cpp_bin = ""          # stale override; auto-detect finds the new build
    cfg.engine = "whisper.cpp"
    return cfg

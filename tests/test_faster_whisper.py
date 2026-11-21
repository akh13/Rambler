"""Runs the real faster-whisper worker (tiny model, CPU) against current PyPI releases, so an upstream
change like PyAV dropping arguments faster-whisper still passes shows up in CI, not on someone's desk.
Skipped when faster-whisper isn't installed."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
fw = pytest.importorskip("faster_whisper")


def test_worker_transcribes_wav_and_mp3(tmp_path):
    from rambler import fw_worker
    src = tmp_path / "speech.wav"
    # speech-like test signal is enough to exercise decoding, VAD and the pipeline end to end
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=180:duration=4", "-af", "volume=0.3", str(src)], check=True)
    mp3 = tmp_path / "speech.mp3"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), str(mp3)], check=True)
    model = fw.WhisperModel("tiny.en", device="cpu", compute_type="int8")
    pipe = fw.BatchedInferencePipeline(model=model)
    for f in (src, mp3):
        audio = fw_worker.decode(str(f))
        assert audio.dtype.name == "float32" and 3.5 * 16000 < len(audio) < 4.5 * 16000
        r = fw_worker.transcribe(pipe, str(f), {"device": "cpu", "language": "en"})
        assert abs(r["duration"] - 4.0) < 0.2 and isinstance(r["segments"], list)

"""faster-whisper worker process.

Runs in its own process so the CUDA libraries from the pip `nvidia-*` wheels can be put on
LD_LIBRARY_PATH before ctranslate2 loads, so a cancel can kill it outright, and so VRAM is released
when a batch ends. The model is loaded once and reused for every file in the batch.

Protocol: JSON lines. stdin  {"cmd": "load", ...opts} | {"cmd": "transcribe", "path": ...} | {"cmd": "download", "model": ...}
          stdout {"event": "ready"|"progress"|"result"|"error"|"log", ...}
"""
from __future__ import annotations

import json
import re
import sys

HALLUCINATIONS = [re.compile(p, re.I) for p in (
    r"thanks? for watching", r"please subscribe", r"subtitles? by", r"amara\.org", r"like and subscribe",
    r"see you (in the )?next video", r"transcribed by", r"in this lecture,? I will", r"www\.[a-z]+\.(com|org)")]


def emit(**kw):
    sys.stdout.write(json.dumps(kw) + "\n")
    sys.stdout.flush()


def main():
    model = pipe = None
    opts: dict = {}
    for line in sys.stdin:
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        cmd = req.get("cmd")
        try:
            if cmd == "download":
                from faster_whisper.utils import download_model
                path = download_model(req["model"])
                emit(event="result", path=path)
            elif cmd == "load":
                from faster_whisper import BatchedInferencePipeline, WhisperModel
                opts = req
                device = opts.get("device", "auto")
                if device == "auto":
                    try:
                        import ctranslate2
                        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                    except Exception:
                        device = "cpu"
                compute = opts.get("compute", "auto")
                if compute == "auto":
                    compute = "float16" if device == "cuda" else "int8"
                emit(event="log", msg=f"loading {opts['model']} on {device} ({compute})")
                model = WhisperModel(opts["model"], device=device, compute_type=compute,
                                     cpu_threads=int(opts.get("threads", 4)))
                pipe = BatchedInferencePipeline(model=model)
                opts["device"], opts["compute"] = device, compute
                emit(event="ready", device=device, compute=compute)
            elif cmd == "transcribe":
                if pipe is None:
                    raise RuntimeError("model not loaded")
                emit(event="result", **transcribe(pipe, req["path"], opts))
            elif cmd == "quit":
                return
        except Exception as e:  # report and keep serving
            emit(event="error", msg=f"{type(e).__name__}: {e}")


def transcribe(pipe, path: str, o: dict) -> dict:
    batch = int(o.get("batch", 8)) if o.get("device") == "cuda" else 1
    lang = o.get("language") or None
    if lang == "auto":
        lang = None
    while True:
        try:
            segs, info = pipe.transcribe(
                path, language=lang, beam_size=int(o.get("beam", 5)), batch_size=batch,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=int(o.get("vad_min_silence_ms", 800)),
                                    speech_pad_ms=int(o.get("vad_speech_pad_ms", 200))),
                condition_on_previous_text=False,
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
                hallucination_silence_threshold=2.0,
                word_timestamps=True,
                hotwords=o.get("hotwords") or None,
                initial_prompt=o.get("prompt") or None)
            out = []
            for s in segs:
                txt = s.text.strip()
                if o.get("filter_hallucinations", True) and len(txt) < 60 and any(h.search(txt) for h in HALLUCINATIONS):
                    continue
                out.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": txt,
                            "logprob": round(s.avg_logprob, 3), "no_speech": round(s.no_speech_prob, 3),
                            "words": [{"w": w.word, "s": round(w.start, 2), "e": round(w.end, 2),
                                       "p": round(w.probability, 3)} for w in (s.words or [])]})
                if info.duration:
                    emit(event="progress", p=min(0.99, s.end / info.duration))
            return {"segments": out, "language": info.language, "duration": info.duration, "batch": batch}
        except Exception as e:
            msg = str(e).lower()
            if ("out of memory" in msg or "cuda" in msg) and batch > 1:
                batch = max(1, batch // 2)
                emit(event="log", msg=f"VRAM pressure, dropping batch to {batch}")
                continue
            raise


if __name__ == "__main__":
    main()

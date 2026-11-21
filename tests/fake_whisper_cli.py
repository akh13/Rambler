#!/usr/bin/env python3
"""Stand-in for whisper.cpp's whisper-cli, used by tests and for trying the UI without a GPU."""
import json
import subprocess
import sys
import time

args = sys.argv[1:]


def opt(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


wav, out = opt("-f"), opt("-of")
dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                            "default=nw=1:nk=1", wav], capture_output=True, text=True).stdout.strip() or 0)
for p in range(0, 101, 20):
    print(f"whisper_print_progress_callback: progress = {p:3d}%", flush=True)
    time.sleep(0.02)
print("whisper_full_with_state: auto-detected language: en (p = 0.98)")
n = max(1, int(dur // 2))
lines = [f"This is fake sentence number {i + 1}." for i in range(n)]
open(out + ".txt", "w").write("\n".join(" " + l for l in lines) + "\n")
if "-osrt" in args:
    def ts(t):
        return f"00:00:{int(t):02d},{int((t % 1) * 1000):03d}"
    blocks = [f"{i + 1}\n{ts(i * 2)} --> {ts(i * 2 + 2)}\n{l}\n" for i, l in enumerate(lines)]
    open(out + ".srt", "w").write("\n".join(blocks))
if "-oj" in args:
    json.dump({"result": {"language": "en"},
               "transcription": [{"offsets": {"from": i * 2000, "to": i * 2000 + 2000}, "text": " " + l}
                                 for i, l in enumerate(lines)]}, open(out + ".json", "w"))

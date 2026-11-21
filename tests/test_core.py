import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rambler import srt
from rambler.config import Config
from rambler.engines import WhisperCppEngine, model_rank
from rambler.library import scan
from rambler.pipeline import expand_drops, import_files, transcribe_one
from rambler.syncer import build_plan, execute

FAKE = Path(__file__).with_name("fake_whisper_cli.py")


def tone(path: Path, seconds=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", str(path)], check=True)


@pytest.fixture
def env(tmp_path):
    a1, a2, t1, t2 = (tmp_path / n for n in ("audioA", "audioB", "txtA", "txtB"))
    for d in (a1, a2, t1, t2):
        d.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (models / "ggml-base.en.bin").write_bytes(b"x")
    (models / "ggml-large-v3-turbo.bin").write_bytes(b"x")
    (models / "ggml-large-v3.bin").write_bytes(b"x")
    os.chmod(FAKE, os.stat(FAKE).st_mode | stat.S_IEXEC)
    cfg = Config(audio_dirs=[str(a1), str(a2), str(tmp_path / "unplugged")],
                 transcript_dirs=[str(t1), str(t2)], whisper_cpp_bin=str(FAKE), models_dir=str(models))
    return cfg, a1, a2, t1, t2


def test_srt_roundtrip():
    cues = srt.parse("1\n00:00:01,000 --> 00:00:02,500\nHello <i>there</i>\n\n2\n00:01:00.000 --> 00:01:02.000\nBye\n")
    assert [c.text for c in cues] == ["Hello there", "Bye"]
    assert srt.parse(srt.dumps(cues))[1].start == 60.0
    idx = srt.CueIndex(cues)
    assert idx.at(1.5) == 0 and idx.at(3) == -1 and idx.at(61) == 1


def test_best_model_rank():
    assert model_rank("ggml-large-v3.bin") > model_rank("ggml-large-v3-turbo.bin") > model_rank("ggml-medium.bin")
    assert model_rank("ggml-medium.en.bin", "auto") < model_rank("ggml-small.bin", "auto")


def test_scan_transcribe_sync(env):
    cfg, a1, a2, t1, t2 = env
    tone(a1 / "2026" / "261114_0001.wav")
    tone(a2 / "261115_0001.mp3")
    # pre-existing transcript in whisper.cpp's default "name.ext.txt" style, flat in transcripts root
    (t2 / "261115_0001.mp3.txt").write_text("old\n")

    recs = scan(cfg)
    by = {r.rel.as_posix(): r for r in recs}
    assert by["2026/261114_0001.wav"].status() == "To be transcribed"
    assert by["261115_0001.mp3"].transcribed and not by["261115_0001.mp3"].srt

    eng = WhisperCppEngine(cfg)
    assert eng.available()[0] and eng.model_label() == "large-v3"
    seen = []
    rec = by["2026/261114_0001.wav"]
    transcribe_one(cfg, eng, rec, True, lambda f, m: seen.append(f), threading.Event())
    assert seen[-1] == 1.0
    assert (t1 / "2026" / "261114_0001.txt").read_text().startswith("This is fake")
    assert srt.load(t1 / "2026" / "261114_0001.srt")[0].text.startswith("This is fake")
    assert (t1 / "2026" / "261114_0001.json").exists()

    plan = build_plan(cfg)
    assert str(Config.expand(cfg.audio_dirs[2])) in [str(Config.expand(s)) for s in plan.skipped_roots]
    dests = {(a.group, a.dst_root.name, a.src.rel.as_posix()) for a in plan.actions}
    assert ("audio", "audioB", "2026/261114_0001.wav") in dests
    assert ("audio", "audioA", "261115_0001.mp3") in dests
    assert ("transcripts", "txtB", "2026/261114_0001.srt") in dests
    assert ("transcripts", "txtA", "261115_0001.mp3.txt") in dests
    assert execute(plan, cfg) == []
    assert build_plan(cfg).actions == []
    # mtimes preserved
    assert abs((a1 / "261115_0001.mp3").stat().st_mtime - (a2 / "261115_0001.mp3").stat().st_mtime) < 0.01


def test_newer_version_replaces_and_keeps_old(env):
    cfg, a1, a2, t1, t2 = env
    (t1 / "x.txt").write_text("old")
    (t2 / "x.txt").write_text("new and longer")
    old = time.time() - 3600
    os.utime(t1 / "x.txt", (old, old))
    plan = build_plan(cfg)
    assert len(plan.actions) == 1 and plan.actions[0].replace is not None
    execute(plan, cfg)
    assert (t1 / "x.txt").read_text() == "new and longer"
    kept = list((t1 / ".rambler-replaced").rglob("x.txt"))
    assert kept and kept[0].read_text() == "old"


def test_same_content_different_mtime_is_not_copied(env):
    cfg, a1, a2, t1, t2 = env
    (t1 / "y.txt").write_text("same")
    (t2 / "y.txt").write_text("same")
    os.utime(t1 / "y.txt", (1, 1))
    assert build_plan(cfg).actions == []


def test_import_drops(env, tmp_path):
    cfg, a1, a2, t1, t2 = env
    cfg.incoming_subdir = "Inbox"
    drop = tmp_path / "Downloads"
    tone(drop / "new.mp3", 2)
    tone(drop / "batch" / "b1.wav", 2)
    (drop / "notes.txt").write_text("ignored")
    items = expand_drops([drop / "new.mp3", drop / "batch", drop / "notes.txt"])
    assert [r.as_posix() for _, r in items] == ["new.mp3", "batch/b1.wav"]
    rels = import_files(cfg, items)
    assert (a1 / "Inbox" / "new.mp3").exists() and (a1 / "Inbox" / "batch" / "b1.wav").exists()
    assert [r.as_posix() for r in rels] == ["Inbox/new.mp3", "Inbox/batch/b1.wav"]
    # dropping again is a no-op
    assert [r.as_posix() for r in import_files(cfg, items)] == ["Inbox/new.mp3", "Inbox/batch/b1.wav"]
    assert not (a1 / "Inbox" / "new (2).mp3").exists()


def test_recorder_import(env, tmp_path, monkeypatch):
    from rambler import recorders as R
    cfg, a1, a2, t1, t2 = env
    monkeypatch.setattr(R, "LEDGER", tmp_path / "ledger.json")
    card = tmp_path / "media" / "X8"
    tone(card / "SOUND" / "261121_0001.wav", 2)
    tone(card / "SOUND" / "261119_0009_MIX.wav", 2)
    tone(card / "SOUND" / "261119_0009_5.wav", 2)
    (card / "SOUND" / "261121_0001.bin").write_bytes(b"x" * 52)
    (card / "SOUNDPAD").mkdir()
    tone(card / "SOUNDPAD" / "pad1.wav", 1)
    (card / "UTILITY").mkdir()
    # one take is already filed in the library (e.g. imported by hand last week)
    tone(card / "SOUND" / "261114_0001.wav", 2)
    (a2 / "2026").mkdir()
    import shutil
    shutil.copy2(card / "SOUND" / "261114_0001.wav", a2 / "2026" / "261114_0001.wav")
    mount = R.Mount(card, "/dev/sdz1", "1234-ABCD", "X8", "exfat")
    rec = R.recorders(cfg)[0]
    assert rec.name == "TASCAM X8"
    assert R.match(rec, [R.Mount(tmp_path, "/dev/sdy1", "", "OTHER", "ext4"), mount]) is mount
    assert sorted(r.name for _, r, _ in R.pending(rec, mount, cfg)) == \
        ["261119_0009_5.wav", "261119_0009_MIX.wav", "261121_0001.wav"]
    got = R.import_recorder(cfg, rec, mount, lambda m: None)
    assert sorted(g.as_posix() for g in got) == ["2026/261119_0009_5.wav", "2026/261119_0009_MIX.wav",
                                                 "2026/261121_0001.wav"]
    assert (a1 / "2026" / "261121_0001.wav").exists()
    assert not (a1 / "2026" / "261114_0001.wav").exists()       # already on the other drive; sync's job
    assert (card / "SOUND" / "261121_0001.wav").exists()         # card untouched
    assert R.pending(rec, mount, cfg) == []
    # file counter reset: same name, different audio -> imported alongside, not over
    tone(card / "SOUND" / "261121_0001.wav", 3)
    got = R.import_recorder(cfg, rec, mount, lambda m: None)
    assert [g.as_posix() for g in got] == ["2026/261121_0001 (2).wav"]
    assert R.match(R.Recorder(label="", uuid="1234-abcd"), [mount]) is mount
    assert R.year_of("2025-11-05_033700_000.wav", 0) == "2025"


def test_library_rules(env, tmp_path):
    from rambler.library import scan
    cfg, a1, a2, t1, t2 = env
    cfg.transcript_layout = "flat"
    tone(a1 / "2024" / "2024-01-01_120000_000.wav" / "track_0.wav", 3)      # TP-7 export folder
    tone(a1 / "2026" / "261119_0009_MIX.wav", 3)
    tone(a1 / "2026" / "261119_0009_1-2.wav", 3)
    tone(a1 / "2026" / "261119_0009_6.wav", 3)
    tone(a1 / "2026" / "blip.wav", 1)
    tone(a1 / "2022" / "221214_0194.wav", 3)
    (t1 / "2024-01-01_120000_000.txt").write_text("x")
    (t2 / "221214 0194.txt").write_text("old-style name with a space")
    by = {r.rel.as_posix(): r for r in scan(cfg)}
    assert by["2024/2024-01-01_120000_000.wav/track_0.wav"].transcribed
    assert by["2022/221214_0194.wav"].transcribed
    assert by["2026/261119_0009_MIX.wav"].pending
    t = by["2026/261119_0009_6.wav"]
    assert not t.pending and t.status() == "Track (MIX pending)"
    assert by["2026/blip.wav"].status() == "Too short" and not by["2026/blip.wav"].pending
    eng = WhisperCppEngine(cfg)
    transcribe_one(cfg, eng, by["2026/261119_0009_MIX.wav"], True, lambda f, m: None, threading.Event())
    assert (t1 / "261119_0009_MIX.txt").exists() and (t1 / "261119_0009_MIX.json").exists()   # flat
    import json
    j = json.loads((t1 / "261119_0009_MIX.json").read_text())
    assert j["audio"] == "2026/261119_0009_MIX.wav" and j["model"] == "large-v3" and j["segments"]
    by = {r.rel.as_posix(): r for r in scan(cfg)}
    assert by["2026/261119_0009_1-2.wav"].status() == "Track (MIX transcribed)"


def test_pipeline_lock():
    from rambler import recorders as R
    with R.pipeline_lock() as a:
        assert a
        import subprocess as sp
        code = "import sys; sys.path.insert(0, '.'); from rambler import recorders as R\n" \
               "with R.pipeline_lock() as g: print(g)"
        out = sp.run([sys.executable, "-c", code], capture_output=True, text=True,
                     cwd=Path(__file__).resolve().parents[1]).stdout.strip()
        assert out == "False"


def test_silent_recording_still_gets_empty_srt(env):
    from rambler.engines import Result
    from rambler.pipeline import write_outputs
    cfg, a1, a2, t1, t2 = env
    tone(a1 / "quiet.wav", 2)
    rec = scan(cfg)[0]
    write_outputs(cfg, rec, Result("\n", "", "en", "large-v3", "faster-whisper", 2.0))
    assert rec.srt is not None and rec.srt.read_text() == ""
    assert scan(cfg)[0].status() == "Transcribed"


def test_mountinfo_parsing():
    from rambler import recorders as R
    info = (
        "36 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
        "512 30 8:33 / /media/anna/X8 rw,nosuid,nodev shared:300 - exfat /dev/sdc1 rw,uid=1000\n"
        "513 30 8:17 / /media/anna/TOSHIBA\\040EXT rw shared:301 - ntfs3 /dev/sdb1 rw\n"
        "514 30 8:49 / /run/media/anna/NOLABEL rw shared:302 - vfat /dev/sdd1 rw\n"
    )
    ms = R.list_mounts(info, labels={"/dev/sdc1": "X8", "/dev/sdb1": "TOSHIBA EXT"},
                       uuids={"/dev/sdc1": "6A1B-2C3D"}, use_findmnt=False)
    by = {str(m.target): m for m in ms}
    assert set(by) == {"/media/anna/X8", "/media/anna/TOSHIBA EXT", "/run/media/anna/NOLABEL"}
    assert by["/media/anna/X8"].label == "X8" and by["/media/anna/X8"].uuid == "6A1B-2C3D"
    assert by["/media/anna/X8"].source == "/dev/sdc1" and by["/media/anna/X8"].fstype == "exfat"
    rec = R.Recorder()
    assert R.match(rec, ms) is by["/media/anna/X8"]
    # label unreadable, but udisks named the folder after it
    ms2 = R.list_mounts(info, labels={}, uuids={}, use_findmnt=False)
    assert R.match(rec, ms2).target.name == "X8"


def test_automount(monkeypatch):
    from rambler import recorders as R
    info_unmounted = "36 1 259:2 / / rw - ext4 /dev/nvme0n1p2 rw\n"
    info_mounted = info_unmounted + "512 30 8:33 / /media/anna/X8 rw - fuseblk /dev/sdc1 rw\n"
    labels = {"/dev/sdc1": "X8", "/dev/sdb1": "TOSHIBA EXT"}
    rec = R.Recorder()
    assert R.unmounted_devices(rec, info_unmounted, labels, {}) == ["/dev/sdc1"]
    assert R.unmounted_devices(rec, info_mounted, labels, {}) == []
    assert R.unmounted_devices(R.Recorder(label="", uuid="AB-12"), info_unmounted, {}, {"/dev/sdc1": "ab-12"}) == ["/dev/sdc1"]

    calls = []
    monkeypatch.setattr(R, "unmounted_devices", lambda r, *a: ["/dev/sdc1"])
    monkeypatch.setattr(R.shutil, "which", lambda x: "/usr/bin/" + x)
    monkeypatch.setattr(R.os.path, "exists", lambda p: True)

    class Done:
        returncode, stdout, stderr = 0, "Mounted /dev/sdc1 at /media/anna/X8", ""
    monkeypatch.setattr(R.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or Done())
    R._no_automount.clear()
    from rambler.config import Config
    cfg = Config()
    assert R.automount(cfg, lambda m: None) == ["/dev/sdc1"]
    assert calls[-1][:3] == ["udisksctl", "mount", "-b"]
    assert R.automount(cfg, lambda m: None) == []          # only once per appearance
    monkeypatch.setattr(R.os.path, "exists", lambda p: False)   # device went away...
    monkeypatch.setattr(R, "unmounted_devices", lambda r, *a: [])
    assert R.automount(cfg, lambda m: None) == []
    monkeypatch.setattr(R.os.path, "exists", lambda p: True)    # ...and came back
    monkeypatch.setattr(R, "unmounted_devices", lambda r, *a: ["/dev/sdc1"])
    assert R.automount(cfg, lambda m: None) == ["/dev/sdc1"]
    # after we unmount it ourselves, leave it alone while it's still plugged in
    R.unmount(R.Mount(R.Path("/media/anna/X8"), "/dev/sdc1", "", "X8", "fuseblk"), lambda m: None)
    assert R.automount(cfg, lambda m: None) == []


def _journal_json(path, segs):
    import json as _j
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_j.dumps({"audio": "x.wav", "model": "large-v3", "duplicates": [], "segments": segs,
                              "text": " ".join(s["text"] for s in segs)}))


def test_transcript_edit_speakers_save(tmp_path):
    from rambler.transcript import Transcript
    j = tmp_path / "t" / "261121_0001.json"
    _journal_json(j, [
        {"start": 0.0, "end": 2.0, "text": "Hello Jordon here", "logprob": -0.2,
         "words": [{"w": " Hello", "s": 0.0, "e": 0.5, "p": 0.99}, {"w": " Jordon", "s": 0.5, "e": 1.0, "p": 0.31},
                   {"w": " here", "s": 1.0, "e": 2.0, "p": 0.97}]},
        {"start": 2.5, "end": 4.0, "text": "Second line", "logprob": -1.4, "no_speech": 0.01},
    ])
    t = Transcript.load(j, None, None)
    assert t.source == "json" and t.index_at(1.2) == 0 and t.index_at(2.2) == -1 and t.index_at(3) == 1
    assert [w.w.strip() for w in t.segments[0].uncertain_words] == ["Jordon"] and t.segments[1].shaky
    assert t.segments[0].word_at(0.7) == 1
    new = t.edit(0, "Hello Jordan here")
    assert new == ["Jordan"]
    w = t.segments[0].words[1]
    assert w.w.strip() == "Jordan" and w.p == 1.0 and (w.s, w.e) == (0.5, 1.0)   # timing kept, now trusted
    t.edit(1, "A completely different second line")       # different word count: no words before, still fine
    t.set_speaker([0], "Jordan")
    written = t.save()
    assert {p.suffix for p in written} == {".json", ".srt", ".txt"}
    t2 = Transcript.load(j, None, None)
    assert t2.segments[0].speaker == "Jordan" and t2.segments[0].edited and t2.meta["speakers"] == ["Jordan"]
    assert t2.segments[1].extra == {"no_speech": 0.01}                     # unknown keys survive
    assert (tmp_path / "t" / "261121_0001.txt").read_text().startswith("Hello Jordan here A completely")
    srt_text = (tmp_path / "t" / "261121_0001.srt").read_text()
    assert "[Jordan] Hello Jordan here" in srt_text
    t3 = Transcript.load(None, tmp_path / "t" / "261121_0001.srt", None)   # SRT-only round trip keeps speaker
    assert t3.segments[0].speaker == "Jordan" and t3.segments[0].text == "Hello Jordan here"


def test_search_index(tmp_path, env):
    from rambler.search import SearchIndex
    cfg, a1, a2, t1, t2 = env
    cfg.transcript_layout = "flat"
    tone(a1 / "2026" / "261121_0001.wav", 1)
    tone(a1 / "2026" / "261121_0002.wav", 1)
    _journal_json(t1 / "261121_0001.json", [{"start": 0, "end": 3, "text": "We fixed the Ceph cluster today",
                                              "speaker": "Alex"},
                                             {"start": 5, "end": 9, "text": "Then a walk in Brighton"}])
    (t1 / "261121_0001.txt").write_text("We fixed the Ceph cluster today Then a walk in Brighton\n")
    (t1 / "261121_0001.summary").write_text("Cluster repair and a walk.")
    (t1 / "261121_0002.txt").write_text("plain text only about cephadm\n")
    recs = scan(cfg)
    idx = SearchIndex(tmp_path / "s.db")
    assert idx.update(recs) == 2
    assert idx.update(recs) == 0                       # unchanged: nothing re-read
    hits = idx.search("ceph")
    assert {h.rel for h in hits} == {"2026/261121_0001.wav", "2026/261121_0002.wav"}   # prefix matches cephadm
    h = [h for h in hits if h.rel.endswith("0001.wav")][0]
    assert h.start == 0 and "<b>Ceph</b>" in h.snippet and h.speaker == "Alex"
    assert [x.start for x in idx.search("brighton")] == [5.0]
    assert idx.search('"walk in brighton"') and not idx.search('"brighton walk"')
    assert [x.rel for x in idx.search("", speaker="Alex")] == ["2026/261121_0001.wav"]
    assert idx.doc_info()["2026/261121_0001.wav"] == ("Cluster repair and a walk.", "Alex")
    assert idx.search('bad "quote') == idx.search("bad quote")   # never throws on odd input
    (t1 / "261121_0002.txt").unlink()
    assert idx.update(scan(cfg)) == 0 and {h.rel for h in idx.search("ceph")} == {"2026/261121_0001.wav"}


def test_peaks(tmp_path, monkeypatch):
    from rambler import peaks
    monkeypatch.setattr(peaks, "CACHE", tmp_path / "peaks")
    f = tmp_path / "a.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=300:duration=2,volume=0.5[a];anullsrc=d=1[b];[a][b]concat=v=0:a=1",
                    str(f)], check=True)
    p = peaks.compute(f)
    assert 70 <= len(p) <= 80                          # 3 s at 25/s
    assert max(p[:40]) == 255 and max(p[-15:]) == 0    # tone then silence
    assert peaks.cached(f) == p


def test_openwebui_push(tmp_path, env, monkeypatch):
    import json as _j
    import threading as _t
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from rambler import openwebui
    calls = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = _j.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            calls.append(("GET", self.path, self.headers.get("Authorization")))
            self._send(200, [{"id": "kb-uuid-1", "name": "Journal"}])

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n)
            calls.append(("POST", self.path, body))
            if self.path.startswith("/api/v1/files/"):
                self._send(200, {"id": f"file-{len([c for c in calls if 'files' in c[1]])}"})
            else:
                self._send(200, {"ok": True})

    srv = HTTPServer(("127.0.0.1", 0), H)
    _t.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(openwebui, "STATE", tmp_path / "owui.json")
    cfg, a1, a2, t1, t2 = env
    cfg.transcript_layout = "flat"
    cfg.owui_url, cfg.owui_key, cfg.owui_knowledge = f"http://127.0.0.1:{srv.server_port}", "sk-test", "kb-uuid-1"
    tone(a1 / "261121_0001.wav", 1)
    _journal_json(t1 / "261121_0001.json", [{"start": 0, "end": 2, "text": "Hello there", "speaker": "Alex"}])
    (t1 / "261121_0001.txt").write_text("Hello there\n")
    (t1 / "261121_0001.summary").write_text("A greeting.")
    c = openwebui.Client(cfg.owui_url, cfg.owui_key)
    assert c.knowledge_bases() == [("kb-uuid-1", "Journal")] and calls[0][2] == "Bearer sk-test"
    recs = scan(cfg)
    assert openwebui.push(cfg, recs, log=lambda m: None) == (1, 0)
    up = [c for c in calls if c[1].startswith("/api/v1/files/")][0][2].decode()
    assert "Summary: A greeting." in up and "Alex:" in up and "Hello there" in up and 'filename="261121_0001.txt"' in up
    assert ("POST", "/api/v1/knowledge/kb-uuid-1/file/add", b'{"file_id": "file-1"}') in calls
    assert openwebui.push(cfg, recs, log=lambda m: None) == (0, 0)          # unchanged: not re-sent
    (t1 / "261121_0001.summary").write_text("A different greeting.")
    assert openwebui.push(cfg, scan(cfg), log=lambda m: None) == (1, 0)    # changed: new file, old removed
    assert ("POST", "/api/v1/knowledge/kb-uuid-1/file/remove", b'{"file_id": "file-1"}') in calls
    srv.shutdown()

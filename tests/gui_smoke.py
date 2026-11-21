"""Headless GUI smoke test: scan -> transcribe (fake engine) -> sync -> player with subtitles -> screenshots."""
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
tmp = Path(tempfile.mkdtemp(prefix="rambler-gui-"))
os.environ["XDG_CONFIG_HOME"] = str(tmp / "config")
os.environ["XDG_DATA_HOME"] = str(tmp / "data")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from rambler.config import Config  # noqa: E402

FAKE = Path(__file__).with_name("fake_whisper_cli.py")
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")


def tone(p, secs):
    p.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"sine=frequency=330:duration={secs}", str(p)], check=True)


dirs = {n: tmp / n for n in ("Drive A/Recordings", "Drive B/Recordings",
                             "Drive A/Transcripts", "Drive B/Transcripts")}
for d in dirs.values():
    d.mkdir(parents=True)
A, B, T1, T2 = dirs.values()
for i, name in enumerate(["261114_0001.wav", "261114_0002.wav", "261115_0001.wav", "2026-11-10 morning.mp3"]):
    tone((A if i % 2 == 0 else B) / name, 6 + i * 3)
(tmp / "models").mkdir()
(tmp / "models" / "ggml-large-v3.bin").write_bytes(b"x")
(tmp / "models" / "ggml-large-v3-turbo.bin").write_bytes(b"x")
Config(audio_dirs=[str(A), str(B), "/media/user/Drive C/Recordings"],
       transcript_dirs=[str(T1), str(T2)], whisper_cpp_bin=str(FAKE), models_dir=str(tmp / "models"),
       confirm_sync_plan=False).save()

app = QApplication(sys.argv)
from rambler.gui.main_window import MainWindow  # noqa: E402

w = MainWindow()
w.resize(1400, 860)
w.show()


def wait_idle(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(QEventLoop.AllEvents, 50)
        if not w.busy() and time.time() - t0 > 0.5:
            # allow chained jobs to start
            for _ in range(10):
                app.processEvents(QEventLoop.AllEvents, 20)
                time.sleep(0.02)
            if not w.busy():
                return
        time.sleep(0.02)
    raise SystemExit("timeout")


wait_idle()
print("scanned:", [(r.name, r.status(), len(r.locations)) for r in w.model.recs])
w.grab().save(str(out_dir / "rambler-1-before.png"))

pending = [r for r in w.model.recs if not r.transcribed]
w.transcribe(pending, want_srt=True, then=lambda: w.start_sync(confirm=False))
# capture mid-job
for _ in range(40):
    app.processEvents(QEventLoop.AllEvents, 20)
    time.sleep(0.02)
w.grab().save(str(out_dir / "rambler-2-working.png"))
wait_idle()
wait_idle()
print("after:", [(r.name, r.status(), len(r.locations)) for r in w.model.recs])
assert all(r.srt for r in w.model.recs), "missing srt"
assert all(len(r.locations) == 2 for r in w.model.recs), "not synced"
assert sorted(p.name for p in T2.iterdir()) == sorted(p.name for p in T1.iterdir())

w.table.selectRow(1)
app.processEvents()
w.player.player.play()
t0 = time.time()
while time.time() - t0 < 4:
    app.processEvents(QEventLoop.AllEvents, 50)
    time.sleep(0.02)
pos = w.player.player.position()
print("player position ms:", pos, "state:", w.player.player.playbackState(), "error:", w.player.player.errorString())
# force a subtitle on screen regardless of audio output availability in the sandbox
w.player._on_position(4500)
w.player._on_duration(12000) if w.player.player.duration() == 0 else None
app.processEvents()
w.grab().save(str(out_dir / "rambler-3-player.png"))
print("subtitle shown:", w.player.screen.sub.text())
print("log:\n" + w.logview.toPlainText())
# ---- TASCAM X8 plugged in as a card reader
from rambler import recorders as R  # noqa: E402
card = tmp / "media" / "X8"
tone(card / "SOUND" / "261121_0001.wav", 5)
(card / "UTILITY").mkdir(parents=True)
unmounted = []
R.LEDGER = tmp / "ledger.json"
R.list_mounts = lambda: [R.Mount(card, "/dev/sdz1", "1234-ABCD", "X8", "exfat")]
R.unmount = lambda m, log: (unmounted.append(m), log("unmounted (test)"), True)[-1]
w._poll_drives()
chip_texts = lambda: [w.drive_bar.itemAt(i).widget().text() for i in range(w.drive_bar.count())
                      if w.drive_bar.itemAt(i).widget()]
print("chips while working:", [c for c in chip_texts() if "TASCAM" in c])
for _ in range(6):
    wait_idle()
names = {r.rel.as_posix(): r for r in w.model.recs}
x8 = names.get("2026/261121_0001.wav")
print("x8:", x8 and (x8.status(), len(x8.locations)), "unmounted:", bool(unmounted))
assert x8 and x8.srt and len(x8.locations) == 2 and unmounted
assert (B / "2026" / "261121_0001.wav").exists() and (T2 / "2026" / "261121_0001.srt").exists()
w.grab().save(str(out_dir / "rambler-4-x8.png"))
for _ in range(3):
    app.processEvents()
print("chips after:", [c for c in chip_texts() if "TASCAM" in c])
assert any("safe to switch back" in c for c in chip_texts())
w._poll_drives()   # still mounted: must not re-trigger
assert not w.busy()
print("log tail:\n" + "\n".join(w.logview.toPlainText().splitlines()[-14:]))

from rambler.gui.settings import SettingsDialog  # noqa: E402
from PySide6.QtWidgets import QTabWidget  # noqa: E402
d = SettingsDialog(w.cfg, w)
d.findChild(QTabWidget).setCurrentIndex(4)
d.show()
app.processEvents()
d.grab().save(str(out_dir / "rambler-5-recorders.png"))
assert d.collect().recorders[0]["label"] == "X8"
tabs = d.findChild(QTabWidget)
tabs.setCurrentIndex(1)
for eng in ("whisper.cpp", "faster-whisper"):
    d.engine.setCurrentIndex(d.engine.findData(eng))
    app.processEvents()
    assert d._tf.isRowVisible(d.hotwords) == (eng == "faster-whisper")
    assert d._tf.isRowVisible(d.extra) == (eng == "whisper.cpp")
    d.grab().save(str(out_dir / f"rambler-6-settings-{eng}.png"))
d.close()
w.close()

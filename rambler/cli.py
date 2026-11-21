"""Command-line entry point. `rambler` opens the GUI; subcommands run headless."""
from __future__ import annotations

import argparse
import sys
import threading

from . import __version__
from .config import CONFIG_PATH, Config


def _bar(frac: float, width: int = 30) -> str:
    n = int(frac * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {frac * 100:5.1f}%"


def cmd_status(cfg: Config, args) -> int:
    from .engines import make_engine
    from .library import scan
    print(f"Config: {CONFIG_PATH}")
    for label, dirs, avail in (("Audio", cfg.audio_dirs, cfg.audio_roots()),
                               ("Transcripts", cfg.transcript_dirs, cfg.transcript_roots())):
        for d in dirs:
            ok = Config.expand(d) in avail
            print(f"  {label:12} {'✓' if ok else '✗ not connected'}  {d}")
    ok, msg = make_engine(cfg).available()
    print(f"Engine: {msg}")
    recs = scan(cfg)
    pending = [r for r in recs if r.pending]
    nosrt = [r for r in recs if r.transcribed and not r.srt]
    print(f"{len(recs)} recordings · {len(pending)} to be transcribed · {len(nosrt)} without SRT")
    if args.verbose:
        for r in pending:
            print(f"  pending: {r.rel}")
    return 0


def cmd_sync(cfg: Config, args) -> int:
    from .syncer import build_plan, execute
    plan = build_plan(cfg)
    print(plan.summary())
    if args.dry_run or not plan.actions:
        return 0
    if plan.space_problems and not args.force:
        print("Not enough free space; use --force to try anyway.")
        return 2
    errs = execute(plan, cfg, lambda d, t, f: print(f"\r{_bar(d / t if t else 1)} {f[-60:]:60}", end="", flush=True))
    print()
    for e in errs:
        print("ERROR", e, file=sys.stderr)
    return 1 if errs else 0


def cmd_transcribe(cfg: Config, args) -> int:
    from .engines import Cancelled, EngineError, make_engine
    from .library import scan
    from .pipeline import transcribe_one
    engine = make_engine(cfg)
    ok, msg = engine.available()
    if not ok:
        print(msg, file=sys.stderr)
        return 2
    want_srt = cfg.generate_srt if args.srt is None else args.srt
    recs = scan(cfg)
    todo = [r for r in recs if r.pending or (want_srt and args.backfill_srt and not r.srt)]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{msg} · {len(todo)} to transcribe" + (" (+SRT)" if want_srt else ""))
    from .pipeline import run_post_hook
    cancel = threading.Event()
    failures = 0
    try:
        for i, r in enumerate(todo, 1):
            try:
                transcribe_one(cfg, engine, r, want_srt,
                               lambda f, m, r=r, i=i: print(f"\r({i}/{len(todo)}) {_bar(f)} {r.name[:40]:40}", end="", flush=True),
                               cancel)
                print()
            except KeyboardInterrupt:
                cancel.set()
                print("\nCancelled.")
                return 130
            except (EngineError, Cancelled, OSError, RuntimeError) as e:
                failures += 1
                print(f"\nFAILED {r.rel}: {e}", file=sys.stderr)
    finally:
        engine.close()
    if len(todo) > failures:
        run_post_hook(cfg, print)
    if args.sync:
        cmd_sync(cfg, argparse.Namespace(dry_run=False, force=False))
    return 1 if failures else 0


def cmd_setup(cfg: Config, args) -> int:
    from . import setup_engine as se
    st = se.status(cfg)
    if st.ready and not args.force:
        print(f"Engine already set up: {st.whisper_cli} · models: {', '.join(st.models)}")
        print("Use --force to rebuild / fetch more models.")
        return 0
    backend = args.backend or se.default_backend()
    models = args.models.split(",") if args.models else ["large-v3-turbo", "large-v3"]
    pkgs = se.missing_packages(backend)
    print(f"Backend: {backend} · models: {', '.join(models)}")
    if pkgs and not args.no_apt:
        print(f"Will install with sudo apt: {' '.join(pkgs)}")
    cancel = threading.Event()
    try:
        cfg = se.run_setup(cfg, backend, models, lambda m: print("\r\033[K" + m),
                           lambda f, m: print(f"\r{_bar(f)} {m[:50]:50}", end="", flush=True),
                           cancel, install_apt=not args.no_apt, ref=args.ref)
    except KeyboardInterrupt:
        cancel.set()
        print("\nCancelled. Run again to resume.")
        return 130
    except (se.SetupError, OSError) as e:
        print(f"\nSetup failed: {e}", file=sys.stderr)
        return 1
    cfg.save()
    print("\nDone. " + make_engine_msg(cfg))
    return 0


def _notify(body: str) -> None:
    import shutil
    import subprocess
    if shutil.which("notify-send"):
        subprocess.Popen(["notify-send", "-a", "Rambler", "-i", "audio-input-microphone", "Rambler", body])


def run_cycle(cfg: Config, rec=None, mount=None, want_srt: bool | None = None) -> int:
    """Import from a recorder (if given), transcribe everything pending, sync, unmount. Used by import/watch."""
    from . import recorders as R
    from .engines import EngineError, Cancelled, make_engine
    from .library import scan
    from .pipeline import transcribe_one
    from .syncer import build_plan, execute
    cancel = threading.Event()
    with R.pipeline_lock() as got:
        if not got:
            print("Another Rambler is busy; skipping this run.")
            return 0
        imported = []
        if rec is not None:
            print(f"== Importing from {rec.name} ({mount.target})")
            imported = R.import_recorder(cfg, rec, mount, print,
                                         lambda d, t, s: print(f"\r{_bar(d / t if t else 1)} {s[-50:]:50}", end="", flush=True),
                                         cancel)
            print(f"\n{len(imported)} new recording(s)")
        engine = make_engine(cfg)
        ok, msg = engine.available()
        want = cfg.generate_srt if want_srt is None else want_srt
        failures = 0
        done = 0
        if ok:
            from .pipeline import run_post_hook
            todo = [r for r in scan(cfg) if r.pending]
            try:
                for i, r in enumerate(todo, 1):
                    print(f"== Transcribing ({i}/{len(todo)}) {r.rel}")
                    try:
                        transcribe_one(cfg, engine, r, want,
                                       lambda f, m: print(f"\r{_bar(f)} {m:20}", end="", flush=True), cancel)
                        print()
                        done += 1
                    except (EngineError, Cancelled, OSError, RuntimeError) as e:
                        failures += 1
                        print(f"\nFAILED {r.rel}: {e}", file=sys.stderr)
            finally:
                engine.close()
            if done:
                run_post_hook(cfg, print)
        else:
            print(f"Not transcribing: {msg}")
        plan = build_plan(cfg)
        print("== Sync\n" + plan.summary())
        errs = execute(plan, cfg) if plan.actions and not plan.space_problems else []
        for e in errs:
            print("ERROR", e, file=sys.stderr)
        if rec is not None and rec.unmount_after:
            R.unmount(mount, print)
        if rec is not None or done:
            _notify(f"{len(imported)} imported, {done} transcribed, {len(plan.actions)} file(s) synced"
                    + (". Card unmounted, safe to switch the recorder back." if rec is not None and rec.unmount_after else "."))
        return 1 if (failures or errs) else 0


def cmd_import(cfg: Config, args) -> int:
    from pathlib import Path
    from . import recorders as R
    if args.source:
        # treat a folder as the card (no mount detection): handy for testing or odd mounts
        recs = R.recorders(cfg)
        rec = next((r for r in recs if r.name == args.recorder), recs[0] if recs else R.Recorder())
        conns = [(rec, R.Mount(Path(args.source), "", "", rec.label, ""))]
    else:
        conns = R.connected(cfg)
    if args.dry_run:
        for rec, m in conns:
            root = m.target / rec.card_subdir if rec.card_subdir else m.target
            todo = R.pending(rec, m, cfg)
            print(f"{rec.name} at {root}: {len(todo)} recording(s) to import")
            dest = cfg.audio_roots()[0] if cfg.audio_roots() else Path("<no recording folder connected>")
            for _, rel, st in todo:
                print(f"  {rel}  ({st.st_size / 1e6:.0f} MB) -> {dest / R.dest_rel(rec, rel, st)}")
        if not conns:
            print("No configured recorder is mounted.")
        return 0
    if not conns:
        print("No configured recorder is mounted. Configured: " + ", ".join(
            f"{r.name} (label {r.label!r})" for r in R.recorders(cfg)))
        for m in R.list_mounts():
            print(f"  mounted: {m.target}  label={m.label!r} uuid={m.uuid}")
        return 1
    rc = 0
    for rec, m in conns:
        rc |= run_cycle(cfg, rec, m)
    return rc


def cmd_recorders(cfg: Config, args) -> int:
    """Show what Rambler sees: removable mounts and which configured recorder each one matches."""
    from . import recorders as R
    mounts = R.list_mounts()
    print("Removable volumes mounted:" if mounts else "No removable volumes mounted under /media or /run/media.")
    for m in mounts:
        hit = next((r.name for r in R.recorders(cfg) if R.match(r, [m]) is m), None)
        print(f"  {m.target}  label={m.label or '-'}  uuid={m.uuid or '-'}  {m.fstype}  {m.source}"
              + (f"  -> {hit}" if hit else ""))
    for r in R.recorders(cfg):
        for dev in R.unmounted_devices(r):
            print(f"  {dev}  label={r.label}  NOT MOUNTED -> {r.name}"
                  + ("  (Rambler will mount it)" if r.auto_mount else "  (auto-mount is off)"))
    print("Configured recorders:")
    for r in R.recorders(cfg):
        m = R.match(r, mounts)
        where = f"connected at {m.target}" if m else "not connected"
        print(f"  {r.name}: label={r.label or '-'} uuid={r.uuid or '-'} folder={r.card_subdir or '(whole card)'} "
              f"-> {where}{'' if r.enabled else ' (disabled)'}")
        if m:
            print(f"    {len(R.pending(r, m, cfg))} recording(s) waiting to import")
    return 0


def cmd_search(cfg: Config, args) -> int:
    import re as _re
    from .library import scan
    from .search import SearchIndex
    idx = SearchIndex()
    n = idx.update(scan(cfg), lambda d, t: print(f"\rIndexing {d}/{t}", end="", flush=True))
    if n:
        print()
    hits = idx.search(" ".join(args.query), args.speaker or "", args.limit)
    for h in hits:
        m, sec = divmod(int(h.start), 60)
        hh, m = divmod(m, 60)
        stamp = f"{hh}:{m:02d}:{sec:02d}" if hh else f"{m}:{sec:02d}"
        text = _re.sub(r"</?b>", "", h.snippet)
        print(f"{h.rel}  {stamp}  {('[' + h.speaker + '] ') if h.speaker else ''}{text}")
    print(f"{len(hits)} match(es)")
    return 0 if hits else 1


def cmd_push_owui(cfg: Config, args) -> int:
    from . import openwebui
    from .library import scan
    if not (cfg.owui_url and cfg.owui_key and cfg.owui_knowledge):
        print("Open WebUI isn't configured (Settings › Open WebUI).", file=sys.stderr)
        return 2
    sent, failed = openwebui.push(cfg, [r for r in scan(cfg) if r.txt], print)
    print(f"{sent} sent, {failed} failed")
    return 1 if failed else 0


def cmd_watch(cfg: Config, args) -> int:
    """Stay running; act when a recorder card or a configured drive appears."""
    import time
    from . import recorders as R
    print("Watching for recorders and drives. Ctrl+C to stop.")
    handled: set[str] = set()
    last_avail = None
    try:
        while True:
            cfg = Config.load()
            R.automount(cfg, print)
            mounts = R.list_mounts()
            present = set()
            for rec, m in R.connected(cfg, mounts):
                key = f"{rec.name}|{m.source}|{m.target}"
                present.add(key)
                if key not in handled and rec.auto_import:
                    handled.add(key)
                    run_cycle(cfg, rec, m)
            handled &= present
            avail = tuple(Config.expand(d).is_dir() for d in cfg.audio_dirs + cfg.transcript_dirs)
            if last_avail is not None and avail != last_avail and cfg.auto_on_drive_connect and \
                    any(a and not b for a, b in zip(avail, last_avail)):
                run_cycle(cfg)
            last_avail = avail
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


def make_engine_msg(cfg: Config) -> str:
    from .engines import make_engine
    return make_engine(cfg).available()[1]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] not in ("status", "sync", "transcribe", "setup", "import", "watch", "recorders",
                                     "search", "push-owui") and not argv[0].startswith("-"):
        from .gui.app import run          # `rambler file1.mp3 ...` (e.g. "Open with Rambler")
        return run(argv)
    p = argparse.ArgumentParser(prog="rambler", description="Transcribe and sync audio recordings.")
    p.add_argument("--version", action="version", version=f"rambler {__version__}")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("status", help="show folders, engine and pending recordings")
    s.add_argument("-v", "--verbose", action="store_true")
    s = sub.add_parser("sync", help="sync audio and transcript folders")
    s.add_argument("-n", "--dry-run", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("transcribe", help="transcribe everything pending")
    s.add_argument("--srt", dest="srt", action="store_true", default=None)
    s.add_argument("--no-srt", dest="srt", action="store_false")
    s.add_argument("--backfill-srt", action="store_true", help="also redo transcribed files that have no SRT")
    s.add_argument("--sync", action="store_true", help="sync folders afterwards")
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("setup", help="install build deps, build whisper.cpp, download models")
    s.add_argument("--backend", choices=["faster-whisper", "cuda", "vulkan", "cpu"],
                   help="faster-whisper (pip, default on NVIDIA) or a whisper.cpp build: cuda, vulkan, cpu")
    s.add_argument("--models", help="comma-separated, default: large-v3-turbo,large-v3")
    s.add_argument("--no-packages", "--no-apt", dest="no_apt", action="store_true",
                   help="don't install system packages (apt/dnf/pacman)")
    s.add_argument("--ref", default="", help="whisper.cpp tag/commit to pin")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("import", help="import from any connected recorder, transcribe, sync, unmount")
    s.add_argument("-n", "--dry-run", action="store_true", help="only list what would be imported")
    s.add_argument("--from", dest="source", help="use this folder as the recorder's card instead of detecting it")
    s.add_argument("--recorder", default="", help="which configured recorder --from refers to (default: first)")
    s = sub.add_parser("search", help="search every transcript")
    s.add_argument("query", nargs="*")
    s.add_argument("--speaker", default="")
    s.add_argument("--limit", type=int, default=50)
    sub.add_parser("push-owui", help="send new/changed transcripts to the configured Open WebUI knowledge base")
    sub.add_parser("recorders", help="show detected cards/volumes and which recorder they match")
    s = sub.add_parser("watch", help="run in the background: auto-import recorders, auto-sync drives")
    s.add_argument("--interval", type=float, default=3.0)
    p.add_argument("--minimized", action="store_true", help="start the GUI minimized")
    args = p.parse_args(argv)
    cfg = Config.load()
    if args.cmd == "status":
        return cmd_status(cfg, args)
    if args.cmd == "sync":
        return cmd_sync(cfg, args)
    if args.cmd == "transcribe":
        return cmd_transcribe(cfg, args)
    if args.cmd == "setup":
        return cmd_setup(cfg, args)
    if args.cmd == "import":
        return cmd_import(cfg, args)
    if args.cmd == "watch":
        return cmd_watch(cfg, args)
    if args.cmd == "recorders":
        return cmd_recorders(cfg, args)
    if args.cmd == "search":
        return cmd_search(cfg, args)
    if args.cmd == "push-owui":
        return cmd_push_owui(cfg, args)
    from .gui.app import run
    return run([], minimized=args.minimized)


if __name__ == "__main__":
    sys.exit(main())

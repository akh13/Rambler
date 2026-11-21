#!/usr/bin/env bash
# Install Rambler for the current user: venv in ~/.local/share/rambler/venv, `rambler` on PATH,
# and a desktop entry so it shows up in the Pop!_OS / GNOME launcher.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
VENV="$DATA/rambler/venv"
BIN="$HOME/.local/bin"

command -v python3 >/dev/null || { echo "python3 missing"; exit 1; }
python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { echo "Rambler needs Python 3.10 or newer."; exit 1; }
if ! python3 -c 'import venv, ensurepip' 2>/dev/null; then
  if command -v apt-get >/dev/null; then echo "Run: sudo apt install python3-venv"
  else echo "Python's venv module is missing; install your distro's python3 venv package."; fi
  exit 1
fi

echo "==> Creating venv at $VENV"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "$HERE"
if [[ "${WITH_FASTER_WHISPER:-0}" == 1 ]]; then
  "$VENV/bin/pip" install --quiet faster-whisper
fi

mkdir -p "$BIN" "$DATA/applications"
ln -sf "$VENV/bin/rambler" "$BIN/rambler"
sed "s|@BIN@|$VENV/bin/rambler|" "$HERE/packaging/io.github.akh13.Rambler.desktop" > "$DATA/applications/io.github.akh13.Rambler.desktop"
update-desktop-database "$DATA/applications" 2>/dev/null || true

echo
echo "Installed. Launch 'Rambler' from the app launcher, or run: rambler"
echo "Headless: rambler status | rambler transcribe --sync | rambler sync --dry-run"
if [[ "${SKIP_ENGINE:-0}" != 1 ]]; then
  echo
  echo "==> Setting up the transcription engine (whisper.cpp + models). SKIP_ENGINE=1 to skip;"
  echo "    you can also do this later from the app or with: rambler setup"
  "$VENV/bin/rambler" setup ${RAMBLER_SETUP_ARGS:-}
fi

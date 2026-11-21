#!/usr/bin/env bash
# Fresh-machine test: install Rambler, build the engine (CPU, small model), transcribe real speech.
# Run by CI on clean Ubuntu, Fedora and Arch. Safe to run locally too (uses a throwaway HOME).
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${RAMBLER_E2E_KEEP_HOME:-0}" != 1 ]]; then
  export HOME="$(mktemp -d)"
  unset XDG_CONFIG_HOME XDG_DATA_HOME XDG_CACHE_HOME
fi
echo "==> HOME=$HOME"

RAMBLER_SETUP_ARGS="--backend cpu --models base.en" ./install.sh
R="$HOME/.local/bin/rambler"
"$R" --version

W="$(mktemp -d)"
mkdir -p "$W/audio/2026" "$W/transcripts" "$W/audio-copy"
cp "$HOME/.local/share/rambler/whisper.cpp/samples/jfk.wav" "$W/audio/2026/jfk.wav"
mkdir -p "$HOME/.config/rambler"
cat > "$HOME/.config/rambler/config.json" <<EOF
{
  "audio_dirs": ["$W/audio", "$W/audio-copy"],
  "transcript_dirs": ["$W/transcripts"],
  "engine": "whisper.cpp",
  "language": "en",
  "threads": 2
}
EOF

"$R" status -v
"$R" transcribe --sync
echo "==> transcript:"
cat "$W/transcripts/2026/jfk.txt"
grep -qi "country" "$W/transcripts/2026/jfk.txt"
test -s "$W/transcripts/2026/jfk.srt"
test -f "$W/transcripts/2026/jfk.json"
test -f "$W/audio-copy/2026/jfk.wav"          # synced to the second folder
"$R" search country | tee /dev/stderr | grep -q "jfk.wav"
"$R" status | grep -q "0 to be transcribed"
echo "==> install + transcribe + sync + search OK"

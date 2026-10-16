#!/usr/bin/env bash
# Build the macOS VoiceProcessingIO capture helper.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/bin"
swiftc -O "$ROOT/scripts/voice-mic.swift" -o "$ROOT/bin/voice-mic"
echo "Built $ROOT/bin/voice-mic"

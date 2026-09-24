#!/usr/bin/env bash
# Vendors and builds audiotee (github.com/makeusabrew/audiotee), a Swift CLI
# that captures macOS system audio via a Core Audio process tap and writes
# raw PCM to stdout. Pinned to the commit six-ddc/livecaption uses.
#
# Idempotent: safe to re-run. Clones into .vendor/audiotee if absent, fetches
# and checks out the pinned commit, builds in release mode, and copies the
# resulting binary to bin/audiotee.
set -euo pipefail

REPO_URL="https://github.com/makeusabrew/audiotee.git"
PINNED_COMMIT="56ac954369a09318e46b88a6eec33c2d2b0d32a3"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/.vendor/audiotee"
BIN_DIR="${ROOT_DIR}/bin"

if [ ! -d "${VENDOR_DIR}/.git" ]; then
    echo "Cloning audiotee into ${VENDOR_DIR}..."
    mkdir -p "$(dirname "${VENDOR_DIR}")"
    git clone "${REPO_URL}" "${VENDOR_DIR}"
fi

cd "${VENDOR_DIR}"

echo "Fetching latest refs..."
git fetch origin

if ! git cat-file -e "${PINNED_COMMIT}^{commit}" 2>/dev/null; then
    echo "ERROR: pinned commit ${PINNED_COMMIT} not found in ${VENDOR_DIR} after fetch." >&2
    exit 1
fi

echo "Checking out pinned commit ${PINNED_COMMIT}..."
git checkout "${PINNED_COMMIT}"

echo "Building audiotee (release)..."
swift build -c release

PRODUCT_PATH="$(swift build -c release --show-bin-path)/audiotee"
if [ ! -f "${PRODUCT_PATH}" ]; then
    echo "ERROR: expected build product not found at ${PRODUCT_PATH}" >&2
    exit 1
fi

mkdir -p "${BIN_DIR}"
cp "${PRODUCT_PATH}" "${BIN_DIR}/audiotee"
chmod +x "${BIN_DIR}/audiotee"

echo "audiotee built: ${BIN_DIR}/audiotee"

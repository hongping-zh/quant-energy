#!/usr/bin/env bash
# EcoCompute one-liner wrapper — secure stub
#
# This file is a thin stub that downloads the real run.sh from a pinned,
# immutable GitHub release, verifies its checksum, then executes it.
#
# Security model:
#   - The actual logic lives in release run-v1 (immutable, cannot be changed)
#   - This stub always downloads from that pinned release URL
#   - Checksum verification prevents tampering
#   - DNS hijack of quantenergy.tech cannot change what executes
#
# Audit this stub: it's ~20 lines below. The real run.sh is at:
#   https://github.com/hongping-zh/ecocompute-mlcube/releases/tag/run-v1
set -euo pipefail

RELEASE_URL="https://github.com/hongping-zh/ecocompute-mlcube/releases/download/run-v1"
CHECKSUM="c56a7a6e30b999f631aeea6bb074bd12d6a376daf0e6023b931ba1b86234785e"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "EcoCompute one-liner: downloading pinned run.sh from release run-v1 ..."
curl -fsSL -o "$TMPDIR/run.sh" "$RELEASE_URL/run.sh" || {
  echo "[error] failed to download run.sh from $RELEASE_URL" >&2
  exit 1
}

echo "Verifying checksum ..."
ACTUAL=$(sha256sum "$TMPDIR/run.sh" | cut -d' ' -f1)
if [ "$ACTUAL" != "$CHECKSUM" ]; then
  echo "[error] checksum mismatch!" >&2
  echo "  expected: $CHECKSUM" >&2
  echo "  actual:   $ACTUAL" >&2
  echo "  This should not happen. The release may have been tampered with." >&2
  exit 1
fi

echo "Checksum OK. Executing run.sh with your arguments: $*"
echo ""

exec bash "$TMPDIR/run.sh" "$@"

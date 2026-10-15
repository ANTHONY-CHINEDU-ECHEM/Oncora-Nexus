#!/usr/bin/env bash
# Assemble a Hugging Face Space folder from this repository.
#   bash deploy/hf_space/prepare_space.sh ../oncora-nexus-space
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:?usage: prepare_space.sh <space-dir>}"
mkdir -p "$DEST"
rsync -a --exclude node_modules --exclude dist --exclude 'data/*' --exclude .git "$SRC/" "$DEST/"
cp "$SRC/deploy/hf_space/Dockerfile" "$DEST/Dockerfile"
cp "$SRC/deploy/hf_space/README.md" "$DEST/README.md"
echo "Space ready in $DEST. Commit and push it to your Space repository."

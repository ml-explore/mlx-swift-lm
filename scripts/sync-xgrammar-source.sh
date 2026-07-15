#!/usr/bin/env bash
#
# Refreshes Libraries/MLXCXGrammar/xgrammar/ from a pinned upstream xgrammar
# revision. Run manually when bumping the pinned sha; NOT invoked by
# swift build. The produced source tree is committed to the repo.
#
# Usage:
#     scripts/sync-xgrammar-source.sh <sha-or-tag> [source-dir]
#
# source-dir defaults to ~/src/xgrammar. If the directory isn't a git
# checkout of https://github.com/mlc-ai/xgrammar, the script aborts.
#
# The script rsyncs only the subtrees SPM needs to compile xgrammar:
#   - cpp/**           (minus cpp/tvm_ffi/; Python bindings)
#   - include/xgrammar/
#   - 3rdparty/picojson/picojson.h
#   - 3rdparty/dlpack/include/dlpack/dlpack.h
#   - LICENSE, NOTICE
#
# The 3rdparty/dlpack/ submodule is auto-initialized if missing.
#
# After syncing, the pinned sha is written to Libraries/MLXCXGrammar/xgrammar/VERSION
# so reviewers can see at a glance which upstream commit is vendored.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <sha-or-tag> [source-dir]" >&2
    exit 64
fi

REQUESTED_REV="$1"
SOURCE_DIR="${2:-$HOME/src/xgrammar}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST_ROOT="$REPO_ROOT/Libraries/MLXCXGrammar/xgrammar"

if [[ ! -d "$SOURCE_DIR/.git" ]]; then
    echo "error: $SOURCE_DIR is not a git checkout." >&2
    echo "       clone https://github.com/mlc-ai/xgrammar.git to $SOURCE_DIR first." >&2
    exit 1
fi

remote_url="$(git -C "$SOURCE_DIR" config --get remote.origin.url || true)"
case "$remote_url" in
    *xgrammar*) ;;
    *)
        echo "error: $SOURCE_DIR remote.origin.url=$remote_url does not look like xgrammar." >&2
        exit 1
        ;;
esac

echo "==> Checking out $REQUESTED_REV in $SOURCE_DIR"
git -C "$SOURCE_DIR" fetch --tags origin >/dev/null
git -C "$SOURCE_DIR" checkout --quiet "$REQUESTED_REV"
RESOLVED_SHA="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
echo "    resolved to $RESOLVED_SHA"

if [[ ! -f "$SOURCE_DIR/3rdparty/dlpack/include/dlpack/dlpack.h" ]]; then
    echo "==> Initializing 3rdparty/dlpack submodule"
    git -C "$SOURCE_DIR" submodule update --init 3rdparty/dlpack >/dev/null
fi

echo "==> Clearing $DEST_ROOT"
rm -rf "$DEST_ROOT"
mkdir -p "$DEST_ROOT"

echo "==> Copying cpp/ (excluding tvm_ffi/)"
# Exclude patterns must precede includes so --include='*/' doesn't pull the
# tvm_ffi/ directory back in before the exclude can reject it.
rsync -a \
    --exclude='tvm_ffi/' \
    --exclude='nanobind/' \
    --include='*/' \
    --include='*.cc' \
    --include='*.h' \
    --exclude='*' \
    "$SOURCE_DIR/cpp/" "$DEST_ROOT/cpp/"

echo "==> Copying include/xgrammar/"
mkdir -p "$DEST_ROOT/include/xgrammar"
rsync -a "$SOURCE_DIR/include/xgrammar/" "$DEST_ROOT/include/xgrammar/"

echo "==> Copying 3rdparty/picojson/picojson.h"
mkdir -p "$DEST_ROOT/3rdparty/picojson"
cp "$SOURCE_DIR/3rdparty/picojson/picojson.h" "$DEST_ROOT/3rdparty/picojson/"

echo "==> Copying 3rdparty/dlpack/include/dlpack/dlpack.h"
mkdir -p "$DEST_ROOT/3rdparty/dlpack/include/dlpack"
cp "$SOURCE_DIR/3rdparty/dlpack/include/dlpack/dlpack.h" \
   "$DEST_ROOT/3rdparty/dlpack/include/dlpack/"

echo "==> Copying LICENSE, NOTICE"
cp "$SOURCE_DIR/LICENSE" "$DEST_ROOT/LICENSE"
cp "$SOURCE_DIR/NOTICE" "$DEST_ROOT/NOTICE"

echo "==> Writing VERSION"
cat > "$DEST_ROOT/VERSION" <<EOF
$REQUESTED_REV

Pinned to the upstream revision $REQUESTED_REV
(resolved SHA $RESOLVED_SHA, informational).

This directory is a vendored snapshot of https://github.com/mlc-ai/xgrammar.
Refresh with: scripts/sync-xgrammar-source.sh <sha-or-tag>

Do not edit files under this directory by hand -- changes will be overwritten
at the next sync. Patches against upstream belong upstream.
EOF

cc_count="$(find "$DEST_ROOT/cpp" -name '*.cc' | wc -l | tr -d ' ')"
h_count="$(find "$DEST_ROOT/cpp" -name '*.h' | wc -l | tr -d ' ')"
echo
echo "Synced $cc_count .cc + $h_count .h files under cpp/"
echo "Pinned to $RESOLVED_SHA"

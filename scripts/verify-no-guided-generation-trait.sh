#!/usr/bin/env bash
#
# Guards against any lingering GuidedGenerationSupport references after the
# trait's removal. The superpowers design/plan docs are allowed to keep
# references (they document the change and prior work), as is this script
# itself (it names the token in its grep pattern).
set -euo pipefail

cd "$(dirname "$0")/.."

matches=$(rg -n "GuidedGenerationSupport" \
    --glob '!.build/**' \
    --glob '!docs/superpowers/**' \
    --glob '!scripts/verify-no-guided-generation-trait.sh' \
    || true)

if [[ -n "$matches" ]]; then
    echo "FAIL: GuidedGenerationSupport references remain:"
    echo "$matches"
    exit 1
fi

echo "PASS: no GuidedGenerationSupport references outside allowed paths"

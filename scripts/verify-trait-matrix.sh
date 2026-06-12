#!/usr/bin/env bash
#
# Builds every trait combination of the package. The FM-off / GG-off arms are
# the regression-prone ones after the MLXGuidedGeneration split: a moved type
# referenced outside the GuidedGenerationSupport gate in MLXFoundationModels
# would break here.
#
# Note: this repo's local dev environment requires --disable-sandbox and
# -Xswiftc -disable-sandbox (the Swift macro plugin server cannot be sandboxed
# here) plus --skip-update (network-restricted). CI uses plain `swift build`.
#
set -euo pipefail

FLAGS=(--disable-sandbox --skip-update -Xswiftc -disable-sandbox)

echo "==> [1/4] both traits ON (defaults)"
swift build "${FLAGS[@]}"

echo "==> [2/4] FM on, GG off"
swift build "${FLAGS[@]}" --traits FoundationModelsIntegration

echo "==> [3/4] FM off, GG on"
swift build "${FLAGS[@]}" --traits GuidedGenerationSupport

echo "==> [4/4] both traits OFF"
swift build "${FLAGS[@]}" --disable-default-traits

echo "PASS: all four trait combinations build"

#!/usr/bin/env bash
#
# Builds both states of the package's sole remaining trait,
# FoundationModelsIntegration. The FM-off arm is the regression-prone one: a
# MLXGuidedGeneration type referenced outside the #if FoundationModelsIntegration
# gate in MLXFoundationModels would break it.
#
# Note: this repo's local dev environment requires --disable-sandbox and
# -Xswiftc -disable-sandbox (the Swift macro plugin server cannot be sandboxed
# here) plus --skip-update (network-restricted). CI uses plain `swift build`.
#
set -euo pipefail

FLAGS=(--disable-sandbox --skip-update -Xswiftc -disable-sandbox)

echo "==> [1/2] FoundationModelsIntegration ON (default)"
swift build "${FLAGS[@]}"

echo "==> [2/2] FoundationModelsIntegration OFF"
swift build "${FLAGS[@]}" --disable-default-traits

echo "PASS: both trait states build"

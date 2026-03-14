#!/bin/bash
set -e

# Idempotent setup for mlx-swift-lm continuous batching mission
# No external services needed - pure Swift Package

cd "$(dirname "$0")/.."

# Resolve SPM dependencies if needed
swift package resolve 2>/dev/null || true

echo "Environment ready."

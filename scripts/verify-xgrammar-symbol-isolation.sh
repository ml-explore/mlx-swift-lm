#!/usr/bin/env bash
#
# Verifies the vendored xgrammar/picojson C++ symbols were namespace-renamed
# so they cannot collide with another xgrammar in the same binary.
#
# PASS criteria:
#   (a) mlx_xgrammar:: and mlx_picojson:: symbols are present in CXGrammar objects
#   (b) no STRONG (defined, external) bare xgrammar:: / picojson:: symbols remain
#
set -euo pipefail

echo "==> Building MLXCXGrammar"
swift build --disable-sandbox --skip-update -Xswiftc -disable-sandbox --target MLXCXGrammar >/dev/null

obj_dir="$(find .build -type d \( -name 'MLXCXGrammar.build' -o -name 'MLXCXGrammar-t.build' \) | head -1)"
if [[ -z "$obj_dir" ]]; then
    echo "FAIL: could not locate MLXCXGrammar.build object directory" >&2
    exit 1
fi
echo "    object dir: $obj_dir"

all_syms="$(find "$obj_dir" -name '*.o' -exec nm -C {} + 2>/dev/null)"

# (a) renamed namespaces present.
# Use `rg -c >/dev/null` (consumes all input) rather than `rg -q` so the
# producer side of the pipe is not killed by SIGPIPE under `pipefail`.
if ! echo "$all_syms" | rg -c 'mlx_xgrammar::' >/dev/null; then
    echo "FAIL: no mlx_xgrammar:: symbols found (rename did not take effect)" >&2
    exit 1
fi
if ! echo "$all_syms" | rg -c 'mlx_picojson::' >/dev/null; then
    echo "FAIL: no mlx_picojson:: symbols found (rename did not take effect)" >&2
    exit 1
fi

# (b) no STRONG bare xgrammar:: / picojson:: symbols.
# nm -C lines look like: "<addr> <type> <demangled name>".
# Strong external defined symbols use uppercase type letters T/D/S/B.
# A bare namespace match is xgrammar::/picojson:: NOT preceded by "mlx_".
leaked="$(echo "$all_syms" \
    | rg '^[0-9a-fA-F]+ [TDSB] ' \
    | rg '(^|[^_A-Za-z0-9])(xgrammar|picojson)::' || true)"
if [[ -n "$leaked" ]]; then
    echo "FAIL: strong bare xgrammar::/picojson:: symbols leaked:" >&2
    echo "$leaked" | head -20 >&2
    exit 1
fi

echo "PASS: symbols isolated under mlx_xgrammar:: / mlx_picojson::"

#!/usr/bin/env python3
"""
inspect_drafter_layout.py — Dump exhaustive weight-key inventory from a
safetensors checkpoint directory.

Produces a sorted list of `key  shape  dtype` lines suitable for pasting into
RESEARCH-NOTES.md question N. Reads tensor metadata only (does not materialize
arrays), so it works on multi-GB checkpoints without loading the full state.

Usage:
    python inspect_drafter_layout.py CHECKPOINT_DIR
    python inspect_drafter_layout.py CHECKPOINT_DIR --plain      # just keys
    python inspect_drafter_layout.py CHECKPOINT_DIR --json       # JSON output

If CHECKPOINT_DIR is a HuggingFace repo id (contains a "/"), resolves via
huggingface_hub.snapshot_download (cached) and inspects the local snapshot.

Examples:
    python inspect_drafter_layout.py mlx-community/gemma-4-31B-it-assistant-bf16
    python inspect_drafter_layout.py mlx-community/gemma-4-26B-A4B-it-assistant-bf16

Prints to stdout. Exit 0 on success, 1 on any I/O or parse error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def resolve_checkpoint_dir(arg: str) -> Path:
    """Resolve a path argument to a local checkpoint directory."""
    path = Path(arg).expanduser()
    if path.is_dir():
        return path
    if "/" in arg and not path.exists():
        # Treat as HF repo id
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            sys.stderr.write(
                f"'{arg}' looks like an HF repo id but huggingface_hub is not "
                f"installed. Install it (`pip install huggingface_hub`) or pass "
                f"a local checkpoint directory instead.\n"
            )
            sys.exit(1)
        return Path(snapshot_download(arg))
    sys.stderr.write(f"checkpoint not found: {arg}\n")
    sys.exit(1)


def list_safetensors(checkpoint_dir: Path) -> list[Path]:
    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        sys.stderr.write(f"no *.safetensors files in {checkpoint_dir}\n")
        sys.exit(1)
    return files


def inspect_safetensors(checkpoint_dir: Path) -> list[dict]:
    """Return [{key, shape, dtype, file}] for every tensor across all shards."""
    try:
        from safetensors import safe_open
    except ImportError:
        sys.stderr.write(
            "safetensors library not installed. `pip install safetensors`\n"
        )
        sys.exit(1)

    entries: list[dict] = []
    for path in list_safetensors(checkpoint_dir):
        with safe_open(str(path), framework="numpy") as f:
            for key in f.keys():
                meta = f.get_slice(key)
                entries.append(
                    {
                        "key": key,
                        "shape": list(meta.get_shape()),
                        "dtype": str(meta.get_dtype()),
                        "file": path.name,
                    }
                )
    entries.sort(key=lambda e: e["key"])
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        help="Local checkpoint directory or HuggingFace repo id (org/name)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Print just the sorted key list, one per line (no shape/dtype)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON [{key, shape, dtype, file}, ...] sorted by key",
    )
    args = parser.parse_args()

    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    entries = inspect_safetensors(checkpoint_dir)

    if args.json:
        json.dump(entries, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    if args.plain:
        for e in entries:
            print(e["key"])
        return

    print(f"# Checkpoint: {checkpoint_dir}")
    print(f"# Tensors: {len(entries)}")
    print(f"# Format: key  shape  dtype  (file)")
    print()
    key_w = max(len(e["key"]) for e in entries)
    shape_w = max(len(str(e["shape"])) for e in entries)
    for e in entries:
        print(
            f"{e['key']:<{key_w}}  {str(e['shape']):<{shape_w}}  "
            f"{e['dtype']:<8}  ({e['file']})"
        )


if __name__ == "__main__":
    main()

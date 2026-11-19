# Copyright © 2026 Apple Inc.

"""Cut a DeepSeek-V4.1 checkpoint down to its first layers.

The tensors are copied byte for byte, so the cut keeps the published names,
dtypes and fp8/fp4 packing. Engram, MTP and vision tensors are dropped, and the
config turns Engram off. Only the shards that hold the kept tensors must exist.

    python3 tools/deepseek-v4-reference/deepseek-v41-truncate.py SOURCE DEST --layers 4
"""

import argparse
import json
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path

DROP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_")
SIDE_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def keep(key, layers):
    if key.startswith(DROP_PREFIXES) or ".engram." in key:
        return False
    match = re.match(r"layers\.(\d+)\.", key)
    return match is None or int(match.group(1)) < layers


def read_header(path):
    with open(path, "rb") as f:
        (size,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(size)), 8 + size


def copy_tensors(source, dest, keys):
    header, data_start = read_header(source)
    out_header, offset = {}, 0
    for key in keys:
        entry = header[key]
        begin, end = entry["data_offsets"]
        out_header[key] = {
            "dtype": entry["dtype"],
            "shape": entry["shape"],
            "data_offsets": [offset, offset + end - begin],
        }
        offset += end - begin
    blob = json.dumps(out_header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(source, "rb") as src, open(dest, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for key in keys:
            begin, end = header[key]["data_offsets"]
            src.seek(data_start + begin)
            remaining = end - begin
            while remaining:
                chunk = src.read(min(remaining, 1 << 26))
                out.write(chunk)
                remaining -= len(chunk)
    return offset


def truncate_config(config, layers):
    text = config.get("text_config", config)
    text["num_hidden_layers"] = layers
    text["compress_ratios"] = text["compress_ratios"][:layers]
    for name in ("kv_source_layer_ids", "index_source_layer_ids"):
        text[name] = [i for i in text.get(name, []) if i < layers]
    if text.get("candidate_source_layer_id", -1) >= layers:
        text["candidate_source_layer_id"] = -1
    text["engram_layer_ids"] = []
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--layers", type=int, default=4)
    args = parser.parse_args()

    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    by_shard = defaultdict(list)
    for key, shard in index["weight_map"].items():
        if keep(key, args.layers):
            by_shard[shard].append(key)

    args.dest.mkdir(parents=True, exist_ok=True)
    weight_map, total = {}, 0
    for n, shard in enumerate(sorted(by_shard), start=1):
        name = f"model-{n:05d}-of-{len(by_shard):05d}.safetensors"
        keys = sorted(by_shard[shard])
        total += copy_tensors(args.source / shard, args.dest / name, keys)
        weight_map.update({key: name for key in keys})
        print(f"{shard} -> {name}: {len(keys)} tensors")

    (args.dest / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)
    )
    config = json.loads((args.source / "config.json").read_text())
    (args.dest / "config.json").write_text(
        json.dumps(truncate_config(config, args.layers), indent=2)
    )
    for name in SIDE_FILES:
        if (args.source / name).exists():
            shutil.copy(args.source / name, args.dest / name)
    print(f"{len(weight_map)} tensors, {total / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()

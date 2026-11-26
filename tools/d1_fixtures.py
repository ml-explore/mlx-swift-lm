#!/usr/bin/env python3
"""
Python references for the d1 tests in Tests/MLXLMTests.

d1-3B's reference is the code in the LiquidAI/d1-3B repository (`modeling_d1.py`, loaded
with `trust_remote_code`). It prepares images with Pillow and transformers'
`Lfm2VlProcessor`, and needs transformers 5.19 or later, torch, torchvision, Pillow,
numpy and huggingface_hub.

    python tools/d1_fixtures.py write --out <dir>
        Writes the directory the gated real-weight suite reads from MLX_D1_FIXTURES. The
        input cats.jpg is downloaded at a pinned revision unless <dir> already holds it, and
        the PNG inputs are derived from it. The checkpoint is LiquidAI/d1-3B at the revision
        below, from the Hugging Face cache or downloaded.

    python tools/d1_fixtures.py readme
        Prints the bfloat16 probabilities `DecisionIntegrationTests` expects for the
        d1-3B README examples.

    python tools/d1_fixtures.py vectors
        Prints the Pillow resize values of `D1Tests`. Needs only Pillow and numpy.

Files written by `write`, besides the inputs:
- d1-3B-float32.json: d1-3B text and image cases (`D1ParityTests`).

Case files name their inputs by file name, relative to the fixture directory. Questions
are lists Swift decodes in order. Probabilities are the reference's, unrounded, in its
order (bool as [yes, no]). Each case carries the tolerance the Swift test applies, and keeps
the prompt texts with their ids, so the Swift tests can use Python's tokenizer output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np

D1_3B = ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432")

# The two cats photo of the model READMEs (COCO val2017 39769).
CATS_JPG = (
    (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/"
        "4539dd3e9faa39d54e606b4c875f85c7561f5423/cats.jpg"
    ),
    "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
)
# PNG inputs: cats.jpg resized with PIL bicubic to each size.
RESIZED = [(400, 600), (1024, 768), (1600, 1200), (720, 1600), (3024, 4032)]

# Questions in the d1-3B reference's schema, which mlx-vlm also reads: bool is "noul",
# choice criteria map labels to descriptions.
TICKET = "I was charged twice, please refund one."
ROOM = "A photo from my living room."
CATS = {
    "type": "choice",
    "instructions": "How many cats are there?",
    "criteria": dict.fromkeys(["one", "two", "more"]),
}
ANIMAL = {"type": "noul", "instructions": "Is there a cat in the picture?"}
CUTE = {
    "type": "score",
    "instructions": "How cute is this picture?",
    "criteria": ["not cute", "somewhat cute", "very cute"],
}
IMAGE = {"cats": CATS, "animal": ANIMAL, "cute": CUTE}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["Can wait", "Today", "Blocking the customer now"],
}
REFUND = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
TEXT_3B = {
    "refund": REFUND,
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds", "technical": "App faults"},
    },
    "urgency": URGENCY,
}

# d1-3B probability tolerances: exact pixels, and JPEG, which Core Image decodes a few
# levels away from libjpeg-turbo.
D1_3B_TOLERANCES = {"exact": 1e-5, "jpeg": 2e-2}


def ordered(questions):
    """Reference question dicts -> a list Swift decodes in order."""
    out = []
    for name, spec in questions.items():
        kind = "bool" if spec["type"] == "noul" else spec["type"]
        criteria = spec.get("criteria")
        entry = {"name": name, "type": kind, "instructions": spec["instructions"]}
        if kind == "choice":
            entry["options"] = [[k, v] for k, v in criteria.items()]
        elif kind == "score":
            entry["levels"] = list(criteria)
        elif criteria is not None:
            entry["when_true"] = criteria.get("true")
            entry["when_false"] = criteria.get("false")
        out.append(entry)
    return out


def snapshot(repo):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo[0], revision=repo[1])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False))
    print("wrote", path, flush=True)


# MARK: - Inputs


def fetch(out, name, source):
    """`name` in `out`, downloaded when missing; checked against its sha256."""
    url, digest = source
    path = out / name
    if not path.exists():
        with urllib.request.urlopen(url, timeout=60) as response, open(path, "wb") as f:
            shutil.copyfileobj(response, f)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"{path}: sha256 {actual}, expected {digest}")


def inputs(out):
    from PIL import Image

    fetch(out, "cats.jpg", CATS_JPG)
    cats = Image.open(out / "cats.jpg").convert("RGB")
    # cats.jpg as PIL decodes it, for exact pixels at the JPEG's own size.
    cats.save(out / "cats_640x480.png")
    for size in RESIZED:
        cats.resize(size, Image.Resampling.BICUBIC).save(
            out / f"cats_{size[0]}x{size[1]}.png"
        )
    # Over a megapixel and square, so d1-3B's pixel cap changes the tile grid.
    big = Image.open(out / "cats_1600x1200.png")
    big.crop((250, 50, 1350, 1150)).save(out / "cats_1100x1100.png")


# MARK: - d1-3B


class Reference:
    """The d1-3B repository's own code on the CPU: its runner (`model.engine`), prompt
    module and image cap, with transformers' processor for the image token ids."""

    def __init__(self, dtype):
        import torch
        from transformers import AutoModel, AutoProcessor

        path = snapshot(D1_3B)
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=True, dtype=getattr(torch, dtype)
        )
        self.engine = self.model.engine
        package = type(self.model).__module__.rsplit(".", 1)[0]
        self.prompt = importlib.import_module(package + ".prompt")
        self.runner = importlib.import_module(package + ".runner")
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
        self.tokenizer = self.engine.tokenizer

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def probabilities(self, state, questions, images):
        """Each question's probabilities and the tokens read, from the runner."""
        parsed = [self.prompt.as_question(q) for q in questions.values()]
        ((probabilities, tokens),) = self.engine.run([(state, parsed, images)])
        return probabilities, tokens

    def case(self, state, questions, images):
        """The prompt texts, ids and vision input the runner builds for one request."""
        engine, prompt = self.engine, self.prompt
        parsed = [prompt.as_question(q) for q in questions.values()]
        capped = [self.runner.cap_pixels(image) for image in images]
        markup = engine._image_markup(len(capped)) if capped else ""
        text = prompt.prefix_text(
            self.tokenizer, state, engine.bos, engine.state_style, engine.system, markup
        )
        suffixes = [
            prompt.suffix_text(self.tokenizer, q, engine.lead, engine.option_style)
            for q in parsed
        ]
        if len(suffixes) == 1:
            text, suffixes = text + suffixes[0], []
        vision = {}
        if capped:
            processed = self.processor(
                text=[text],
                images=[capped],
                return_tensors="pt",
                add_special_tokens=False,
            )
            trunk = processed["input_ids"][0].tolist()
            # The runner keeps the patch rows up to the longest view.
            mask = processed["pixel_attention_mask"]
            n = int(mask.sum(1).max())
            vision = {
                "pixel_values": processed["pixel_values"][:, :n],
                "pixel_attention_mask": mask[:, :n],
                "spatial_shapes": processed["spatial_shapes"],
            }
        else:
            trunk = self.encode(text)
        return {
            "capped_sizes": [list(image.size) for image in capped],
            "prompt_text": text,
            "suffix_texts": suffixes,
            "trunk_ids": trunk,
            "branch_ids": [self.encode(s) for s in suffixes],
            "vision": vision,
        }


def load_images(out, names):
    from transformers.image_utils import load_image

    return [load_image(str(out / name)) for name in names]


def d1_3b(out):
    reference = Reference("float32")
    cases = [
        ("text", TICKET, TEXT_3B, []),
        ("text-single", TICKET, {"team": TEXT_3B["team"]}, []),
        ("image", None, IMAGE, ["cats.jpg"]),
        ("image+state", ROOM, {"cats": CATS}, ["cats.jpg"]),
        ("png-400x600", None, IMAGE, ["cats_400x600.png"]),
        ("png-1024x768", ROOM, IMAGE, ["cats_1024x768.png"]),
        ("png-1600x1200-single", None, {"cats": CATS}, ["cats_1600x1200.png"]),
        ("png-1600x1200", None, IMAGE, ["cats_1600x1200.png"]),
        # The pixel cap changes the grid: 2x2 tiles at 1024x1024, 3x3 without it.
        (
            "png-1100x1100",
            None,
            {"cats": CATS, "animal": ANIMAL},
            ["cats_1100x1100.png"],
        ),
        ("png-720x1600", None, IMAGE, ["cats_720x1600.png"]),
        (
            "png-3024x4032",
            None,
            {"animal": ANIMAL, "cute": CUTE},
            ["cats_3024x4032.png"],
        ),
        (
            "two-images",
            # Keys sorted: Swift tests read states as JSONValue, which keeps no order.
            {"note": "two photos of my cats", "room": "living room"},
            {
                "cats": CATS,
                "same": {
                    "type": "noul",
                    "instructions": "Do both pictures show the same cats?",
                    "criteria": {
                        "true": "the same animals",
                        "false": "different animals",
                    },
                },
            },
            ["cats_400x600.png", "cats_1024x768.png"],
        ),
        (
            "two-images-single",
            None,
            {"cats": CATS},
            ["cats_720x1600.png", "cats_400x600.png"],
        ),
        # Two single views of different sizes: rows are cut to the longer view.
        ("two-views", ROOM, IMAGE, ["cats_400x600.png", "cats_640x480.png"]),
    ]
    results = []
    for tag, state, questions, names in cases:
        images = load_images(out, names)
        case = reference.case(state, questions, images)
        probabilities, tokens = reference.probabilities(state, questions, images)
        read = len(case["trunk_ids"]) + sum(map(len, case["branch_ids"]))
        if tokens != read:
            raise SystemExit(f"{tag}: the runner read {tokens} tokens, not {read}")
        vision = case.pop("vision")
        mask = np.asarray(vision.get("pixel_attention_mask", np.zeros((0, 0))))
        results.append(
            {
                "tag": tag,
                "images": names,
                "capped_sizes": case["capped_sizes"],
                "state": state,
                "questions": ordered(questions),
                "prompt_text": case["prompt_text"],
                "suffix_texts": case["suffix_texts"],
                "trunk_ids": case["trunk_ids"],
                "branch_ids": case["branch_ids"],
                "vision_shapes": {k: list(v.shape) for k, v in vision.items()},
                "spatial_shapes": np.asarray(
                    vision.get("spatial_shapes", np.zeros((0, 2)))
                ).tolist(),
                "mask_sums": mask.sum(1).tolist(),
                "probabilities": probabilities,
                "input_tokens": tokens,
                "tolerance": D1_3B_TOLERANCES[
                    "jpeg" if any(n.endswith(".jpg") for n in names) else "exact"
                ],
            }
        )
        print(
            tag,
            tokens,
            [[round(p, 4) for p in row] for row in probabilities],
            flush=True,
        )
    write_json(
        out / "d1-3B-float32.json",
        {
            "model": f"{D1_3B[0]}@{D1_3B[1]}",
            "reference": "modeling_d1.D1Model (trust_remote_code), torch CPU",
            "dtype": "float32",
            "cases": results,
        },
    )


def write(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs(out)
    d1_3b(out)


def readme(args):
    """The README examples in bfloat16, the precision d1-3B ships in."""
    import tempfile

    reference = Reference("bfloat16")
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        fetch(out, "cats.jpg", CATS_JPG)
        requests = [
            ("text", TICKET, TEXT_3B, []),
            ("image", None, {"cats": CATS}, load_images(out, ["cats.jpg"])),
        ]
        for tag, state, questions, images in requests:
            probabilities, tokens = reference.probabilities(state, questions, images)
            print(tag, tokens)
            for name, p in zip(questions, probabilities):
                # Swift's test reads a bool as [P(true)].
                print(f"  {name}: {p[:1] if questions[name]['type'] == 'noul' else p}")


def vectors(args):
    """`D1Tests` Pillow bicubic cases: in, out, byte sum, first 6 and last 3 bytes."""
    from PIL import Image

    def pattern(w, h):
        y, x, c = np.meshgrid(np.arange(h), np.arange(w), np.arange(3), indexing="ij")
        values = (x * 31 + y * 17 + c * 59 + ((x * y) % 7) * 23) % 256
        return Image.fromarray(values.astype(np.uint8))

    # 3x500 takes the height-first path `Image.resize` uses for very tall images.
    sizes = [
        (37, 53, 15, 20),
        (64, 48, 23, 17),
        (50, 40, 50, 13),
        (50, 40, 7, 40),
        (3, 500, 2, 37),
        (3, 500, 3, 41),
        (300, 2, 31, 3),
        (2048, 1024, 1448, 724),
    ]
    for w, h, ow, oh in sizes:
        out = np.asarray(pattern(w, h).resize((ow, oh), Image.Resampling.BICUBIC))
        flat = out.reshape(-1).tolist()
        print(f"(({w}, {h}), ({ow}, {oh}), {sum(flat)}, {flat[:6]}, {flat[-3:]}),")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("write", help="write the fixture directory")
    run.add_argument("--out", type=Path, required=True)
    run.set_defaults(handler=write)
    commands.add_parser("readme", help="print the README example answers").set_defaults(
        handler=readme
    )
    commands.add_parser("vectors", help="print the Pillow resize values").set_defaults(
        handler=vectors
    )
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

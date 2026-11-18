#!/usr/bin/env python3
"""
Python references for the d1 tests in Tests/MLXLMTests.

d1-3B's reference is the code in the LiquidAI/d1-3B repository (`modeling_d1.py`, loaded
with `trust_remote_code`). It prepares images with Pillow and transformers'
`Lfm2VlProcessor`, and needs transformers 5.19 or later, torch and torchvision.
d1-omni-600M's reference is mlx-vlm's `mlx_vlm.models.d1_omni` (merged at 370f4ccf). Both
need Pillow, numpy and huggingface_hub.

    python tools/d1_fixtures.py write --out <dir> [--only ...]
        Writes the directory the gated real-weight suites read from MLX_D1_FIXTURES. The
        inputs (cats.jpg, jfk.wav) are downloaded at pinned revisions unless <dir> already
        holds them, and the PNG inputs are derived from cats.jpg. The checkpoints are
        LiquidAI/d1-3B and LiquidAI/d1-omni-600M at the revisions below, from the Hugging
        Face cache or downloaded. `--only d1-3b` or `--only omni` writes the files of one
        model, so each can run in its own environment.

    python tools/d1_fixtures.py readme
        Prints the bfloat16 probabilities `DecisionIntegrationTests` expects for the
        d1-3B README examples.

    python tools/d1_fixtures.py vectors
        Prints the Pillow resize values of `D1Tests`. Needs only Pillow and numpy.

    python tools/d1_fixtures.py tiny-audio
        Prints the tiny audio model golden values in `D1OmniAudioTests` (no checkpoint).

Files written by `write`, besides the inputs:
- d1-3B-float32.json: d1-3B text and image cases (`D1ParityTests`).
- d1-omni-600M-<dtype>-media.{json,safetensors} for float32, float16 and bfloat16: d1-omni
  text, image and audio cases (`D1OmniParityTests`).
- d1-omni-600M-float32-tower.safetensors: SigLIP2 tower inputs and output on cats.jpg
  (`Siglip2NaFlexParityTests`).
- d1-omni-600M-float32-audio.safetensors: audio prefixes of clips cut from jfk.wav
  (`D1OmniAudioParityTests`).

Case files name their inputs by file name, relative to the fixture directory. Questions
are lists Swift decodes in order. Probabilities are the reference's, unrounded, in its
order (d1-3B bool as [yes, no], d1-omni bool as [false, true]). Each case carries the
tolerance the Swift test applies. d1-3B cases keep the prompt texts with their ids, and
d1-omni files map every string the prompt encoder tokenized to its ids, so the Swift tests
can use Python's tokenizer output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import shutil
import urllib.request
import wave
from pathlib import Path

import numpy as np

D1_3B = ("LiquidAI/d1-3B", "051bcc464b01b9f92942b364d9586b0ef5912432")
D1_OMNI = ("LiquidAI/d1-omni-600M", "02b55d7076f15129e59ab3f94783f32c4b088674")

# The two cats photo of the model READMEs (COCO val2017 39769) and the JFK speech clip.
CATS_JPG = (
    (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/"
        "4539dd3e9faa39d54e606b4c875f85c7561f5423/cats.jpg"
    ),
    "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e",
)
JFK_WAV = (
    (
        "https://raw.githubusercontent.com/ggml-org/whisper.cpp/"
        "b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
    ),
    "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e",
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

TEXT_OMNI = {
    "refund": REFUND,
    "team": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Charges, refunds, invoices",
            "technical": "App or site faults",
            "fraud": "Suspected unauthorised use",
        },
    },
    "urgency": URGENCY,
}
AUDIO = {
    "speaking": {"type": "noul", "instructions": "Is anyone speaking in this clip?"},
    "speaker": {
        "type": "choice",
        "instructions": "Who is speaking?",
        "criteria": dict.fromkeys(["man", "woman", "child"]),
    },
    "topic": {
        "type": "choice",
        "instructions": "What is the speech about?",
        "criteria": {
            "politics": "Government, country, citizens",
            "sports": "Games and athletes",
            "cooking": "Food and recipes",
        },
    },
}

# d1-3B probability tolerances: exact pixels, and JPEG, which Core Image decodes a few
# levels away from libjpeg-turbo.
D1_3B_TOLERANCES = {"exact": 1e-5, "jpeg": 2e-2}
# d1-omni (max |probability difference|, prefix relative L2). Half precision allows for
# its rounding noise.
OMNI_TOLERANCES = {
    "float32": {"exact": (1e-4, 1e-4), "jpeg": (3e-2, 5e-2)},
    "float16": {"exact": (1e-2, 1e-2), "jpeg": (5e-2, 5e-2)},
    "bfloat16": {"exact": (1e-1, 5e-2), "jpeg": (1e-1, 1e-1)},
}
OMNI_SPECIAL = ["<|startoftext|>", "<|mask|>"] + [
    f"<|reserved_{i}|>" for i in range(7, 12)
]


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


def read_wav(path):
    with wave.open(str(path), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (16000, 1, 2)
        return np.frombuffer(f.readframes(f.getnframes()), np.int16)


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
    fetch(out, "jfk.wav", JFK_WAV)
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


# MARK: - d1-omni


class RecordingTokenizer:
    """Forwards to the real tokenizer and records every string it encodes."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.table = {}

    def convert_tokens_to_ids(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    def __call__(self, text, add_special_tokens=False):
        ids = self.tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
        self.table[text] = list(ids)
        return {"input_ids": ids}


def omni_media(out, path, dtype):
    """Text, image and audio requests, with their media prefixes."""
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.d1_omni.d1_omni import _image

    model, processor = load(path)
    if dtype != "float32":
        model.set_dtype(getattr(mx, dtype))
    tokenizer = RecordingTokenizer(getattr(processor, "tokenizer", processor))

    tensors = {"audio-array.samples": mx.array(read_wav(out / "jfk.wav")[16000:96000])}
    cases = [
        ("text", TICKET, TEXT_OMNI, [], None),
        ("jpeg", None, IMAGE, ["cats.jpg"], None),
        ("png-400x600", None, IMAGE, ["cats_400x600.png"], None),
        ("png-1024x768", None, IMAGE, ["cats_1024x768.png"], None),
        ("png-720x1600", ROOM, IMAGE, ["cats_720x1600.png"], None),
        (
            "png-two",
            # Keys sorted: Swift tests read states as JSONValue, which keeps no order.
            {"photos": 2, "room": "living room"},
            {"cats": CATS, "animal": ANIMAL},
            ["cats_400x600.png", "cats_1024x768.png"],
            None,
        ),
        ("audio", None, AUDIO, [], "jfk.wav"),
        ("audio-array", {"channel": "phone"}, AUDIO, [], "audio-array.samples"),
    ]
    results = []
    for tag, state, questions, names, audio in cases:
        images = [str(out / name) for name in names]
        if audio in tensors:
            clip, audio_name, samples = np.array(tensors[audio]), None, audio
        else:
            clip = None if audio is None else str(out / audio)
            audio_name, samples = audio, None
        request = (state, list(questions.values()), images or None, clip)
        [(probabilities, tokens)] = model.probabilities(tokenizer, [request])
        if images:
            tensors[f"{tag}.prefix"] = model.vision([_image(p) for p in images])
        elif clip is not None:
            tensors[f"{tag}.prefix"] = model.audio(clip)[0]
        tolerance, prefix_tolerance = OMNI_TOLERANCES[dtype][
            "jpeg" if tag == "jpeg" else "exact"
        ]
        results.append(
            {
                "tag": tag,
                "state": state,
                "questions": ordered(questions),
                "images": names,
                "audio": audio_name,
                "audio_samples": samples,
                "probabilities": probabilities,
                "input_tokens": tokens,
                "prefix": f"{tag}.prefix" if f"{tag}.prefix" in tensors else None,
                "tolerance": tolerance,
                "prefix_tolerance": prefix_tolerance,
            }
        )
        print(
            dtype,
            tag,
            tokens,
            [[round(p, 4) for p in row] for row in probabilities],
            flush=True,
        )

    special = {t: tokenizer.convert_tokens_to_ids(t) for t in OMNI_SPECIAL}
    stem = out / f"d1-omni-600M-{dtype}-media"
    write_json(
        stem.with_suffix(".json"),
        {
            "model": f"{D1_OMNI[0]}@{D1_OMNI[1]}",
            "dtype": dtype,
            "special": special,
            "tokens": tokenizer.table,
            "cases": results,
        },
    )
    mx.eval(tensors)
    mx.save_safetensors(
        str(stem.with_suffix(".safetensors")),
        {
            k: v.astype(mx.float32) if v.dtype in (mx.float16, mx.bfloat16) else v
            for k, v in tensors.items()
        },
    )


def omni_tower(out, path):
    """SigLIP2 tower inputs and last hidden state for cats.jpg, from mlx-vlm's pixels."""
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.d1_omni.vision import preprocess
    from mlx_vlm.utils import load_image

    model, _ = load(path)
    inputs = preprocess(load_image(str(out / "cats.jpg")))
    tensors = {f"vision.{k}": mx.array(v) for k, v in inputs.items()}
    *_, tensors["vision.tower_out"] = model.vision.tower(
        tensors["vision.pixel_values"],
        spatial_shapes=tensors["vision.spatial_shapes"],
        pixel_attention_mask=tensors["vision.pixel_attention_mask"],
    )
    mx.eval(tensors)
    mx.save_safetensors(str(out / "d1-omni-600M-float32-tower.safetensors"), tensors)


def audio_clips(jfk):
    """Clips around the front end's edges: 0.5 s padding, 30 s cut, int16 input."""
    pcm16 = read_wav(jfk)
    speech = pcm16.astype(np.float32) / np.float32(32768)
    long = np.tile(speech, 3)
    t = np.arange(250000) / 16000
    rng = np.random.default_rng(1)
    return {
        "jfk11s": speech,
        "jfk30s": long[:480000],
        "jfk1p04s": speech[:16640],
        "jfk0p3s": speech[:4800],
        "jfk479999": long[:479999],
        "int16jfk50000": pcm16[5000:55000].copy(),
        "sweep250000": (0.3 * np.sin(2 * np.pi * (50 + 3000 * t / t[-1]) * t)).astype(
            np.float32
        ),
        "noise8001": rng.normal(0, 0.2, 8001).astype(np.float32),
    }


def omni_audio(out, path):
    """Each clip `name` as `name.input`, with mlx-vlm's `name.mel` (T, 128),
    `name.mel_valid`, `name.enc` (T_enc, 512), `name.enc_valid` and `name.prefix` (P, 1024)."""
    import mlx.core as mx
    from mlx_vlm import load
    from mlx_vlm.models.d1_omni.audio import waveform

    model, _ = load(path)
    audio = model.audio
    tensors = {}
    for name, x in audio_clips(out / "jfk.wav").items():
        # Audio.__call__, keeping the intermediates.
        mel, mask, _ = audio.frontend([waveform(x)])
        enc, valid = audio.encoder(mel.astype(audio.adapter.norm.weight.dtype), mask)
        length = int(valid.sum())
        prefix = audio.residual(audio.adapter(enc[:, :length]))
        tensors[f"{name}.input"] = mx.array(x)
        tensors[f"{name}.mel"] = mel[0]
        tensors[f"{name}.mel_valid"] = mx.array(int(mask.sum()), dtype=mx.int32)
        tensors[f"{name}.enc"] = enc[0]
        tensors[f"{name}.enc_valid"] = mx.array(length, dtype=mx.int32)
        tensors[f"{name}.prefix"] = prefix[0]
        mx.eval(tensors)
        print("audio", name, tuple(prefix[0].shape), flush=True)
    mx.save_safetensors(str(out / "d1-omni-600M-float32-audio.safetensors"), tensors)


def write(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs(out)
    if "d1-3b" in args.only:
        d1_3b(out)
    if "omni" in args.only:
        path = snapshot(D1_OMNI)
        omni_tower(out, path)
        omni_audio(out, path)
        for dtype in ("float32", "float16", "bfloat16"):
            omni_media(out, path, dtype)


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


# MARK: - Tiny audio model

# `audio_config` of the tiny d1_omni case in mlx_vlm/tests/extraction_cases.json.
TINY_AUDIO_CONFIG = {
    "feat_in": 16,
    "n_layers": 1,
    "d_model": 16,
    "subsampling_conv_channels": 4,
    "ff_expansion_factor": 2,
    "n_heads": 2,
    "conv_kernel_size": 3,
    "residual_width": 8,
}


def hashed(count: int, seed: int) -> np.ndarray:
    """32-bit integer hash of 0..<count; `hashed` in D1OmniAudioTests.swift."""
    mask = np.uint64(0xFFFFFFFF)
    h = np.arange(count, dtype=np.uint64) * np.uint64(0x9E3779B1)
    h = (h + np.uint64(seed) * np.uint64(0x85EBCA77)) & mask
    h ^= h >> np.uint64(16)
    h = (h * np.uint64(0x7FEB352D)) & mask
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x846CA68B)) & mask
    h ^= h >> np.uint64(16)
    return h


def uniform(count: int, seed: int) -> np.ndarray:
    """Values in [-0.5, 0.5), exact in float32."""
    bits = (hashed(count, seed) >> np.uint64(8)).astype(np.float32)
    return bits / np.float32(2**24) - np.float32(0.5)


def pcm(count: int, seed: int) -> np.ndarray:
    """int16 noise in [-8192, 8191]."""
    return (hashed(count, seed) >> np.uint64(16)).astype(np.uint16).view(np.int16) >> 2


def tiny_audio(_args):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_vlm.models.d1_omni.audio import Audio
    from mlx_vlm.models.d1_omni.config import AudioConfig

    mx.set_default_device(mx.cpu)
    model = Audio(AudioConfig(**TINY_AUDIO_CONFIG), 64)
    weights = []
    parameters = sorted(tree_flatten(model.parameters()), key=lambda item: item[0])
    for index, (key, value) in enumerate(parameters):
        w = uniform(value.size, index).reshape(value.shape)
        weights.append(
            (key, mx.array(np.abs(w) + 0.5 if key.endswith("running_var") else w))
        )
    model.load_weights(weights, strict=True)
    model.eval()
    print(f"// {len(weights)} parameters")
    for count in (12000, 12800):
        prefix = np.array(model(mx.array(pcm(count, count)))[0])
        print(f"// {count} samples: prefix {prefix.shape}, prefix[:, ::16]")
        for row in prefix[:, ::16]:
            print(", ".join(f"{v:.6g}" for v in row) + ",")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("write", help="write the fixture directory")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--only",
        nargs="+",
        choices=["d1-3b", "omni"],
        default=["d1-3b", "omni"],
        help="the models to write fixtures for",
    )
    run.set_defaults(handler=write)
    commands.add_parser("readme", help="print the README example answers").set_defaults(
        handler=readme
    )
    commands.add_parser("vectors", help="print the Pillow resize values").set_defaults(
        handler=vectors
    )
    commands.add_parser(
        "tiny-audio", help="print the tiny audio golden values"
    ).set_defaults(handler=tiny_audio)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

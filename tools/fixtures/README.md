# MTP fixtures

Reference tensors for the Gemma 4 MTP speculative-decoding parity tests
(Rungs 1–4 plus the bidirectional-mask suite). The 43 `.safetensors` files
themselves are **not vendored in this repo** — they live in the
[`angelsbrood/gemma4-mtp-fixtures`](https://huggingface.co/datasets/angelsbrood/gemma4-mtp-fixtures)
public HuggingFace dataset and are fetched on demand by the integration
tests in `IntegrationTesting/IntegrationTestingTests/`.

The integration tests pin a specific dataset commit SHA so byte-exact
parity assertions are reproducible. Bumping the dataset requires updating
the `fixturesRevision` constant at the top of each consuming test file.

## What's in this directory

- `FIXTURE-SCHEMA.md` — per-fixture tensor schema (keys, shapes, dtypes)
- `FIXTURE-MANIFEST.json` — per-file SHA-256 manifest of intended contents
- `generate_mtp_fixtures.py` (one level up at `tools/`) — regeneration script
- `inspect_drafter_layout.py` (one level up at `tools/`) — debugging utility

## Regenerating the dataset

1. Run `tools/generate_mtp_fixtures.py` against `mlx-vlm@d49d428` with
   target `mlx-community/gemma-4-31b-it-8bit` and drafter
   `mlx-community/gemma-4-31B-it-assistant-bf16`. Output lands under
   `tools/fixtures/{masks,drafter_forward,drafter_block,end_to_end}/`.
2. Upload the regenerated directory tree to the HF dataset (e.g. via
   `huggingface-cli upload`). Note the new commit SHA.
3. Update `fixturesRevision` in each integration-test file under
   `IntegrationTesting/IntegrationTestingTests/`.
4. Update `FIXTURE-MANIFEST.json` to reflect the new SHA-256 values.

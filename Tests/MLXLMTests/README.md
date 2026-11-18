The two includes resources are for testing the MediaProcessing pipeline for correctness and validation

The video file was created via FFMPEG via

```
ffmpeg \
-f lavfi \
-i smptebars=size=1920x1080:rate=30:duration=5.0 \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-vf "drawtext=text='Frame\\: %{frame_num}': start_number=1: x=(w-tw)/2: y=h-(2*lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:alpha=0.5:box=1:boxborderw=4,drawtext=text='TC':x=(w-tw)/2:y=(lh):fontfile='Inconsolata-Regular.ttf':fontsize=40:fontcolor=white:timecode='01\\:00\\:00\\:00':timecode_rate=(30)" \
-c:v libx264 \
-c:a aac \
-crf 23 \
-preset medium \
-pix_fmt yuv420p \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y 1080p_30.mov
```

and the audio only file 

```
ffmpeg \
-f lavfi \
-i sine=frequency=1000:duration=5.0 \
-c:a aac \
-crf 23 \
-preset medium \
-fflags +shortest \
-t 5 \
-timecode 01:00:00:00 \
-write_tmcd true \
-y audio_only.mov
```

## LFM2-VL test vectors

The expected values in `ImagePixelsTests.swift`, `LFM2VLImageProcessingTests.swift`,
`LFM2VLProcessorTests.swift`, `LFM2VLTests.swift` and `Siglip2NaFlexTests.swift` come from
Hugging Face transformers, except the hashes in
`LFM2VLProcessorTests.severalImagesKeepTheirTokensAndPixels`, which are regression values from
this port. To regenerate them:

```
pip install transformers torch torchvision pillow
python tools/lfm2_vl_test_vectors.py
```

The script prints the values grouped by test. It needs no model weights.

## d1 parity tests on real weights

Four suites compare the LiquidAI d1 decision models with their Python references on the
released checkpoints: d1-3B in float32, and d1-omni-600M in float32, float16 and bfloat16.
d1-3B's reference is the checkpoint's own code (`modeling_d1.py`, loaded with
`trust_remote_code`), which prepares images with Pillow and transformers' `Lfm2VlProcessor`.
d1-omni's reference is mlx-vlm. The suites skip unless their environment variables are set:

| Variable | Value | Suites |
| --- | --- | --- |
| `MLX_D1_FIXTURES` | The directory `tools/d1_fixtures.py` writes | all of them |
| `MLX_D1_3B_MODEL` | A `LiquidAI/d1-3B` snapshot | `D1ParityTests` |
| `MLX_D1_OMNI_MODEL` | A `LiquidAI/d1-omni-600M` snapshot | `D1OmniParityTests`, `D1OmniAudioParityTests`, `Siglip2NaFlexParityTests` |

The suites check answers, token counts, image layouts, media prefixes and vision tower
outputs. The tests feed Python's token ids through a lookup table, so the swift-transformers
tokenizer does not take part. d1-3B text and PNG cases must match to 1e-5, and d1-omni float32
cases to 1e-4; float16 and bfloat16 allow for their rounding. JPEG cases match loosely, because
Core Image decodes JPEG a few levels away from libjpeg-turbo.

### Write the fixtures

The d1-3B fixtures come from its reference on the CPU, with transformers 5.19 or later and
Pillow 12. Pillow 13 resizes some images differently. The d1-omni fixtures come from mlx-vlm
at commit 370f4ccf (the merge of the d1 models) or later:

```bash
pip install "transformers>=5.19" torch torchvision "pillow<13" numpy huggingface_hub \
    "git+https://github.com/Blaizzy/mlx-vlm@370f4ccfa5db854f2640d982b8595ae7e39b9f1f"

python tools/d1_fixtures.py write --out ~/d1-fixtures
```

The script downloads `cats.jpg` and `jfk.wav` at pinned revisions and makes the PNG inputs
from `cats.jpg`. It loads d1-3B and d1-omni-600M at the revisions the tests expect, from the
Hugging Face cache or the Hub, and writes about 150 MB. `--only d1-3b` or `--only omni`
writes the files of one model, so each can run in its own environment.
`python tools/d1_fixtures.py vectors` prints the Pillow resize values of `D1Tests.swift`.

### Run the tests

`xcodebuild` passes variables that start with `TEST_RUNNER_` to the tests, without the
prefix:

```bash
snapshot() {
    python -c "from huggingface_hub import snapshot_download as s; print(s('$1', revision='$2'))"
}
export TEST_RUNNER_MLX_D1_FIXTURES=~/d1-fixtures
export TEST_RUNNER_MLX_D1_3B_MODEL=$(snapshot LiquidAI/d1-3B 051bcc464b01b9f92942b364d9586b0ef5912432)
export TEST_RUNNER_MLX_D1_OMNI_MODEL=$(snapshot LiquidAI/d1-omni-600M 02b55d7076f15129e59ab3f94783f32c4b088674)

xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' \
    -skipPackagePluginValidation \
    -only-testing:MLXLMTests/D1ParityTests \
    -only-testing:MLXLMTests/D1OmniParityTests \
    -only-testing:MLXLMTests/D1OmniAudioParityTests \
    -only-testing:MLXLMTests/Siglip2NaFlexParityTests
```

They take under a minute on an M3 Ultra, not counting the build.

`python tools/d1_fixtures.py tiny-audio` prints the expected values of the tiny audio
model test in `D1OmniAudioTests`. It needs no checkpoint.

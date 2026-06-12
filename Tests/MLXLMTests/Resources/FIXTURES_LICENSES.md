# Gemma 4 media test fixtures - sources and licenses

All media used by the Gemma 4 audio and video integration tests, with
provenance and license. CC0/Public-Domain files need no attribution; CC-BY
files are attributed below.

| File | Source | License | Notes |
|---|---|---|---|
| `gemma_audio_librispeech.wav` | LibriSpeech `test`/`dev-clean`, utterance `1272-128104-0000` via `hf-internal-testing/librispeech_asr_dummy` | **CC-BY-4.0** | Real human read speech with ground-truth transcript: "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL". 5.86 s, 16 kHz mono. LibriSpeech by Vassil Panayotov et al., derived from LibriVox public-domain recordings. |
| `gemma_speech_test.wav` | Generated locally with macOS `say` and `afconvert` | **CC0 / our own** | "The quick brown fox jumps over the lazy dog near the river bank." Synthetic TTS. |
| `gemma_speech_long.wav` | Generated locally with macOS `say` and `afconvert` | **CC0 / our own** | "The weather forecast predicts heavy rain tomorrow afternoon ..." Synthetic TTS. |
| `gemma_video_bbb.mp4` | Big Buck Bunny, Blender Foundation, 10 s 360p sample via test-videos.co.uk | **CC-BY-3.0** | Copyright Blender Foundation / peach.blender.org. |
| `1080p_30.mov`, `audio_only.mov` | Pre-existing upstream fixtures, added in mlx-swift-lm PR #64 | Upstream fixture | Color-bar test pattern plus tone. Not added by this branch. |

Whisper large-v3-turbo transcribes these clips correctly, confirming the audio
files are usable. The Gemma 4 E4B tests assert audio perception and key content
words instead of verbatim ASR quality, because verbatim transcription quality is
a model property rather than an integration requirement.

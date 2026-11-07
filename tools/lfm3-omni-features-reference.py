#!/usr/bin/env python3
"""Print full log-mel reference values for LFM3OmniAudioFeaturesTests.

Generated with numpy 2.5.4 and transformers 5.22.2. Uses the Slaney filter-bank
utility and NumPy FFT with an exactly representable broadband input.
Reference: LiquidAI/LFM3-Omni-2B, revision 21d75123f86c1f7ffb4bef1ea202710c4e7ebe67.
"""
import numpy as np
from transformers.audio_utils import mel_filter_bank


def reference():
    samples = (np.arange(721) % 31).astype(np.float32) / 32 - 0.5
    samples = np.pad(samples, (0, 880 - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, 400)[::160]
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 400)).astype(np.float32)
    power = abs(np.fft.rfft(frames * window)) ** 2
    filters = mel_filter_bank(201, 128, 0, 8000, 16000, norm="slaney", mel_scale="slaney")
    return (np.maximum(np.log10(np.maximum(power @ filters, 1e-30)), -12) + 4) / 4


if __name__ == "__main__":
    for row in reference():
        for start in range(0, len(row), 8):
            print("            " + ", ".join(f"{v:.9g}" for v in row[start:start + 8]) + ",")

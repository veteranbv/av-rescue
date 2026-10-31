#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "nara-wpe>=0.0.11",
#   "numpy>=1.26",
#   "scipy>=1.11",  # imported by nara-wpe but missing from its metadata
#   "soundfile>=0.12",
# ]
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""Single-channel WPE dereverberation for speech.

Weighted prediction error (WPE) estimates the late reverberation tail with
per-frequency linear prediction in the STFT domain and subtracts it. It is a
filter, so it cannot invent or garble words, but it is mild: it shortens
heavy reverb rather than removing it.

Long files are processed in overlapping blocks joined with a linear
crossfade, which bounds memory. A recording made in one room is stationary,
so neighbouring blocks converge to the same filter and the seams are
inaudible. Output is mono 32-bit float at the input sample rate, the same
length as the input.
"""

from __future__ import annotations

import argparse

import numpy as np
import soundfile as sf
from nara_wpe.utils import istft, stft
from nara_wpe.wpe import wpe

# STFT geometry in seconds so taps and delay (counted in frames) mean the same
# thing at any sample rate: 32 ms windows, 8 ms hop.
WINDOW_SECONDS = 0.032
HOP_SECONDS = 0.008


def dereverb_block(
    y: np.ndarray, size: int, shift: int, taps: int, delay: int, iterations: int
) -> np.ndarray:
    spec = stft(y[None, :], size=size, shift=shift).transpose(2, 0, 1)  # F, D, T
    clean = wpe(
        spec, taps=taps, delay=delay, iterations=iterations, statistics_mode="full"
    ).transpose(1, 2, 0)  # D, T, F
    z = istft(clean, size=size, shift=shift)[0]
    return np.pad(z, (0, max(0, len(y) - len(z))))[: len(y)]


def dereverb(
    y: np.ndarray,
    rate: int,
    taps: int,
    delay: int,
    iterations: int,
    block_seconds: float,
    overlap_seconds: float,
) -> np.ndarray:
    size = round(WINDOW_SECONDS * rate)
    shift = round(HOP_SECONDS * rate)
    block = round(block_seconds * rate)
    overlap = round(overlap_seconds * rate)
    if not 0 < overlap < block:
        raise ValueError("overlap must be positive and shorter than the block")

    def process(segment: np.ndarray) -> np.ndarray:
        return dereverb_block(segment, size, shift, taps, delay, iterations)

    if len(y) <= block:
        return process(y)

    out = np.empty_like(y)
    fade_in = np.linspace(0.0, 1.0, overlap)
    starts = range(0, len(y), block - overlap)
    for i, start in enumerate(starts):
        end = min(start + block, len(y))
        z = process(y[start:end])
        if i == 0:
            out[start:end] = z
        else:
            # Every block after the first is longer than the overlap: the block
            # before it ended short of the input, so this one spans > overlap.
            head = slice(start, start + overlap)
            out[head] = out[head] * (1 - fade_in) + z[:overlap] * fade_in
            out[start + overlap : end] = z[overlap:]
        print(f"  block {i + 1}: {start / rate:7.1f}-{end / rate:7.1f} s", flush=True)
        if end == len(y):
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument(
        "--taps",
        type=int,
        default=10,
        help="prediction order in frames (default: 10, i.e. 80 ms)",
    )
    p.add_argument(
        "--delay",
        type=int,
        default=2,
        help="frames of direct sound and early reflections to keep "
        "(default: 2, i.e. 16 ms)",
    )
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--block-seconds", type=float, default=240.0)
    p.add_argument("--overlap-seconds", type=float, default=6.0)
    a = p.parse_args()

    y, rate = sf.read(a.input, dtype="float64", always_2d=True)
    z = dereverb(
        y.mean(axis=1),
        rate,
        a.taps,
        a.delay,
        a.iterations,
        a.block_seconds,
        a.overlap_seconds,
    )
    sf.write(a.output, z.astype(np.float32), rate, subtype="FLOAT")
    print(f"wrote {a.output}: {len(z) / rate:.1f} s at {rate} Hz")


if __name__ == "__main__":
    main()

#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy>=1.26", "scipy>=1.11", "soundfile>=0.12"]
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""Stack spectrograms of several versions of a recording for A/B review.

Reverb shows up as energy smeared to the right of each syllable, noise as a
haze between words. Compare the same stretch of each file:

  uv run restore/compare.py out.png raw.wav restored.wav --start 600 --duration 20

Each panel is scaled to its own loudest bins, so the figure compares spectral
shape rather than level. Panels share a frequency axis up to the highest
Nyquist frequency, which makes a bandwidth difference visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from matplotlib.figure import Figure
from scipy.signal import spectrogram

DYNAMIC_RANGE_DB = 70


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("output", type=Path, help="image to write, e.g. compare.png")
    p.add_argument("files", nargs="+", type=Path, help="WAV, FLAC, or OGG files")
    p.add_argument("--labels", nargs="+", help="one label per file (default: names)")
    p.add_argument("--title", default="")
    p.add_argument("--start", type=float, default=0.0, help="seconds (default: 0)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds (default: 30)")
    a = p.parse_args()

    labels = a.labels or [f.name for f in a.files]
    if len(labels) != len(a.files):
        p.error("give exactly one label per file")

    clips = []
    for path in a.files:
        rate = sf.info(str(path)).samplerate
        y, _ = sf.read(
            str(path),
            start=round(a.start * rate),
            frames=round(a.duration * rate),
            always_2d=True,
        )
        if not len(y):
            p.error(f"{path} is shorter than --start")
        clips.append((y.mean(axis=1), rate))
    top_khz = max(rate for _, rate in clips) / 2000

    fig = Figure(figsize=(12, 2.7 * len(clips)), layout="constrained")
    axes = fig.subplots(len(clips), 1, squeeze=False, sharex=True)
    for ax, (y, rate), label in zip(axes[:, 0], clips, labels):
        window = 1 << round(np.log2(0.032 * rate))
        f, t, power = spectrogram(y, rate, nperseg=window, noverlap=window * 3 // 4)
        db = 10 * np.log10(power + 1e-12)
        vmax = np.percentile(db, 99)
        ax.pcolormesh(
            a.start + t,
            f / 1000,
            db,
            shading="auto",
            cmap="magma",
            vmin=vmax - DYNAMIC_RANGE_DB,
            vmax=vmax,
            rasterized=True,
        )
        ax.set_ylim(0, top_khz)
        ax.set_facecolor("black")
        ax.set_ylabel("kHz")
        ax.set_title(label, loc="left", fontsize=11)
    axes[-1, 0].set_xlabel("seconds")
    if a.title:
        fig.suptitle(a.title, fontsize=13)
    fig.savefig(a.output, dpi=110)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()

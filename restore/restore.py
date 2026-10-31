#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Clean up a speech recording: denoise, level, and master it to a loudness.

The default chain never invents audio. DeepFilterNet removes noise, slow
leveling evens out near and far talkers, and a fixed gain into a limiter sets
the final loudness. Two optional stages go further on reverb:

  --wpe    linear-prediction dereverb. Mild, cannot change what was said.
  --sidon  generative restoration. Removes reverb that filters cannot and
           extends bandwidth to 48 kHz, but it regenerates the voice and can
           garble words in very quiet or noisy stretches. Listen to all of it.

The output is mono with the same duration and timeline as the input, so it
drops into a video edit in place of the original track.

Needs ffmpeg/ffprobe and the DeepFilterNet `deep-filter` binary on PATH.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent

# Slow, non-pumping leveling: 500 ms frames, a 71-frame (~35 s) Gaussian
# window, 0.9 peak target, at most 20x (26 dB) gain, light compression. Tuned
# on an hour-long meeting whose regions ranged from -48 to -32 LUFS, where
# fast loudness normalization pumped the noise and reverb back up.
LEVEL_FILTER = "dynaudnorm=f=500:g=71:p=0.9:m=20:s=8"

# The limiter runs at 4x the sample rate, the oversampling BS.1770 uses to
# measure true peak, so its ceiling approximates a true-peak ceiling.
OVERSAMPLE = 4
PEAK_CEILING_DB = -1.5
LIMITER = "alimiter=limit={limit:.4f}:attack=5:release=50:level=false:latency=1"

# DeepFilterNet's suppression depends on input level: a 30 s clip at -52 LUFS
# came out silent, speech and all. Measured across quiet and loud stretches of
# a real meeting, -30 LUFS kept the most speech for the noise removed. Peaks
# stay under full scale because its output is 16-bit.
DENOISE_INPUT_LUFS = -30.0
DENOISE_PEAK_DB = -1.0

# DeepFilterNet works up to 48 kHz; higher input rates are resampled to this.
MAX_WORK_RATE = 48_000

# Pad audio that starts after its container does (common in screen
# recordings) so the output timeline begins where the file's does.
FROM_ZERO = "aresample=async=1:first_pts=0"
SIDON_RATE = 48_000
SILENCE_LUFS = -70.0
# EBU R128's tolerance for meeting a loudness target. Misses under half of it
# are not worth another mastering pass.
LOUDNESS_TOLERANCE = 1.0
CORRECTION_THRESHOLD = LOUDNESS_TOLERANCE / 2


def fail(message: str) -> NoReturn:
    sys.exit(f"error: {message}")


def ffmpeg(*args: str | Path) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", *map(str, args)]
    subprocess.run(cmd, check=True)


def probe(path: Path, entries: str) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            entries,
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    streams = json.loads(out).get("streams", [])
    if not streams:
        fail(f"{path} has no audio stream")
    return streams[0]


def parse_loudnorm(stderr: str) -> tuple[float, float]:
    """Return (integrated LUFS, true peak dBTP) from loudnorm's JSON report."""
    report = json.loads(stderr[stderr.rindex("{") : stderr.rindex("}") + 1])
    return float(report["input_i"]), float(report["input_tp"])


def measure(path: Path) -> tuple[float, float]:
    stderr = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "loudnorm=print_format=json",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stderr
    return parse_loudnorm(stderr)


def master_filter(gain_db: float, rate: int, samples: int) -> str:
    """Fixed gain, oversampled peak limiter, then pad/trim to exact length."""
    limit = 10 ** (PEAK_CEILING_DB / 20)
    return ",".join(
        [
            f"volume={gain_db:.2f}dB",
            f"aresample={rate * OVERSAMPLE}",
            LIMITER.format(limit=limit),
            f"aresample={rate}",
            f"apad=whole_len={samples}",
            f"atrim=end_sample={samples}",
        ]
    )


def uv_script(name: str, *args: str | Path) -> None:
    subprocess.run(
        ["uv", "run", "--script", str(HERE / name), *map(str, args)], check=True
    )


def denoise(src: Path, dst: Path, deep_filter: str) -> None:
    loudness, peak = measure(src)
    if not math.isfinite(loudness) or loudness < SILENCE_LUFS:
        fail("the recording is silent")
    staged = dst.with_name("denoise-input.wav")
    gain = min(DENOISE_INPUT_LUFS - loudness, DENOISE_PEAK_DB - peak)
    ffmpeg("-i", src, "-af", f"volume={gain:.2f}dB", "-c:a", "pcm_f32le", staged)
    # -D compensates the model's 30 ms lookahead so the timeline is unchanged.
    out_dir = dst.parent / "deep-filter"
    subprocess.run(
        [deep_filter, "--pf", "-D", "-o", str(out_dir), str(staged)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    (out_dir / staged.name).rename(dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input", type=Path, help="audio or video file")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output WAV (default: INPUT.restored.wav beside the input)",
    )
    p.add_argument("--wpe", action="store_true", help="add WPE dereverb first")
    p.add_argument(
        "--sidon",
        action="store_true",
        help="add generative Sidon restoration (slow on CPU)",
    )
    p.add_argument(
        "--no-level",
        action="store_true",
        help="keep the original dynamics instead of leveling",
    )
    p.add_argument(
        "--loudness",
        type=float,
        default=-18.0,
        metavar="LUFS",
        help="integrated loudness target (default: -18)",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="device for --sidon (default: auto)",
    )
    p.add_argument(
        "--deep-filter",
        default="deep-filter",
        metavar="PATH",
        help="DeepFilterNet binary (default: deep-filter on PATH)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = args.input.resolve()
    dst = (args.output or src.with_name(f"{src.stem}.restored.wav")).resolve()
    if not src.is_file():
        fail(f"{src} not found")
    if dst == src:
        fail("output would overwrite the input")

    needed = ["ffmpeg", "ffprobe"] + (["uv"] if args.wpe or args.sidon else [])
    missing = [tool for tool in needed if not shutil.which(tool)]
    deep_filter = shutil.which(args.deep_filter)
    if not deep_filter:
        missing.append(
            f"{args.deep_filter} (https://github.com/Rikorose/DeepFilterNet/releases)"
        )
    if missing:
        fail("missing " + ", ".join(missing))

    rate = min(int(probe(src, "stream=sample_rate")["sample_rate"]), MAX_WORK_RATE)
    plan = [
        ("decode", True),
        ("wpe", args.wpe),
        ("denoise", True),
        ("level", not args.no_level),
        ("sidon", args.sidon),
        ("master", True),
    ]
    stages = [name for name, enabled in plan if enabled]

    def step(name: str) -> None:
        print(f"[{stages.index(name) + 1}/{len(stages)}] {name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="av-rescue-") as tmp:
        work = Path(tmp)
        stage = work / "0-input.wav"
        step("decode")
        ffmpeg(
            "-i",
            src,
            "-map",
            "0:a:0",
            "-af",
            FROM_ZERO,
            "-ac",
            "1",
            "-ar",
            rate,
            "-c:a",
            "pcm_f32le",
            stage,
        )
        samples = int(probe(stage, "stream=duration_ts")["duration_ts"])

        if args.wpe:
            step("wpe")
            uv_script("wpe.py", stage, work / "1-wpe.wav")
            stage = work / "1-wpe.wav"

        step("denoise")
        denoise(stage, work / "2-denoise.wav", deep_filter)
        stage = work / "2-denoise.wav"

        if not args.no_level:
            step("level")
            ffmpeg(
                "-i",
                stage,
                "-af",
                LEVEL_FILTER,
                "-c:a",
                "pcm_f32le",
                work / "3-level.wav",
            )
            stage = work / "3-level.wav"

        if args.sidon:
            step("sidon")
            uv_script("sidon.py", stage, work / "4-sidon.wav", "--device", args.device)
            stage = work / "4-sidon.wav"
            samples = round(samples * SIDON_RATE / rate)
            rate = SIDON_RATE

        step("master")
        loudness, _ = measure(stage)
        if not math.isfinite(loudness) or loudness < SILENCE_LUFS:
            fail("the recording is silent after cleanup; nothing to master")
        # Limiting costs loudness, so a second pass adds back what the first
        # fell short by, as loudnorm's two-pass mode does.
        gain = args.loudness - loudness
        for _ in range(2):
            ffmpeg(
                "-i",
                stage,
                "-af",
                master_filter(gain, rate, samples),
                "-c:a",
                "pcm_s16le",
                dst,
            )
            final_lufs, final_tp = measure(dst)
            if args.loudness - final_lufs <= CORRECTION_THRESHOLD:
                break
            gain += args.loudness - final_lufs

    print(
        f"wrote {dst}\n  {samples / rate:.2f} s at {rate} Hz, "
        f"{final_lufs:.1f} LUFS, {final_tp:.1f} dBTP"
    )
    if args.loudness - final_lufs > LOUDNESS_TOLERANCE:
        advice = (
            "Leveling first gets closer: drop --no-level."
            if args.no_level
            else "That is as loud as this recording goes cleanly."
        )
        print(
            f"  note: reaching {args.loudness:g} LUFS would take heavy limiting, "
            f"so the limiter stopped at {final_lufs:.1f}. {advice}"
        )


if __name__ == "__main__":
    # Exit normally on SIGTERM so the temp directory of intermediate audio,
    # possibly from a private recording, is removed rather than left behind.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        main()
    except subprocess.CalledProcessError as e:
        fail(f"{Path(e.cmd[0]).name} failed (exit {e.returncode})")
    except KeyboardInterrupt:
        sys.exit(130)

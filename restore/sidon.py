#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface-hub>=0.25",
#   "numpy>=1.26",
#   "soundfile>=0.12",
#   "torch>=2.5",
#   "torchaudio>=2.5",
#   "transformers>=4.45",
# ]
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""Generative speech restoration with Sidon (sarulab-speech/sidon-v0.1, MIT).

Sidon re-synthesizes the voice from w2v-BERT 2.0 speech features, so it
removes reverb and noise that filters cannot and outputs 48 kHz audio. The
cost of regenerating the waveform: it can garble words where the input is
very quiet or noisy. Spot-check the whole file, not just the opening.

The inference recipe follows the authors' demo Space,
huggingface.co/spaces/sarulab-speech/sidon_demo_beta. The Sidon repository's
infer.py targets the two-speaker DialogueSidon model and is not this path.

Weights (about 1 GB) download from the Hugging Face Hub on first run and are
cached. Pass --feature-extractor and --decoder to use local copies offline.
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
import transformers
from huggingface_hub import hf_hub_download

# Model files are pinned to the commits this script was tested with, so a
# change upstream cannot swap in different code-bearing TorchScript files.
REPO = "sarulab-speech/sidon-v0.1"
REPO_REVISION = "b3b02d8bbd55fdbc410e6e46e76ef95ace4fbf52"
FEATURE_MODEL = "facebook/w2v-bert-2.0"
FEATURE_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
RATE_IN = 16_000
RATE_OUT = 48_000

# Values from the reference implementation.
CHUNK_SECONDS = 96  # decode in 96 s chunks, carrying one feature frame over
EDGE_PAD = 160  # 10 ms of zeros either side of each chunk before features
TAIL_PAD = 24_000  # 1.5 s of zeros so the last chunk decodes completely
FRAME_OUT = 960  # one 20 ms feature frame at 48 kHz, trimmed per chunk
HIGHPASS_HZ = 50
PEAK = 0.9

# The published weights exist only as TorchScript, so torch's deprecation
# notice for jit.load is not actionable; exclude-newer pins a torch that loads
# them.
warnings.filterwarnings("ignore", message=r"`torch\.jit\.load` is deprecated")


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def load_models(device: str, fe_path: str | None, decoder_path: str | None):
    # TorchScript exports are per device family; CUDA weights only run on CUDA.
    variant = "cuda" if device.startswith("cuda") else "cpu"
    fe_path = fe_path or hf_hub_download(
        REPO, f"feature_extractor_{variant}.pt", revision=REPO_REVISION
    )
    decoder_path = decoder_path or hf_hub_download(
        REPO, f"decoder_{variant}.pt", revision=REPO_REVISION
    )
    preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
        FEATURE_MODEL, revision=FEATURE_REVISION
    )
    fe = torch.jit.load(fe_path, map_location=device).eval()
    decoder = torch.jit.load(decoder_path, map_location=device).eval()
    return preprocessor, fe, decoder


@torch.inference_mode()
def restore(audio: np.ndarray, rate: int, models, device: str) -> np.ndarray:
    """Restore mono float audio at any rate; return 48 kHz audio, same duration."""
    preprocessor, fe, decoder = models
    wav = torch.from_numpy(audio).float().view(1, -1)
    wav = PEAK * wav / wav.abs().max().clamp_min(1e-6)
    out_len = round(wav.shape[-1] * RATE_OUT / rate)

    wav = AF.highpass_biquad(wav, rate, HIGHPASS_HZ)
    wav = AF.resample(wav, rate, RATE_IN)
    wav = torch.nn.functional.pad(wav, (0, TAIL_PAD)).view(-1)

    chunks = wav.split(RATE_IN * CHUNK_SECONDS)
    restored, carry, started = [], None, time.monotonic()
    for i, chunk in enumerate(chunks, 1):
        inputs = preprocessor(
            torch.nn.functional.pad(chunk, (EDGE_PAD, EDGE_PAD)),
            sampling_rate=RATE_IN,
            return_tensors="pt",
        )
        feat = fe(inputs["input_features"].to(device))["last_hidden_state"]
        if carry is not None:
            feat = torch.cat([carry, feat], dim=1)
        restored.append(decoder(feat.transpose(1, 2)).view(-1)[:-FRAME_OUT].cpu())
        carry = feat[:, -1:]
        elapsed = time.monotonic() - started
        eta = elapsed / i * (len(chunks) - i)
        print(
            f"  chunk {i}/{len(chunks)}  elapsed {elapsed:5.0f} s  eta {eta:5.0f} s",
            flush=True,
        )

    return torch.cat(restored)[:out_len].numpy()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("input")
    p.add_argument("output", help="48 kHz mono 32-bit float WAV")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--feature-extractor",
        metavar="PATH",
        help="local feature_extractor_{cpu,cuda}.pt",
    )
    p.add_argument("--decoder", metavar="PATH", help="local decoder_{cpu,cuda}.pt")
    a = p.parse_args()

    device = pick_device(a.device)
    audio, rate = sf.read(a.input, dtype="float32", always_2d=True)
    print(f"sidon on {device}: {len(audio) / rate:.1f} s of audio", flush=True)
    models = load_models(device, a.feature_extractor, a.decoder)
    out = restore(audio.mean(axis=1), rate, models, device)
    sf.write(a.output, out, RATE_OUT, subtype="FLOAT")
    print(f"wrote {a.output}: {len(out) / RATE_OUT:.1f} s at {RATE_OUT} Hz")


if __name__ == "__main__":
    main()

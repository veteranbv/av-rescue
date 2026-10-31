import importlib.util
import shutil
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    # dataclasses look their module up in sys.modules while it executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def speech_like(seconds: float, rate: int, seed: int) -> np.ndarray:
    """Noise bursts with syllable-like lengths, gaps, and loudness."""
    rng = np.random.default_rng(seed)
    gate = np.zeros(round(seconds * rate))
    t = 0.0
    while t < seconds:
        t += rng.uniform(0.05, 0.6)
        length = rng.uniform(0.08, 0.35)
        gate[round(t * rate) : round(min(seconds, t + length) * rate)] = rng.uniform(
            0.2, 1
        )
        t += length
    return 0.3 * gate * rng.standard_normal(len(gate))


def write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture(scope="session")
def sync():
    return load("av_sync", "sync/sync.py")


@pytest.fixture(scope="session")
def restore():
    return load("av_restore", "restore/restore.py")


@pytest.fixture
def ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("needs ffmpeg and ffprobe")

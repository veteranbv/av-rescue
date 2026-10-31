import subprocess
import wave

import numpy as np
import pytest
from conftest import speech_like, write_wav

LOUDNORM_REPORT = """
[Parsed_loudnorm_0 @ 0x600]
{
	"input_i" : "-25.31",
	"input_tp" : "-3.02",
	"input_lra" : "11.50",
	"input_thresh" : "-35.80",
	"output_i" : "-24.00",
	"output_tp" : "-2.00",
	"output_lra" : "9.10",
	"output_thresh" : "-34.40",
	"normalization_type" : "dynamic",
	"target_offset" : "0.10"
}
"""


def test_parse_loudnorm(restore):
    assert restore.parse_loudnorm("ffmpeg log noise\n" + LOUDNORM_REPORT) == (
        -25.31,
        -3.02,
    )


def test_parse_loudnorm_silence(restore):
    report = LOUDNORM_REPORT.replace('"-25.31"', '"-inf"')
    assert restore.parse_loudnorm(report)[0] == float("-inf")


def test_master_filter_keeps_length_and_timing(restore):
    chain = restore.master_filter(4.5, 16_000, 320_000)
    assert chain.startswith("volume=4.50dB,aresample=64000,alimiter=")
    assert "latency=1" in chain
    assert chain.endswith("apad=whole_len=320000,atrim=end_sample=320000")


def read_wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)


def test_master_stage_is_sample_exact(restore, ffmpeg, tmp_path):
    """Mastering adds gain and limits peaks without moving or resizing audio."""
    rate = 16_000
    source = speech_like(12, rate, seed=7) * 0.2
    src = write_wav(tmp_path / "in.wav", source, rate)
    out = tmp_path / "out.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-af",
            restore.master_filter(12.0, rate, len(source)),
            str(out),
        ],
        check=True,
    )
    a, b = read_wav(src), read_wav(out)
    assert len(b) == len(a)
    corr = np.correlate(b[1000:-1000], a[1000 + 50 : -1000 - 50], mode="valid")
    assert int(np.argmax(corr)) - 50 == 0
    lufs, true_peak = restore.measure(out)
    assert true_peak <= restore.PEAK_CEILING_DB + 0.5
    assert lufs == pytest.approx(restore.measure(src)[0] + 12, abs=1.5)

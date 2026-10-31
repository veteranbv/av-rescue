import http.client
import json
import subprocess
import threading
from pathlib import Path

import numpy as np
import pytest
from conftest import speech_like, write_wav

ANALYSIS_RATE = 8_000


def envelope_db(x: np.ndarray, sync) -> np.ndarray:
    """The server's envelope, computed directly from samples at 8 kHz."""
    hop = ANALYSIS_RATE // sync.ENV_RATE
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    return (10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)).astype(np.float32)


def camera_track(
    source: np.ndarray, rate: float, offset: float, seconds: float, seed: int
) -> np.ndarray:
    """What a second device hears: source at audio time rate*t + offset, plus noise."""
    t = np.arange(round(seconds * ANALYSIS_RATE)) / ANALYSIS_RATE
    src_t = np.arange(len(source)) / ANALYSIS_RATE
    heard = np.interp(rate * t + offset, src_t, source, left=0, right=0)
    noise = np.random.default_rng(seed).standard_normal(len(t)) * 0.003
    return 0.4 * heard + noise


# --- auto-sync -------------------------------------------------------------------


def test_estimate_recovers_offset(sync):
    source = speech_like(400, ANALYSIS_RATE, seed=1)
    camera = camera_track(source, 1.0, 50.01375, 300, seed=2)
    result = sync.estimate(envelope_db(camera, sync), envelope_db(source, sync))
    (point,) = result["anchors"]
    assert point["audio"] - point["video"] == pytest.approx(50.01375, abs=0.003)
    assert result["confident"]
    assert result["driftPpm"] == 0


def test_estimate_measures_drift(sync):
    source = speech_like(700, ANALYSIS_RATE, seed=3)
    camera = camera_track(source, 1.0003, 20.0, 600, seed=4)
    result = sync.estimate(envelope_db(camera, sync), envelope_db(source, sync))
    assert result["driftPpm"] == pytest.approx(300, abs=20)
    for point in result["anchors"]:
        assert point["audio"] == pytest.approx(
            1.0003 * point["video"] + 20.0, abs=0.005
        )


def test_estimate_rejects_silence(sync):
    silent = np.full(10_000, -120, np.float32)
    with pytest.raises(sync.ApiError):
        sync.estimate(silent, silent)


# --- sync model ----------------------------------------------------------------------


def test_linear_map_single_point(sync):
    assert sync.linear_map([{"video": 10, "audio": 55.5}]) == (1.0, pytest.approx(45.5))


def test_linear_map_two_points(sync):
    rate, offset = sync.linear_map(
        [{"video": 0, "audio": 5}, {"video": 1000, "audio": 1005.2}]
    )
    assert rate == pytest.approx(1.0002)
    assert offset == pytest.approx(5)


@pytest.mark.parametrize(
    "anchors",
    [
        [],
        [{"video": 1, "audio": 2}] * 3,
        [{"video": -1, "audio": 2}],
        [{"video": "x", "audio": 2}],
        [{"video": 1}],
        [{"video": 10, "audio": 20}, {"video": 10.5, "audio": 21}],  # too close
        [{"video": 10, "audio": 20}, {"video": 100, "audio": 200}],  # 2x speed
    ],
)
def test_parse_anchors_rejects(sync, anchors):
    with pytest.raises(sync.ApiError):
        sync.parse_anchors(anchors)


def test_parse_anchors_sorts(sync):
    anchors = sync.parse_anchors(
        [{"video": 90, "audio": 100}, {"video": 10, "audio": 20}]
    )
    assert [p["video"] for p in anchors] == [10, 90]


# --- export ------------------------------------------------------------------------------


def media(sync, path: str, is_video: bool, duration: float, rate: int | None):
    return sync.Media("m", Path(path), Path(path).name, is_video, duration, rate)


def filters(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


def test_export_trims_audio_that_starts_early(sync):
    video = media(sync, "/v/talk.mp4", True, 120, 48_000)
    audio = media(sync, "/a/rec.wav", False, 600, 16_000)
    cmd = sync.export_command(
        video, audio, [{"video": 0, "audio": 2.5}], Path("/v/out.mp4")
    )
    assert filters(cmd) == (
        "[1:a:0]aresample=async=1:first_pts=0,atrim=start_sample=40000,"
        "asetpts=PTS-STARTPTS,aresample=48000,apad[a]"
    )
    assert cmd[-1] == "/v/out.mp4" and "+faststart" in cmd
    assert cmd[cmd.index("-t") + 1] == "120.000000"


def test_export_delays_late_audio_and_corrects_drift(sync):
    video = media(sync, "/v/talk.mkv", True, 120, None)
    audio = media(sync, "/a/rec.wav", False, 600, 16_000)
    anchors = [{"video": 0, "audio": -1.5}, {"video": 100, "audio": 98.6}]
    graph = filters(sync.export_command(video, audio, anchors, Path("/v/out.mkv")))
    assert graph == (
        "[1:a:0]aresample=async=1:first_pts=0,"
        "aresample=384000,asetrate=384384,aresample=48000,"
        f"adelay=delays={round(1.5 / 1.001 * 48_000)}S:all=1,apad[a]"
    )


def test_export_container_choice(sync):
    for name, expected in [
        ("a.mov", "a.synced.mov"),
        ("a.MP4", "a.synced.mp4"),
        ("a.avi", "a.synced.mkv"),
        ("a.webm", "a.synced.webm"),
    ]:
        assert (
            sync.export_path(media(sync, f"/v/{name}", True, 10, None)).name == expected
        )
    webm = media(sync, "/v/a.webm", True, 10, None)
    cmd = sync.export_command(
        webm,
        media(sync, "/a/r.wav", False, 10, 8000),
        [{"video": 0, "audio": 0}],
        sync.export_path(webm),
    )
    assert "libopus" in cmd and "+faststart" not in cmd


@pytest.mark.parametrize("rate,offset", [(1.0, 7.25), (1.004, -3.0)])
def test_export_end_to_end(sync, ffmpeg, tmp_path, rate, offset):
    """Exported audio lines up with the camera's own track across the whole video."""
    seconds = 200
    source = speech_like(seconds + 20, ANALYSIS_RATE, seed=5)
    recorder = write_wav(tmp_path / "recorder.wav", source, ANALYSIS_RATE)
    scratch = write_wav(
        tmp_path / "scratch.wav",
        camera_track(source, rate, offset, seconds, seed=6),
        ANALYSIS_RATE,
    )
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s=64x48:r=5:d={seconds}",
            "-i",
            str(scratch),
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(clip),
        ],
        check=True,
    )
    video = sync.Media("v", clip, clip.name, *sync.probe(clip))
    audio = sync.Media("a", recorder, recorder.name, *sync.probe(recorder))
    anchors = [{"video": v, "audio": rate * v + offset} for v in (10, 190)]
    out = tmp_path / "out.mp4"
    subprocess.run(
        sync.export_command(video, audio, anchors, out), check=True, capture_output=True
    )

    result = sync.estimate(sync.envelope(clip), sync.envelope(out))
    assert result["confident"]
    assert abs(result["driftPpm"]) < 100
    for point in result["anchors"]:
        assert point["audio"] == pytest.approx(point["video"], abs=0.005)


def test_envelope_keeps_late_audio_in_place(sync, ffmpeg, tmp_path):
    """Audio that starts after the picture is padded, not pulled to time 0."""
    tone = write_wav(
        tmp_path / "tone.wav", np.full(ANALYSIS_RATE * 5, 0.3), ANALYSIS_RATE
    )
    clip = tmp_path / "late.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=64x48:r=5:d=10",
         "-itsoffset", "1.5", "-i", str(tone), "-map", "0:v", "-map", "1:a",
         "-c:v", "mpeg4", "-c:a", "aac", str(clip)],
        check=True,
    )  # fmt: skip
    db = sync.envelope(clip)
    onset = int(np.argmax(db > db.max() - 20)) / sync.ENV_RATE
    assert onset == pytest.approx(1.5, abs=0.05)


# --- saved state -----------------------------------------------------------------------


def test_state_round_trip(sync, tmp_path):
    video = media(sync, str(tmp_path / "talk.mp4"), True, 60, None)
    audio = media(sync, str(tmp_path / "rec.wav"), False, 60, 16_000)
    anchors = [{"video": 1.0, "audio": 3.5}]
    sync.save_state(video, audio, anchors)
    saved = json.loads((tmp_path / "talk.mp4.av-sync.json").read_text())
    assert saved == {"audio": {"rec.wav": {"anchors": anchors}}}
    sync.save_state(video, audio, [])
    assert sync.load_state(video) == {"audio": {}}


def test_invalid_saved_points_are_ignored(sync, tmp_path):
    video = media(sync, str(tmp_path / "talk.mp4"), True, 60, None)
    audio = media(sync, str(tmp_path / "rec.wav"), False, 60, 16_000)
    state = tmp_path / "talk.mp4.av-sync.json"
    state.write_text(json.dumps({"audio": {"rec.wav": {"anchors": [
        {"video": 1, "audio": 2}, {"video": 1.2, "audio": 9}]}}}))  # fmt: skip
    assert sync.saved_anchors(video, audio) is None
    state.write_text("[]")
    with pytest.raises(sync.ApiError):
        sync.load_state(video)


# --- HTTP ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, None),
        ("", None),
        ("bytes=0-99", (0, 99)),
        ("bytes=100-", (100, 999)),
        ("bytes=-100", (900, 999)),
        ("bytes=-5000", (0, 999)),
        ("bytes=990-5000", (990, 999)),
        ("bytes=0-1,5-6", None),
        ("items=0-1", None),
        ("bytes=abc", None),
        ("bytes=50-10", None),
    ],
)
def test_parse_range(sync, header, expected):
    assert sync.parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=-0"])
def test_parse_range_unsatisfiable(sync, header):
    with pytest.raises(ValueError):
        sync.parse_range(header, 1000)


@pytest.fixture
def server(sync, tmp_path):
    data = tmp_path / "rec.wav"
    data.write_bytes(bytes(range(256)) * 4)
    app = sync.App(
        [
            media(sync, str(tmp_path / "v.mp4"), True, 10, None),
            sync.Media("m1", data, "rec.wav", False, 10, 8000),
        ]
    )
    handler = type("TestHandler", (sync.Handler,), {"app": app})
    httpd = sync.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def request(port: int, path: str, headers: dict | None = None):
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", path, headers=headers or {})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_serves_media_list(server):
    status, _, body = request(server, "/api/media")
    assert status == 200
    assert [m["kind"] for m in json.loads(body)["media"]] == ["video", "audio"]


def test_serves_byte_ranges(server):
    status, headers, body = request(server, "/media/m1", {"Range": "bytes=10-19"})
    assert status == 206
    assert headers["Content-Range"] == "bytes 10-19/1024"
    assert body == bytes(range(10, 20))


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://example.com"},
        {"Host": "attacker.example"},
    ],
)
def test_refuses_other_sites(server, headers):
    status, _, _ = request(server, "/api/media", headers)
    assert status == 403


def test_unknown_media_is_404(server):
    assert request(server, "/media/nope")[0] == 404


def test_bad_content_length_is_400(server):
    conn = http.client.HTTPConnection("127.0.0.1", server)
    conn.putrequest("POST", "/api/autosync")
    conn.putheader("Content-Length", "lots")
    conn.endheaders()
    assert conn.getresponse().status == 400

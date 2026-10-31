#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.26"]
# [tool.uv]
# exclude-newer = "2026-10-31T00:00:00Z"
# ///
"""Line up separately recorded audio with a video, then export the result.

  uv run sync/sync.py screen.mp4 meeting.wav [more audio ...]
  uv run sync/sync.py ~/Recordings/standup     # every media file in a folder

Opens a page in your browser. Slide the audio under the video until the lips
match, or press Auto-sync when the video has its own scratch audio. Two sync
points correct clock drift between devices. Export writes a copy of the video
with the new track; the picture is copied, not re-encoded.

Alignment is saved beside the video (talk.mp4 gets talk.mp4.av-sync.json), so
you can stop and come back. The server listens on 127.0.0.1 only. Needs ffmpeg
and ffprobe.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
import webbrowser
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

PAGE = Path(__file__).resolve().parent / "index.html"
STATE_SUFFIX = ".av-sync.json"

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aif": "audio/aiff",
    ".aiff": "audio/aiff",
    ".caf": "audio/x-caf",
}
# Containers that take a copied video stream plus AAC (or Opus for WebM). Any
# other input container is exported as Matroska, which accepts every codec.
EXPORT_CONTAINERS = {".mp4", ".m4v", ".mov", ".mkv", ".webm"}
EXPORT_AUDIO_RATE = 48_000
# Drift is corrected by resampling, which is sample-accurate (atempo's
# time-stretching wanders by tens of ms). asetrate only takes whole rates, so
# the change happens at 384 kHz, where rounding costs at most 1.3 ppm (5 ms
# per hour). The pitch shift at clock-drift ratios is a fraction of a cent.
DRIFT_RESAMPLE_RATE = 384_000

# Pad audio that starts after the container does (common in screen recordings)
# so sample 0 is time 0, as it is for the picture, the page, and the export.
FROM_ZERO = "aresample=async=1:first_pts=0"

# Analysis: a speech-band loudness envelope with 10 ms frames.
DECODE_RATE = 8_000
ENV_RATE = 100
ENV_RANGE_DB = 60

# Auto-sync: correlate onset envelopes over the whole overlap, then re-check
# the lag in segments across it to detect drift between the two clocks.
MIN_OVERLAP_S = 30
PEAK_GUARD_S = 1.0
CONFIDENT_PSR = 8.0
SEGMENTS = 8
MIN_SEGMENT_S = 60
SEGMENT_SEARCH_S = 2.0
SEGMENT_GUARD_S = 0.2
SEGMENT_PSR = 6.0
OUTLIER_S = 0.03
MIN_DRIFT_S = 0.02

# Sanity bound on the audio/video speed ratio. Clock drift is well under 0.1%;
# 10% still admits PAL speed-up (4%) while rejecting misplaced sync points.
MAX_RATE_ERROR = 0.10
MIN_ANCHOR_GAP_S = 1.0
MAX_BODY = 64 * 1024


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Media:
    id: str
    path: Path
    name: str
    is_video: bool
    duration: float
    sample_rate: int | None  # None when the file has no audio stream

    def public(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "kind": "video" if self.is_video else "audio",
            "hasAudio": self.sample_rate is not None,
        }


# --- media discovery --------------------------------------------------------


def probe(path: Path) -> tuple[bool, float, int | None] | None:
    """(has picture, duration, audio sample rate) or None if not usable media."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-of",
            "json",
            "-show_entries",
            "format=duration:stream=codec_type,sample_rate:stream_disposition=attached_pic",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        return None
    info = json.loads(result.stdout)
    streams = info.get("streams", [])
    is_video = any(
        s.get("codec_type") == "video"
        and not s.get("disposition", {}).get("attached_pic")
        for s in streams
    )
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    try:
        duration = float(info.get("format", {}).get("duration") or 0)
    except ValueError:  # "N/A" for some live-style containers
        return None
    if duration < 1 or not (is_video or audio):
        return None
    return is_video, duration, int(audio[0]["sample_rate"]) if audio else None


def discover(inputs: list[Path]) -> list[Media]:
    files: list[Path] = []
    for item in inputs:
        item = item.expanduser().resolve()
        if item.is_dir():
            files += sorted(
                p
                for p in item.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in MEDIA_TYPES
            )
        elif item.is_file():
            files.append(item)
        else:
            sys.exit(f"error: {item} not found")
    files = list(dict.fromkeys(files))
    names = [p.name for p in files]

    media = []
    for path in files:
        info = probe(path)
        if info is None:
            print(f"skipping {path.name}: not a readable audio or video file")
            continue
        name = (
            path.name
            if names.count(path.name) == 1
            else f"{path.parent.name}/{path.name}"
        )
        media.append(Media(f"m{len(media)}", path, name, *info))
    return media


# --- analysis ---------------------------------------------------------------


def envelope(path: Path) -> np.ndarray:
    """Speech-band loudness in dB, ENV_RATE frames per second."""
    hop = DECODE_RATE // ENV_RATE
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-af",
        f"{FROM_ZERO},highpass=f=150,lowpass=f=3800",
        "-ar",
        str(DECODE_RATE),
        "-f",
        "f32le",
        "-",
    ]
    power, rest = [], np.empty(0, np.float32)
    with tempfile.TemporaryFile() as errors:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=errors) as proc:
            while chunk := proc.stdout.read(hop * 4 * 8192):
                x = np.concatenate([rest, np.frombuffer(chunk, np.float32)])
                whole = len(x) // hop * hop
                power.append(
                    np.mean(
                        np.square(x[:whole].reshape(-1, hop), dtype=np.float64), axis=1
                    )
                )
                rest = x[whole:]
        if proc.returncode:
            errors.seek(0)
            message = errors.read().decode(errors="replace").strip()
            raise ApiError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"could not decode {path.name}: {message}",
            )
    return (10 * np.log10(np.concatenate(power) + 1e-12)).astype(np.float32)


def display_envelope(db: np.ndarray) -> bytes:
    """Envelope quantized to 0-255 over ENV_RANGE_DB below the loud frames."""
    top = np.percentile(db, 99.5)
    scaled = (db - (top - ENV_RANGE_DB)) / ENV_RANGE_DB
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8).tobytes()


def onsets(db: np.ndarray) -> np.ndarray:
    """Normalized rises in loudness: the feature two microphones agree on."""
    level = np.maximum(db, np.percentile(db, 99.5) - ENV_RANGE_DB)
    rise = np.maximum(np.diff(level, prepend=level[:1]), 0)
    spread = rise.std()
    if spread < 1e-6:
        raise ApiError(HTTPStatus.UNPROCESSABLE_ENTITY, "one of the tracks is silent")
    return (rise - rise.mean()) / spread


def peak_offset(y: np.ndarray, i: int) -> float:
    """Sub-frame position of a peak by parabolic interpolation."""
    if 0 < i < len(y) - 1:
        left, mid, right = y[i - 1], y[i], y[i + 1]
        curve = left - 2 * mid + right
        if curve < 0:
            return float(0.5 * (left - right) / curve)
    return 0.0


def sharpness(c: np.ndarray, i: int, guard: int) -> float:
    """Peak-to-sidelobe ratio: how far the peak stands above the rest."""
    side = np.concatenate([c[: max(0, i - guard)], c[i + guard + 1 :]])
    if len(side) < 2 or side.std() == 0:
        return 0.0
    return float((c[i] - side.mean()) / side.std())


def segment_lag(
    v: np.ndarray, a: np.ndarray, start: int, stop: int, center: int, radius: int
) -> tuple[float, float]:
    """(lag, sharpness) for video frames [start, stop), searching center +- radius."""
    lo, hi = start + center - radius, stop + center + radius
    window = np.pad(a[max(0, lo) : max(0, hi)], (max(0, -lo), max(0, hi - len(a))))
    c = np.correlate(window, v[start:stop], mode="valid")
    i = int(np.argmax(c))
    return center - radius + i + peak_offset(c, i), sharpness(
        c, i, round(SEGMENT_GUARD_S * ENV_RATE)
    )


def estimate(v_db: np.ndarray, a_db: np.ndarray) -> dict:
    """Sync points mapping video time to audio time, from two envelopes.

    Lag k means audio frame t + k sounds at video frame t.
    """
    v, a = onsets(v_db), onsets(a_db)
    size = 1 << (len(v) + len(a) - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, size) * np.conj(np.fft.rfft(v, size)), size)
    lags = np.arange(-(len(v) - 1), len(a))
    overlap = np.minimum(len(v), len(a) - lags) - np.maximum(0, -lags)
    usable = overlap >= MIN_OVERLAP_S * ENV_RATE
    if not usable.any():
        raise ApiError(
            HTTPStatus.UNPROCESSABLE_ENTITY,
            f"the tracks need at least {MIN_OVERLAP_S} s in common",
        )
    lags, c = lags[usable], corr[lags[usable] % size]
    i = int(np.argmax(c))
    lag = lags[i] + peak_offset(c, i)
    confidence = sharpness(c, i, round(PEAK_GUARD_S * ENV_RATE))

    start = max(0, -lags[i])
    span = min(len(v), len(a) - lags[i]) - start
    count = min(SEGMENTS, span // (MIN_SEGMENT_S * ENV_RATE))
    points = []
    for s in range(count):
        lo = start + s * span // count
        hi = start + (s + 1) * span // count
        seg, seg_sharpness = segment_lag(
            v, a, lo, hi, int(lags[i]), round(SEGMENT_SEARCH_S * ENV_RATE)
        )
        if seg_sharpness >= SEGMENT_PSR:
            points.append(((lo + hi) / 2, seg))

    anchors = [(start + span / 2, start + span / 2 + lag)]
    drift = 0.0
    if len(points) >= 3:
        t, k = np.array(points).T
        slope, intercept = np.polyfit(t, k, 1)
        keep = np.abs(k - (slope * t + intercept)) <= OUTLIER_S * ENV_RATE
        if keep.sum() >= 3:
            t, k = t[keep], k[keep]
            slope, intercept = np.polyfit(t, k, 1)
            if abs(slope * (t[-1] - t[0])) >= MIN_DRIFT_S * ENV_RATE:
                drift = float(slope)
                anchors = [(x, x + slope * x + intercept) for x in (t[0], t[-1])]

    return {
        "anchors": [{"video": x / ENV_RATE, "audio": y / ENV_RATE} for x, y in anchors],
        "confidence": round(confidence, 1),
        "confident": confidence >= CONFIDENT_PSR,
        "driftPpm": round(drift * 1e6, 1),
    }


# --- sync points and export --------------------------------------------------


def parse_anchors(raw: object) -> list[dict]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 2:
        raise ApiError(HTTPStatus.BAD_REQUEST, "expected one or two sync points")
    anchors = []
    for point in raw:
        try:
            video, audio = float(point["video"]), float(point["audio"])
        except (KeyError, TypeError, ValueError):
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "sync points need video and audio times"
            )
        if not (math.isfinite(video) and math.isfinite(audio)) or video < 0:
            raise ApiError(HTTPStatus.BAD_REQUEST, "sync point times are out of range")
        anchors.append({"video": video, "audio": audio})
    anchors.sort(key=lambda p: p["video"])
    linear_map(anchors)
    return anchors


def linear_map(anchors: list[dict]) -> tuple[float, float]:
    """(rate, offset) with audio_time = rate * video_time + offset."""
    first = anchors[0]
    if len(anchors) == 1:
        return 1.0, first["audio"] - first["video"]
    last = anchors[-1]
    gap = last["video"] - first["video"]
    if gap < MIN_ANCHOR_GAP_S:
        raise ApiError(HTTPStatus.BAD_REQUEST, "sync points must be at least 1 s apart")
    rate = (last["audio"] - first["audio"]) / gap
    if abs(rate - 1) > MAX_RATE_ERROR:
        raise ApiError(
            HTTPStatus.BAD_REQUEST,
            "those sync points imply a speed change over 10%; one is misplaced",
        )
    return rate, first["audio"] - rate * first["video"]


def export_path(video: Media) -> Path:
    suffix = video.path.suffix.lower()
    suffix = suffix if suffix in EXPORT_CONTAINERS else ".mkv"
    return video.path.with_name(f"{video.path.stem}.synced{suffix}")


def export_command(
    video: Media, audio: Media, anchors: list[dict], out: Path
) -> list[str]:
    rate, offset = linear_map(anchors)
    chain = [FROM_ZERO]
    if offset > 0:
        chain += [
            f"atrim=start_sample={round(offset * audio.sample_rate)}",
            "asetpts=PTS-STARTPTS",
        ]
    if rate != 1:
        chain += [
            f"aresample={DRIFT_RESAMPLE_RATE}",
            f"asetrate={round(DRIFT_RESAMPLE_RATE * rate)}",
        ]
    chain.append(f"aresample={EXPORT_AUDIO_RATE}")
    if offset < 0:
        chain.append(
            f"adelay=delays={round(-offset / rate * EXPORT_AUDIO_RATE)}S:all=1"
        )
    chain.append("apad")

    if out.suffix == ".webm":
        codec = ["-c:a", "libopus", "-b:a", "128k"]
    else:
        codec = ["-c:a", "aac", "-b:a", "192k"]
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(video.path),
        "-i",
        str(audio.path),
        "-filter_complex",
        f"[1:a:0]{','.join(chain)}[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        *codec,
        "-t",
        f"{video.duration:.6f}",
    ]
    if out.suffix in {".mp4", ".m4v", ".mov"}:
        cmd += ["-movflags", "+faststart"]
    return cmd + [str(out)]


class ExportJob:
    """One ffmpeg export at a time, with progress for the page to poll."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: dict = {"state": "idle"}
        self._proc: subprocess.Popen | None = None
        self._partial: Path | None = None

    def cancel(self) -> None:
        """Stop a running export and remove its partial file."""
        with self._lock:
            proc, partial = self._proc, self._partial
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait()
            partial.unlink(missing_ok=True)

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def start(self, cmd: list[str], out: Path, duration: float) -> None:
        with self._lock:
            if self._status["state"] == "running":
                raise ApiError(HTTPStatus.CONFLICT, "an export is already running")
            self._status = {
                "state": "running",
                "progress": 0.0,
                "output": str(out),
                "command": shlex.join(cmd),
            }
        threading.Thread(
            target=self._run, args=(cmd, out, duration), daemon=True
        ).start()

    def _update(self, **changes) -> None:
        with self._lock:
            self._status.update(changes)

    def _run(self, cmd: list[str], out: Path, duration: float) -> None:
        # Write beside the target and rename on success, so a failed export
        # never leaves a truncated file where a good one used to be.
        partial = out.with_name(f".{out.stem}.partial{out.suffix}")
        try:
            self._ffmpeg(
                [
                    *cmd[:-1],
                    "-v",
                    "error",
                    "-nostats",
                    "-progress",
                    "pipe:1",
                    str(partial),
                ],
                partial,
                out,
                duration,
            )
        except Exception as e:  # noqa: BLE001  thread boundary: report anything, or the page waits forever
            partial.unlink(missing_ok=True)
            self._update(state="error", error=str(e))

    def _ffmpeg(
        self, argv: list[str], partial: Path, out: Path, duration: float
    ) -> None:
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(
                argv, stdout=subprocess.PIPE, stderr=errors, text=True
            )
            with self._lock:
                self._proc, self._partial = proc, partial
            for line in proc.stdout:
                key, _, value = line.strip().partition("=")
                if key == "out_time_us" and value.isdigit():
                    self._update(progress=min(1.0, int(value) / 1e6 / duration))
            proc.wait()
            if proc.returncode == 0:
                partial.replace(out)
                self._update(state="done", progress=1.0)
                return
            errors.seek(0)
            tail = errors.read().decode(errors="replace").strip().splitlines()[-12:]
            partial.unlink(missing_ok=True)
            self._update(
                state="error",
                error="\n".join(tail) or f"ffmpeg exited {proc.returncode}",
            )


# --- saved sync points --------------------------------------------------------


def state_file(video: Media) -> Path:
    return video.path.with_name(video.path.name + STATE_SUFFIX)


def state_key(video: Media, audio: Media) -> str:
    return os.path.relpath(audio.path, video.path.parent)


def load_state(video: Media) -> dict:
    try:
        state = json.loads(state_file(video).read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"cannot read saved sync: {e}")
    if not isinstance(state, dict) or not isinstance(state.get("audio", {}), dict):
        raise ApiError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            f"{state_file(video).name} is not a saved sync file",
        )
    return state


def saved_anchors(video: Media, audio: Media) -> list[dict] | None:
    """The pair's saved sync points, or None if missing or no longer valid."""
    entry = load_state(video).get("audio", {}).get(state_key(video, audio))
    try:
        return parse_anchors(entry["anchors"])
    except (ApiError, KeyError, TypeError):
        return None


# Requests are handled on threads; serialize the file's read-modify-write.
_state_lock = threading.Lock()


def save_state(video: Media, audio: Media, anchors: list[dict]) -> None:
    """Save the pair's sync points; an empty list forgets them."""
    with _state_lock:
        _write_state(video, audio, anchors)


def _write_state(video: Media, audio: Media, anchors: list[dict]) -> None:
    state = load_state(video)
    pairs = state.setdefault("audio", {})
    if anchors:
        pairs[state_key(video, audio)] = {"anchors": anchors}
    else:
        pairs.pop(state_key(video, audio), None)
    path = state_file(video)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(path)
    except OSError as e:
        raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR, f"cannot save sync: {e}")


# --- HTTP ---------------------------------------------------------------------


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Byte range from a Range header, None to send the whole file.

    Raises ValueError when the range cannot be satisfied (HTTP 416). Multiple
    ranges and malformed headers are ignored, which RFC 9110 allows.
    """
    if not header or not header.startswith("bytes=") or "," in header:
        return None
    first, sep, last = header[6:].strip().partition("-")
    if not sep or not (first.isdigit() or last.isdigit()):
        return None
    if not first:  # suffix range: the final N bytes
        length = int(last)
        if length == 0 or size == 0:
            raise ValueError("empty suffix range")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last.isdigit() else size - 1
    if start >= size:
        raise ValueError("range starts past the end")
    return None if start > end else (start, end)


class App:
    def __init__(self, media: list[Media]):
        self.media = {m.id: m for m in media}
        self.export = ExportJob()
        self._envelopes: dict[str, np.ndarray] = {}
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def get(self, media_id: str | None, *, video: bool = False) -> Media:
        item = self.media.get(media_id or "")
        if item is None or (video and not item.is_video):
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown media")
        return item

    def pair(self, video_id: str | None, audio_id: str | None) -> tuple[Media, Media]:
        video, audio = self.get(video_id, video=True), self.get(audio_id)
        if audio.sample_rate is None or audio.id == video.id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "choose a different audio file")
        return video, audio

    def envelope(self, item: Media) -> np.ndarray:
        if item.sample_rate is None:
            raise ApiError(HTTPStatus.NOT_FOUND, f"{item.name} has no audio")
        with self._locks_guard:
            lock = self._locks[item.id]
        with lock:
            if item.id not in self._envelopes:
                self._envelopes[item.id] = envelope(item.path)
            return self._envelopes[item.id]


class Handler(BaseHTTPRequestHandler):
    app: App
    server_version = "av-sync"

    # Routing --------------------------------------------------------------

    def do_GET(self) -> None:
        self._handle(self._get)

    def do_PUT(self) -> None:
        self._handle(self._put)

    def do_POST(self) -> None:
        self._handle(self._post)

    def _handle(self, route) -> None:
        url = urlsplit(self.path)
        try:
            self._check_origin()
            route(url.path, {k: v[0] for k, v in parse_qs(url.query).items()})
        except ApiError as e:
            self._json({"error": str(e)}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001  answer the page rather than drop it
            traceback.print_exc()
            self._json(
                {"error": f"server error: {e}"}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def _get(self, path: str, query: dict) -> None:
        app = self.app
        if path == "/":
            self._file(PAGE, "text/html; charset=utf-8")
        elif path.startswith("/media/"):
            item = app.get(path.removeprefix("/media/"))
            self._file(item.path, MEDIA_TYPES.get(item.path.suffix.lower()))
        elif path == "/api/media":
            self._json(
                {"envRate": ENV_RATE, "media": [m.public() for m in app.media.values()]}
            )
        elif path.startswith("/api/envelope/"):
            item = app.get(path.removeprefix("/api/envelope/"))
            self._send(
                HTTPStatus.OK,
                display_envelope(app.envelope(item)),
                "application/octet-stream",
            )
        elif path == "/api/sync":
            video, audio = app.pair(query.get("video"), query.get("audio"))
            self._json({"anchors": saved_anchors(video, audio)})
        elif path == "/api/export":
            self._json(app.export.status())
        else:
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    def _put(self, path: str, query: dict) -> None:
        if path != "/api/sync":
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")
        body = self._body()
        video, audio = self.app.pair(body.get("video"), body.get("audio"))
        raw = body.get("anchors")
        save_state(video, audio, parse_anchors(raw) if raw != [] else [])
        self._json({"ok": True})

    def _post(self, path: str, query: dict) -> None:
        body = self._body()
        video, audio = self.app.pair(body.get("video"), body.get("audio"))
        if path == "/api/autosync":
            if video.sample_rate is None:
                raise ApiError(
                    HTTPStatus.BAD_REQUEST, "this video has no audio to sync from"
                )
            self._json(estimate(self.app.envelope(video), self.app.envelope(audio)))
        elif path == "/api/export":
            anchors = parse_anchors(body.get("anchors"))
            out = export_path(video)
            if out in {video.path, audio.path}:
                raise ApiError(HTTPStatus.CONFLICT, f"{out.name} is one of the sources")
            cmd = export_command(video, audio, anchors, out)
            self.app.export.start(cmd, out, video.duration)
            self._json(self.app.export.status())
        else:
            raise ApiError(HTTPStatus.NOT_FOUND, "not found")

    # Guards ---------------------------------------------------------------

    def _check_origin(self) -> None:
        # Refuse requests that reach this port through another site: a
        # rebinding DNS name (wrong Host) or a cross-site form (wrong Origin).
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in allowed or (
            origin is not None and urlsplit(origin).netloc not in allowed
        ):
            raise ApiError(HTTPStatus.FORBIDDEN, "forbidden")

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if not 0 < length <= MAX_BODY:
            raise ApiError(HTTPStatus.BAD_REQUEST, "missing or oversized body")
        try:
            body = json.loads(self.rfile.read(length))
        except ValueError:
            raise ApiError(HTTPStatus.BAD_REQUEST, "body is not JSON")
        if not isinstance(body, dict):
            raise ApiError(HTTPStatus.BAD_REQUEST, "body must be a JSON object")
        return body

    # Responses ------------------------------------------------------------

    def _send(self, status: HTTPStatus, data: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def _file(self, path: Path, ctype: str | None) -> None:
        size = path.stat().st_size
        try:
            span = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = span or (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if span else HTTPStatus.OK)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if span:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0 and (chunk := f.read(min(1 << 16, remaining))):
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format: str, *args) -> None:
        pass


def serve(app: App, port: int, open_browser: bool) -> None:
    handler = type("BoundHandler", (Handler,), {"app": app})
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"av-sync running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            app.export.cancel()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "media", nargs="+", type=Path, help="video and audio files, or folders"
    )
    p.add_argument(
        "--port", type=int, default=8765, help="default: 8765, or any free port"
    )
    p.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = p.parse_args()

    missing = [tool for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    if missing:
        sys.exit(
            f"error: missing {', '.join(missing)} (https://ffmpeg.org/download.html)"
        )
    media = discover(args.media)
    if not any(
        v.is_video and any(a is not v and a.sample_rate for a in media) for v in media
    ):
        sys.exit("error: need a video and at least one other file with audio")
    serve(App(media), args.port, not args.no_browser)


if __name__ == "__main__":
    main()

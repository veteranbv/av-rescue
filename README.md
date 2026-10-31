# av-rescue

You recorded the video on one device and the audio on another. Maybe the screen recorder caught the picture and a separate recorder caught the sound, or you started the camera halfway through the meeting. Now you need the two back together, and the audio probably sounds like it was recorded in a stairwell.

This repo has two tools for that:

- **sync** opens a local web page where you line the audio up under the video and export a synced copy. When the video has its own scratch audio, Auto-sync does the lining up for you, including clock drift between the two devices.
- **restore** cleans up a speech recording: denoise, even out the levels, master to a sensible loudness. Optional stages go after reverb.

Both run entirely on your machine.

## Requirements

- [uv](https://docs.astral.sh/uv/). Each script declares its own dependencies (PEP 723), so there is nothing to install and no shared environment to break.
- [ffmpeg](https://ffmpeg.org/download.html), including `ffprobe`.
- For restore only: the DeepFilterNet `deep-filter` binary from its [releases page](https://github.com/Rikorose/DeepFilterNet/releases), on your PATH or passed with `--deep-filter`.

## Sync audio to video

```bash
uv run sync/sync.py screen-recording.mp4 meeting.wav
```

Give it files or folders. Every video is offered as the picture, and every other file with audio as the sound. If you have a few versions of the audio (raw, cleaned, cleaned harder), pass them all and switch between them in the page.

The page shows both tracks as loudness lanes. If the video has scratch audio, press **Auto-sync**. Otherwise:

1. In **Find** mode, play the video to a moment you can recognize, like someone starting a sentence.
2. Click or drag the audio lane until you hear the same moment, then **Lock these together**.
3. Press Play and watch the lips. Slide with the buttons or the arrow keys (10 ms, or 100 ms with Shift), or drag the audio in the close-up.
4. For a long recording, jump near the end and check again. If it has drifted, add a **second sync point** there and slide until it matches. The first point stays put.
5. **Export video.** The picture is copied untouched and the new audio goes in as AAC (Opus for WebM). The result lands next to the video as `NAME.synced.mp4`.

Your sync points are saved next to the video (`talk.mp4` gets `talk.mp4.av-sync.json`), so you can close the page and pick up later. The page also shows the ffmpeg command, so you can rerun or adapt it yourself.

### How Auto-sync works

Both tracks are reduced to a 100 Hz speech-band loudness envelope. The rises in loudness (syllable onsets) are what two different microphones agree on, even when one is a laptop mic across the room. Cross-correlating those onsets over the whole overlap gives the offset. The tool then re-measures the offset in up to eight segments across the recording and fits a line through them. A slope means the two devices' clocks run at slightly different speeds, which is normal and adds up over an hour.

The tests in `tests/` require offsets within 3 ms and drift within 20 ppm on synthetic speech. On a 10-minute clip built from a real meeting, with the scratch track degraded to sound like a laptop mic, it found a 21-minute offset within 1 ms of the true value.

The confidence score is a peak-to-sidelobe ratio. 8 or more means the match is unambiguous. Below that, check by ear.

### Drift correction

With two sync points, the export resamples the audio by the measured ratio. Resampling is sample-accurate. ffmpeg's `atempo` was tested and rejected: its time-stretching splices audio in overlapping windows, and the timing wandered by 9 to 26 ms. The pitch change from resampling at clock-drift ratios (a few hundred ppm) is a fraction of a cent, far below what anyone can hear.

## Restore speech audio

```bash
uv run restore/restore.py meeting.wav            # writes meeting.restored.wav
uv run restore/restore.py talk.mp4 -o talk.wav   # video in, audio out
```

The default chain:

1. **Denoise** with DeepFilterNet, after setting the input to the level where it keeps the most speech.
2. **Level** with a slow leveler, so quiet and loud talkers end up closer without pumping.
3. **Master** with a fixed gain to -18 LUFS (change it with `--loudness`) into a 4x-oversampled peak limiter that keeps true peak under -1 dBTP. If a target is out of reach without heavy limiting, it says so.

The output is a mono WAV with the same duration and timeline as the input. If you already synced the original, the restored file drops in at the same offset. Inputs above 48 kHz come out at 48 kHz, and `--sidon` always outputs 48 kHz. Its timing stayed within 4 ms of the input across a 400-second test spanning four of its processing chunks.

The default chain took about 5 minutes for a 62-minute recording on an Apple M3 Max, and the result lined up with the original within 1 ms across the whole hour.

### Reverb

Reverb is the hard part. Denoisers do not remove it, and the filters that do are weak. There are two optional stages:

| Flag | What it is | Good | Bad |
|---|---|---|---|
| `--wpe` | Weighted prediction error dereverb, a linear filter | Cannot change what was said | Shortens heavy reverb, does not remove it |
| `--sidon` | [Sidon](https://huggingface.co/sarulab-speech/sidon-v0.1), generative speech restoration | Removes reverb filters cannot touch, outputs 48 kHz | Regenerates the voice, so it can garble words |

These flags come from cleaning up one real hour-long meeting recorded with heavy room echo. Sidon sounded remarkable on the first stretch. Around 22 minutes in, where the speaker got quieter, it started producing garbled words. Leveling the input first, then also running WPE and denoising before Sidon, never got it to a version we trusted. The version that shipped was the default chain: echo still audible, every word intact.

So the default never invents audio. If you try `--sidon`, listen to all of it, not just the first minute.

Sidon downloads about 1 GB of weights on first use and runs on CPU or CUDA. On an Apple Silicon Mac, a 62-minute file took 15 to 45 minutes on CPU.

### Comparing versions

```bash
uv run restore/compare.py compare.png raw.wav meeting.restored.wav --start 600 --duration 20
```

Stacked spectrograms of the same stretch. Reverb shows up as energy smeared to the right of each syllable.

The individual stages also run on their own: `restore/wpe.py` and `restore/sidon.py` each take an input and output WAV. Run any script with `--help` for its options.

## Things that bit us

These are the details that cost real time, kept here so they do not cost yours.

- **DeepFilterNet delays its output by 30 ms** unless you pass `-D` (`--compensate-delay`). The restore chain does. Auto-sync caught this on the original meeting: the cleaned track was 30 ms late against the raw one.
- **DeepFilterNet's suppression depends on input level.** A quiet 30-second clip at -52 LUFS came out silent, speech and all. Across quiet and loud stretches of that meeting, -30 LUFS input kept the most speech for the noise removed, so the chain sets that level first. On a full hour the difference is small; on short or quiet files it is the difference between speech and silence.
- **Screen recordings often start their audio a little after the picture.** A plain decode drops that gap and shifts everything by it. Both tools decode with `aresample=async=1:first_pts=0`, which pads the gap so audio time 0 is the file's time 0.
- **ffmpeg's `alimiter` adds 5 ms of latency** unless you set `latency=1`.
- **Fast loudness normalization pumps the room back up.** That meeting's loudness ranged from -48 to -32 LUFS across stretches. One-pass `loudnorm` to -16 LUFS applied up to 32 dB of moving gain and made it sound like a louder version of the original. Slow leveling followed by a fixed gain does not.
- **Sidon's repository `infer.py` is the wrong entry point** for single-speaker cleanup. It is the two-speaker DialogueSidon model and needs CUDA-only flash-attn. The working recipe is the authors' [demo Space](https://huggingface.co/spaces/sarulab-speech/sidon_demo_beta), which `restore/sidon.py` follows.
- **`nara-wpe` imports scipy without declaring it**, so `restore/wpe.py` lists scipy itself.

## Privacy

The sync server listens on 127.0.0.1 only and refuses requests whose Host or Origin header points anywhere else, which blocks other websites from reaching it through your browser. Nothing is uploaded. Downloads are limited to what uv fetches on each script's first run (Python packages, and a Python interpreter if you lack a suitable one) and, for `--sidon`, the Sidon and w2v-BERT 2.0 weights from Hugging Face.

## Development

```bash
uv run --no-project --with pytest --with numpy pytest
uvx ruff check . && uvx ruff format --check .
```

The tests use synthetic signals only. Tests that need ffmpeg skip themselves when it is missing.

## Credits

- [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), MIT or Apache-2.0
- [nara-wpe](https://github.com/fgnt/nara_wpe), MIT
- [Sidon](https://github.com/sarulab-speech/Sidon) and its [weights](https://huggingface.co/sarulab-speech/sidon-v0.1), MIT
- [w2v-BERT 2.0](https://huggingface.co/facebook/w2v-bert-2.0), MIT

Models are downloaded from their sources at run time, not redistributed here.

## License

MIT. See [LICENSE](LICENSE).

# SoundGraft

Replace video clip audio with matched segments from a dedicated audio recording.

When you record an event with both a camera and a separate high-quality audio recorder, SoundGraft automatically aligns each video clip to the audio timeline and swaps in the better audio track. It also cleans up the replacement audio by attenuating applause and impulse noise, then peak-normalizes the result.

## How it works

1. **File classification** — Scans the input directory and sorts files into audio recordings and video clips by extension.
2. **Event grouping** — Groups consecutive audio segments into events using a file-size heuristic (a short segment followed by a full-length one marks an event boundary).
3. **Alignment** — Uses [Chromaprint](https://acoustid.org/chromaprint) fingerprint correlation to find where each video clip sits within the audio timeline. Metadata timestamps provide a search hint when available; falls back to a full scan otherwise.
4. **Audio processing** — For each matched clip:
   - Cuts the corresponding segment from the audio recording
   - Detects and attenuates applause sections (spectral flatness analysis)
   - Detects and attenuates impulse noise (transient peak analysis)
   - Peak-normalizes to 0 dBFS
5. **Muxing** — Replaces the video's audio track with the processed audio using ffmpeg.

Per-clip `.log` files are written to the output directory with alignment, applause, impulse, and normalization details.

## Installation

```
pip install soundgraft
```

You also need these system tools:

- `ffmpeg` / `ffprobe`
- `sox`
- `fpcalc` (from `libchromaprint-tools`)

On Debian/Ubuntu:

```
sudo apt install ffmpeg sox libchromaprint-tools
```

## Usage

```
soundgraft --input <input_dir> --output <output_dir> [options]
```

Place all raw audio recordings and video clips in a single input directory. Output videos (with replaced audio) and per-clip log files are written to the output directory.

### Options

| Flag | Description |
|------|-------------|
| `--input DIR` | Directory containing raw audio and video files (required) |
| `--output DIR` | Directory for output files (required) |
| `--clip N` | Process only video clip number N (1-indexed) |
| `--from-clip N` | Process video clips from N onwards |
| `--it-is-what-it-is` | Include low-confidence matches instead of skipping them |
| `--no-hint` | Skip metadata timestamp heuristic, always do full scan |
| `--keep-original-audio` | Keep the original video audio as a second track |
| `--temp-dir DIR` | Directory for temporary files (default: system temp) |

### Supported formats

- **Audio:** `.wav`, `.flac`, `.mp3`, `.ogg`, `.aac`
- **Video:** `.mp4`, `.mov`, `.avi`, `.mkv`, `.mts`

## Example

```
soundgraft --input ./concert-raw --output ./concert-fixed
```

This scans `concert-raw/` for audio and video files, aligns each video clip to the audio recording, processes the audio, and writes the final videos to `concert-fixed/`.

## License

Apache-2.0

# Sound Fixer — Design Spec

## Overview

A Python CLI tool that automates the post-processing of event recordings captured with separate audio and video devices. It reconstitutes fragmented audio recordings, aligns them with video clips, and produces final videos with high-quality replaced audio.

## Problem

When recording events (e.g., live music), a dedicated audio recorder captures the full event in high quality, while a camera captures short video clips. The audio recorder may split recordings into multiple files due to file size limits. The goal is to replace each video clip's lower-quality audio with the corresponding segment from the high-quality audio recording, volume-optimized.

## Dependencies

- **Python 3.10+**
- **System tools**: `ffmpeg`, `sox` (must be pre-installed)
- **Python packages**:
  - `audio-offset-finder` — MFCC-based audio alignment (BBC)
  - `numpy` — impulse detection and peak normalization
  - `tqdm` — progress bars

## CLI Interface

```
python sound_fixer.py --input DIR --output DIR [options]
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input DIR` | Yes | Directory containing raw audio and video files |
| `--output DIR` | Yes | Directory for output video files |
| `--clip N` | No | Process only video clip number N |
| `--from-clip N` | No | Process video clips from N onwards |
| `--it-is-what-it-is` | No | Include low-confidence matches in output instead of skipping |

Default behavior (no clip flags) processes all clips. Clip numbering is based on video files sorted by creation time (1-indexed).

## Pipeline

### Phase 1: Identify Files

Scan the input directory and classify files by extension:

- **Audio extensions**: `.wav`, `.flac`, `.mp3`, `.ogg`, `.aac`
- **Video extensions**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.mts`
- **Unknown extensions**: log warning, skip

For each recognized file, extract metadata via `ffprobe`:
- `creation_time` / `date`
- `duration`
- `encoded_by` (device identifier)
- File size

Output: two sorted lists (by creation time) — audio files and video files — with metadata attached.

### Phase 2: Reconstitute Audio

**Grouping segments into events:**

1. Sort audio files by `creation_time` metadata.
2. Determine "full size" dynamically: max file size among all audio files is the reference. Any file >= 98% of that is considered "full."
3. Walk through sorted files, accumulating segments into the current event. When a "short" segment (< 98%) is followed by a "full" segment, that boundary marks a new event.
4. The last segment is always the end of its event regardless of size.

**Concatenation:**

For each event with multiple segments, concatenate in order using `sox`:
```
sox segment1.wav segment2.wav segment3.wav event_1.wav
```

Single-segment events skip concatenation and use the file directly.

Output: one WAV file per event, stored as intermediate files in a temp directory.

### Phase 3: Align Video Clips to Audio

For each video clip (respecting `--clip` / `--from-clip` filters):

1. **Extract audio** from the video to a temp WAV using `ffmpeg`.

2. **Metadata-hinted search (fast path)**: Compare the video's `creation_time` against each event's time range. If timestamps look plausible (same date, reasonable hour), search a +/- 30 minute window around the expected offset using `audio-offset-finder`. Show progress bar during search. Log clearly: "Trying metadata-hinted window (HH:MM - HH:MM) for clip N..."

3. **Full scan (fallback)**: If timestamps look unreliable (wrong date, year 2000, missing) or the narrow search confidence is below threshold, search the entire audio recording. Log clearly: "Narrow search inconclusive (score: X.XX), falling back to full scan..." Show progress bar.

4. **Multi-event**: If multiple reconstituted events exist, try alignment against each. Pick the best confidence match across all events.

5. **Low confidence**: If the best match is below a confidence threshold, log a warning. Skip the clip unless `--it-is-what-it-is` is set. When skipped, log the clip number and score so the user knows. The threshold value should be determined empirically during implementation by testing against the sample data, and hardcoded as a sensible default.

Output: a mapping of each video clip to (event audio file, offset in seconds, confidence score).

### Phase 4: Replace Audio

For each successfully matched video clip:

1. **Cut matching segment**: Extract audio from the reconstituted event starting at the aligned offset, matching the video clip's duration, using `ffmpeg`.

2. **Impulse detection and attenuation**:
   - Analyze the audio segment in 50ms windows.
   - For each window, compare its peak amplitude to the average peak of surrounding windows (~500ms on each side).
   - If a window's peak exceeds its neighbors by more than a threshold (default: ~15 dB), attenuate that window to match its surroundings.
   - **Console warning**: print a red (ANSI color) warning for each detected impulse, including timestamp and levels.
   - **Impulse report**: write all detected impulses to `<output_dir>/impulse_report.txt` with columns: timestamp, original peak (dB), surrounding average (dB), attenuation applied (dB). Format is designed for easy cross-reference with Audacity.

3. **Peak normalization**: After impulse cleanup, find the maximum sample amplitude and scale the entire segment so the peak reaches 0 dBFS. Implemented with `numpy`.

4. **Mux final video**:
   ```
   ffmpeg -i original_video.mp4 -i normalized_audio.wav \
     -c:v copy -map 0:v -map 1:a <output_dir>/<original_filename>
   ```
   Video stream is copied (no re-encoding). Original video audio is excluded via stream mapping (only `0:v` is selected from the video input). Output filename matches the input video filename, placed in the output directory.

## Logging

- All phases produce clear log messages describing what is happening.
- Impulse detections produce red ANSI-colored console warnings.
- A detailed `impulse_report.txt` is written to the output directory.
- Alignment results (offset, confidence) are logged for each clip.
- Skipped clips (low confidence) are logged with their scores.

## Project Structure

```
sound-fixer/
  sound_fixer.py        # Main script — all pipeline phases
  requirements.txt      # Python dependencies
  CLAUDE.md             # Project conventions
  AGENTS.md             # Future extensions / known issues
  docs/                 # Design docs
  test-input/           # Sample input data
```

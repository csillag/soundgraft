# Sound Fixer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI tool that replaces video clip audio with matched, peak-normalized segments from a dedicated audio recording of the same event.

**Architecture:** Single Python script (`sound_fixer.py`) with a 4-phase pipeline: identify files by extension, reconstitute fragmented audio using sox, align video audio to the long recording using `audio-offset-finder`, then mux the normalized audio back into the video with ffmpeg. All orchestration in Python; heavy lifting delegated to external tools.

**Tech Stack:** Python 3.10+, ffmpeg, sox, audio-offset-finder (BBC), numpy, tqdm

---

## File Structure

```
sound_fixer.py          # Main script — CLI, pipeline orchestration, all 4 phases
requirements.txt        # Python dependencies
tests/
  test_identify.py      # Tests for Phase 1 (file identification)
  test_reconstitute.py  # Tests for Phase 2 (audio grouping + concatenation)
  test_impulse.py       # Tests for Phase 4 impulse detection logic
  test_cli.py           # Tests for CLI argument parsing
```

Note: Phase 3 (alignment) and the muxing part of Phase 4 are integration-heavy (they call ffmpeg/audio-offset-finder on real files). We'll test those manually against test-input data rather than unit testing them. The impulse detection is pure numpy math — that we unit test.

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `sound_fixer.py` (skeleton)
- Create: `tests/test_cli.py`

- [ ] **Step 1: Create requirements.txt**

```
audio-offset-finder
numpy
tqdm
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 3: Write CLI argument parsing test**

Create `tests/test_cli.py`:

```python
import subprocess
import sys


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "sound_fixer.py", "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--clip" in result.stdout
    assert "--from-clip" in result.stdout
    assert "--it-is-what-it-is" in result.stdout


def test_missing_required_args():
    result = subprocess.run(
        [sys.executable, "sound_fixer.py"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL — `sound_fixer.py` doesn't exist yet or has no argument parsing.

- [ ] **Step 5: Write the CLI skeleton**

Create `sound_fixer.py`:

```python
#!/usr/bin/env python3
"""Sound Fixer — replace video audio with high-quality dedicated recordings."""

import argparse
import sys


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replace video clip audio with matched segments from a dedicated audio recording."
    )
    parser.add_argument("--input", required=True, help="Directory containing raw audio and video files")
    parser.add_argument("--output", required=True, help="Directory for output video files")
    parser.add_argument("--clip", type=int, help="Process only video clip number N (1-indexed)")
    parser.add_argument("--from-clip", type=int, help="Process video clips from N onwards (1-indexed)")
    parser.add_argument(
        "--it-is-what-it-is",
        action="store_true",
        help="Include low-confidence alignment matches in output instead of skipping them",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt sound_fixer.py tests/test_cli.py
git commit -m "feat: project setup with CLI skeleton and argument parsing"
```

---

### Task 2: Phase 1 — File Identification

**Files:**
- Modify: `sound_fixer.py`
- Create: `tests/test_identify.py`

- [ ] **Step 1: Write the test for file classification**

Create `tests/test_identify.py`:

```python
import os
import tempfile
from unittest.mock import patch

# We'll import after writing the function
from sound_fixer import classify_files


def _touch(path):
    open(path, "w").close()


def test_classify_by_extension():
    with tempfile.TemporaryDirectory() as d:
        _touch(os.path.join(d, "rec001.wav"))
        _touch(os.path.join(d, "rec002.wav"))
        _touch(os.path.join(d, "clip.mp4"))
        _touch(os.path.join(d, "notes.txt"))

        audio, video, skipped = classify_files(d)

        audio_names = [os.path.basename(a["path"]) for a in audio]
        video_names = [os.path.basename(v["path"]) for v in video]

        assert sorted(audio_names) == ["rec001.wav", "rec002.wav"]
        assert video_names == ["clip.mp4"]
        assert len(skipped) == 1
        assert "notes.txt" in skipped[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_identify.py::test_classify_by_extension -v`
Expected: FAIL — `classify_files` not defined.

- [ ] **Step 3: Write the implementation**

Add to `sound_fixer.py`:

```python
import json
import os
import subprocess


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mts"}


def run_ffprobe(filepath):
    """Extract metadata from a media file using ffprobe. Returns a dict."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {}
    return json.loads(result.stdout)


def get_creation_time(probe_data):
    """Extract creation_time from ffprobe data. Returns ISO string or None."""
    fmt = probe_data.get("format", {})
    tags = fmt.get("tags", {})
    # Try format-level tags first
    for key in ("creation_time", "date"):
        if key in tags:
            return tags[key]
    # Try stream-level tags
    for stream in probe_data.get("streams", []):
        stream_tags = stream.get("tags", {})
        if "creation_time" in stream_tags:
            return stream_tags["creation_time"]
    return None


def get_file_metadata(filepath):
    """Get metadata dict for a media file."""
    probe = run_ffprobe(filepath)
    fmt = probe.get("format", {})
    tags = fmt.get("tags", {})
    return {
        "path": filepath,
        "size": os.path.getsize(filepath),
        "duration": float(fmt.get("duration", 0)),
        "creation_time": get_creation_time(probe),
        "encoded_by": tags.get("encoded_by", ""),
        "probe_data": probe,
    }


def classify_files(input_dir):
    """Classify files in input_dir into audio, video, and skipped lists.

    Returns (audio_files, video_files, skipped_files) where audio_files and
    video_files are lists of metadata dicts sorted by creation_time, and
    skipped_files is a list of file paths.
    """
    audio_files = []
    video_files = []
    skipped_files = []

    for entry in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, entry)
        if not os.path.isfile(filepath):
            continue
        ext = os.path.splitext(entry)[1].lower()
        if ext in AUDIO_EXTENSIONS:
            audio_files.append(get_file_metadata(filepath))
        elif ext in VIDEO_EXTENSIONS:
            video_files.append(get_file_metadata(filepath))
        else:
            skipped_files.append(filepath)

    # Sort by creation_time (None sorts first, which is fine — fallback to name order)
    audio_files.sort(key=lambda f: f["creation_time"] or "")
    video_files.sort(key=lambda f: f["creation_time"] or "")

    return audio_files, video_files, skipped_files
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_identify.py::test_classify_by_extension -v`
Expected: PASS

- [ ] **Step 5: Write a test for metadata extraction with real files**

Add to `tests/test_identify.py`:

```python
def test_metadata_extraction_on_test_input():
    """Integration test using actual test-input files."""
    input_dir = "test-input"
    if not os.path.isdir(input_dir):
        import pytest
        pytest.skip("test-input directory not available")

    audio, video, skipped = classify_files(input_dir)

    assert len(audio) >= 2
    assert len(video) >= 1

    # Audio files should have creation_time metadata
    for a in audio:
        assert a["creation_time"] is not None, f"{a['path']} missing creation_time"
        assert a["duration"] > 0

    # Video files should have creation_time and duration
    for v in video:
        assert v["creation_time"] is not None, f"{v['path']} missing creation_time"
        assert v["duration"] > 0
```

- [ ] **Step 6: Run the integration test**

Run: `pytest tests/test_identify.py::test_metadata_extraction_on_test_input -v`
Expected: PASS (requires test-input directory with sample files)

- [ ] **Step 7: Commit**

```bash
git add sound_fixer.py tests/test_identify.py
git commit -m "feat: Phase 1 — file identification by extension with ffprobe metadata"
```

---

### Task 3: Phase 2 — Reconstitute Audio

**Files:**
- Modify: `sound_fixer.py`
- Create: `tests/test_reconstitute.py`

- [ ] **Step 1: Write the test for event grouping**

Create `tests/test_reconstitute.py`:

```python
from sound_fixer import group_audio_into_events


def _file(name, size, creation_time):
    return {"path": f"/fake/{name}", "size": size, "creation_time": creation_time, "duration": 3600.0}


def test_single_event_two_segments():
    """Two files: one full, one short — single event."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 1_700_000_000, "2026-03-31T10:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 2


def test_two_events():
    """full, short, full, short — two events."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 1_000_000_000, "2026-03-31T10:00:00"),
        _file("003.wav", 2_000_000_000, "2026-03-31T14:00:00"),
        _file("004.wav", 1_500_000_000, "2026-03-31T15:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 2
    assert len(events[0]) == 2
    assert len(events[1]) == 2


def test_single_file():
    """One file — one event."""
    files = [_file("001.wav", 500_000_000, "2026-03-31T09:00:00")]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 1


def test_three_full_then_short():
    """full, full, short — one event with three segments."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 2_000_000_000, "2026-03-31T10:00:00"),
        _file("003.wav", 800_000_000, "2026-03-31T11:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reconstitute.py -v`
Expected: FAIL — `group_audio_into_events` not defined.

- [ ] **Step 3: Write the grouping implementation**

Add to `sound_fixer.py`:

```python
def group_audio_into_events(audio_files):
    """Group sorted audio files into events using the 98% size heuristic.

    A 'short' segment followed by a 'full' segment marks an event boundary.
    Returns a list of events, each event being a list of file metadata dicts.
    """
    if not audio_files:
        return []

    max_size = max(f["size"] for f in audio_files)
    full_threshold = max_size * 0.98

    events = []
    current_event = []

    for i, f in enumerate(audio_files):
        current_event.append(f)
        is_last = i == len(audio_files) - 1
        is_short = f["size"] < full_threshold

        if is_short and not is_last:
            # Short segment — check if next is full (= new event)
            next_is_full = audio_files[i + 1]["size"] >= full_threshold
            if next_is_full:
                events.append(current_event)
                current_event = []

    if current_event:
        events.append(current_event)

    return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reconstitute.py -v`
Expected: PASS

- [ ] **Step 5: Write the concatenation function**

Add to `sound_fixer.py`:

```python
import tempfile
import shutil


def reconstitute_audio(events, temp_dir):
    """Concatenate audio segments for each event using sox.

    Returns a list of dicts, one per event:
      {
        "path": path to the concatenated WAV (or original if single segment),
        "segments": list of original file metadata dicts,
        "start_time": creation_time of the first segment,
        "total_duration": sum of segment durations,
      }
    """
    result = []

    for i, segments in enumerate(events):
        total_duration = sum(s["duration"] for s in segments)
        start_time = segments[0]["creation_time"]

        if len(segments) == 1:
            event_path = segments[0]["path"]
        else:
            event_path = os.path.join(temp_dir, f"event_{i + 1}.wav")
            input_paths = [s["path"] for s in segments]
            cmd = ["sox"] + input_paths + [event_path]
            print(f"  Concatenating {len(segments)} segments into {os.path.basename(event_path)}...")
            subprocess.run(cmd, check=True)

        result.append({
            "path": event_path,
            "segments": segments,
            "start_time": start_time,
            "total_duration": total_duration,
        })

    return result
```

- [ ] **Step 6: Commit**

```bash
git add sound_fixer.py tests/test_reconstitute.py
git commit -m "feat: Phase 2 — audio event grouping and sox concatenation"
```

---

### Task 4: Phase 3 — Align Video Clips to Audio

**Files:**
- Modify: `sound_fixer.py`

This task is integration-heavy (real audio files + audio-offset-finder). We test it manually against test-input.

- [ ] **Step 1: Write the alignment function**

Add to `sound_fixer.py`:

```python
from datetime import datetime
from tqdm import tqdm
from audio_offset_finder.audio_offset_finder import find_offset_between_files


# Confidence threshold: scores >= 10 are likely correct per audio-offset-finder docs.
# Scores < 5 are unlikely correct. We use 8 as a reasonable middle ground.
CONFIDENCE_THRESHOLD = 8


def parse_timestamp(ts_string):
    """Parse a creation_time string to a datetime. Returns None on failure."""
    if not ts_string:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_string, fmt)
        except ValueError:
            continue
    return None


def timestamps_look_plausible(video_time, event_start_time):
    """Check if two timestamps are plausibly from the same event (same date, within 24h)."""
    if video_time is None or event_start_time is None:
        return False
    # Check for obviously wrong timestamps (year < 2010 = clock was reset)
    if video_time.year < 2010 or event_start_time.year < 2010:
        return False
    # Same date check
    return video_time.date() == event_start_time.date()


def extract_video_audio(video_path, temp_dir):
    """Extract audio from a video file to a temp WAV. Returns path to WAV."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    wav_path = os.path.join(temp_dir, f"{base}_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1",
        wav_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return wav_path


def align_clip_to_event(video_wav, event, video_meta, temp_dir):
    """Try to align a video clip's audio to an event's audio.

    First tries a metadata-hinted narrow search, then falls back to full scan.
    Returns (offset_seconds, confidence_score) or (None, 0) on failure.
    """
    event_path = event["path"]
    video_time = parse_timestamp(video_meta["creation_time"])
    event_start = parse_timestamp(event["start_time"])

    # Try metadata-hinted search first
    if timestamps_look_plausible(video_time, event_start):
        offset_hint = (video_time - event_start).total_seconds()
        # Search window: +/- 30 minutes around expected position
        trim_start = max(0, offset_hint - 1800)
        trim_end = offset_hint + 1800 + video_meta["duration"]

        print(f"    Trying metadata-hinted window "
              f"({trim_start:.0f}s - {trim_end:.0f}s of event audio)...")

        # Use trim parameter to limit search to the window
        # audio-offset-finder's trim parameter trims from the start, so we need
        # to create a trimmed version of the event audio for the windowed search
        trimmed_path = os.path.join(temp_dir, "trimmed_event.wav")
        cmd = [
            "ffmpeg", "-y", "-i", event_path,
            "-ss", str(trim_start), "-t", str(trim_end - trim_start),
            "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1",
            trimmed_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        print(f"    Aligning (narrow search)...")
        for _ in tqdm(range(1), desc="    Narrow alignment", bar_format="{desc}: {bar} {elapsed}"):
            results = find_offset_between_files(trimmed_path, video_wav)

        score = results.get("standard_score", 0)
        offset = results.get("time_offset", 0)

        if score >= CONFIDENCE_THRESHOLD:
            # Offset is relative to the trimmed audio — add back the trim start
            actual_offset = offset + trim_start
            print(f"    Match found! Offset: {actual_offset:.2f}s, confidence: {score:.2f}")
            return actual_offset, score
        else:
            print(f"    Narrow search inconclusive (score: {score:.2f}), falling back to full scan...")

    # Full scan fallback
    print(f"    Aligning (full scan of {event['total_duration']:.0f}s event audio)...")
    # Convert full event audio to the format audio-offset-finder expects
    full_path = os.path.join(temp_dir, "full_event_8k.wav")
    cmd = [
        "ffmpeg", "-y", "-i", event_path,
        "-acodec", "pcm_s16le", "-ar", "8000", "-ac", "1",
        full_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    for _ in tqdm(range(1), desc="    Full alignment", bar_format="{desc}: {bar} {elapsed}"):
        results = find_offset_between_files(full_path, video_wav)

    score = results.get("standard_score", 0)
    offset = results.get("time_offset", 0)

    if score >= CONFIDENCE_THRESHOLD:
        print(f"    Match found! Offset: {offset:.2f}s, confidence: {score:.2f}")
    else:
        print(f"    Low confidence match. Offset: {offset:.2f}s, confidence: {score:.2f}")

    return offset, score


def align_all_clips(video_files, events, temp_dir, clip_filter=None, from_clip=None, it_is_what_it_is=False):
    """Align video clips to events. Returns list of alignment results.

    Each result is a dict:
      {
        "video": video file metadata,
        "clip_number": 1-indexed clip number,
        "event": matched event dict,
        "offset": offset in seconds within the event audio,
        "confidence": confidence score,
        "skipped": True if low confidence and not --it-is-what-it-is,
      }
    """
    alignments = []

    for i, video in enumerate(video_files):
        clip_num = i + 1

        # Apply clip filters
        if clip_filter is not None and clip_num != clip_filter:
            continue
        if from_clip is not None and clip_num < from_clip:
            continue

        print(f"\n  Clip {clip_num}: {os.path.basename(video['path'])} ({video['duration']:.1f}s)")

        # Extract audio from video
        video_wav = extract_video_audio(video["path"], temp_dir)

        # Try each event, pick best match
        best_offset = None
        best_score = 0
        best_event = None

        for j, event in enumerate(events):
            print(f"    Trying event {j + 1} ({event['total_duration']:.0f}s)...")
            offset, score = align_clip_to_event(video_wav, event, video, temp_dir)
            if score > best_score:
                best_offset = offset
                best_score = score
                best_event = event

        skipped = best_score < CONFIDENCE_THRESHOLD and not it_is_what_it_is

        if skipped:
            print(f"  \033[33mWARNING: Clip {clip_num} skipped — "
                  f"best confidence {best_score:.2f} below threshold {CONFIDENCE_THRESHOLD}\033[0m")

        alignments.append({
            "video": video,
            "clip_number": clip_num,
            "event": best_event,
            "offset": best_offset,
            "confidence": best_score,
            "skipped": skipped,
        })

    return alignments
```

- [ ] **Step 2: Manual integration test**

Run against test-input:
```bash
python -c "
from sound_fixer import classify_files, group_audio_into_events, reconstitute_audio, align_all_clips
import tempfile, os

audio, video, _ = classify_files('test-input')
print(f'Audio: {len(audio)} files, Video: {len(video)} files')

events = group_audio_into_events(audio)
print(f'Events: {len(events)}')

with tempfile.TemporaryDirectory() as tmp:
    recon = reconstitute_audio(events, tmp)
    for e in recon:
        print(f'  Event: {e[\"total_duration\"]:.0f}s starting {e[\"start_time\"]}')

    alignments = align_all_clips(video, recon, tmp)
    for a in alignments:
        print(f'  Clip {a[\"clip_number\"]}: offset={a[\"offset\"]:.2f}s, confidence={a[\"confidence\"]:.2f}, skipped={a[\"skipped\"]}')
"
```

Expected: Alignment completes with a confidence score. Check that the offset is reasonable (the video was created at 09:19 and the first audio segment starts at 09:38, so the video might actually precede the audio — or timestamps may differ). Inspect results and adjust `CONFIDENCE_THRESHOLD` if needed.

- [ ] **Step 3: Commit**

```bash
git add sound_fixer.py
git commit -m "feat: Phase 3 — audio alignment with metadata-hinted search and full-scan fallback"
```

---

### Task 5: Phase 4 — Impulse Detection and Peak Normalization

**Files:**
- Modify: `sound_fixer.py`
- Create: `tests/test_impulse.py`

- [ ] **Step 1: Write tests for impulse detection**

Create `tests/test_impulse.py`:

```python
import numpy as np
from sound_fixer import detect_impulses, attenuate_impulses


def test_no_impulse_in_steady_signal():
    """A constant-amplitude signal should have no impulses detected."""
    # 1 second of steady sine wave at 44100 Hz
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 0


def test_detect_single_impulse():
    """A short spike in an otherwise quiet signal should be detected."""
    sr = 44100
    # 2 seconds of quiet signal
    signal = np.full(sr * 2, 0.01, dtype=np.float32)
    # Insert a 2ms spike at t=1.0s
    spike_start = sr  # sample at 1.0s
    spike_len = int(0.002 * sr)  # 2ms
    signal[spike_start:spike_start + spike_len] = 0.9

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 1
    # The impulse should be near t=1.0s
    assert abs(impulses[0]["timestamp"] - 1.0) < 0.1


def test_loud_passage_not_detected():
    """A sustained loud section should NOT be detected as an impulse."""
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    # Quiet-loud-quiet pattern, each section ~1 second
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    signal[:sr] *= 0.1        # quiet first second
    signal[sr:2*sr] *= 0.8    # loud second second (musical climax)
    signal[2*sr:] *= 0.1      # quiet third second

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 0


def test_attenuate_reduces_spike():
    """After attenuation, the spike should be close to the surrounding level."""
    sr = 44100
    signal = np.full(sr * 2, 0.01, dtype=np.float32)
    spike_start = sr
    spike_len = int(0.002 * sr)
    signal[spike_start:spike_start + spike_len] = 0.9

    impulses = detect_impulses(signal, sr)
    cleaned = attenuate_impulses(signal.copy(), impulses, sr)

    # The spike area should now be much lower
    assert np.max(np.abs(cleaned[spike_start:spike_start + spike_len])) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_impulse.py -v`
Expected: FAIL — `detect_impulses` and `attenuate_impulses` not defined.

- [ ] **Step 3: Write the impulse detection implementation**

Add to `sound_fixer.py`:

```python
import numpy as np


# ANSI color codes
RED = "\033[91m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def detect_impulses(signal, sample_rate, window_ms=50, context_ms=500, threshold_db=15):
    """Detect anomalous impulse events in an audio signal.

    Analyzes the signal in windows of `window_ms` milliseconds. If a window's
    peak exceeds the average peak of surrounding windows (within `context_ms`
    on each side) by more than `threshold_db` dB, it's flagged as an impulse.

    Returns a list of dicts:
      {
        "timestamp": time in seconds,
        "window_idx": index of the window,
        "peak_db": peak level of the window in dB,
        "context_db": average peak of surrounding windows in dB,
        "excess_db": how much the peak exceeds context in dB,
      }
    """
    window_samples = int(sample_rate * window_ms / 1000)
    context_windows = int(context_ms / window_ms)

    # Split signal into windows and compute peak amplitude per window
    num_windows = len(signal) // window_samples
    if num_windows == 0:
        return []

    # Reshape into windows (drop trailing samples that don't fill a window)
    trimmed = signal[:num_windows * window_samples]
    windows = np.abs(trimmed.reshape(num_windows, window_samples))
    peaks = windows.max(axis=1)

    # Avoid log of zero
    peaks_safe = np.maximum(peaks, 1e-10)
    peaks_db = 20 * np.log10(peaks_safe)

    impulses = []

    for i in range(num_windows):
        # Get context window indices (excluding the current window)
        ctx_start = max(0, i - context_windows)
        ctx_end = min(num_windows, i + context_windows + 1)
        context_indices = list(range(ctx_start, i)) + list(range(i + 1, ctx_end))

        if not context_indices:
            continue

        context_avg_db = np.mean(peaks_db[context_indices])
        excess = peaks_db[i] - context_avg_db

        if excess > threshold_db:
            impulses.append({
                "timestamp": i * window_ms / 1000,
                "window_idx": i,
                "peak_db": float(peaks_db[i]),
                "context_db": float(context_avg_db),
                "excess_db": float(excess),
            })

    return impulses


def attenuate_impulses(signal, impulses, sample_rate, window_ms=50):
    """Attenuate detected impulse windows to match their surrounding level.

    Modifies and returns the signal array.
    """
    window_samples = int(sample_rate * window_ms / 1000)

    for imp in impulses:
        idx = imp["window_idx"]
        start = idx * window_samples
        end = start + window_samples

        # Scale this window down so its peak matches the context level
        current_peak = np.max(np.abs(signal[start:end]))
        if current_peak > 0:
            target_peak = 10 ** (imp["context_db"] / 20)
            scale = target_peak / current_peak
            signal[start:end] *= scale

    return signal
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_impulse.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sound_fixer.py tests/test_impulse.py
git commit -m "feat: impulse detection and attenuation with windowed peak analysis"
```

---

### Task 6: Phase 4 — Audio Replacement and Muxing

**Files:**
- Modify: `sound_fixer.py`

- [ ] **Step 1: Write the replace/mux function**

Add to `sound_fixer.py`:

```python
import wave


def peak_normalize(signal):
    """Normalize signal so that the peak amplitude is 1.0 (0 dBFS).

    Returns the normalized signal and the gain factor applied.
    """
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal, 1.0
    gain = 1.0 / peak
    return signal * gain, gain


def read_audio_as_float(filepath, temp_dir):
    """Read an audio file and return (signal_as_float32_array, sample_rate, n_channels).

    Uses ffmpeg to convert to 32-bit float WAV first, avoiding issues with
    exotic formats (24-bit, high sample rates, etc.).
    """
    # Get original file info for channel count
    probe = run_ffprobe(filepath)
    n_channels = 2  # default
    original_sr = 48000  # default
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            n_channels = int(stream.get("channels", 2))
            original_sr = int(stream.get("sample_rate", 48000))
            break

    # Convert to 32-bit float WAV via ffmpeg (handles any input format)
    float_path = os.path.join(temp_dir, "float_convert.wav")
    cmd = [
        "ffmpeg", "-y", "-i", filepath,
        "-acodec", "pcm_f32le", "-ar", str(original_sr),
        float_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    # Read the float32 WAV
    with wave.open(float_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.float32)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)

    return samples, sr, n_channels


def write_wav_from_float(filepath, signal, sample_rate, n_channels):
    """Write a float32 signal array to a WAV file (32-bit float format)."""
    # Clip to [-1, 1] to prevent overflow
    signal = np.clip(signal, -1.0, 1.0).astype(np.float32)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(4)  # 4 bytes = 32-bit float
        wf.setframerate(sample_rate)
        wf.writeframes(signal.tobytes())


def write_impulse_report(impulses_by_clip, output_dir):
    """Write the impulse detection report to output_dir/impulse_report.txt."""
    report_path = os.path.join(output_dir, "impulse_report.txt")
    with open(report_path, "w") as f:
        f.write("Impulse Detection Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Clip':<8} {'Timestamp':<12} {'Peak (dB)':<12} {'Context (dB)':<14} {'Attenuation (dB)':<18}\n")
        f.write("-" * 70 + "\n")

        for clip_name, impulses in impulses_by_clip:
            for imp in impulses:
                f.write(
                    f"{clip_name:<8} "
                    f"{imp['timestamp']:>8.3f}s   "
                    f"{imp['peak_db']:>8.1f}    "
                    f"{imp['context_db']:>10.1f}      "
                    f"{imp['excess_db']:>12.1f}\n"
                )

    return report_path


def replace_audio_for_clip(alignment, temp_dir, output_dir):
    """Cut, clean, normalize, and mux audio for one video clip.

    Returns (output_path, impulses_list) or (None, []) if skipped.
    """
    if alignment["skipped"]:
        return None, []

    video_path = alignment["video"]["path"]
    video_duration = alignment["video"]["duration"]
    event_path = alignment["event"]["path"]
    offset = alignment["offset"]
    clip_num = alignment["clip_number"]
    basename = os.path.basename(video_path)

    print(f"\n  Processing clip {clip_num}: {basename}")

    # Step 1: Cut the matching audio segment from the event
    cut_path = os.path.join(temp_dir, f"clip_{clip_num}_cut.wav")
    cmd = [
        "ffmpeg", "-y", "-i", event_path,
        "-ss", str(offset), "-t", str(video_duration),
        cut_path
    ]
    print(f"    Cutting audio at offset {offset:.2f}s for {video_duration:.1f}s...")
    subprocess.run(cmd, capture_output=True, check=True)

    # Step 2: Read the audio segment
    signal, sr, nch = read_audio_as_float(cut_path, temp_dir)

    # For impulse detection, work with mono (average channels if stereo)
    if nch > 1:
        mono = signal.mean(axis=1)
    else:
        mono = signal

    # Step 3: Detect and attenuate impulses
    impulses = detect_impulses(mono, sr)

    if impulses:
        for imp in impulses:
            print(f"    {RED}IMPULSE DETECTED{RESET} at {imp['timestamp']:.3f}s — "
                  f"peak: {imp['peak_db']:.1f} dB, context: {imp['context_db']:.1f} dB, "
                  f"excess: {imp['excess_db']:.1f} dB")

        # Attenuate impulses in all channels
        if nch > 1:
            for ch in range(nch):
                signal[:, ch] = attenuate_impulses(signal[:, ch], impulses, sr)
        else:
            signal = attenuate_impulses(signal, impulses, sr)
    else:
        print(f"    No impulses detected.")

    # Step 4: Peak normalize
    if nch > 1:
        flat = signal.flatten()
    else:
        flat = signal
    flat, gain = peak_normalize(flat)
    if nch > 1:
        signal = flat.reshape(-1, nch)
    else:
        signal = flat
    gain_db = 20 * np.log10(gain) if gain > 0 else 0
    print(f"    Peak normalized (gain: {gain_db:+.1f} dB)")

    # Step 5: Write normalized audio to temp file
    norm_path = os.path.join(temp_dir, f"clip_{clip_num}_normalized.wav")
    write_wav_from_float(norm_path, signal, sr, nch)

    # Step 6: Mux into final video
    output_path = os.path.join(output_dir, basename)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", norm_path,
        "-c:v", "copy",
        "-map", "0:v",
        "-map", "1:a",
        output_path,
    ]
    print(f"    Muxing final video to {output_path}...")
    subprocess.run(cmd, capture_output=True, check=True)

    print(f"    Done: {output_path}")
    return output_path, impulses


- [ ] **Step 2: Manual test with test-input**

```bash
python sound_fixer.py --input test-input --output test-output
```

Check:
- Output directory contains the video file
- Play the output video — audio should come from the dedicated recorder
- Check `test-output/impulse_report.txt` if any impulses were detected
- Verify no re-encoding artifacts in video (file size should be similar to input)

- [ ] **Step 3: Commit**

```bash
git add sound_fixer.py
git commit -m "feat: Phase 4 — impulse detection, peak normalization, and audio muxing"
```

---

### Task 7: Wire Up the Main Pipeline

**Files:**
- Modify: `sound_fixer.py`

- [ ] **Step 1: Write the main pipeline function**

Replace the `main()` function in `sound_fixer.py`:

```python
def main():
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    # Validate input directory
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Phase 1: Identify
    print("=" * 60)
    print("Phase 1: Identifying files")
    print("=" * 60)
    audio_files, video_files, skipped = classify_files(input_dir)

    print(f"  Audio files: {len(audio_files)}")
    for a in audio_files:
        print(f"    {os.path.basename(a['path'])} — {a['duration']:.0f}s, {a['creation_time']}")
    print(f"  Video files: {len(video_files)}")
    for v in video_files:
        print(f"    {os.path.basename(v['path'])} — {v['duration']:.0f}s, {v['creation_time']}")
    if skipped:
        print(f"  Skipped: {len(skipped)}")
        for s in skipped:
            print(f"    {YELLOW}WARNING: Skipping unrecognized file: {os.path.basename(s)}{RESET}")

    if not audio_files:
        print("Error: No audio files found in input directory.")
        sys.exit(1)
    if not video_files:
        print("Error: No video files found in input directory.")
        sys.exit(1)

    # Phase 2: Reconstitute
    print(f"\n{'=' * 60}")
    print("Phase 2: Reconstituting audio")
    print("=" * 60)
    events = group_audio_into_events(audio_files)
    print(f"  Detected {len(events)} event(s)")

    with tempfile.TemporaryDirectory(prefix="sound_fixer_") as temp_dir:
        reconstituted = reconstitute_audio(events, temp_dir)
        for i, e in enumerate(reconstituted):
            print(f"  Event {i + 1}: {e['total_duration']:.0f}s "
                  f"({len(e['segments'])} segment(s)), starts {e['start_time']}")

        # Phase 3: Align
        print(f"\n{'=' * 60}")
        print("Phase 3: Aligning video clips to audio")
        print("=" * 60)
        alignments = align_all_clips(
            video_files, reconstituted, temp_dir,
            clip_filter=args.clip,
            from_clip=args.from_clip,
            it_is_what_it_is=args.it_is_what_it_is,
        )

        # Phase 4: Replace
        print(f"\n{'=' * 60}")
        print("Phase 4: Replacing audio in video clips")
        print("=" * 60)
        impulses_by_clip = []

        for alignment in alignments:
            output_path, impulses = replace_audio_for_clip(alignment, temp_dir, output_dir)
            if impulses:
                clip_name = os.path.basename(alignment["video"]["path"])
                impulses_by_clip.append((clip_name, impulses))

        # Write impulse report if any were detected
        if impulses_by_clip:
            report_path = write_impulse_report(impulses_by_clip, output_dir)
            print(f"\n  {RED}Impulse report written to: {report_path}{RESET}")
            print(f"  Review detected impulses in Audacity to verify they are genuine artifacts.")

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print("=" * 60)
    processed = [a for a in alignments if not a["skipped"]]
    skipped_clips = [a for a in alignments if a["skipped"]]
    print(f"  Processed: {len(processed)} clip(s)")
    print(f"  Skipped:   {len(skipped_clips)} clip(s)")
    for s in skipped_clips:
        print(f"    Clip {s['clip_number']}: {os.path.basename(s['video']['path'])} "
              f"(confidence: {s['confidence']:.2f})")
    print(f"\n  Output directory: {output_dir}")
    print("  Done!")
```

- [ ] **Step 2: End-to-end test**

```bash
mkdir -p test-output
python sound_fixer.py --input test-input --output test-output
```

Expected output:
- All 4 phases run with clear log messages
- Output video in `test-output/`
- Impulse report if impulses were detected
- Summary at the end

Verify:
- Play the output video — audio should be from the dedicated recorder
- Compare audio quality to the original video's audio
- Check the impulse report if generated

- [ ] **Step 3: Test CLI flags**

```bash
# Single clip
python sound_fixer.py --input test-input --output test-output --clip 1

# With it-is-what-it-is
python sound_fixer.py --input test-input --output test-output --it-is-what-it-is
```

- [ ] **Step 4: Commit**

```bash
git add sound_fixer.py
git commit -m "feat: wire up main pipeline — end-to-end audio replacement workflow"
```

---

### Task 8: Final Polish and All Tests

**Files:**
- Modify: `sound_fixer.py` (minor fixes from testing)

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 2: Run end-to-end on test-input**

```bash
rm -rf test-output && mkdir test-output
python sound_fixer.py --input test-input --output test-output
```

Verify output video plays correctly with replaced audio.

- [ ] **Step 3: Fix any issues discovered during testing**

Address any failures or unexpected behavior found in steps 1-2.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final polish — all tests passing, end-to-end verified"
```

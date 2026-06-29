#!/usr/bin/env python3
"""SoundGraft — replace video audio with high-quality dedicated recordings."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime


def check_dependencies():
    """Check that all required dependencies are available."""
    missing_python = []
    missing_system = []

    for module, pip_name in [("numpy", "numpy"), ("tqdm", "tqdm"), ("librosa", "librosa")]:
        try:
            __import__(module)
        except ImportError:
            missing_python.append(pip_name)

    for cmd, pkg_hint in [("ffmpeg", "ffmpeg"), ("ffprobe", "ffmpeg"), ("sox", "sox"), ("fpcalc", "libchromaprint-tools")]:
        if shutil.which(cmd) is None:
            missing_system.append((cmd, pkg_hint))

    if missing_python or missing_system:
        print("Missing dependencies:\n")
        if missing_python:
            print(f"  Python packages:  pip install {' '.join(missing_python)}")
        for cmd, pkg in missing_system:
            print(f"  System tool '{cmd}':  sudo apt install {pkg}")
        print()
        sys.exit(1)


check_dependencies()

import numpy as np

from tqdm import tqdm


# Confidence threshold for chromaprint correlation (0.0-1.0).
# Scores are bit-match ratios: 0.5 = random chance, higher = better match.
# For different-device recordings of the same event, 0.3+ is a good match.
CONFIDENCE_THRESHOLD = 0.25

RED = "\033[91m"
YELLOW = "\033[33m"
RESET = "\033[0m"


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


def build_event_metadata(events):
    """Build event metadata from grouped segments (no concatenation needed).

    Returns a list of dicts, one per event:
      {
        "segments": list of original file metadata dicts,
        "start_time": creation_time of the first segment,
        "total_duration": sum of segment durations,
      }
    """
    result = []
    for segments in events:
        total_duration = sum(s["duration"] for s in segments)
        start_time = segments[0]["creation_time"]
        result.append({
            "segments": segments,
            "start_time": start_time,
            "total_duration": total_duration,
        })
    return result


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


FPCALC_ITEM_DURATION = 0.1238  # seconds per fingerprint item (4096 / 11025 / 3)
ALIGNMENT_OFFSET_CORRECTION = 0.5  # empirical correction for chromaprint alignment bias

# Non-maximum suppression window (in fingerprint items) for top-N peak
# extraction. ~16 items ≈ 2.0 s: small enough to keep a true peak a few
# seconds from a spurious one, large enough to collapse one peak's shoulder.
NMS_WINDOW_ITEMS = 16

# Minimum fingerprint-item overlap for a lag to be considered a real match.
# ~40 items ≈ 5 s. Prevents a tiny accidental overlap from scoring a
# spuriously perfect bit-match and beating the true alignment.
MIN_OVERLAP_ITEMS_HARD = 40

# Default product floor for how much of a clip must be covered by audio to be
# worth emitting (seconds). Overridable via --min-overlap.
DEFAULT_MIN_OVERLAP_SEC = 10.0


def effective_min_overlap_items(min_overlap_sec):
    """Convert a min-overlap in seconds to fingerprint items, never below the
    correlation-sanity hard floor."""
    return max(MIN_OVERLAP_ITEMS_HARD,
               math.ceil(min_overlap_sec / FPCALC_ITEM_DURATION))


def compute_overlap(lag_items, video_dur, audio_dur):
    """Geometry SSOT: turn a signed correlation lag into video-time overlap.

    lag_items is the index in the audio fingerprint aligned with the clip's
    first item (signed; negative means the video leads). Returns a dict:
      audio_start_in_video : where the audio recording begins in video time
                             (>0 = audio starts partway into the video)
      ov_start, ov_end     : overlap span in video time
      ov_dur               : ov_end - ov_start (<=0 means no overlap)
      audio_cut_start      : offset into the audio recording for the overlap
    The offset correction is applied here, exactly once.
    """
    audio_start_in_video = -lag_items * FPCALC_ITEM_DURATION + ALIGNMENT_OFFSET_CORRECTION
    ov_start = max(0.0, audio_start_in_video)
    ov_end = min(video_dur, audio_start_in_video + audio_dur)
    ov_dur = ov_end - ov_start
    audio_cut_start = ov_start - audio_start_in_video
    return {
        "audio_start_in_video": audio_start_in_video,
        "ov_start": ov_start,
        "ov_end": ov_end,
        "ov_dur": ov_dur,
        "audio_cut_start": audio_cut_start,
    }


def first_keyframe_at_or_after(keyframe_times, ov_start):
    """Return the first keyframe timestamp >= ov_start, or None if none.
    keyframe_times must be sorted ascending."""
    for t in keyframe_times:
        if t >= ov_start:
            return t
    return None


def classify_alignment_skip(score, overlap_dur, min_overlap_sec, it_is_what_it_is):
    """Return a skip reason string, or None to keep the clip.
    Min-overlap is a hard product floor (not overridable by it_is_what_it_is);
    low-confidence is overridable."""
    if overlap_dur < min_overlap_sec:
        return "below-min-overlap"
    if score < CONFIDENCE_THRESHOLD and not it_is_what_it_is:
        return "low-confidence"
    return None


def get_fingerprint(filepath):
    """Get raw chromaprint fingerprint for an audio file using fpcalc.

    Returns a list of 32-bit integers.
    """
    cmd = ["fpcalc", "-raw", "-length", "0", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    {RED}fpcalc failed (exit {result.returncode}): {result.stderr.strip()}{RESET}")
        return []
    for line in result.stdout.strip().split("\n"):
        if line.startswith("FINGERPRINT="):
            return [int(x) for x in line.split("=", 1)[1].split(",")]
    return []


def popcnt(x):
    """Count the number of set bits in a 32-bit integer."""
    return bin(x & 0xFFFFFFFF).count("1")


def correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items,
                                min_overlap_items, search_start=None, search_end=None):
    """Cross-correlate fp_clip against fp_ref over all signed lags.

    A lag L is the index in fp_ref aligned with fp_clip[0]; L may be negative
    (the clip leads). Each lag is scored on its overlapping items only:
    1 - bit_errors / (32 * overlap). Lags whose overlap is below
    min_overlap_items are excluded entirely. Returns up to n (lag, score)
    peaks, descending score, separated by non-maximum suppression.

    search_start / search_end, when given, are inclusive bounds on the lag.
    """
    clip_len = len(fp_clip)
    ref_len = len(fp_ref)
    if clip_len == 0 or ref_len == 0:
        return []

    lo_lag = -(clip_len - 1)
    hi_lag = ref_len - 1
    if search_start is not None:
        lo_lag = max(lo_lag, search_start)
    if search_end is not None:
        hi_lag = min(hi_lag, search_end)
    if lo_lag > hi_lag:
        return []

    n_lags = hi_lag - lo_lag + 1
    scores = np.full(n_lags, -1.0)

    for lag in tqdm(range(lo_lag, hi_lag + 1),
                    desc="    Aligning", unit="lag",
                    bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        i_start = max(0, -lag)
        i_end = min(clip_len, ref_len - lag)
        overlap = i_end - i_start
        if overlap < min_overlap_items:
            continue
        bit_errors = 0
        for i in range(i_start, i_end):
            bit_errors += popcnt(fp_ref[lag + i] ^ fp_clip[i])
        scores[lag - lo_lag] = 1.0 - (bit_errors / (32 * overlap))

    results = []
    work = scores.copy()
    for _ in range(n):
        idx = int(np.argmax(work))
        if work[idx] < 0:
            break
        results.append((idx + lo_lag, float(scores[idx])))
        lo = max(0, idx - nms_window_items)
        hi = min(len(work), idx + nms_window_items + 1)
        work[lo:hi] = -1.0

    return results


def correlate_fingerprints(fp_ref, fp_clip, min_overlap_items, search_start=None, search_end=None):
    """Best single signed lag. Returns (lag, score), or (0, 0.0) if none."""
    peaks = correlate_fingerprints_topn(
        fp_ref, fp_clip, 1, NMS_WINDOW_ITEMS, min_overlap_items, search_start, search_end)
    if not peaks:
        return 0, 0.0
    return peaks[0]


def fingerprint_events(events):
    """Fingerprint all events upfront. Returns a list of fingerprint arrays, one per event."""
    event_fingerprints = []
    total_segments = sum(len(e["segments"]) for e in events)

    print(f"  Fingerprinting {total_segments} audio segment(s) across {len(events)} event(s)...")
    with tqdm(total=total_segments, desc="  Fingerprinting",
              bar_format="{desc}: {n}/{total} [{elapsed}]") as pbar:
        for event in events:
            fp = []
            for seg in event["segments"]:
                fp_seg = get_fingerprint(seg["path"])
                if not fp_seg:
                    print(f"  {RED}ERROR: Could not fingerprint {os.path.basename(seg['path'])}{RESET}")
                    fp = []
                    break
                fp.extend(fp_seg)
                pbar.update(1)
            event_fingerprints.append(fp)

    for i, fp in enumerate(event_fingerprints):
        if fp:
            print(f"  Event {i + 1}: {len(fp)} fingerprint items ({len(fp) * FPCALC_ITEM_DURATION:.0f}s)")
        else:
            print(f"  Event {i + 1}: {RED}fingerprinting failed{RESET}")

    return event_fingerprints


def align_clip_to_event(video_path, event, fp_ref, video_meta, no_hint=False, min_overlap_items=None):
    """Align a video clip to an event using chromaprint fingerprint correlation.

    fp_ref is the pre-computed fingerprint for the event. Returns
    (lag_items, score), where lag_items is the signed offset of the clip's
    first item within the event audio. Returns (None, 0) on failure.
    """
    if min_overlap_items is None:
        min_overlap_items = effective_min_overlap_items(DEFAULT_MIN_OVERLAP_SEC)

    video_time = parse_timestamp(video_meta["creation_time"])
    event_start = parse_timestamp(event["start_time"])

    print(f"    Fingerprinting clip...")
    fp_clip = get_fingerprint(video_path)

    if not fp_ref or not fp_clip:
        print(f"    {RED}ERROR: Could not generate fingerprints{RESET}")
        return None, 0

    print(f"    Reference: {len(fp_ref)} items ({len(fp_ref) * FPCALC_ITEM_DURATION:.0f}s), "
          f"Clip: {len(fp_clip)} items ({len(fp_clip) * FPCALC_ITEM_DURATION:.0f}s)")

    # Try metadata-hinted search first (expressed as a signed-lag window).
    if not no_hint and timestamps_look_plausible(video_time, event_start):
        offset_hint = (video_time - event_start).total_seconds()
        hint_start = int((offset_hint - 1800) / FPCALC_ITEM_DURATION)
        hint_end = int((offset_hint + 1800) / FPCALC_ITEM_DURATION)

        print(f"    Trying metadata-hinted window "
              f"({hint_start * FPCALC_ITEM_DURATION:.0f}s - {hint_end * FPCALC_ITEM_DURATION:.0f}s)...")

        lag_items, score = correlate_fingerprints(
            fp_ref, fp_clip, min_overlap_items, hint_start, hint_end)

        if score >= CONFIDENCE_THRESHOLD:
            print(f"    Match found! Lag: {lag_items} items, confidence: {score:.4f}")
            return lag_items, score
        else:
            print(f"    Narrow search inconclusive (score: {score:.4f}), falling back to full scan...")

    # Full scan over all signed lags.
    print(f"    Full scan of {len(fp_ref) * FPCALC_ITEM_DURATION:.0f}s event audio...")
    lag_items, score = correlate_fingerprints(fp_ref, fp_clip, min_overlap_items)

    if score >= CONFIDENCE_THRESHOLD:
        print(f"    Match found! Lag: {lag_items} items, confidence: {score:.4f}")
    else:
        print(f"    Low confidence match. Lag: {lag_items} items, confidence: {score:.4f}")

    return lag_items, score


def shotgun_align_clip(video, events, event_fingerprints, clip_num, n, no_hint=False, min_overlap_items=None):
    """Build up to n candidate alignments for one clip (shotgun mode).

    Chooses the single best event using the normal per-event selection, then
    runs a FULL scan of that event and returns its top-n correlation peaks as
    candidate alignment dicts (ranked by descending score). The metadata hint
    is intentionally ignored for the peak survey — the hinted auto-pick is what
    produced the bad match in the first place.
    """
    if min_overlap_items is None:
        min_overlap_items = effective_min_overlap_items(DEFAULT_MIN_OVERLAP_SEC)

    fp_clip = get_fingerprint(video["path"])
    if not fp_clip:
        return []

    # Pick the best event (respecting the hint just for event selection).
    best_event_idx = None
    best_score = -1.0
    for j, event in enumerate(events):
        fp_ref = event_fingerprints[j]
        if not fp_ref:
            continue
        _, score = align_clip_to_event(
            video["path"], event, fp_ref, video, no_hint=no_hint,
            min_overlap_items=min_overlap_items)
        if score > best_score:
            best_score = score
            best_event_idx = j

    if best_event_idx is None:
        return []

    event = events[best_event_idx]
    fp_ref = event_fingerprints[best_event_idx]

    print(f"    Shotgun: full-scan top-{n} peaks of event {best_event_idx + 1}...")
    peaks = correlate_fingerprints_topn(
        fp_ref, fp_clip, n, NMS_WINDOW_ITEMS, min_overlap_items)

    candidates = []
    for rank, (lag_items, score) in enumerate(peaks, start=1):
        geo = compute_overlap(lag_items, video["duration"], event["total_duration"])
        candidates.append({
            "video": video,
            "clip_number": clip_num,
            "event": event,
            "offset": geo["audio_start_in_video"],
            "confidence": score,
            "skipped": False,
            "skip_reason": None,
            "ov_start": geo["ov_start"],
            "ov_end": geo["ov_end"],
            "overlap_dur": geo["ov_dur"],
            "audio_cut_start": geo["audio_cut_start"],
            "audio_start_in_video": geo["audio_start_in_video"],
            "candidate": {
                "rank": rank,
                "raw_offset_items": lag_items,
                "correction": ALIGNMENT_OFFSET_CORRECTION,
            },
        })
    return candidates


def align_all_clips(video_files, events, temp_dir, clip_filter=None, from_clip=None, it_is_what_it_is=False, no_hint=False, shotgun=None, min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC):
    """Align video clips to events. Returns list of alignment results."""
    # Fingerprint all events once upfront
    event_fingerprints = fingerprint_events(events)

    min_overlap_items = effective_min_overlap_items(min_overlap_sec)

    alignments = []

    for i, video in enumerate(video_files):
        clip_num = i + 1

        # Apply clip filters
        if clip_filter is not None and clip_num != clip_filter:
            continue
        if from_clip is not None and clip_num < from_clip:
            continue

        print(f"\n  Clip {clip_num}: {os.path.basename(video['path'])} ({video['duration']:.1f}s)")

        if shotgun:
            candidates = shotgun_align_clip(
                video, events, event_fingerprints, clip_num, shotgun, no_hint=no_hint,
                min_overlap_items=min_overlap_items)
            if not candidates:
                print(f"  {YELLOW}WARNING: Clip {clip_num} produced no shotgun candidates{RESET}")
            else:
                print(f"    {len(candidates)} candidate(s): " +
                      ", ".join(f"#{c['candidate']['rank']} @ {c['offset']:.2f}s "
                                f"(conf {c['confidence']:.3f})" for c in candidates))
            alignments.extend(candidates)
            continue

        # Try each event, pick best match
        best_lag = None
        best_score = 0
        best_event = None

        for j, event in enumerate(events):
            fp_ref = event_fingerprints[j]
            if not fp_ref:
                continue
            print(f"    Trying event {j + 1} ({event['total_duration']:.0f}s)...")
            lag, score = align_clip_to_event(video["path"], event, fp_ref, video, no_hint=no_hint,
                                             min_overlap_items=min_overlap_items)
            if score > best_score:
                best_lag = lag
                best_score = score
                best_event = event

        geo = None
        if best_lag is not None and best_event is not None:
            geo = compute_overlap(best_lag, video["duration"], best_event["total_duration"])

        overlap_dur = geo["ov_dur"] if geo else 0.0
        skip_reason = classify_alignment_skip(
            best_score, overlap_dur, min_overlap_sec, it_is_what_it_is)
        if best_event is None:
            skip_reason = "low-confidence"
        skipped = skip_reason is not None

        if skipped:
            print(f"  \033[33mWARNING: Clip {clip_num} skipped — {skip_reason} "
                  f"(confidence {best_score:.2f}, overlap {overlap_dur:.1f}s)\033[0m")

        alignments.append({
            "video": video,
            "clip_number": clip_num,
            "event": best_event,
            "offset": geo["audio_start_in_video"] if geo else None,
            "confidence": best_score,
            "skipped": skipped,
            "skip_reason": skip_reason,
            "ov_start": geo["ov_start"] if geo else None,
            "ov_end": geo["ov_end"] if geo else None,
            "overlap_dur": overlap_dur,
            "audio_cut_start": geo["audio_cut_start"] if geo else None,
            "audio_start_in_video": geo["audio_start_in_video"] if geo else None,
        })

    return alignments


def detect_impulses(signal, sample_rate, window_ms=50, context_ms=500, threshold_db=15, exclude_regions=None):
    """Detect impulse events. exclude_regions is a list of (start_sec, end_sec) to skip."""
    window_samples = int(sample_rate * window_ms / 1000)
    context_windows = int(context_ms / window_ms)

    num_windows = len(signal) // window_samples
    if num_windows == 0:
        return []

    trimmed = signal[:num_windows * window_samples]
    windows = np.abs(trimmed.reshape(num_windows, window_samples))
    peaks = windows.max(axis=1)

    peaks_safe = np.maximum(peaks, 1e-10)
    peaks_db = 20 * np.log10(peaks_safe)

    impulses = []

    for i in range(num_windows):
        timestamp = i * window_ms / 1000

        # Skip windows inside excluded regions (e.g., applause blocks)
        if exclude_regions:
            in_excluded = False
            for ex_start, ex_end in exclude_regions:
                if ex_start <= timestamp < ex_end:
                    in_excluded = True
                    break
            if in_excluded:
                continue

        ctx_start = max(0, i - context_windows)
        ctx_end = min(num_windows, i + context_windows + 1)
        context_indices = list(range(ctx_start, i)) + list(range(i + 1, ctx_end))

        if not context_indices:
            continue

        context_avg_db = np.mean(peaks_db[context_indices])
        excess = peaks_db[i] - context_avg_db

        if excess > threshold_db:
            impulses.append({
                "timestamp": timestamp,
                "window_idx": i,
                "peak_db": float(peaks_db[i]),
                "context_db": float(context_avg_db),
                "excess_db": float(excess),
            })

    return impulses


def attenuate_impulses(signal, impulses, sample_rate, window_ms=50):
    window_samples = int(sample_rate * window_ms / 1000)

    for imp in impulses:
        idx = imp["window_idx"]
        start = idx * window_samples
        end = start + window_samples

        current_peak = np.max(np.abs(signal[start:end]))
        if current_peak > 0:
            target_peak = 10 ** (imp["context_db"] / 20)
            scale = target_peak / current_peak
            signal[start:end] *= scale

    return signal


APPLAUSE_FLATNESS_THRESHOLD = 0.03  # spectral flatness above this = applause
APPLAUSE_WINDOW_SEC = 0.5  # analysis window for spectral flatness
APPLAUSE_FADE_SEC = 0.5  # crossfade duration for gain transitions


def detect_applause(signal, sample_rate):
    """Detect applause sections using spectral flatness.

    Downsamples to 22050Hz for consistent flatness values regardless of source SR.
    Returns a list of (start_sec, end_sec) tuples for each applause block.
    """
    import librosa

    # Downsample to 22050Hz for consistent spectral flatness behavior.
    # Higher sample rates spread energy across more bins, diluting flatness.
    analysis_sr = 22050
    if sample_rate != analysis_sr:
        analysis_signal = librosa.resample(signal.astype(np.float32), orig_sr=sample_rate, target_sr=analysis_sr)
    else:
        analysis_signal = signal.astype(np.float32)

    # Compute spectral flatness in windows
    n_fft = int(analysis_sr * APPLAUSE_WINDOW_SEC)
    hop_length = n_fft  # non-overlapping windows

    if len(analysis_signal) < n_fft:
        return []

    flatness = librosa.feature.spectral_flatness(
        y=analysis_signal, n_fft=n_fft, hop_length=hop_length
    )[0]

    # Label each window as applause or not
    is_applause = flatness >= APPLAUSE_FLATNESS_THRESHOLD

    # Group consecutive applause windows into blocks
    blocks = []
    in_block = False
    block_start = 0

    for i, appl in enumerate(is_applause):
        if appl and not in_block:
            block_start = i
            in_block = True
        elif not appl and in_block:
            start_sec = block_start * APPLAUSE_WINDOW_SEC
            end_sec = i * APPLAUSE_WINDOW_SEC
            blocks.append((start_sec, end_sec))
            in_block = False

    if in_block:
        start_sec = block_start * APPLAUSE_WINDOW_SEC
        end_sec = len(is_applause) * APPLAUSE_WINDOW_SEC
        blocks.append((start_sec, end_sec))

    # Merge blocks that are within 2 seconds of each other
    merged = []
    for start, end in blocks:
        if merged and start - merged[-1][1] <= 2.0:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Extend each block backward by 0.5s to catch onset claps
    extended = [(max(0, start - 0.5), end) for start, end in merged]

    return extended


def attenuate_applause(signal, sample_rate, applause_blocks):
    """Attenuate applause sections so their peak matches the peak of music sections.

    Applies a smoothed gain curve to avoid abrupt volume changes.
    Returns the modified signal.
    """
    if not applause_blocks:
        return signal

    total_samples = len(signal)
    window_samples = int(sample_rate * APPLAUSE_WINDOW_SEC)

    # Compute per-window peak amplitude
    n_windows = total_samples // window_samples
    if n_windows == 0:
        return signal

    trimmed = np.abs(signal[:n_windows * window_samples].reshape(n_windows, window_samples))
    window_peaks = trimmed.max(axis=1)

    # Build a mask: True for music windows, False for applause windows
    music_mask = np.ones(n_windows, dtype=bool)
    for start_sec, end_sec in applause_blocks:
        start_win = int(start_sec / APPLAUSE_WINDOW_SEC)
        end_win = int(end_sec / APPLAUSE_WINDOW_SEC)
        music_mask[start_win:end_win] = False

    # Find the peak of music sections, then set applause ceiling 1 dB below
    music_peaks = window_peaks[music_mask]
    if len(music_peaks) == 0:
        return signal
    music_peak = np.max(music_peaks)

    if music_peak == 0:
        return signal

    # 4 dB below music peak: multiply by 10^(-4/20) ≈ 0.631
    applause_ceiling = music_peak * (10 ** (-4.0 / 20))

    # Build per-window gain: 1.0 for music, reduced for applause
    gains = np.ones(n_windows, dtype=np.float32)
    for i in range(n_windows):
        if not music_mask[i] and window_peaks[i] > applause_ceiling:
            gains[i] = applause_ceiling / window_peaks[i]

    # Smooth the gain curve to avoid abrupt transitions
    fade_windows = max(1, int(APPLAUSE_FADE_SEC / APPLAUSE_WINDOW_SEC))
    kernel = np.ones(fade_windows * 2 + 1) / (fade_windows * 2 + 1)
    gains_smoothed = np.convolve(gains, kernel, mode="same").astype(np.float32)
    # Ensure we never amplify — only attenuate
    gains_smoothed = np.minimum(gains_smoothed, 1.0)

    # Apply per-window gain to the signal
    for i in range(n_windows):
        if gains_smoothed[i] < 1.0:
            start = i * window_samples
            end = start + window_samples
            signal[start:end] *= gains_smoothed[i]

    # Second pass: ensure no applause window still exceeds ceiling after smoothing
    for i in range(n_windows):
        if not music_mask[i]:
            start = i * window_samples
            end = start + window_samples
            actual_peak = np.max(np.abs(signal[start:end]))
            if actual_peak > applause_ceiling:
                signal[start:end] *= applause_ceiling / actual_peak

    return signal


def peak_normalize(signal):
    """Normalize signal so that the peak amplitude is 1.0 (0 dBFS).
    Returns the normalized signal and the gain factor applied.
    """
    if len(signal) == 0:
        return signal, 1.0
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
    probe = run_ffprobe(filepath)
    n_channels = 2
    original_sr = 48000
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "audio":
            n_channels = int(stream.get("channels", 2))
            original_sr = int(stream.get("sample_rate", 48000))
            break

    pcm_path = os.path.join(temp_dir, "pcm_convert.wav")
    cmd = [
        "ffmpeg", "-y", "-i", filepath,
        "-acodec", "pcm_s16le", "-ar", str(original_sr),
        pcm_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    with wave.open(pcm_path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)

    return samples, sr, n_channels


def write_wav_from_float(filepath, signal, sample_rate, n_channels):
    """Write a float32 signal array to a 16-bit PCM WAV file."""
    signal = np.clip(signal, -1.0, 1.0)
    int_signal = (signal * 32767).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_signal.tobytes())



class ClipLogger:
    """Log to both console and a per-clip log file."""

    def __init__(self, log_path):
        self.log_file = open(log_path, "w")

    def log(self, msg, console_msg=None):
        """Write msg to log file (without ANSI codes) and console_msg (or msg) to console."""
        # Strip ANSI codes for the log file
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', msg)
        self.log_file.write(clean + "\n")
        print(console_msg if console_msg is not None else msg)

    def close(self):
        self.log_file.close()


def candidate_suffix(alignment):
    """Filename suffix distinguishing shotgun candidates.

    Empty string for a normal (non-candidate) alignment, so default-mode
    output, temp, and log filenames are unchanged.
    """
    cand = alignment.get("candidate")
    if not cand:
        return ""
    return f"_cand{cand['rank']}_{alignment['offset']:.1f}s"


def get_keyframe_times(video_path):
    """Return sorted keyframe presentation timestamps (seconds) for the video
    stream, using ffprobe. Empty list on failure."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pts_time",
        "-of", "csv=print_section=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return []
    return parse_keyframe_times(result.stdout)


def parse_keyframe_times(stdout):
    """Parse ffprobe csv output into sorted keyframe times (seconds).

    Each line's FIRST comma-separated field is the pts_time. ffprobe may append
    extra fields per frame (e.g. H.265 "User Data Unregistered SEI" side-data),
    so only the first field is the timestamp; the rest must be ignored.
    """
    times = []
    for line in stdout.strip().split("\n"):
        field = line.split(",", 1)[0].strip()
        if not field:
            continue
        try:
            times.append(float(field))
        except ValueError:
            continue
    return sorted(times)


def process_audio_for_clip(alignment, temp_dir, output_dir, min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC):
    """Phase 4: keyframe-snap the overlap, trim the video (copy, no re-encode),
    cut the matching audio, detect applause, attenuate impulses, peak normalize.

    Returns (trimmed_video_path, normalized_wav_path, applause_blocks, impulses)
    or (None, None, [], []) if skipped.
    """
    if alignment["skipped"]:
        return None, None, [], []

    clip_num = alignment["clip_number"]
    basename = os.path.basename(alignment["video"]["path"])
    video_path = alignment["video"]["path"]
    suffix = candidate_suffix(alignment)
    log_name = os.path.splitext(basename)[0] + suffix + ".log"
    logger = ClipLogger(os.path.join(output_dir, log_name))

    print(f"\n  Clip {clip_num}: {basename}")
    logger.log(f"    Confidence: {alignment['confidence']:.4f}, "
               f"overlap {alignment['overlap_dur']:.1f}s, "
               f"audio starts at {alignment['audio_start_in_video']:.2f}s in video")
    if alignment.get("candidate"):
        cand = alignment["candidate"]
        logger.log(f"    Shotgun candidate #{cand['rank']} — "
                   f"raw lag {cand['raw_offset_items']} items, "
                   f"correction applied {cand['correction']:.2f}s")

    ov_start = alignment["ov_start"]
    ov_end = alignment["ov_end"]

    # Snap the overlap start forward to the first keyframe >= ov_start so we can
    # cut the video with -c:v copy (no re-encode) and still be frame-accurate.
    keyframe_times = get_keyframe_times(video_path)
    kf_start = first_keyframe_at_or_after(keyframe_times, ov_start)
    if kf_start is None or kf_start >= ov_end:
        logger.log(f"    {RED}SKIP: no keyframe in overlap [{ov_start:.2f}s, {ov_end:.2f}s]{RESET}")
        logger.close()
        alignment["skipped"] = True
        alignment["skip_reason"] = "no-keyframe-in-overlap"
        return None, None, [], []

    effective_overlap = ov_end - kf_start
    if effective_overlap < min_overlap_sec:
        logger.log(f"    {RED}SKIP: overlap after keyframe snap "
                   f"({effective_overlap:.1f}s) below min ({min_overlap_sec:.1f}s){RESET}")
        logger.close()
        alignment["skipped"] = True
        alignment["skip_reason"] = "below-min-overlap"
        return None, None, [], []

    # Trim the video, copying the stream (no quality loss). Measure the actual
    # duration afterward, since -c copy ends near a packet boundary.
    trimmed_video = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_trim{os.path.splitext(basename)[1]}")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(kf_start),
        "-i", video_path,
        "-t", str(ov_end - kf_start),
        "-c:v", "copy", "-an",
        trimmed_video,
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    trimmed_meta = get_file_metadata(trimmed_video)
    D = trimmed_meta["duration"]
    alignment["trim_start"] = kf_start
    alignment["trim_dur"] = D
    logger.log(f"    Trimmed video from keyframe {kf_start:.2f}s, duration {D:.2f}s "
               f"(dropped {kf_start - ov_start:.2f}s lead-in)")

    # Audio cut starts at the audio-time corresponding to kf_start, length D.
    audio_cut_start = alignment["audio_cut_start"] + (kf_start - ov_start)
    cut_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_cut.wav")
    segments = alignment["event"]["segments"]
    logger.log(f"    Cutting audio at {audio_cut_start:.2f}s for {D:.2f}s...")

    remaining_offset = audio_cut_start
    remaining_duration = D
    cut_parts = []
    part_idx = 0
    for seg in segments:
        if remaining_duration <= 0:
            break
        seg_dur = seg["duration"]
        if remaining_offset >= seg_dur:
            remaining_offset -= seg_dur
            continue
        take_duration = min(remaining_duration, seg_dur - remaining_offset)
        part_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_part_{part_idx}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", seg["path"],
            "-ss", str(remaining_offset), "-t", str(take_duration),
            part_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        cut_parts.append(part_path)
        remaining_duration -= take_duration
        remaining_offset = 0
        part_idx += 1

    if len(cut_parts) == 0:
        logger.log(f"    {RED}ERROR: audio cut start {audio_cut_start:.2f}s is beyond the event audio{RESET}")
        logger.close()
        alignment["skipped"] = True
        alignment["skip_reason"] = "audio-beyond-event"
        return None, None, [], []
    elif len(cut_parts) == 1:
        os.rename(cut_parts[0], cut_path)
    else:
        cmd = ["sox"] + cut_parts + [cut_path]
        subprocess.run(cmd, check=True)
        for p in cut_parts:
            os.remove(p)

    # Step 2: Read the audio segment
    signal, sr, nch = read_audio_as_float(cut_path, temp_dir)

    # Work with mono for detection algorithms
    if nch > 1:
        mono = signal.mean(axis=1)
    else:
        mono = signal

    # Step 3: Detect and attenuate applause
    applause_blocks = detect_applause(mono, sr)

    if applause_blocks:
        logger.log(f"    Applause detected:")
        for start, end in applause_blocks:
            logger.log(f"      {RED}APPLAUSE{RESET} {start:.1f}s - {end:.1f}s ({end - start:.1f}s)")
        if nch > 1:
            for ch in range(nch):
                signal[:, ch] = attenuate_applause(signal[:, ch], sr, applause_blocks)
        else:
            signal = attenuate_applause(signal, sr, applause_blocks)
        logger.log(f"    Applause attenuated to match music level.")
    else:
        logger.log(f"    No applause detected.")

    # Recompute mono after applause attenuation
    if nch > 1:
        mono = signal.mean(axis=1)
    else:
        mono = signal

    # Step 4: Detect and attenuate impulses
    impulses = detect_impulses(mono, sr, exclude_regions=applause_blocks)

    if impulses:
        for imp in impulses:
            logger.log(f"      {RED}IMPULSE{RESET} at {imp['timestamp']:.3f}s — "
                       f"peak: {imp['peak_db']:.1f} dB, context: {imp['context_db']:.1f} dB, "
                       f"excess: {imp['excess_db']:.1f} dB")
        if nch > 1:
            for ch in range(nch):
                signal[:, ch] = attenuate_impulses(signal[:, ch], impulses, sr)
        else:
            signal = attenuate_impulses(signal, impulses, sr)
    else:
        logger.log(f"    No impulses detected.")

    # Step 5: Peak normalize
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
    logger.log(f"    Peak normalized (gain: {gain_db:+.1f} dB)")

    # Step 6: Write normalized audio to temp file
    norm_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_normalized.wav")
    write_wav_from_float(norm_path, signal, sr, nch)

    logger.close()
    return trimmed_video, norm_path, applause_blocks, impulses


def mux_clip(alignment, trimmed_video_path, norm_path, output_dir, keep_original_audio=False):
    """Phase 5: mux the normalized audio onto the trimmed (copied) video."""
    if alignment["skipped"] or trimmed_video_path is None or norm_path is None:
        return None

    clip_num = alignment["clip_number"]
    basename = os.path.basename(alignment["video"]["path"])
    stem, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, stem + candidate_suffix(alignment) + ext)

    inputs = ["-i", trimmed_video_path, "-i", norm_path]
    map_args = ["-map", "0:v", "-map", "1:a"]
    if keep_original_audio:
        inputs += ["-ss", str(alignment["trim_start"]), "-t", str(alignment["trim_dur"]),
                   "-i", alignment["video"]["path"]]
        map_args += ["-map", "2:a"]
    cmd = ["ffmpeg", "-y"] + inputs + ["-c:v", "copy"] + map_args + [output_path]
    print(f"    Clip {clip_num}: {basename} -> {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)

    return output_path


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
    parser.add_argument("--temp-dir", help="Directory for temporary files (default: system temp)")
    parser.add_argument("--no-hint", action="store_true", help="Skip metadata timestamp heuristic, always do full scan")
    parser.add_argument("--keep-original-audio", action="store_true", help="Keep original video audio as a second track (for verifying alignment)")
    parser.add_argument(
        "--shotgun",
        type=int,
        metavar="N",
        help="Emit N candidate outputs per clip (different alignment offsets) "
             "instead of auto-picking one. Use to recover from a bad match.",
    )
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP_SEC,
        metavar="SEC",
        help=f"Minimum seconds of audio/video overlap to emit a clip "
             f"(default: {DEFAULT_MIN_OVERLAP_SEC}). Below this, the clip is skipped.",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    input_dir = args.input
    output_dir = args.output

    if args.shotgun is not None and args.shotgun < 2:
        print("Error: --shotgun N requires N >= 2.")
        sys.exit(2)

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

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

    print(f"\n{'=' * 60}")
    print("Phase 2: Grouping audio segments")
    print("=" * 60)
    events = group_audio_into_events(audio_files)
    reconstituted = build_event_metadata(events)
    print(f"  Detected {len(reconstituted)} event(s)")
    for i, e in enumerate(reconstituted):
        print(f"  Event {i + 1}: {e['total_duration']:.0f}s "
              f"({len(e['segments'])} segment(s)), starts {e['start_time']}")

    if args.temp_dir:
        os.makedirs(args.temp_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sound_fixer_", dir=args.temp_dir) as temp_dir:
        print(f"\n{'=' * 60}")
        print("Phase 3: Aligning video clips to audio")
        print("=" * 60)
        alignments = align_all_clips(
            video_files, reconstituted, temp_dir,
            clip_filter=args.clip,
            from_clip=args.from_clip,
            it_is_what_it_is=args.it_is_what_it_is,
            no_hint=args.no_hint,
            shotgun=args.shotgun,
            min_overlap_sec=args.min_overlap,
        )

        # Phase 4: Process audio
        print(f"\n{'=' * 60}")
        print("Phase 4: Processing audio")
        print("=" * 60)
        clip_results = []

        for alignment in alignments:
            trimmed_video, norm_path, applause_blocks, impulses = process_audio_for_clip(
                alignment, temp_dir, output_dir, min_overlap_sec=args.min_overlap)
            clip_results.append((alignment, trimmed_video, norm_path))

        # Phase 5: Mux final videos
        print(f"\n{'=' * 60}")
        print("Phase 5: Muxing final videos")
        print("=" * 60)
        for alignment, trimmed_video, norm_path in clip_results:
            mux_clip(alignment, trimmed_video, norm_path, output_dir,
                     keep_original_audio=args.keep_original_audio)

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


if __name__ == "__main__":
    main()

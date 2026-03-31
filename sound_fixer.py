#!/usr/bin/env python3
"""Sound Fixer — replace video audio with high-quality dedicated recordings."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

from tqdm import tqdm
from audio_offset_finder.audio_offset_finder import find_offset_between_files


# Confidence threshold: scores >= 10 are likely correct per audio-offset-finder docs.
# Scores < 5 are unlikely correct. We use 8 as a reasonable middle ground.
CONFIDENCE_THRESHOLD = 8


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

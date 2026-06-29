import subprocess
import sys
from unittest.mock import patch, MagicMock

from soundgraft.cli import (
    parse_args, DEFAULT_MIN_OVERLAP_SEC, align_all_clips,
    effective_min_overlap_items,
)


def test_shotgun_arg_parsed():
    args = parse_args(["--input", "in", "--output", "out", "--shotgun", "3"])
    assert args.shotgun == 3


def test_shotgun_defaults_to_none():
    args = parse_args(["--input", "in", "--output", "out"])
    assert args.shotgun is None


def test_help_flag():
    result = subprocess.run(
        [sys.executable, "-m", "soundgraft.cli", "--help"],
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
        [sys.executable, "-m", "soundgraft.cli"],
        capture_output=True, text=True
    )
    assert result.returncode != 0


def test_min_overlap_default():
    args = parse_args(["--input", "in", "--output", "out"])
    assert args.min_overlap == DEFAULT_MIN_OVERLAP_SEC


def test_min_overlap_parsed():
    args = parse_args(["--input", "in", "--output", "out", "--min-overlap", "25"])
    assert args.min_overlap == 25.0


def test_min_overlap_items_threaded_into_align_clip_to_event():
    """align_all_clips must forward min_overlap_items derived from min_overlap_sec
    to align_clip_to_event (normal mode) and shotgun_align_clip (shotgun mode)."""
    event = {"segments": [], "start_time": None, "total_duration": 30.0}
    video = {"path": "clip.mp4", "duration": 10.0, "creation_time": None}
    min_overlap_sec = 20.0
    expected_items = effective_min_overlap_items(min_overlap_sec)

    captured = {}

    def fake_align_clip_to_event(video_path, event, fp_ref, video_meta,
                                 no_hint=False, min_overlap_items=None, **kwargs):
        captured["min_overlap_items"] = min_overlap_items
        return (0, 0.9)

    def fake_shotgun_align_clip(video, events, event_fingerprints, clip_num, n,
                                no_hint=False, min_overlap_items=None, **kwargs):
        captured["shotgun_min_overlap_items"] = min_overlap_items
        return []

    fake_fingerprints = [[1, 2, 3]]

    with patch("soundgraft.cli.fingerprint_events", return_value=fake_fingerprints), \
         patch("soundgraft.cli.align_clip_to_event", side_effect=fake_align_clip_to_event), \
         patch("soundgraft.cli.shotgun_align_clip", side_effect=fake_shotgun_align_clip):
        # Normal mode
        align_all_clips([video], [event], "/tmp", min_overlap_sec=min_overlap_sec)
        assert captured.get("min_overlap_items") == expected_items, (
            f"align_clip_to_event got min_overlap_items={captured.get('min_overlap_items')}, "
            f"expected {expected_items}"
        )

        # Shotgun mode
        align_all_clips([video], [event], "/tmp", shotgun=2, min_overlap_sec=min_overlap_sec)
        assert captured.get("shotgun_min_overlap_items") == expected_items, (
            f"shotgun_align_clip got min_overlap_items={captured.get('shotgun_min_overlap_items')}, "
            f"expected {expected_items}"
        )


def test_offset_correction_default():
    from soundgraft.cli import ALIGNMENT_OFFSET_CORRECTION
    args = parse_args(["--input", "in", "--output", "out"])
    assert args.offset_correction == ALIGNMENT_OFFSET_CORRECTION


def test_offset_correction_parsed():
    args = parse_args(["--input", "in", "--output", "out", "--offset-correction", "-0.19"])
    assert args.offset_correction == -0.19

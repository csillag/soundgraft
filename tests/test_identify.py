import os
import tempfile
from unittest.mock import patch

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

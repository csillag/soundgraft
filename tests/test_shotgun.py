from unittest.mock import patch
from soundgraft.cli import candidate_suffix, shotgun_align_clip


def test_candidate_suffix_empty_for_normal_alignment():
    alignment = {"offset": 12.3, "candidate": None}
    assert candidate_suffix(alignment) == ""


def test_candidate_suffix_absent_key_is_empty():
    assert candidate_suffix({"offset": 12.3}) == ""


def test_candidate_suffix_for_candidate():
    alignment = {"offset": 16.4, "candidate": {"rank": 2, "raw_offset_items": 130, "correction": 0.5}}
    assert candidate_suffix(alignment) == "_cand2_16.4s"


# Reuse the planted-peak helper shape from test_correlate.
_CLIP = [0x0F0F0F0F, 0x33333333, 0x55555555]
_MISS = 0xFFFFFFFF


def _ref_with_peaks(positions, length):
    ref = [_MISS] * length
    for p in positions:
        ref[p:p + len(_CLIP)] = _CLIP
    return ref


def test_shotgun_align_clip_returns_n_candidates():
    fp_ref = _ref_with_peaks([5, 25], length=40)
    event = {"segments": [], "start_time": None, "total_duration": 5.0}
    video = {"path": "clip.mp4", "duration": 0.4, "creation_time": None}

    # get_fingerprint is called for the clip (and inside align_clip_to_event);
    # always return the planted clip pattern.
    with patch("soundgraft.cli.get_fingerprint", return_value=_CLIP):
        candidates = shotgun_align_clip(
            video, [event], [fp_ref], clip_num=1, n=2, no_hint=True, min_overlap_items=2)

    assert len(candidates) == 2
    # Distinct offsets, ranked, candidate metadata present, never skipped.
    assert candidates[0]["candidate"]["rank"] == 1
    assert candidates[1]["candidate"]["rank"] == 2
    assert candidates[0]["clip_number"] == 1
    assert all(c["skipped"] is False for c in candidates)
    assert all(c["event"] is event for c in candidates)
    offsets = {round(c["offset"], 3) for c in candidates}
    assert len(offsets) == 2
    # Raw offsets recorded for the correction experiment.
    raw = {c["candidate"]["raw_offset_items"] for c in candidates}
    assert raw == {5, 25}
    from soundgraft.cli import compute_overlap
    for c in candidates:
        expected = compute_overlap(
            c["candidate"]["raw_offset_items"], video["duration"], event["total_duration"]
        )["audio_start_in_video"]
        assert abs(c["offset"] - expected) < 1e-9


def test_shotgun_candidates_carry_geometry_fields():
    from unittest.mock import patch
    _CLIP = [0x0F0F0F0F, 0x33333333, 0x55555555]
    _MISS = 0xFFFFFFFF
    fp_ref = [_MISS] * 40
    fp_ref[5:8] = _CLIP
    fp_ref[25:28] = _CLIP
    event = {"segments": [], "start_time": None, "total_duration": 5.0}
    video = {"path": "clip.mp4", "duration": 0.4, "creation_time": None}
    with patch("soundgraft.cli.get_fingerprint", return_value=_CLIP):
        cands = shotgun_align_clip(
            video, [event], [fp_ref], clip_num=1, n=2, no_hint=True, min_overlap_items=2)
    for c in cands:
        for key in ("ov_start", "ov_end", "overlap_dur", "audio_cut_start",
                    "audio_start_in_video", "skip_reason"):
            assert key in c
        assert c["overlap_dur"] == c["ov_end"] - c["ov_start"]

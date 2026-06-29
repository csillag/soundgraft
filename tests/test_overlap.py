from soundgraft.cli import (
    compute_overlap,
    first_keyframe_at_or_after,
    effective_min_overlap_items,
    classify_alignment_skip,
    MIN_OVERLAP_ITEMS_HARD,
    FPCALC_ITEM_DURATION,
    ALIGNMENT_OFFSET_CORRECTION,
)


def test_effective_min_overlap_uses_hard_floor_when_seconds_small():
    # 1s / 0.1238 ≈ 9 items, below the hard floor of 40
    assert effective_min_overlap_items(1.0) == MIN_OVERLAP_ITEMS_HARD


def test_effective_min_overlap_scales_with_seconds():
    import math
    assert effective_min_overlap_items(20.0) == math.ceil(20.0 / FPCALC_ITEM_DURATION)


def test_compute_overlap_audio_contains_video():
    # lag > 0: video[0] aligns deep into audio -> audio starts before video.
    # audio_start_in_video = -100*DUR + 0.5
    r = compute_overlap(lag_items=100, video_dur=30.0, audio_dur=600.0)
    asv = -100 * FPCALC_ITEM_DURATION + ALIGNMENT_OFFSET_CORRECTION
    assert abs(r["audio_start_in_video"] - asv) < 1e-9
    assert r["ov_start"] == 0.0            # audio covers the whole video start
    assert abs(r["ov_end"] - 30.0) < 1e-9  # video fully covered
    assert abs(r["ov_dur"] - 30.0) < 1e-9
    assert abs(r["audio_cut_start"] - (0.0 - asv)) < 1e-9


def test_compute_overlap_video_leads_partial():
    # lag < 0: video leads; audio starts partway into the video.
    # audio_start_in_video = -(-200)*DUR + 0.5 = 200*DUR + 0.5 (positive)
    r = compute_overlap(lag_items=-200, video_dur=120.0, audio_dur=30.0)
    asv = 200 * FPCALC_ITEM_DURATION + ALIGNMENT_OFFSET_CORRECTION
    assert abs(r["audio_start_in_video"] - asv) < 1e-9
    assert abs(r["ov_start"] - asv) < 1e-9                 # overlap starts where audio starts
    assert abs(r["ov_end"] - min(120.0, asv + 30.0)) < 1e-9
    assert r["ov_dur"] > 0
    assert abs(r["audio_cut_start"] - 0.0) < 1e-9          # cut audio from its beginning


def test_compute_overlap_no_overlap_is_nonpositive():
    # Audio ends before video starts.
    r = compute_overlap(lag_items=-100000, video_dur=10.0, audio_dur=5.0)
    assert r["ov_dur"] <= 0


def test_first_keyframe_at_or_after_picks_first_ge():
    times = [0.0, 1.0, 2.0, 3.5, 5.0]
    assert first_keyframe_at_or_after(times, 2.0) == 2.0
    assert first_keyframe_at_or_after(times, 2.1) == 3.5
    assert first_keyframe_at_or_after(times, 0.0) == 0.0


def test_first_keyframe_none_when_past_end():
    assert first_keyframe_at_or_after([0.0, 1.0], 5.0) is None


def test_classify_skip_reasons():
    # Below min overlap takes priority.
    assert classify_alignment_skip(0.9, overlap_dur=3.0, min_overlap_sec=10.0, it_is_what_it_is=False) == "below-min-overlap"
    # Low confidence when overlap ok and score under threshold.
    assert classify_alignment_skip(0.10, overlap_dur=30.0, min_overlap_sec=10.0, it_is_what_it_is=False) == "low-confidence"
    # it_is_what_it_is overrides low confidence but NOT min-overlap.
    assert classify_alignment_skip(0.10, overlap_dur=30.0, min_overlap_sec=10.0, it_is_what_it_is=True) is None
    assert classify_alignment_skip(0.10, overlap_dur=3.0, min_overlap_sec=10.0, it_is_what_it_is=True) == "below-min-overlap"
    # Good match: no skip.
    assert classify_alignment_skip(0.9, overlap_dur=30.0, min_overlap_sec=10.0, it_is_what_it_is=False) is None

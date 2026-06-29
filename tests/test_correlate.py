from soundgraft.cli import correlate_fingerprints, correlate_fingerprints_topn

CLIP = [0x0F0F0F0F, 0x33333333, 0x55555555]
MISS = 0xFFFFFFFF


def _ref_with_peaks(positions, length):
    ref = [MISS] * length
    for p in positions:
        ref[p:p + len(CLIP)] = CLIP
    return ref


def test_positive_lag_match():
    ref = _ref_with_peaks([10], length=40)
    lag, score = correlate_fingerprints(ref, CLIP, min_overlap_items=3)
    assert lag == 10
    assert abs(score - 1.0) < 1e-9


def test_negative_lag_when_clip_leads():
    # Clip's tail overlaps the start of ref: only the last 2 clip items overlap.
    # Build ref so that ref[0:2] == CLIP[1:3]; that is lag = -1.
    ref = [MISS] * 40
    ref[0] = CLIP[1]
    ref[1] = CLIP[2]
    peaks = correlate_fingerprints_topn(ref, CLIP, n=1, nms_window_items=3, min_overlap_items=2)
    assert peaks[0][0] == -1
    assert abs(peaks[0][1] - 1.0) < 1e-9


def test_min_overlap_excludes_short_overlaps():
    # Plant a perfect 2-item overlap at the extreme negative lag, but require 3.
    ref = [MISS] * 40
    ref[0] = CLIP[1]
    ref[1] = CLIP[2]
    # Also a full 3-item match at lag 20.
    ref[20:23] = CLIP
    peaks = correlate_fingerprints_topn(ref, CLIP, n=5, nms_window_items=3, min_overlap_items=3)
    lags = [p[0] for p in peaks]
    assert 20 in lags
    assert -1 not in lags  # the 2-item overlap is below the floor


def test_normalized_score_uses_overlap_count():
    # Partial overlap of 2 items, both exact -> score 1.0 despite clip_len 3.
    ref = [MISS] * 40
    ref[0] = CLIP[1]
    ref[1] = CLIP[2]
    peaks = correlate_fingerprints_topn(ref, CLIP, n=1, nms_window_items=3, min_overlap_items=2)
    assert abs(peaks[0][1] - 1.0) < 1e-9  # normalized over 2, not 3


def test_topn_two_separated_peaks():
    ref = _ref_with_peaks([5, 25], length=40)
    peaks = correlate_fingerprints_topn(ref, CLIP, n=2, nms_window_items=3, min_overlap_items=3)
    assert {p[0] for p in peaks} == {5, 25}


def test_empty_inputs():
    assert correlate_fingerprints_topn([], CLIP, n=2, nms_window_items=3, min_overlap_items=2) == []
    assert correlate_fingerprints(CLIP, [], min_overlap_items=2) == (0, 0.0)

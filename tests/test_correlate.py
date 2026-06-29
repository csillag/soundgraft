from soundgraft.cli import correlate_fingerprints, correlate_fingerprints_topn

# Three-item "clip" pattern with a mix of bits.
CLIP = [0x0F0F0F0F, 0x33333333, 0x55555555]
MISS = 0xFFFFFFFF  # XOR vs any CLIP item yields many bit errors -> low score


def _ref_with_peaks(positions, length):
    """Build a reference fingerprint of MISS values with CLIP planted at each position."""
    ref = [MISS] * length
    for p in positions:
        ref[p:p + len(CLIP)] = CLIP
    return ref


def test_topn_returns_two_separated_peaks_in_score_order():
    ref = _ref_with_peaks([5, 20], length=30)
    peaks = correlate_fingerprints_topn(ref, CLIP, n=2, nms_window_items=3)
    assert len(peaks) == 2
    offsets = {p[0] for p in peaks}
    assert offsets == {5, 20}
    # Both are exact matches -> score 1.0
    assert all(abs(score - 1.0) < 1e-9 for _, score in peaks)
    # Sorted descending by score
    assert peaks[0][1] >= peaks[1][1]


def test_nms_suppresses_nearby_twin_peak():
    # Two exact length-1 matches 2 items apart; NMS window 3 must suppress the
    # twin so it does not appear as a second peak.
    single = [0x0F0F0F0F]
    ref = [MISS] * 30
    ref[5] = single[0]
    ref[7] = single[0]
    peaks = correlate_fingerprints_topn(ref, single, n=2, nms_window_items=3)
    offsets = [o for o, _ in peaks]
    assert offsets[0] == 5          # highest peak
    assert 7 not in offsets         # twin suppressed by NMS despite exact match


def test_topn_empty_on_empty_input():
    assert correlate_fingerprints_topn([], CLIP, n=2, nms_window_items=3) == []
    assert correlate_fingerprints_topn(CLIP, [], n=2, nms_window_items=3) == []


def test_correlate_wrapper_matches_top1():
    ref = _ref_with_peaks([5, 20], length=30)
    offset, score = correlate_fingerprints(ref, CLIP)
    top1 = correlate_fingerprints_topn(ref, CLIP, n=1, nms_window_items=3)[0]
    assert (offset, score) == top1
    assert offset == 5
    assert abs(score - 1.0) < 1e-9

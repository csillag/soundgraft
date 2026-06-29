# Partial-Overlap Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bounded "audio always contains video" slide with a general cross-correlation that aligns video and audio at any signed lag, then cut the video (no re-encode, keyframe-snapped) to the audio-covered overlap.

**Architecture:** Generalize the correlation primitive to score every signed lag over its overlapping region only (normalized, with a minimum-overlap floor). A pure `compute_overlap` function turns the chosen lag into video-time overlap bounds and an audio cut offset. Phase 4 snaps the overlap start to the first keyframe, trims the video with `-c:v copy`, and cuts the matching audio to the trimmed duration. The alignment data contract gains geometry fields; the legacy `offset` field is kept through the producer change and removed when Phase 4 switches over.

**Tech Stack:** Python 3, numpy, tqdm, chromaprint (`fpcalc`), ffmpeg/ffprobe, sox. Tests use pytest.

## Global Constraints

- No LLM co-author trailers in git commit messages.
- All project rules live in AGENTS.md; never create a CLAUDE.md.
- Never re-encode the video stream — output must use `-c:v copy`.
- Single source of truth: one correlation primitive (`correlate_fingerprints` wraps `correlate_fingerprints_topn`); one geometry function (`compute_overlap`); no duplicated correlation or geometry logic.
- The offset correction is applied exactly once, in `compute_overlap`, using the existing constant `ALIGNMENT_OFFSET_CORRECTION = 0.5` (value unchanged; empirical 0.5-vs-1.0 question stays deferred).
- `FPCALC_ITEM_DURATION = 0.1238` is the seconds-per-item constant; reuse it, never re-derive.

---

## File Structure

- `src/soundgraft/cli.py` — all production changes (single-module project):
  - New constants `MIN_OVERLAP_ITEMS_HARD`, `DEFAULT_MIN_OVERLAP_SEC`; `import math`.
  - New pure functions `effective_min_overlap_items`, `compute_overlap`, `first_keyframe_at_or_after`, `classify_alignment_skip`.
  - Generalized `correlate_fingerprints_topn` + `correlate_fingerprints` wrapper (signed lags, normalized score, min-overlap floor).
  - `align_clip_to_event`, `shotgun_align_clip`, `align_all_clips` rewired to lag + geometry.
  - New `get_keyframe_times`; `process_audio_for_clip` + `mux_clip` reworked for keyframe-snapped trim.
  - `--min-overlap` CLI option threaded through.
- `tests/test_overlap.py` — **new**: `compute_overlap`, `first_keyframe_at_or_after`, `effective_min_overlap_items`, `classify_alignment_skip`.
- `tests/test_correlate.py` — **extend/replace**: signed-lag cross-correlation tests.
- `tests/test_shotgun.py` — **update**: shotgun candidates under the new lag semantics.
- `tests/test_cli.py` — **extend**: `--min-overlap` parsing.
- `README.md`, `AGENTS.md` — docs.

---

## Task 1: Pure geometry and threshold helpers

**Files:**
- Modify: `src/soundgraft/cli.py` (add `import math` near the top imports; add constants after `NMS_WINDOW_ITEMS`; add four pure functions after the constants, before `get_fingerprint`)
- Test: `tests/test_overlap.py` (create)

**Interfaces:**
- Consumes: `FPCALC_ITEM_DURATION`, `ALIGNMENT_OFFSET_CORRECTION`, `CONFIDENCE_THRESHOLD` (existing module constants).
- Produces:
  - `MIN_OVERLAP_ITEMS_HARD = 40`, `DEFAULT_MIN_OVERLAP_SEC = 10.0`
  - `effective_min_overlap_items(min_overlap_sec: float) -> int`
  - `compute_overlap(lag_items: int, video_dur: float, audio_dur: float) -> dict` with keys `audio_start_in_video, ov_start, ov_end, ov_dur, audio_cut_start`
  - `first_keyframe_at_or_after(keyframe_times: list[float], ov_start: float) -> float | None`
  - `classify_alignment_skip(score, overlap_dur, min_overlap_sec, it_is_what_it_is) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_overlap.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_overlap.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_overlap'`.

- [ ] **Step 3: Add `import math`**

At the top of `src/soundgraft/cli.py`, the existing imports include `import json`, `import os`, etc. Add `import math` alongside them (alphabetical order: after `import json`).

- [ ] **Step 4: Add constants**

Immediately after the `NMS_WINDOW_ITEMS = 16` line (and its comment block), add:

```python
# Minimum fingerprint-item overlap for a lag to be considered a real match.
# ~40 items ≈ 5 s. Prevents a tiny accidental overlap from scoring a
# spuriously perfect bit-match and beating the true alignment.
MIN_OVERLAP_ITEMS_HARD = 40

# Default product floor for how much of a clip must be covered by audio to be
# worth emitting (seconds). Overridable via --min-overlap.
DEFAULT_MIN_OVERLAP_SEC = 10.0
```

- [ ] **Step 5: Add the four pure functions**

Immediately after the constants from Step 4 (before `def get_fingerprint`), add:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_overlap.py -v`
Expected: PASS (9 tests).

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all existing tests still PASS (no wiring changed yet).

- [ ] **Step 8: Commit**

```bash
git add src/soundgraft/cli.py tests/test_overlap.py
git commit -m "feat: add pure overlap geometry and threshold helpers"
```

---

## Task 2: Generalize the correlation primitive to signed lags

**Files:**
- Modify: `src/soundgraft/cli.py` (`correlate_fingerprints_topn` lines 242-289; `correlate_fingerprints` lines 292-302; the two `correlate_fingerprints` call sites in `align_clip_to_event` lines 363/374; the `correlate_fingerprints_topn` call in `shotgun_align_clip` line 417; add a `min_overlap_items` parameter to `align_clip_to_event` and `shotgun_align_clip`)
- Test: `tests/test_correlate.py` (replace contents), `tests/test_shotgun.py` (update)

**Interfaces:**
- Consumes: `effective_min_overlap_items`, `DEFAULT_MIN_OVERLAP_SEC`, `NMS_WINDOW_ITEMS` (Task 1 / existing).
- Produces:
  - `correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items, min_overlap_items, search_start=None, search_end=None) -> list[(lag:int, score:float)]` — `lag` is signed, scores normalized over overlap, lags below `min_overlap_items` excluded.
  - `correlate_fingerprints(fp_ref, fp_clip, min_overlap_items, search_start=None, search_end=None) -> (lag:int, score:float)`
  - `align_clip_to_event(video_path, event, fp_ref, video_meta, no_hint=False, min_overlap_items=None) -> (lag_items:int, score:float)` — now returns a **lag**, not seconds; no correction applied here.
  - `shotgun_align_clip(..., min_overlap_items=None)` — candidate `offset` is computed via `compute_overlap` (see Task 3 finalizes the dict; this task keeps it working by storing `audio_start_in_video` as `offset`).

- [ ] **Step 1: Replace `tests/test_correlate.py`**

Overwrite `tests/test_correlate.py` with signed-lag tests:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_correlate.py -v`
Expected: FAIL (signature mismatch / wrong results — `min_overlap_items` is not yet a parameter).

- [ ] **Step 3: Replace `correlate_fingerprints_topn` and `correlate_fingerprints`**

Replace the whole existing `correlate_fingerprints_topn` (lines 242-289) and `correlate_fingerprints` (lines 292-302) with:

```python
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
```

- [ ] **Step 4: Update `align_clip_to_event` to return a lag**

Replace the `align_clip_to_event` body from its signature (line 334) through its `return` (line 382). The new version takes `min_overlap_items`, returns `(lag_items, score)`, applies no correction, and uses signed-lag hint bounds:

```python
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
```

- [ ] **Step 5: Update `shotgun_align_clip` for the new signatures**

In `shotgun_align_clip` (lines 385-435): add the `min_overlap_items` parameter, pass it through, and build candidate offsets via `compute_overlap`. Replace the signature line and the body from the event-selection loop onward:

Replace the signature (line 385):
```python
def shotgun_align_clip(video, events, event_fingerprints, clip_num, n, no_hint=False, min_overlap_items=None):
```

Immediately after the docstring, before `fp_clip = get_fingerprint(...)`, add:
```python
    if min_overlap_items is None:
        min_overlap_items = effective_min_overlap_items(DEFAULT_MIN_OVERLAP_SEC)
```

In the event-selection loop, change the `align_clip_to_event` call to pass the floor:
```python
        _, score = align_clip_to_event(
            video["path"], event, fp_ref, video, no_hint=no_hint,
            min_overlap_items=min_overlap_items)
```

Change the top-N call (line 417) to pass the floor:
```python
    peaks = correlate_fingerprints_topn(
        fp_ref, fp_clip, n, NMS_WINDOW_ITEMS, min_overlap_items)
```

Replace the candidate-building loop (lines 419-434) with a version that uses `compute_overlap` and stores `audio_start_in_video` as the legacy `offset`:
```python
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
            "candidate": {
                "rank": rank,
                "raw_offset_items": lag_items,
                "correction": ALIGNMENT_OFFSET_CORRECTION,
            },
        })
    return candidates
```

- [ ] **Step 6: Update `align_all_clips`'s single-match path for the lag return**

In `align_all_clips` (lines 468-497), the non-shotgun path calls `align_clip_to_event` and stores `best_offset`. Update it to convert the returned lag to the legacy `offset` (= `audio_start_in_video`) so Phase 4 keeps working unchanged for now. Replace the event loop and dict build (lines 468-497) with:

```python
        # Try each event, pick best match
        best_lag = None
        best_score = 0
        best_event = None

        for j, event in enumerate(events):
            fp_ref = event_fingerprints[j]
            if not fp_ref:
                continue
            print(f"    Trying event {j + 1} ({event['total_duration']:.0f}s)...")
            lag, score = align_clip_to_event(video["path"], event, fp_ref, video, no_hint=no_hint)
            if score > best_score:
                best_lag = lag
                best_score = score
                best_event = event

        skipped = best_score < CONFIDENCE_THRESHOLD and not it_is_what_it_is

        if skipped:
            print(f"  \033[33mWARNING: Clip {clip_num} skipped — "
                  f"best confidence {best_score:.2f} below threshold {CONFIDENCE_THRESHOLD}\033[0m")

        offset = None
        if best_lag is not None and best_event is not None:
            offset = compute_overlap(
                best_lag, video["duration"], best_event["total_duration"])["audio_start_in_video"]

        alignments.append({
            "video": video,
            "clip_number": clip_num,
            "event": best_event,
            "offset": offset,
            "confidence": best_score,
            "skipped": skipped,
        })
```

- [ ] **Step 7: Update `tests/test_shotgun.py` for the new lag semantics**

In `tests/test_shotgun.py`, the existing `test_shotgun_align_clip_returns_n_candidates` plants peaks and asserts `raw == {5, 25}`. Under cross-correlation those lags are still found, but the call now needs `min_overlap_items`. Update the `shotgun_align_clip` call in that test to pass a small floor, and assert candidate offsets come from `compute_overlap`:

Change the call to:
```python
        candidates = shotgun_align_clip(
            video, [event], [fp_ref], clip_num=1, n=2, no_hint=True, min_overlap_items=2)
```

Keep the existing assertions on `rank`, `clip_number`, `skipped is False`, `event is event`, and `raw == {5, 25}`. Replace any assertion that compared `offset` to a `*DUR + correction` formula with:
```python
    from soundgraft.cli import compute_overlap
    for c in candidates:
        expected = compute_overlap(
            c["candidate"]["raw_offset_items"], video["duration"], event["total_duration"]
        )["audio_start_in_video"]
        assert abs(c["offset"] - expected) < 1e-9
```
(Set `event["total_duration"]` and `video["duration"]` to concrete values in the test fixture if not already present — e.g. `event = {"segments": [], "start_time": None, "total_duration": 5.0}` and `video = {"path": "clip.mp4", "duration": 0.4, "creation_time": None}`.)

- [ ] **Step 8: Run the updated tests**

Run: `pytest tests/test_correlate.py tests/test_shotgun.py -v`
Expected: PASS.

- [ ] **Step 9: Run the full suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add src/soundgraft/cli.py tests/test_correlate.py tests/test_shotgun.py
git commit -m "feat: generalize correlation to signed-lag cross-correlation"
```

---

## Task 3: Geometry and skip reasons in the alignment result

**Files:**
- Modify: `src/soundgraft/cli.py` (`align_all_clips` single-match dict build; `shotgun_align_clip` candidate dict build — both add geometry fields and `skip_reason`)
- Test: `tests/test_shotgun.py` (extend)

**Interfaces:**
- Consumes: `compute_overlap`, `classify_alignment_skip`, `effective_min_overlap_items`, `DEFAULT_MIN_OVERLAP_SEC`.
- Produces: every alignment/candidate dict additionally carries `ov_start`, `ov_end`, `overlap_dur`, `audio_cut_start`, `audio_start_in_video`, and `skip_reason` (str or None). `align_all_clips` gains a `min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC` parameter.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shotgun.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shotgun.py::test_shotgun_candidates_carry_geometry_fields -v`
Expected: FAIL with `assert 'ov_start' in c` (KeyError-style assertion failure).

- [ ] **Step 3: Add geometry to `shotgun_align_clip` candidates**

In `shotgun_align_clip`'s candidate loop (the version from Task 2 Step 5), expand each appended dict to include the full geometry. Replace that loop with:

```python
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
```

(Shotgun deliberately emits candidates regardless of overlap/confidence, so `skip_reason` stays `None` here — the user is overriding the auto-decision.)

- [ ] **Step 4: Add geometry + skip reasons to `align_all_clips`**

Change the `align_all_clips` signature to add `min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC`:
```python
def align_all_clips(video_files, events, temp_dir, clip_filter=None, from_clip=None, it_is_what_it_is=False, no_hint=False, shotgun=None, min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC):
```

Replace the single-match dict build (from Task 2 Step 6, the block starting `skipped = best_score < ...` through `alignments.append({...})`) with:

```python
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
```

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_shotgun.py -v`
Expected: PASS (all shotgun tests, including the new geometry test).

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/soundgraft/cli.py tests/test_shotgun.py
git commit -m "feat: carry overlap geometry and skip reasons in alignment results"
```

---

## Task 4: Keyframe-snapped video trim and audio cut

**Files:**
- Modify: `src/soundgraft/cli.py` (add `get_keyframe_times`; rewrite the cut section of `process_audio_for_clip`; change its return to include a trimmed-video path; update `mux_clip`; update the Phase 4/5 loop in `main`)
- Test: `tests/test_overlap.py` (extend with a keyframe-extraction-independent skip-branch test is not feasible without a refactor; instead unit-test the pure pieces already covered in Task 1 and rely on manual media testing here — see Testing note)

**Interfaces:**
- Consumes: `compute_overlap` outputs already on the alignment dict (`ov_start`, `ov_end`, `overlap_dur`, `audio_cut_start`), `first_keyframe_at_or_after`, `effective_min_overlap_items`, `DEFAULT_MIN_OVERLAP_SEC`.
- Produces:
  - `get_keyframe_times(video_path) -> list[float]` (sorted keyframe PTS seconds)
  - `process_audio_for_clip(alignment, temp_dir, output_dir, min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC) -> (trimmed_video_path, norm_path, applause_blocks, impulses)`; returns `(None, None, [], [])` when skipped.
  - `mux_clip(alignment, trimmed_video_path, norm_path, output_dir, keep_original_audio=False)`

- [ ] **Step 1: Add `get_keyframe_times`**

Immediately before `process_audio_for_clip`, add:

```python
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
    times = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip().rstrip(",")
        if not line:
            continue
        try:
            times.append(float(line))
        except ValueError:
            continue
    return sorted(times)
```

- [ ] **Step 2: Rewrite the cut section of `process_audio_for_clip`**

Replace the body of `process_audio_for_clip` from its signature down to the end of the audio-cut block (the original lines 804-878, i.e. signature through the `else: # Concatenate parts with sox` block) with the version below. The downstream applause/impulse/normalize steps (original lines 880-945) stay; only their `return` at the end changes (Step 3).

```python
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
        return None, None, [], []

    effective_overlap = ov_end - kf_start
    if effective_overlap < min_overlap_sec:
        logger.log(f"    {RED}SKIP: overlap after keyframe snap "
                   f"({effective_overlap:.1f}s) below min ({min_overlap_sec:.1f}s){RESET}")
        logger.close()
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
        return None, None, [], []
    elif len(cut_parts) == 1:
        os.rename(cut_parts[0], cut_path)
    else:
        cmd = ["sox"] + cut_parts + [cut_path]
        subprocess.run(cmd, check=True)
        for p in cut_parts:
            os.remove(p)
```

- [ ] **Step 3: Update `process_audio_for_clip`'s return**

The remaining steps (read audio, applause, impulses, normalize, write norm) are unchanged. Change the final `return` of the function (original line 945) from:
```python
    logger.close()
    return norm_path, applause_blocks, impulses
```
to:
```python
    logger.close()
    return trimmed_video, norm_path, applause_blocks, impulses
```

- [ ] **Step 4: Update `mux_clip` to mux the trimmed video**

Replace `mux_clip` (lines 948-971) with:

```python
def mux_clip(alignment, trimmed_video_path, norm_path, output_dir, keep_original_audio=False):
    """Phase 5: mux the normalized audio onto the trimmed (copied) video."""
    if alignment["skipped"] or trimmed_video_path is None or norm_path is None:
        return None

    clip_num = alignment["clip_number"]
    basename = os.path.basename(alignment["video"]["path"])
    stem, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, stem + candidate_suffix(alignment) + ext)

    map_args = ["-map", "0:v", "-map", "1:a"]
    if keep_original_audio:
        map_args += ["-map", "0:a"]
    cmd = [
        "ffmpeg", "-y",
        "-i", trimmed_video_path,
        "-i", norm_path,
        "-c:v", "copy",
    ] + map_args + [output_path]
    print(f"    Clip {clip_num}: {basename} -> {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)

    return output_path
```

(Note: `keep_original_audio` now maps the *trimmed* video's audio — which we stripped with `-an`. Since the trimmed file has no audio track, `-map 0:a` would fail. Drop the original-audio remap by treating the trimmed video as video-only: if `keep_original_audio` is set, instead re-trim with audio. To keep this task focused, leave `keep_original_audio` mapping `0:a` against the *source* clip is out of scope — see Step 5.)

- [ ] **Step 5: Fix `keep_original_audio` to use the source clip's audio**

Because the trimmed video is video-only (`-an`), keep-original-audio must pull audio from the source clip at the same window. Update the `mux_clip` map logic to add the source clip as a third input when needed:

```python
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
    cmd = ["ffmpeg", "-y"] + inputs + ["-c:v", "copy"] + map_args + [output_path]
    print(f"    Clip {clip_num}: {basename} -> {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)

    return output_path
```

(YAGNI: `--keep-original-audio` was a debugging aid for the old full-length mux. With the trimmed-and-stripped video it no longer has a clean meaning; this task drops its effect from `mux_clip`. Task 5 removes the now-dead `keep_original_audio` plumbing if the reviewer agrees, or leaves the flag as a no-op — flagged for the reviewer.)

- [ ] **Step 6: Update the Phase 4/5 loop in `main`**

In `main`, locate the Phase 4 loop (anchor: `norm_path, applause_blocks, impulses = process_audio_for_clip(`) and the Phase 5 loop (anchor: `mux_clip(alignment, norm_path, output_dir,`). Replace them with the trimmed-video-threaded versions:

Phase 4 loop:
```python
        clip_results = []
        for alignment in alignments:
            trimmed_video, norm_path, applause_blocks, impulses = process_audio_for_clip(
                alignment, temp_dir, output_dir, min_overlap_sec=args.min_overlap)
            clip_results.append((alignment, trimmed_video, norm_path))
```

Phase 5 loop:
```python
        for alignment, trimmed_video, norm_path in clip_results:
            mux_clip(alignment, trimmed_video, norm_path, output_dir,
                     keep_original_audio=args.keep_original_audio)
```

(`args.min_overlap` is added in Task 5; until then this references an attribute that does not exist. To keep Task 4's suite green, temporarily pass `min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC` here and switch to `args.min_overlap` in Task 5.)

So for THIS task, the Phase 4 loop line reads:
```python
            trimmed_video, norm_path, applause_blocks, impulses = process_audio_for_clip(
                alignment, temp_dir, output_dir, min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC)
```

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all PASS (no unit test exercises the ffmpeg path; the pure helpers it relies on are covered by Task 1).

- [ ] **Step 8: Commit**

```bash
git add src/soundgraft/cli.py
git commit -m "feat: keyframe-snapped video trim and matched audio cut"
```

**Testing note:** the ffmpeg trim/cut/mux is validated manually against real media (next task wires the CLI; final manual validation uses the actual partial-overlap dataset). The pure decision logic (`first_keyframe_at_or_after`, `compute_overlap`, min-overlap floor) is unit-tested in Tasks 1-3.

---

## Task 5: `--min-overlap` CLI flag, threading, and docs

**Files:**
- Modify: `src/soundgraft/cli.py` (`parse_args`; the `align_all_clips` call in `main`; the Phase 4 loop `min_overlap_sec` argument)
- Modify: `README.md`, `AGENTS.md`
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `DEFAULT_MIN_OVERLAP_SEC`, `align_all_clips(min_overlap_sec=...)`, `process_audio_for_clip(min_overlap_sec=...)`.
- Produces: `--min-overlap SEC` (float, default `DEFAULT_MIN_OVERLAP_SEC`), threaded into alignment and Phase 4.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
from soundgraft.cli import parse_args, DEFAULT_MIN_OVERLAP_SEC


def test_min_overlap_default():
    args = parse_args(["--input", "in", "--output", "out"])
    assert args.min_overlap == DEFAULT_MIN_OVERLAP_SEC


def test_min_overlap_parsed():
    args = parse_args(["--input", "in", "--output", "out", "--min-overlap", "25"])
    assert args.min_overlap == 25.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v -k min_overlap`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'min_overlap'`.

- [ ] **Step 3: Add the argument**

In `parse_args`, after the `--shotgun` argument, add:

```python
    parser.add_argument(
        "--min-overlap",
        type=float,
        default=DEFAULT_MIN_OVERLAP_SEC,
        metavar="SEC",
        help=f"Minimum seconds of audio/video overlap to emit a clip "
             f"(default: {DEFAULT_MIN_OVERLAP_SEC}). Below this, the clip is skipped.",
    )
```

- [ ] **Step 4: Thread `--min-overlap` into `align_all_clips`**

In `main`, the `align_all_clips(...)` call (anchor: `alignments = align_all_clips(`) — add `min_overlap_sec=args.min_overlap,` to its keyword arguments.

- [ ] **Step 5: Thread `--min-overlap` into Phase 4**

In `main`, change the Phase 4 loop's `process_audio_for_clip(...)` call from `min_overlap_sec=DEFAULT_MIN_OVERLAP_SEC` (Task 4 placeholder) to `min_overlap_sec=args.min_overlap`.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_cli.py -v -k min_overlap`
Expected: PASS (2 tests).

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 8: Update docs**

In `README.md`, add to the Options table (after `--shotgun N`):
```markdown
| `--min-overlap SEC` | Minimum seconds of audio/video overlap required to emit a clip (default: 10). Clips with less overlap are skipped. |
```

In `README.md`, update the "How it works" alignment bullet (point 3) to note partial overlap. Replace the existing alignment sentence with:
```markdown
3. **Alignment** — Cross-correlates [Chromaprint](https://acoustid.org/chromaprint) fingerprints to find where each video clip overlaps the audio timeline, at any relative offset (the audio need not contain the whole clip). Metadata timestamps provide a search hint when available. The output video is trimmed to the audio-covered overlap, snapped to a keyframe so the video stream is copied without re-encoding.
```

In `AGENTS.md`, under "## Future Extensions & Known Issues", update the offset-correction note to reflect that the correction is now applied once, in `compute_overlap`:
```markdown
### Offset-Correction Value (pending data)
- `compute_overlap` applies `ALIGNMENT_OFFSET_CORRECTION` (0.5s) exactly once when converting a correlation lag to video time. The historical hinted-vs-full-scan double-apply asymmetry is gone. Whether 0.5s is the correct empirical value is still open; resolve by running a known-good clip and observing which offset syncs, then adjust the single constant.
```

- [ ] **Step 9: Commit**

```bash
git add src/soundgraft/cli.py tests/test_cli.py README.md AGENTS.md
git commit -m "feat: add --min-overlap flag and document partial-overlap alignment"
```

---

## Self-Review Notes

- **Spec coverage:** generalized signed-lag correlation with normalized score + min-overlap floor (Task 2); `compute_overlap` geometry SSOT (Task 1, used in Tasks 2-4); keyframe-snapped `-c:v copy` trim + matched audio cut + measure-actual-duration (Task 4); `--min-overlap` with hard floor (Tasks 1+5); distinct skip reasons low-confidence / below-min-overlap / no-keyframe (Tasks 3-4); correction applied once in `compute_overlap` (Task 1); hint window as signed-lag bounds (Task 2); shotgun benefits from the larger lag space (Task 2); multi-event unchanged. Testing section: pure units for overlap/keyframe/floor/skip + signed-lag correlation; real-media manual for ffmpeg paths.
- **Transient legacy field:** `offset` (= `audio_start_in_video`) is retained through Tasks 2-3 so Phase 4 stays green, and Phase 4 stops reading it in Task 4 (it now uses `ov_start`/`ov_end`/`audio_cut_start`). The field remains harmlessly in the dict; no consumer depends on it after Task 4.
- **Open reviewer decision:** Task 4 Step 5 changes `--keep-original-audio` semantics (trimmed video is audio-stripped). Flagged for the task reviewer / final review: keep the flag as a documented no-op, or remove its plumbing. Not silently dropped.
- **Manual validation (out of automated scope):** the ffmpeg keyframe extraction, trim, cut, and mux, plus end-to-end validation on the real partial-overlap dataset and the offset-correction (0.5 vs 1.0) measurement.

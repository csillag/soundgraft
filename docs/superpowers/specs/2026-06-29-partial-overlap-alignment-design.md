# SoundGraft — Partial-Overlap Alignment

**Date:** 2026-06-29
**Status:** Design approved, pending implementation

## Problem

The alignment algorithm assumed the audio recording always fully encompasses
every video clip — a video clip could only slide *within* the audio timeline.
That assumption is false. Real data exists where the video and audio merely
*overlap*: the video may start before the audio and stop after it, or the two
may partially overlap with neither containing the other.

Observed symptom: with a video clip whose audio recording is barely longer
than the clip, `correlate_fingerprints` had a search range of only 7 items
(~0.87 s). Shotgun mode collapsed to a single candidate (NMS window 16 ≫ range
7), and the chosen offset sat at the boundary of the searchable range — a sign
the true alignment lies *outside* what the bounded slide can represent. A
multi-second misalignment cannot exist within a sub-second search window, so
the failure is structural, not a peak-selection problem.

## Goal

Replace the bounded slide with a general cross-correlation that aligns video
and audio at **any lag** and handles all four overlap configurations with one
mechanism:

- (A) audio ⊇ video — the old assumption
- (B) video ⊇ audio — video started first, stopped last
- (C) audio leads, partial overlap
- (D) video leads, partial overlap

Output only the region for which good audio exists: **cut the video** to the
overlap, swap in the matched audio, never re-encode the video.

## Non-Goals

- Re-encoding the video for frame-accurate trimming (explicitly rejected — the
  output feeds further processing; quality must be preserved).
- Hybrid tracks / silence padding for uncovered video regions (rejected — we
  cut to the covered region instead).
- Resolving the empirical offset-correction value (0.5 s vs 1.0 s); this
  rewrite collapses the correction to one site, leaving the value as the
  existing constant and the empirical question still open (now trivially
  tunable).

## Design

### 1. Generalized correlation primitive

`correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items,
search_start=None, search_end=None)` is extended from a bounded forward slide
to a full cross-correlation over signed lags.

- Lag `L` = the index in `fp_ref` aligned with `fp_clip[0]`.
  Range: `L ∈ [-(clip_len - 1), ref_len - 1]` (today: `[0, ref_len - clip_len]`).
- Overlap indices for a lag: `i ∈ [max(0, -L), min(clip_len, ref_len - L))`;
  `overlap_count` is that width.
- **Normalized score:** `1 - bit_errors / (32 * overlap_count)` — divided by
  the *actual* overlap, so scores are comparable across lags with different
  overlap widths.
- **Min-overlap floor:** lags with `overlap_count < floor` are excluded from
  the search entirely (never scored, never returned).
  `floor = max(MIN_OVERLAP_ITEMS_HARD, ceil(min_overlap_sec / FPCALC_ITEM_DURATION))`.
- `MIN_OVERLAP_ITEMS_HARD ≈ 40` items (~5 s) — a correlation-sanity constant
  that prevents a tiny overlap from scoring a spuriously perfect match.
- Returns `(lag, score)` peaks as before, but `lag` is now signed.
- `correlate_fingerprints` remains a thin `n=1` wrapper (SSOT preserved).
- The `search_start`/`search_end` window (used by the metadata hint) is
  reinterpreted as an inclusive lag window; passing `None` searches the full
  signed range.

This removes the `7/7` collapse: the lag space is now `ref_len + clip_len - 1`
positions, giving shotgun's top-N + NMS room to return distinct candidates.

**Cost:** ~`ref_len + clip_len` lags vs `ref_len - clip_len` today; per-lag
work scales with `overlap_count`. Same order when audio ≫ video; ~2× when the
two are similar in length. Acceptable.

### 2. Lag → geometry (pure function)

`compute_overlap(lag_items, video_dur, audio_dur)` is a new pure function — the
single source of truth for the geometry. With `DUR = FPCALC_ITEM_DURATION` and
the offset correction applied once:

```
audio_start_in_video = -lag_items * DUR + ALIGNMENT_OFFSET_CORRECTION
ov_start = max(0, audio_start_in_video)
ov_end   = min(video_dur, audio_start_in_video + audio_dur)
ov_dur   = ov_end - ov_start
audio_cut_start = ov_start - audio_start_in_video
```

Returns `{audio_start_in_video, ov_start, ov_end, ov_dur, audio_cut_start}`.

Interpretation of `audio_start_in_video`: `> 0` means the audio recording
begins partway into the video (video leads — cases B/D); `< 0` means the audio
began before the video (cases A/C). The formula is uniform across all four
configurations.

If `ov_dur <= 0` the clip does not overlap the audio at all → skip.

### 3. Keyframe-snapped trim (no re-encode)

To keep `-c:v copy` (no quality loss) while preserving exact A/V sync, the
output starts at the first keyframe at or after `ov_start`:

1. `first_keyframe_at_or_after(keyframe_times, ov_start)` (pure) selects the
   first keyframe PTS `kf_start ≥ ov_start`; returns `None` if none lies within
   the overlap.
2. Keyframe timestamps come from `ffprobe -select_streams v -skip_frame nokey
   -show_frames -show_entries frame=pkt_pts_time` (wrapped in its own function
   so the pure selector stays testable).
3. Trim video copied: `ffmpeg -ss kf_start -to ov_end -i clip -c:v copy -an
   <tmp_video>`. Input-seek lands cleanly on the keyframe.
4. **Measure** the trimmed file's actual duration `D` with ffprobe — with
   `-c copy` the end lands near a packet boundary, so `D` is measured, not
   assumed.
5. Cut audio to match exactly: start `audio_cut_start + (kf_start - ov_start)`
   in video-time terms, i.e. cut the audio span beginning at the audio-time
   corresponding to `kf_start`, length `D`. Reuses the existing
   segment-walking cut.
6. Process the cut audio (applause / impulse / normalize) as today.
7. Mux: trimmed video (`-c:v copy`) + processed audio. Output runs `D`,
   anchored at `kf_start` on both streams — no drift; only the ≤ 1-GOP
   lead-in (`kf_start - ov_start`) is discarded.

**Guards:** the effective overlap after the snap is `ov_end - kf_start`;
re-apply the min-overlap floor here. If a large GOP eats the overlap below the
floor, or no keyframe lies at/after `ov_start` within the overlap, skip the
clip with the appropriate reason.

This replaces the cut path in `process_audio_for_clip`, which assumed
`offset >= 0` and that the whole `video_dur` fits inside the audio. The old
`if offset < 0: clamp to 0` warning-hack is removed — the overlap math
supersedes it.

### 4. CLI, confidence, skip reasons

- **`--min-overlap SEC`** (float, default 10.0). Effective floor in items =
  `max(MIN_OVERLAP_ITEMS_HARD, ceil(SEC / FPCALC_ITEM_DURATION))`. Applied in
  the correlation search and re-checked after the keyframe snap.
- **Confidence:** the normalized score keeps the `[0, 1]` scale; existing
  `CONFIDENCE_THRESHOLD` and `--it-is-what-it-is` semantics are unchanged.
- **Skip reasons** are distinct and logged: `low-confidence`,
  `below-min-overlap`, `no-keyframe-in-overlap`. The alignment result dict
  carries `ov_start` / `ov_end` / `overlap_dur` and a `skip_reason`.
- **Offset-correction:** the rewrite collapses the seconds conversion to one
  site (`compute_overlap`), so the historical double-vs-single asymmetry
  between the hinted and full-scan paths disappears by construction. Correction
  is applied once, value = the existing `ALIGNMENT_OFFSET_CORRECTION = 0.5`.
  The empirical "0.5 vs 1.0" question remains open but is now tunable in one
  place; the AGENTS.md note is updated accordingly.

### 5. Interaction with existing features

- **Hint window:** the metadata timestamp still yields an expected lag; the
  narrow search around it is expressed as an inclusive signed-lag window and
  works unchanged.
- **Shotgun:** logic unchanged, but now genuinely useful — the large lag space
  yields real distinct top-N candidates instead of collapsing to one.
- **Multi-event:** unchanged — each event is correlated and the best is picked.

## Testing

Pure units (no ffmpeg):

- **Generalized `correlate_fingerprints_topn`:** synthetic fingerprints with a
  planted partial overlap at a *negative* lag (video leads). Assert the best
  lag and normalized score; assert lags below the min-overlap floor are
  excluded; assert the legacy `audio ⊇ video` case still resolves (regression).
- **`compute_overlap`:** table test over all four configs (A/B/C/D) with
  concrete numbers and signed lags, asserting every returned field. Geometry
  SSOT.
- **`first_keyframe_at_or_after`:** picks the first keyframe ≥ `ov_start`;
  returns `None` when none lies in range.
- **Min-overlap floor:** below-floor lag excluded during correlation; post-snap
  effective-overlap re-check triggers a `below-min-overlap` skip.
- **Skip-reason classification:** `low-confidence` vs `below-min-overlap` vs
  `no-keyframe-in-overlap`.

Manual (real media, not unit-tested): the keyframe ffprobe extraction, the
`-c:v copy` trim + audio cut + mux end-to-end, and validation against the
actual partial-overlap dataset.

## Out-of-scope follow-ups

- Resolve the offset-correction value (0.5 vs 1.0) empirically, now a
  one-line change.

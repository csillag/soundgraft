# SoundGraft — Shotgun Mode: Multiple Alignment Candidates

**Date:** 2026-06-29
**Status:** Design approved, pending implementation

## Problem

Chromaprint alignment can produce a confident-but-wrong match. Observed
failure: a single-clip / single-audio-file job where the correct event was
chosen but the whole clip was offset by ~4 seconds (≈32 fingerprint items).

Root cause class: `correlate_fingerprints` returns only the single
**argmax** offset. When a spurious correlation peak scores marginally higher
than the true sync peak (common with musical/repetitive content or
cross-device mic differences), the wrong peak silently wins. The true sync
point is typically the runner-up peak, which the current code discards.

There is no way for the user to recover the correct alignment short of
guessing `--offset-adjust` values.

## Goals

1. Improve robustness by exposing the top-N correlation peaks instead of only
   the global maximum.
2. Add a `--shotgun N` mode that produces N candidate output videos (each a
   different alignment), so the user can eyeball which one syncs and keep it.

Default behavior is **unchanged**: without `--shotgun`, the tool auto-picks
the single best match exactly as today.

## Non-Goals

- Interactive picker / preview UI.
- Automatic peak disambiguation heuristics (margin scoring, peak sharpness).
  Out of scope; shotgun delegates the choice to the human.
- Cross-event candidate pooling. Shotgun operates on the best event only.

## Design

### 1. Core: top-N peak extraction

New function:

```
correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items,
                            search_start=0, search_end=None)
    -> list[(offset_items, score)]   # sorted by score descending, len <= n
```

- Performs the same single sliding-correlation pass as today, but retains the
  full per-position score array rather than only the running maximum.
- Extracts the top-N peaks via **non-maximum suppression (NMS)**: take the
  highest-scoring position, suppress all positions within
  `±nms_window_items` of it, take the next-highest remaining, repeat up to N
  times (or until no positions remain).
- NMS prevents returning N adjacent positions that all belong to the same
  peak's shoulder.

NMS window constant: `NMS_WINDOW_ITEMS`, default ≈ 16 items (~2.0 s). Small
enough to retain a true peak ~4 s away from a spurious one; large enough to
collapse a single peak's shoulder into one entry. Documented as a tunable
module constant.

The existing `correlate_fingerprints` is refactored to call
`correlate_fingerprints_topn(..., n=1, ...)` and return its first element as
`(offset_items, score)` — no duplicated correlation logic. This preserves the
current single-match code path byte-for-byte in behavior.

Cost: the correlation pass is unchanged O(positions × clip_len). N-peak
extraction is cheap post-processing over the score array. No extra `fpcalc`
or correlation work.

### 2. Shotgun mode

CLI flag: `--shotgun N` (integer ≥ 2; N = number of candidate outputs).

Behavior when active, per clip:

1. Choose the best event using the existing per-event selection logic
   (best-event-only).
2. Run a **full scan** of that event (ignore the metadata-timestamp hint).
   Rationale: the auto-pick — which uses the hinted window — is exactly what
   produced the bad result, so shotgun deliberately surveys the full peak
   landscape.
3. Call `correlate_fingerprints_topn` with `n = N` to get the top-N peaks.
4. For **each** candidate offset, run the full existing audio pipeline
   (cut → applause attenuation → impulse attenuation → peak-normalize) and
   mux a separate output video.
5. Bypass the confidence-threshold skip: shotgun always emits N outputs
   regardless of score, because the user is overriding the auto-decision.

Interaction with other flags:
- Combines with `--clip N` / `--from-clip N` (shotgun applies to each
  selected clip).
- `--shotgun` implies "emit regardless of confidence" for the selected
  clips; `--it-is-what-it-is` is not required.

### 3. Output naming and logging

Per candidate `k` (1-indexed, by descending score):

- Output video: `<clipname>_cand<k>_<offset>s.<ext>`
  e.g. `clip01_cand1_12.3s.mp4`, `clip01_cand2_16.4s.mp4`.
- Log file: matching `<clipname>_cand<k>_<offset>s.log`, recording:
  - candidate rank `k`,
  - confidence (bit-match ratio),
  - applied offset in seconds (with correction),
  - **raw `offset_items` (pre-correction)** and the correction value applied.

The raw-offset record exists to support the offset-correction experiment
(below): from whichever candidate actually syncs, the true correction can be
computed directly.

### 4. Deferred: offset-correction inconsistency

Pre-existing asymmetry found while reading the code:

- Hinted path (`align_clip_to_event`): `offset_secs = items*DUR + CORRECTION + CORRECTION` (0.5 applied twice = 1.0 s).
- Full-scan path: `offset_secs = items*DUR + CORRECTION` (0.5 once).

Git history (`09efc7a`) shows both lines were introduced in the **same
commit**, already asymmetric — so history alone cannot prove which value was
empirically tuned. The asymmetry suggests deliberate hand-tuning on one path
(accidental duplication would have made both identical).

**Decision: leave the offset-correction code unchanged in this work.** The
correct value will be determined empirically by the user running normal
(hinted) and `--no-hint` (full-scan) modes against a known-good clip after
shotgun lands, then observing which offset syncs. Once known, a follow-up
unifies both paths to the validated value. The shotgun raw-offset logging
(Section 3) directly supports this measurement.

## Testing

- Unit test `correlate_fingerprints_topn`: synthetic fingerprint arrays with
  planted peaks at known positions and known separations.
  - Top-N returned in descending score order.
  - NMS does not return two positions within one peak (separation < window).
  - Two well-separated peaks (e.g. 4 s apart) both survive NMS.
- Unit test the `n=1` refactor: `correlate_fingerprints` returns the same
  `(offset, score)` as the prior implementation for a given input — guards the
  unchanged default path.
- Existing test suite (`test_identify`, `test_impulse`, `test_reconstitute`,
  `test_cli`) stays green.
- Shotgun output generation and muxing are validated manually against real
  media (requires ffmpeg/sox and actual recordings); not unit-tested.

## Out-of-scope follow-ups (not part of this work)

- Unify the offset-correction paths once test data identifies the right value.
- Optional: in default mode, log the runner-up peak and the margin to the
  winner, so the user knows when a result is ambiguous enough to warrant
  `--shotgun`.

# Shotgun Mode: Multiple Alignment Candidates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--shotgun N` flag that emits N candidate output videos (each a different alignment offset) so the user can pick the one that syncs, while leaving default single-match behavior byte-for-byte unchanged.

**Architecture:** Extract the top-N correlation peaks (with non-maximum suppression) instead of only the global argmax. In shotgun mode, build N candidate `alignment` dicts per clip and run each through the existing audio-processing and muxing pipeline, distinguished by a filename suffix. Default mode produces an empty suffix, preserving current paths and names exactly.

**Tech Stack:** Python 3, numpy, tqdm, chromaprint (`fpcalc`), ffmpeg/sox. Tests use pytest.

## Global Constraints

- No CLAUDE.md file; all project rules live in AGENTS.md.
- No LLM co-author trailers in git commit messages.
- Default (no `--shotgun`) behavior must remain identical: same output filenames, same temp filenames, same log filenames, same offsets.
- Single source of truth: `correlate_fingerprints` must not duplicate correlation logic — it wraps the new top-N function.
- Do **not** change the offset-correction code (the `ALIGNMENT_OFFSET_CORRECTION` double-vs-single asymmetry). That fix is deferred pending empirical test data.

---

## File Structure

- `src/soundgraft/cli.py` — all production changes (single-module project):
  - New constant `NMS_WINDOW_ITEMS`.
  - New `correlate_fingerprints_topn(...)`; refactor `correlate_fingerprints(...)` to wrap it.
  - New `candidate_suffix(alignment)` helper; thread suffix through `process_audio_for_clip` and `mux_clip`.
  - New `shotgun_align_clip(...)`; add `shotgun` param to `align_all_clips`.
  - Add `--shotgun` to `parse_args`; pass through in `main`.
- `tests/test_correlate.py` — **new** unit tests for top-N correlation and the refactor.
- `tests/test_shotgun.py` — **new** unit tests for `candidate_suffix` and `shotgun_align_clip`.
- `tests/test_cli.py` — extend with `--shotgun` argument-parsing/validation tests.

---

## Task 1: Top-N correlation with non-maximum suppression

**Files:**
- Modify: `src/soundgraft/cli.py` (add constant near line 212; add `correlate_fingerprints_topn`; refactor `correlate_fingerprints` at lines 237-274)
- Test: `tests/test_correlate.py` (create)

**Interfaces:**
- Consumes: `popcnt`, `np`, `tqdm` (already in module).
- Produces:
  - `NMS_WINDOW_ITEMS = 16` (module constant, ~2.0 s).
  - `correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items, search_start=0, search_end=None) -> list[(offset_items:int, score:float)]` — descending score order, length ≤ n.
  - `correlate_fingerprints(fp_ref, fp_clip, search_start=0, search_end=None) -> (offset_items:int, score:float)` — unchanged signature/return, now a wrapper.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_correlate.py`:

```python
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


def test_nms_suppresses_adjacent_position_in_same_peak():
    # Two exact matches one item apart; NMS window 3 must collapse them to one.
    ref = _ref_with_peaks([5, 6], length=30)
    peaks = correlate_fingerprints_topn(ref, CLIP, n=2, nms_window_items=3)
    assert len(peaks) == 1
    assert peaks[0][0] == 5


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_correlate.py -v`
Expected: FAIL with `ImportError: cannot import name 'correlate_fingerprints_topn'`.

- [ ] **Step 3: Add the constant**

In `src/soundgraft/cli.py`, just after line 213 (`ALIGNMENT_OFFSET_CORRECTION = 0.5 ...`), add:

```python
# Non-maximum suppression window (in fingerprint items) for top-N peak
# extraction. ~16 items ≈ 2.0 s: small enough to keep a true peak a few
# seconds from a spurious one, large enough to collapse one peak's shoulder.
NMS_WINDOW_ITEMS = 16
```

- [ ] **Step 4: Add `correlate_fingerprints_topn` and refactor `correlate_fingerprints`**

Replace the entire existing `correlate_fingerprints` function (lines 237-274) with:

```python
def correlate_fingerprints_topn(fp_ref, fp_clip, n, nms_window_items,
                                search_start=0, search_end=None):
    """Slide fp_clip over fp_ref and return the top-N correlation peaks.

    Returns a list of (offset_items, score) tuples sorted by score descending,
    length <= n. Peaks are separated by non-maximum suppression: after each
    pick, all positions within +/- nms_window_items are suppressed so the
    next pick comes from a different peak rather than the same peak's shoulder.
    """
    clip_len = len(fp_clip)
    ref_len = len(fp_ref)

    if clip_len == 0 or ref_len == 0:
        return []

    if search_end is None:
        search_end = ref_len - clip_len

    search_end = min(search_end, ref_len - clip_len)
    search_start = max(0, search_start)

    if search_start >= search_end:
        return []

    total_bits = 32 * clip_len
    # Score array indexed by absolute offset; unscanned positions stay -1.0.
    scores = np.full(search_end, -1.0)

    for offset in tqdm(range(search_start, search_end),
                       desc="    Aligning", unit="pos",
                       bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
        bit_errors = 0
        for i in range(clip_len):
            bit_errors += popcnt(fp_ref[offset + i] ^ fp_clip[i])
        scores[offset] = 1.0 - (bit_errors / total_bits)

    results = []
    work = scores.copy()
    for _ in range(n):
        idx = int(np.argmax(work))
        if work[idx] < 0:
            break
        results.append((idx, float(scores[idx])))
        lo = max(0, idx - nms_window_items)
        hi = min(len(work), idx + nms_window_items + 1)
        work[lo:hi] = -1.0

    return results


def correlate_fingerprints(fp_ref, fp_clip, search_start=0, search_end=None):
    """Slide fp_clip over fp_ref and find the offset with highest correlation.

    Returns (best_offset_items, best_score) where best_offset_items is the
    position in fp_ref where fp_clip best matches, and best_score is 0.0-1.0.
    """
    peaks = correlate_fingerprints_topn(
        fp_ref, fp_clip, 1, NMS_WINDOW_ITEMS, search_start, search_end)
    if not peaks:
        return 0, 0.0
    return peaks[0]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_correlate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Verify no regression in the existing suite**

Run: `pytest -q`
Expected: all existing tests still PASS.

- [ ] **Step 7: Commit**

```bash
git add src/soundgraft/cli.py tests/test_correlate.py
git commit -m "feat: extract top-N correlation peaks with non-maximum suppression"
```

---

## Task 2: Candidate filename suffix

**Files:**
- Modify: `src/soundgraft/cli.py` (add `candidate_suffix` helper before `process_audio_for_clip` at line 699; thread suffix through `process_audio_for_clip` lines 699-834 and `mux_clip` lines 837-859)
- Test: `tests/test_shotgun.py` (create)

**Interfaces:**
- Consumes: an `alignment` dict that may contain an optional `"candidate"` key of shape `{"rank": int, "raw_offset_items": int, "correction": float}` (produced in Task 3). Absent/None for normal matches.
- Produces: `candidate_suffix(alignment) -> str` — `""` when no candidate, else `f"_cand{rank}_{offset:.1f}s"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_shotgun.py`:

```python
from soundgraft.cli import candidate_suffix


def test_candidate_suffix_empty_for_normal_alignment():
    alignment = {"offset": 12.3, "candidate": None}
    assert candidate_suffix(alignment) == ""


def test_candidate_suffix_absent_key_is_empty():
    assert candidate_suffix({"offset": 12.3}) == ""


def test_candidate_suffix_for_candidate():
    alignment = {"offset": 16.4, "candidate": {"rank": 2, "raw_offset_items": 130, "correction": 0.5}}
    assert candidate_suffix(alignment) == "_cand2_16.4s"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shotgun.py -v`
Expected: FAIL with `ImportError: cannot import name 'candidate_suffix'`.

- [ ] **Step 3: Add the helper**

In `src/soundgraft/cli.py`, immediately before `def process_audio_for_clip` (line 699), add:

```python
def candidate_suffix(alignment):
    """Filename suffix distinguishing shotgun candidates.

    Empty string for a normal (non-candidate) alignment, so default-mode
    output, temp, and log filenames are unchanged.
    """
    cand = alignment.get("candidate")
    if not cand:
        return ""
    return f"_cand{cand['rank']}_{alignment['offset']:.1f}s"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_shotgun.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Thread the suffix through `process_audio_for_clip`**

In `process_audio_for_clip`, after the line `basename = os.path.basename(alignment["video"]["path"])` (line 710), add:

```python
    suffix = candidate_suffix(alignment)
```

Change the log name line (711) from:

```python
    log_name = os.path.splitext(basename)[0] + ".log"
```

to:

```python
    log_name = os.path.splitext(basename)[0] + suffix + ".log"
```

Change the cut temp path (line 725) from:

```python
    cut_path = os.path.join(temp_dir, f"clip_{clip_num}_cut.wav")
```

to:

```python
    cut_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_cut.wav")
```

Change the part temp path (line 744) from:

```python
        part_path = os.path.join(temp_dir, f"clip_{clip_num}_part_{part_idx}.wav")
```

to:

```python
        part_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_part_{part_idx}.wav")
```

Change the normalized temp path (line 830) from:

```python
    norm_path = os.path.join(temp_dir, f"clip_{clip_num}_normalized.wav")
```

to:

```python
    norm_path = os.path.join(temp_dir, f"clip_{clip_num}{suffix}_normalized.wav")
```

Then, immediately after the existing `logger.log(f"    Match found at offset ...")` line (716), add candidate provenance logging:

```python
    if alignment.get("candidate"):
        cand = alignment["candidate"]
        logger.log(f"    Shotgun candidate #{cand['rank']} — "
                   f"raw offset {cand['raw_offset_items']} items, "
                   f"correction applied {cand['correction']:.2f}s")
```

- [ ] **Step 6: Thread the suffix through `mux_clip`**

In `mux_clip`, change the output path line (845) from:

```python
    output_path = os.path.join(output_dir, basename)
```

to:

```python
    stem, ext = os.path.splitext(basename)
    output_path = os.path.join(output_dir, stem + candidate_suffix(alignment) + ext)
```

- [ ] **Step 7: Run the full suite to verify default behavior is preserved**

Run: `pytest -q`
Expected: all tests PASS (default-mode names unchanged because suffix is `""`).

- [ ] **Step 8: Commit**

```bash
git add src/soundgraft/cli.py tests/test_shotgun.py
git commit -m "feat: candidate filename suffix for shotgun outputs"
```

---

## Task 3: Shotgun alignment builder

**Files:**
- Modify: `src/soundgraft/cli.py` (add `shotgun_align_clip` after `align_clip_to_event` at line 354; add `shotgun` param + branch to `align_all_clips` lines 357-406)
- Test: `tests/test_shotgun.py` (extend)

**Interfaces:**
- Consumes: `get_fingerprint`, `align_clip_to_event`, `correlate_fingerprints_topn`, `FPCALC_ITEM_DURATION`, `ALIGNMENT_OFFSET_CORRECTION`, `NMS_WINDOW_ITEMS`. Event dicts have `segments`, `start_time`, `total_duration`. Video dicts have `path`, `duration`, `creation_time`.
- Produces:
  - `shotgun_align_clip(video, events, event_fingerprints, clip_num, n, no_hint=False) -> list[alignment]` — up to `n` candidate dicts, each `{"video", "clip_number", "event", "offset", "confidence", "skipped": False, "candidate": {"rank", "raw_offset_items", "correction"}}`, ranked by descending score.
  - `align_all_clips(..., shotgun=None)` — when `shotgun` is an int ≥ 2, each selected clip contributes its candidate list (flattened) instead of a single alignment.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_shotgun.py`:

```python
from unittest.mock import patch
from soundgraft.cli import shotgun_align_clip

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
            video, [event], [fp_ref], clip_num=1, n=2, no_hint=True)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_shotgun.py -v`
Expected: FAIL with `ImportError: cannot import name 'shotgun_align_clip'`.

- [ ] **Step 3: Add `shotgun_align_clip`**

In `src/soundgraft/cli.py`, immediately after `align_clip_to_event` (after line 354, before `align_all_clips`), add:

```python
def shotgun_align_clip(video, events, event_fingerprints, clip_num, n, no_hint=False):
    """Build up to n candidate alignments for one clip (shotgun mode).

    Chooses the single best event using the normal per-event selection, then
    runs a FULL scan of that event and returns its top-n correlation peaks as
    candidate alignment dicts (ranked by descending score). The metadata hint
    is intentionally ignored for the peak survey — the hinted auto-pick is what
    produced the bad match in the first place.
    """
    fp_clip = get_fingerprint(video["path"])
    if not fp_clip:
        return []

    # Pick the best event (respecting the hint just for event selection).
    best_event_idx = None
    best_score = -1.0
    for j, event in enumerate(events):
        fp_ref = event_fingerprints[j]
        if not fp_ref:
            continue
        _, score = align_clip_to_event(video["path"], event, fp_ref, video, no_hint=no_hint)
        if score > best_score:
            best_score = score
            best_event_idx = j

    if best_event_idx is None:
        return []

    event = events[best_event_idx]
    fp_ref = event_fingerprints[best_event_idx]

    print(f"    Shotgun: full-scan top-{n} peaks of event {best_event_idx + 1}...")
    peaks = correlate_fingerprints_topn(fp_ref, fp_clip, n, NMS_WINDOW_ITEMS)

    candidates = []
    for rank, (offset_items, score) in enumerate(peaks, start=1):
        offset_secs = offset_items * FPCALC_ITEM_DURATION + ALIGNMENT_OFFSET_CORRECTION
        candidates.append({
            "video": video,
            "clip_number": clip_num,
            "event": event,
            "offset": offset_secs,
            "confidence": score,
            "skipped": False,
            "candidate": {
                "rank": rank,
                "raw_offset_items": offset_items,
                "correction": ALIGNMENT_OFFSET_CORRECTION,
            },
        })
    return candidates
```

- [ ] **Step 4: Add the `shotgun` branch to `align_all_clips`**

Change the `align_all_clips` signature (line 357) from:

```python
def align_all_clips(video_files, events, temp_dir, clip_filter=None, from_clip=None, it_is_what_it_is=False, no_hint=False):
```

to:

```python
def align_all_clips(video_files, events, temp_dir, clip_filter=None, from_clip=None, it_is_what_it_is=False, no_hint=False, shotgun=None):
```

Then, inside the per-clip loop, immediately after the `print(f"\n  Clip {clip_num}: ...")` line (373) and before `best_offset = None`, add the shotgun short-circuit:

```python
        if shotgun:
            candidates = shotgun_align_clip(
                video, events, event_fingerprints, clip_num, shotgun, no_hint=no_hint)
            if not candidates:
                print(f"  {YELLOW}WARNING: Clip {clip_num} produced no shotgun candidates{RESET}")
            else:
                print(f"    {len(candidates)} candidate(s): " +
                      ", ".join(f"#{c['candidate']['rank']} @ {c['offset']:.2f}s "
                                f"(conf {c['confidence']:.3f})" for c in candidates))
            alignments.extend(candidates)
            continue
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_shotgun.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/soundgraft/cli.py tests/test_shotgun.py
git commit -m "feat: shotgun candidate alignment builder"
```

---

## Task 4: Wire up the `--shotgun` CLI flag

**Files:**
- Modify: `src/soundgraft/cli.py` (`parse_args` lines 862-878; `main` — argument validation and the `align_all_clips` call at lines 932-938)
- Test: `tests/test_cli.py` (extend)

**Interfaces:**
- Consumes: `parse_args`, `align_all_clips(..., shotgun=...)` from Task 3.
- Produces: `--shotgun N` CLI option (`args.shotgun`, int, default `None`); rejected with a clear error and exit code 2 when `N < 2`.

- [ ] **Step 1: Write the failing tests**

Read the existing `tests/test_cli.py` to match its invocation style, then append:

```python
from soundgraft.cli import parse_args


def test_shotgun_arg_parsed():
    args = parse_args(["--input", "in", "--output", "out", "--shotgun", "3"])
    assert args.shotgun == 3


def test_shotgun_defaults_to_none():
    args = parse_args(["--input", "in", "--output", "out"])
    assert args.shotgun is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py -v -k shotgun`
Expected: FAIL with `AttributeError: 'Namespace' object has no attribute 'shotgun'`.

- [ ] **Step 3: Add the argument**

In `parse_args`, after the `--no-hint` argument (line 876), add:

```python
    parser.add_argument(
        "--shotgun",
        type=int,
        metavar="N",
        help="Emit N candidate outputs per clip (different alignment offsets) "
             "instead of auto-picking one. Use to recover from a bad match.",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v -k shotgun`
Expected: PASS (2 tests).

- [ ] **Step 5: Validate `N >= 2` and pass through in `main`**

In `main`, immediately after `output_dir = args.output` (line 884), add validation:

```python
    if args.shotgun is not None and args.shotgun < 2:
        print("Error: --shotgun N requires N >= 2.")
        sys.exit(2)
```

Then change the `align_all_clips(...)` call (lines 932-938) to pass the flag — add `shotgun=args.shotgun,` to the keyword arguments:

```python
        alignments = align_all_clips(
            video_files, reconstituted, temp_dir,
            clip_filter=args.clip,
            from_clip=args.from_clip,
            it_is_what_it_is=args.it_is_what_it_is,
            no_hint=args.no_hint,
            shotgun=args.shotgun,
        )
```

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: all PASS.

- [ ] **Step 7: Update documentation**

In `README.md`, add a row to the Options table (after `--keep-original-audio`):

```markdown
| `--shotgun N` | Emit N candidate outputs per clip (different alignment offsets) instead of auto-picking one; use to recover from a bad match. Output files are named `<clip>_cand<k>_<offset>s.<ext>`. |
```

In `AGENTS.md`, under "## Future Extensions & Known Issues", append a short note recording the deferred decision:

```markdown
### Offset-Correction Path Asymmetry (pending data)
- Hinted path applies `ALIGNMENT_OFFSET_CORRECTION` twice (1.0s); full-scan applies it once (0.5s). Both introduced together in commit 09efc7a, so history can't say which was empirically tuned. To be resolved by running normal vs `--no-hint` against a known-good clip and observing which syncs; shotgun `.log` files record the raw pre-correction offset to support this. Then unify both paths to the validated value.
```

- [ ] **Step 8: Commit**

```bash
git add src/soundgraft/cli.py tests/test_cli.py README.md AGENTS.md
git commit -m "feat: add --shotgun CLI flag and document candidate outputs"
```

---

## Self-Review Notes

- **Spec coverage:** top-N + NMS (Task 1), best-event-only full-scan candidates (Task 3), per-candidate output + `.log` with raw offset (Tasks 2-3), N≥2 validation and full-scan-ignores-hint (Tasks 3-4), default behavior preserved via empty suffix (Task 2), deferred offset-correction left untouched and documented (Task 4). Testing section: unit tests for top-N, NMS separation, n=1 refactor equivalence, and suffix all present.
- **Manual validation (out of automated scope):** real-media shotgun run, and the normal-vs-`--no-hint` correction experiment, are performed by the user with actual recordings.

# Sound Fixer — Future Extensions & Known Issues

## TODO

### Impulse Detection Tuning
- The default impulse detection threshold (~15 dB above neighbors) and window sizes (50ms analysis, 500ms context) are initial guesses. These may need tuning based on real-world results. Consider making them CLI-configurable if the defaults prove inadequate.

### Audio Segment Reconstruction Ambiguity
- Current approach: simple concatenation, assuming negligible gap between segments. In reality there could be a small gap or even overlap at segment boundaries. A future improvement could use cross-correlation on the boundary regions to detect and compensate for gaps/overlaps.

### Timestamp Reliability
- Metadata timestamps are used as a heuristic for narrowing alignment search. Battery-powered recorders may have reset clocks. Current fallback (full scan) handles this, but a smarter heuristic could detect clock-reset patterns (e.g., all files dated 2000-01-01) and skip the narrow search entirely.

### Multi-Event Alignment
- When multiple events are detected, each video clip is tried against all events. If events have very similar audio content (e.g., two sets of the same concert), alignment may pick the wrong event. No solution currently — relies on confidence scores.

### Audio Format Mismatches
- Concatenation assumes all audio segments from the same recorder have identical format (sample rate, bit depth, channels). If a user mixes recorders, this will fail silently or produce garbled audio.

### Chromaprint Alignment ~500ms Fixed Offset
- The chromaprint-based alignment consistently produces an offset that is ~500ms late. The hardcoded `FPCALC_ITEM_DURATION = 0.1238` constant is more accurate than computing from fpcalc's reported duration (which made it worse — ~2.5s off). The ~500ms error is likely a fixed bias from chromaprint's analysis window or the pipeline, not cumulative drift. Currently compensated manually via `--offset-adjust`. Investigate root cause: could be window centering, fpcalc startup buffering, or sox concatenation artifact.

### Progress Bar for Alignment
- The progress bar during alignment is real (iterates over correlation positions). But for fingerprint generation, it's just a 1/2, 2/2 counter since fpcalc is an opaque external call. Could be improved by monitoring fpcalc's output or estimating based on file size.

### DST / Timezone Handling
- Metadata timestamps from different devices may disagree due to DST changes or timezone settings. The `--no-hint` flag bypasses the timestamp heuristic entirely. A smarter approach could try multiple timezone offsets (+/- 1h, +/- 2h) in the narrow search window.

### Loudness Normalization Option
- Current design uses peak normalization only (correct for music). A future flag could offer EBU R128 loudness normalization for speech/conference recordings where perceptual consistency matters more than preserving dynamics.

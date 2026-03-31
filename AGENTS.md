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

### Loudness Normalization Option
- Current design uses peak normalization only (correct for music). A future flag could offer EBU R128 loudness normalization for speech/conference recordings where perceptual consistency matters more than preserving dynamics.

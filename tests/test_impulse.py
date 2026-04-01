import numpy as np
from soundgraft.cli import detect_impulses, attenuate_impulses


def test_no_impulse_in_steady_signal():
    """A constant-amplitude signal should have no impulses detected."""
    sr = 44100
    t = np.linspace(0, 1, sr, endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 0


def test_detect_single_impulse():
    """A short spike in an otherwise quiet signal should be detected."""
    sr = 44100
    signal = np.full(sr * 2, 0.01, dtype=np.float32)
    spike_start = sr
    spike_len = int(0.002 * sr)
    signal[spike_start:spike_start + spike_len] = 0.9

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 1
    assert abs(impulses[0]["timestamp"] - 1.0) < 0.1


def test_loud_passage_not_detected():
    """A sustained loud section should NOT be detected as an impulse."""
    sr = 44100
    t = np.linspace(0, 3, sr * 3, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    signal[:sr] *= 0.1
    signal[sr:2*sr] *= 0.8
    signal[2*sr:] *= 0.1

    impulses = detect_impulses(signal, sr)
    assert len(impulses) == 0


def test_attenuate_reduces_spike():
    """After attenuation, the spike should be close to the surrounding level."""
    sr = 44100
    signal = np.full(sr * 2, 0.01, dtype=np.float32)
    spike_start = sr
    spike_len = int(0.002 * sr)
    signal[spike_start:spike_start + spike_len] = 0.9

    impulses = detect_impulses(signal, sr)
    cleaned = attenuate_impulses(signal.copy(), impulses, sr)

    assert np.max(np.abs(cleaned[spike_start:spike_start + spike_len])) < 0.1

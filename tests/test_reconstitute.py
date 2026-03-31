from sound_fixer import group_audio_into_events


def _file(name, size, creation_time):
    return {"path": f"/fake/{name}", "size": size, "creation_time": creation_time, "duration": 3600.0}


def test_single_event_two_segments():
    """Two files: one full, one short — single event."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 1_700_000_000, "2026-03-31T10:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 2


def test_two_events():
    """full, short, full, short — two events."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 1_000_000_000, "2026-03-31T10:00:00"),
        _file("003.wav", 2_000_000_000, "2026-03-31T14:00:00"),
        _file("004.wav", 1_500_000_000, "2026-03-31T15:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 2
    assert len(events[0]) == 2
    assert len(events[1]) == 2


def test_single_file():
    """One file — one event."""
    files = [_file("001.wav", 500_000_000, "2026-03-31T09:00:00")]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 1


def test_three_full_then_short():
    """full, full, short — one event with three segments."""
    files = [
        _file("001.wav", 2_000_000_000, "2026-03-31T09:00:00"),
        _file("002.wav", 2_000_000_000, "2026-03-31T10:00:00"),
        _file("003.wav", 800_000_000, "2026-03-31T11:00:00"),
    ]
    events = group_audio_into_events(files)
    assert len(events) == 1
    assert len(events[0]) == 3

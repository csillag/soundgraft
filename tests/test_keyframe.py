from soundgraft.cli import parse_keyframe_times


def test_parse_ignores_trailing_sei_side_data_fields():
    # ffprobe csv leaks H.265 SEI side-data as extra comma fields after pts_time.
    out = (
        "0.000000,H.26[45] User Data Unregistered SEI message\n"
        "0.935000,H.26[45] User Data Unregistered SEI message,H.26[45] User Data Unregistered SEI message\n"
        "1.868333,H.26[45] User Data Unregistered SEI message\n"
    )
    assert parse_keyframe_times(out) == [0.0, 0.935, 1.868333]


def test_parse_plain_lines_and_skips_bad():
    out = "0.0\n1.5\nN/A\n\n2.0\n"
    assert parse_keyframe_times(out) == [0.0, 1.5, 2.0]


def test_parse_sorts_output():
    out = "2.0\n0.5\n1.0\n"
    assert parse_keyframe_times(out) == [0.5, 1.0, 2.0]


def test_parse_empty():
    assert parse_keyframe_times("") == []

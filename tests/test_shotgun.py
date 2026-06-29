from soundgraft.cli import candidate_suffix


def test_candidate_suffix_empty_for_normal_alignment():
    alignment = {"offset": 12.3, "candidate": None}
    assert candidate_suffix(alignment) == ""


def test_candidate_suffix_absent_key_is_empty():
    assert candidate_suffix({"offset": 12.3}) == ""


def test_candidate_suffix_for_candidate():
    alignment = {"offset": 16.4, "candidate": {"rank": 2, "raw_offset_items": 130, "correction": 0.5}}
    assert candidate_suffix(alignment) == "_cand2_16.4s"

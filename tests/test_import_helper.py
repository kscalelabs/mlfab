"""Tests the root import helper module."""

import mlfab


def test_import_helper() -> None:
    all_names, map_names = set(mlfab.__all__), set(mlfab.NAME_MAP)
    both_names = all_names & map_names
    assert all_names == both_names
    assert map_names == both_names
    for name in both_names:
        assert getattr(mlfab, name) is not None

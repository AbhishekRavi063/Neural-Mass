from pathlib import Path

import pytest

from kcomplex_detector.utils.dreams_io import read_scoring_file, read_signal_txt


DREAMS_DIR = Path("data/dreams/DatabaseKcomplexes")


@pytest.mark.skipif(not DREAMS_DIR.exists(), reason="DREAMS dataset not downloaded")
def test_dreams_excerpt_and_scoring_load():
    signal = read_signal_txt(DREAMS_DIR / "excerpt1.txt")
    events = read_scoring_file(DREAMS_DIR / "Visual_scoring1_excerpt1.txt")

    assert len(signal) == 360000
    assert len(events) > 0
    assert {"onset", "end", "duration"} <= set(events[0])

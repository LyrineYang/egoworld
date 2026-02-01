from egoworld.utils.video import validate_time_alignment


def test_time_alignment():
    assert validate_time_alignment(1.0, 2.0, 30, 60, 30.0)

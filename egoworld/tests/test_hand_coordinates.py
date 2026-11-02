import numpy as np
import pytest
from egoworld.hand_coordinates import dex_wrist_coordinates


@pytest.mark.parametrize("side", ["left", "right"])
def test_wrist_coordinates_remove_camera_orientation_and_translation(side):
    rng = np.random.default_rng(4)
    hand = rng.normal(size=(21, 3)) * .02
    hand[0] = 0
    hand[5] = [.02, .06, 0]
    hand[9] = [0, .07, 0]
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(rotation) < 0:
        rotation[:, 0] *= -1
    expected = dex_wrist_coordinates(hand, side)
    actual = dex_wrist_coordinates(hand @ rotation.T + [1, 2, 3], side)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), np.linalg.norm(hand, axis=1), atol=1e-12)


def test_wrist_coordinates_reject_degenerate_palm():
    with pytest.raises(ValueError, match="Degenerate"):
        dex_wrist_coordinates(np.zeros((21, 3)), "right")

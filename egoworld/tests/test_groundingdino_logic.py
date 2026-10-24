import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from egoworld.operators.groundingdino_op import GroundingDINOOperator


def test_model_api_uses_bgr_and_two_return_values():
    rgb = np.tile(np.array([[[10, 20, 30]]], dtype=np.uint8), (100, 100, 1))

    def predict(image, caption, **kwargs):
        np.testing.assert_array_equal(image, rgb[:, :, ::-1])
        assert image.flags.c_contiguous
        return SimpleNamespace(
            xyxy=np.array([[10, 15, 20, 25], [30, 35, 40, 45]]),
            confidence=np.array([0.4, 0.9]),
        ), ["hand", "cup"]

    op = GroundingDINOOperator("unused", "unused")
    op._model = SimpleNamespace(predict_with_caption=predict)
    op._use_model_class = True
    result = op.predict(rgb, "hand . cup .", max_boxes=1)
    assert len(result) == 1
    assert result[0].box_xyxy == (30, 35, 40, 45)
    assert result[0].phrase == "cup"
    assert result[0].score == 0.9


def test_function_api_converts_cxcywh_and_preserves_device(monkeypatch, tmp_path):
    inference = ModuleType("groundingdino.util.inference")
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    model = object()

    def predict(**kwargs):
        assert kwargs["model"] is model
        assert kwargs["device"] == "cpu"
        # Width exceeds center x; guessing the format from x2 < x1 fails here.
        return np.array([[0.2, 0.2, 0.6, 0.6]]), np.array([0.8]), ["hand"]

    inference.predict = predict
    inference.load_image = lambda path: (image, object())
    for name in ("groundingdino", "groundingdino.util"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "groundingdino.util.inference", inference)
    monkeypatch.setattr("egoworld.operators.groundingdino_op._write_image", lambda *args: None)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    op = GroundingDINOOperator("unused", "unused", device="cpu")
    op._model = model
    result = op.predict(image, "hand .")
    np.testing.assert_allclose(result[0].box_xyxy, [0, 0, 50, 50], atol=1e-5)
    assert list(tmp_path.iterdir()) == []

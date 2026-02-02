import json
from pathlib import Path


def _load_example_config() -> dict:
    path = Path(__file__).resolve().parents[1] / "configs" / "example.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_example_config_has_operator_contracts() -> None:
    config = _load_example_config()
    operators = config.get("operators")
    assert isinstance(operators, dict)

    sam2 = operators.get("sam2")
    assert isinstance(sam2, dict)
    assert sam2.get("enabled") is True
    params = sam2.get("params")
    assert isinstance(params, dict)

    assert isinstance(params.get("checkpoint"), str)
    assert isinstance(params.get("config"), str)
    assert isinstance(params.get("device"), str)
    assert isinstance(params.get("precision"), str)

    prompting = params.get("prompting")
    assert isinstance(prompting, dict)
    assert isinstance(prompting.get("source"), str)
    assert isinstance(prompting.get("gd_config"), str)
    assert isinstance(prompting.get("gd_checkpoint"), str)

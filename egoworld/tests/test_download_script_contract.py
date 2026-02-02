from pathlib import Path


def test_download_models_script_matches_defaults() -> None:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_models.sh"
    content = script_path.read_text(encoding="utf-8")

    expected = [
        "sam2.1_hiera_small.pt",
        "sam2.1_hiera_s.yaml",
        "groundingdino_swint_ogc.pth",
        "GroundingDINO_SwinT_OGC.py",
    ]
    for name in expected:
        assert name in content

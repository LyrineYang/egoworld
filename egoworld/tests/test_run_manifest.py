import json

from egoworld.config import PipelineConfig


def test_run_manifest_serialization() -> None:
    config = PipelineConfig(model_versions={"sam2": "v1"})
    manifest = config.to_run_manifest()

    assert isinstance(manifest["parquet_params"], str)
    assert isinstance(manifest["model_versions"], str)

    parquet_params = json.loads(manifest["parquet_params"])
    model_versions = json.loads(manifest["model_versions"])

    assert parquet_params["compression"] == "zstd"
    assert model_versions["sam2"] == "v1"

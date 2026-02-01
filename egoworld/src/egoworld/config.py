"""Configuration for the offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class ParquetConfig:
    compression: str = "zstd"
    row_group_size: int = 256 * 1024 * 1024
    data_page_size: int = 8 * 1024 * 1024
    partition: List[str] = field(default_factory=lambda: ["run_id", "video_id", "clip_id"])


@dataclass
class BackpressureConfig:
    max_in_flight_cpu: Optional[int] = None
    max_in_flight_gpu: Optional[int] = None
    max_in_flight_write: Optional[int] = None

    def resolve(self, num_gpus: int) -> "BackpressureConfig":
        multiplier = 2
        return BackpressureConfig(
            max_in_flight_cpu=self.max_in_flight_cpu or (multiplier * num_gpus),
            max_in_flight_gpu=self.max_in_flight_gpu or (multiplier * num_gpus),
            max_in_flight_write=self.max_in_flight_write or (multiplier * num_gpus),
        )


@dataclass
class RetryPolicy:
    max_retries: int = 3
    base_delay_s: float = 5.0
    backoff: float = 3.0

    def next_delay(self, attempt: int) -> float:
        return self.base_delay_s * (self.backoff ** max(0, attempt - 1))


@dataclass
class SceneDetectConfig:
    method: str = "scenedetect"
    min_scene_len_s: float = 1.0
    fallback_full_clip: bool = True
    overlap_s: float = 1.0


@dataclass
class CoordinateSpec:
    spec_version: str = "v1"
    time_base: str = "seconds"
    mask_encoding: str = "rle"
    length_unit: str = "meters"
    handedness: str = "right"
    quat_order: str = "wxyz"
    frame_index_base: int = 0
    axis_order: str = "x,y,z"
    coord_frame: str = "camera"


@dataclass
class MetricsThresholds:
    gpu_util_min: float = 0.60
    gpu_util_window_s: int = 600
    failure_rate_max: float = 0.01
    empty_mask_rate_max: float = 0.20


@dataclass
class PathsConfig:
    data_root: str = "./data"
    output_root: str = "./output"
    manifest_path: str = "./manifests"
    state_db_path: str = "./state/pipeline.db"
    runlog_path: str = "./runlog.md"


@dataclass
class OperatorConfig:
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperatorsConfig:
    sam2: OperatorConfig = field(default_factory=lambda: OperatorConfig(enabled=True))
    hamer: OperatorConfig = field(default_factory=lambda: OperatorConfig(enabled=False))
    foundationpose: OperatorConfig = field(default_factory=lambda: OperatorConfig(enabled=False))
    dex_retarget: OperatorConfig = field(default_factory=lambda: OperatorConfig(enabled=False))
    fast3r: OperatorConfig = field(default_factory=lambda: OperatorConfig(enabled=False))


@dataclass
class PipelineConfig:
    num_gpus: int = 1
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    backpressure: BackpressureConfig = field(default_factory=BackpressureConfig)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    scenedetect: SceneDetectConfig = field(default_factory=SceneDetectConfig)
    coordinates: CoordinateSpec = field(default_factory=CoordinateSpec)
    metrics: MetricsThresholds = field(default_factory=MetricsThresholds)
    paths: PathsConfig = field(default_factory=PathsConfig)
    operators: OperatorsConfig = field(default_factory=OperatorsConfig)
    run_id: Optional[str] = None
    model_versions: Dict[str, str] = field(default_factory=dict)
    dataset_hash: Optional[str] = None
    code_git_hash: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def resolved(self) -> "PipelineConfig":
        resolved = PipelineConfig(
            num_gpus=self.num_gpus,
            parquet=self.parquet,
            backpressure=self.backpressure.resolve(self.num_gpus),
            retry=self.retry,
            scenedetect=self.scenedetect,
            coordinates=self.coordinates,
            metrics=self.metrics,
            paths=self.paths,
            operators=self.operators,
            run_id=self.run_id,
            model_versions=self.model_versions,
            dataset_hash=self.dataset_hash,
            code_git_hash=self.code_git_hash,
            extra=self.extra,
        )
        return resolved

    def to_run_manifest(self) -> Dict[str, Any]:
        data = asdict(self)
        data["parquet_params"] = json.dumps(asdict(self.parquet), ensure_ascii=True)
        data["model_versions"] = json.dumps(self.model_versions, ensure_ascii=True)
        data["coordinate_spec_version"] = self.coordinates.spec_version
        data["mask_encoding"] = self.coordinates.mask_encoding
        data["time_base"] = self.coordinates.time_base
        return data


def load_config(path: str) -> PipelineConfig:
    content = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        data = json.loads(content)
    else:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML config requires pyyaml") from exc
        data = yaml.safe_load(content)
    operators_raw = data.get("operators", {}) or {}
    operators = OperatorsConfig(
        sam2=_load_operator_config(operators_raw.get("sam2"), OperatorsConfig().sam2),
        hamer=_load_operator_config(operators_raw.get("hamer"), OperatorsConfig().hamer),
        foundationpose=_load_operator_config(
            operators_raw.get("foundationpose"), OperatorsConfig().foundationpose
        ),
        dex_retarget=_load_operator_config(
            operators_raw.get("dex_retarget"), OperatorsConfig().dex_retarget
        ),
        fast3r=_load_operator_config(operators_raw.get("fast3r"), OperatorsConfig().fast3r),
    )

    return PipelineConfig(
        num_gpus=data.get("num_gpus", 1),
        parquet=ParquetConfig(**data.get("parquet", {})),
        backpressure=BackpressureConfig(**data.get("backpressure", {})),
        retry=RetryPolicy(**data.get("retry", {})),
        scenedetect=SceneDetectConfig(**data.get("scenedetect", {})),
        coordinates=CoordinateSpec(**data.get("coordinates", {})),
        metrics=MetricsThresholds(**data.get("metrics", {})),
        paths=PathsConfig(**data.get("paths", {})),
        operators=operators,
        run_id=data.get("run_id"),
        model_versions=data.get("model_versions", {}),
        dataset_hash=data.get("dataset_hash"),
        code_git_hash=data.get("code_git_hash"),
        extra=data.get("extra", {}),
    )


def _load_operator_config(raw: Optional[Dict[str, Any]], default: OperatorConfig) -> OperatorConfig:
    raw = raw or {}
    params = raw.get("params")
    if params is None:
        params = dict(default.params)
    return OperatorConfig(
        enabled=raw.get("enabled", default.enabled),
        params=params,
    )

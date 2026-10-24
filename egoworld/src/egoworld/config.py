"""Configuration for the offline pipeline."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math


@dataclass
class ParquetConfig:
    compression: str = "zstd"
    row_group_size: int = 65536  # rows, not bytes
    data_page_size: int = 8 * 1024 * 1024
    partition: List[str] = field(default_factory=lambda: ["run_id", "video_id", "clip_id"])


@dataclass
class BackpressureConfig:
    max_in_flight_cpu: Optional[int] = None
    max_in_flight_gpu: Optional[int] = None
    max_in_flight_write: Optional[int] = None

    def resolve(self, num_gpus: int) -> "BackpressureConfig":
        multiplier = 2
        num_gpus = max(1, num_gpus)
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
    max_clip_duration_s: float = 15.0


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
class ExecutionConfig:
    cpu_workers: int = 2
    write_workers: int = 2
    cpu_operator_workers: int = 1
    max_active_clips: int = 4
    max_cached_models: int = 1
    stage_timeout_s: float = 900.0
    poll_timeout_s: float = 0.1
    keep_prepared: bool = False
    verify_hashes_on_resume: bool = True
    gpu_stage_groups: List[List[str]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    num_gpus: int = 1
    parquet: ParquetConfig = field(default_factory=ParquetConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
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
        self.validate()
        resolved = PipelineConfig(
            num_gpus=self.num_gpus,
            parquet=self.parquet,
            execution=self.execution,
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

    def validate(self) -> None:
        if not isinstance(self.num_gpus, int) or isinstance(self.num_gpus, bool) or self.num_gpus < 0:
            raise ValueError("num_gpus must be a nonnegative integer")
        for name in ("max_in_flight_cpu", "max_in_flight_gpu", "max_in_flight_write"):
            value = getattr(self.backpressure, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"backpressure.{name} must be a positive integer")
        if self.scenedetect.max_clip_duration_s <= 0:
            raise ValueError("scenedetect.max_clip_duration_s must be positive")
        if not 0 <= self.scenedetect.overlap_s < self.scenedetect.max_clip_duration_s:
            raise ValueError("scenedetect.overlap_s must be nonnegative and shorter than the clip duration")
        if self.retry.max_retries < 0 or self.retry.base_delay_s < 0 or self.retry.backoff < 1:
            raise ValueError("Invalid retry policy")
        if self.parquet.row_group_size < 1 or not 0 <= self.metrics.empty_mask_rate_max <= 1:
            raise ValueError("Invalid Parquet row group size or empty-mask threshold")
        from egoworld.contracts import enabled_stages
        graph = enabled_stages(self)
        from egoworld.contracts import GPU_STAGES
        if self.num_gpus == 0 and set(graph) & GPU_STAGES:
            raise ValueError("num_gpus must be positive when GPU stages are enabled")
        fixed = {"time_base": "seconds", "mask_encoding": "rle", "frame_index_base": 0}
        for name, value in fixed.items():
            if getattr(self.coordinates, name) != value:
                raise ValueError(f"coordinates.{name} currently supports only {value!r}")
        if self.parquet.partition != ["run_id", "video_id", "clip_id"]:
            raise ValueError("Only run_id/video_id/clip_id partitioning is supported")
        for name in ("cpu_workers", "write_workers", "cpu_operator_workers", "max_active_clips", "max_cached_models"):
            if not isinstance(getattr(self.execution, name), int) or isinstance(getattr(self.execution, name), bool) or getattr(self.execution, name) < 1:
                raise ValueError(f"execution.{name} must be positive")
        if self.execution.stage_timeout_s <= 0 or self.execution.poll_timeout_s <= 0:
            raise ValueError("Execution timeouts must be positive")
        for value in (self.execution.stage_timeout_s, self.execution.poll_timeout_s,
                      self.scenedetect.max_clip_duration_s, self.scenedetect.overlap_s,
                      self.retry.base_delay_s, self.retry.backoff):
            if not math.isfinite(value):
                raise ValueError("Durations and retry parameters must be finite")
        if self.execution.gpu_stage_groups:
            from egoworld.contracts import GPU_STAGES
            if len(self.execution.gpu_stage_groups) != self.num_gpus:
                raise ValueError("gpu_stage_groups must contain one group per GPU")
            available = set().union(*map(set, self.execution.gpu_stage_groups))
            required = set(enabled_stages(self)) & GPU_STAGES
            if required - available or available - GPU_STAGES:
                raise ValueError("gpu_stage_groups must cover enabled GPU stages and contain only GPU stages")
        required_params = {"hamer": ("checkpoint",), "foundationpose": ("assets_manifest", "objects"),
                           "dex_retarget": ("config", "robot_id", "human_to_robot_rotation", "handedness", "joint_units"),
                           "fast3r": ("model_name_or_path",)}
        for name, keys in required_params.items():
            operator = getattr(self.operators, name)
            if operator.enabled:
                for key in keys:
                    if key not in operator.params:
                        raise ValueError(f"operators.{name}.params.{key} is required")

    def to_run_manifest(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id or "",
            "created_at": "",
            "config_path": "",
            "code_git_hash": self.code_git_hash or "",
            "model_versions": json.dumps(self.model_versions, ensure_ascii=True),
            "dataset_hash": self.dataset_hash or "",
            "parquet_params": json.dumps(asdict(self.parquet), ensure_ascii=True),
            "coordinate_spec_version": self.coordinates.spec_version,
            "mask_encoding": self.coordinates.mask_encoding,
            "time_base": self.coordinates.time_base,
        }


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
    accepted = {f.name for f in __import__("dataclasses").fields(PipelineConfig)}
    if unknown := set(data) - accepted:
        raise ValueError(f"Unknown config fields: {sorted(unknown)}")
    operators_raw = data.get("operators", {}) or {}
    if unknown := set(operators_raw) - {"sam2", "hamer", "foundationpose", "dex_retarget", "fast3r"}:
        raise ValueError(f"Unknown operators: {sorted(unknown)}")
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

    config = PipelineConfig(
        num_gpus=data.get("num_gpus", 1),
        parquet=ParquetConfig(**data.get("parquet", {})),
        execution=ExecutionConfig(**data.get("execution", {})),
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
    # Relative paths follow the invocation directory, as in repository examples.
    # Workers may change directory to an upstream model repository afterwards.
    for name, value in vars(config.paths).items():
        setattr(config.paths, name, str(Path(value).expanduser().resolve()))
    path_keys = {"checkpoint", "config", "gd_config", "gd_checkpoint", "repo_path", "working_dir", "assets_manifest", "mesh_path", "urdf_dir"}
    def resolve_params(params):
        for name, value in params.items():
            if isinstance(value, dict):
                resolve_params(value)
            elif isinstance(value, str) and (name in path_keys or
                  (name == "model_name_or_path" and (value.startswith((".", "/", "~")) or Path(value).exists()))):
                params[name] = str(Path(value).expanduser().resolve())
            elif name == "runner_command" and isinstance(value, list):
                params[name] = [str(Path(arg).expanduser().resolve()) if isinstance(arg, str) and
                               (arg.startswith(("./", "../", "~")) or Path(arg).is_file()) else arg for arg in value]
    for operator in vars(config.operators).values():
        resolve_params(operator.params)
    return config


def _load_operator_config(raw: Optional[Dict[str, Any]], default: OperatorConfig) -> OperatorConfig:
    raw = raw or {}
    if unknown := set(raw) - {"enabled", "params"}:
        raise ValueError(f"Unknown operator config fields: {sorted(unknown)}")
    if not isinstance(raw.get("enabled", default.enabled), bool):
        raise ValueError("Operator enabled must be boolean")
    params = raw.get("params")
    if params is None:
        params = dict(default.params)
    if not isinstance(params, dict):
        raise ValueError("Operator params must be an object")
    return OperatorConfig(
        enabled=raw.get("enabled", default.enabled),
        params=params,
    )

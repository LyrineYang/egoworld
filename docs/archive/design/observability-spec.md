# Observability Specialization (Spec)

目标：为 TB 级批处理管线提供**可量化、低开销、可扩展**的观测能力，确保跑得快、跑得稳、跑得对。
范围：单机 4×H100；Manifest 生成（CPU/IO）+ 推理/写入（Ray/GPU）；Parquet 输出；SAM2.1 small 为默认分割模型。

---

## 1. 设计目标
- **可定位瓶颈**：CPU/IO/GPU 三段性能可量化。
- **可验证质量**：空结果、异常分布可被自动发现。
- **低侵入**：不开启监控时为 no-op，最低运行开销。
- **可扩展**：新增算子/模型时不破坏现有指标规范。
- **可追溯**：指标与 run_id / config / model_versions 对齐。

## 2. 非目标
- 多机/跨节点监控聚合
- 实时强一致告警系统（如 PagerDuty）
- 业务级数据可视化平台

---

## 3. 观测层输出类型
### 3.1 Prometheus Metrics（实时）
用于实时监控与告警，默认可关闭；无 prometheus_client 时降级为 no-op。

**指标命名约定**
- 目标全局前缀：`egoworld_`（避免冲突）
- 现有实现如存在无前缀指标，需在落地阶段统一改名（必要时保留兼容别名）
- 禁止高基数 label（例如 video_id/clip_id 不进入 metrics label）

**核心指标（MVP）**
- `egoworld_clips_processed_total{stage, status}`
  clip 处理计数（stage: gpu/write；status: ok/failed）
- `egoworld_queue_length{stage}`
  队列长度（gpu/write；CPU 预处理不在运行期队列内）
- `egoworld_stage_latency_seconds{stage}`
  各阶段处理耗时（观察分布）
- `egoworld_failure_total{stage, error_class}`
  失败计数（error_class 由 classify_error 提供）
- `egoworld_gpu_utilization{gpu}`
  GPU 利用率（0–1 比例）

**CPU/IO 预处理指标（Manifest 阶段）**
- `egoworld_manifest_videos_total`
  处理的视频数
- `egoworld_manifest_clips_total`
  生成的 clip 数
- `egoworld_manifest_stage_latency_seconds{stage}`
  预处理阶段耗时（stage: ffprobe/scenedetect/manifest）
- `egoworld_manifest_fallback_total`
  scenedetect fallback 次数
- `egoworld_manifest_io_mb_per_s`
  IO 吞吐（按视频文件大小/耗时估算）

**可选指标（扩展）**
- `egoworld_clip_latency_seconds`（端到端时延）
- `egoworld_bytes_written_total{artifact}`（masks/poses 等）
- `egoworld_empty_mask_rate`（窗口均值）

### 3.2 运行期 JSONL 事件（可追溯）
用于离线分析/回放，避免在 Prometheus 中放高基数维度。

**事件格式（每行一条 JSON）**
```
{
  "ts": "2026-02-03T12:00:00Z",
  "run_id": "YYYYMMDD_HHMMSS",
  "clip_id": "...",
  "video_id": "...",
  "stage": "gpu|write|pipeline|manifest",
  "status": "ok|failed|retry",
  "attempt": 0,
  "duration_s": 1.23,
  "error_class": "io|oom|decode|model|unknown",
  "metrics": { "frames": 240, "rows": 240, "bytes": 123456 }
}
```

**输出位置**
- `output/run_id=.../metrics/events.jsonl`（推理阶段）
- `manifests/manifest_metrics.json` 或 `events.jsonl`（预处理阶段）
（单写入点：由 driver/manifest 进程聚合写入，避免多 actor 并发写文件）

### 3.3 QC 汇总（质量探针）
基于 clip 输出的轻量统计，不要求解码全 mask。

**最小 QC 指标**
- `empty_mask_rate`：空 mask 比例（按帧）
- `mask_rle_len_mean`：RLE 字符串长度均值（代理指标；不解码，仅统计字符串长度）
- `frames_count`：输出帧数

**输出位置**
- `output/run_id=.../qc/qc_summary.json`（必需）
- 可选：`output/run_id=.../qc/clip_qc.jsonl`（默认开启）

---

## 4. 指标采集点（基于当前 pipeline）
### 4.1 Manifest/SceneDetect（CPU/IO 批处理）
说明：该阶段不在运行时 pipeline 队列内，但仍需要性能指标。
记录：
- scene detect 耗时
- 生成 clip 数量
- fallback_full_clip 次数
输出：
- `manifest_metrics.json`（单文件聚合）

### 4.2 Driver（主调度）
采集：
- `queue_length`（pending_gpu / pending_write）
- `stage_latency`（gpu：提交 actor → ray.get；write：写入开始 → 完成）
- `clip_latency`（提交 clip → 写入完成，若启用）
- `failure_total`
- `clips_processed_total`

### 4.3 GPU Actor（SAM2 等）
采集：
- 单 clip 推理耗时（最好由 actor 内部计时并回传）
- 输出帧数/行数

### 4.4 Writer
采集：
- 写入耗时
- 写入字节量（按文件类型）

---

## 5. GPU Util 采样规范
### 5.1 优先方案：NVML（pynvml）
- 采样周期：1–5s
- 指标维度：GPU index
- 失败时降级为 no-op

### 5.2 备选：DCGM exporter
- 适合长期部署场景
- 由运维独立运行，pipeline 只消费指标

---

## 6. 阈值与告警策略
阈值来自 `config.metrics`：
- `gpu_util_min` + `gpu_util_window_s`
- `failure_rate_max`
- `empty_mask_rate_max`

**告警形式（MVP）**
- 写入 `alerts.jsonl` + WARN 日志
- 不阻断主流程，但标记 run 需要复核

---

## 7. 配置扩展建议（不破坏兼容）
`metrics` 仍用于阈值；新增 `observability` 作为运行控制：
```
metrics:
  gpu_util_min: 0.60
  gpu_util_window_s: 600
  failure_rate_max: 0.01
  empty_mask_rate_max: 0.20

observability:
  enabled: false
  exporter: "prometheus"
  port: 9100
  sample_interval_s: 2
  write_events: true
  write_qc: true
  write_clip_qc: true
```

---

## 8. 低开销原则
- **不在 metrics label 中引入高基数 ID**
- QC 计算避免 full mask decode（优先 RLE 长度、空值）
- 能在 driver 聚合就不在 actor 分散埋点

---

## 9. 兼容性与扩展
- 每个新算子只需输出 `timings` + `rows`，无需改 metrics 逻辑
- QC 支持扩展 “mask area / object count / pose magnitude” 等

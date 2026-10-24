# Observability Implementation Plan

目标：在不影响主流程稳定性的前提下，逐步落地指标、质量探针与告警能力。
原则：低开销、可关闭、可追溯。

---

## Phase 0：对齐规范（当前）
**输入**：`docs/design/observability-spec.md`
**产出**：统一指标命名、事件格式、QC 口径、阈值定义与配置命名空间（metrics vs observability）。

验收：
- 规范文件评审完成
- 无代码变更

---

## Phase 1：Driver 级别的指标骨架
**目标**：最小可用的吞吐/队列/失败指标

工作项：
1) 在 `pipeline/driver.py` 增加事件统计（submit/complete/failed）
2) 统一指标命名为 `egoworld_` 前缀（如需保留旧名，提供兼容别名）
3) 引入 `observability` 运行开关（默认关闭）
4) 由 driver 单点写入 `events.jsonl`（避免并发写入冲突）

验收：
- 运行后生成 `output/run_id=.../metrics/events.jsonl`
- Prometheus exporter 可选启用

---

## Phase 1.5：Manifest 预处理指标
**目标**：量化 CPU/IO 预处理阶段性能

工作项：
1) 在 manifest 构建流程中记录 ffprobe/scenedetect/整体耗时
2) 输出视频数、clip 数、fallback 次数
3) 估算 IO 吞吐（文件大小/耗时）

验收：
- 生成 `manifest_metrics.json` 或写入 `events.jsonl`（stage=manifest）
- 可统计视频/clip 吞吐与平均耗时

---

## Phase 2：Writer/QC 汇总
**目标**：输出质量摘要 + 写入指标

工作项：
1) Writer 统计写入耗时、输出行数/字节
2) 生成 `qc_summary.json`（empty_mask_rate + rle_len_mean）
3) 可选输出 `clip_qc.jsonl`（默认开启）
4) 将 QC 与 run_id 关联

验收：
- run 结束后生成 `qc/qc_summary.json`
- 若开启 `clip_qc.jsonl`，QC 可回溯每个 clip

---

## Phase 3：GPU Util 采样
**目标**：GPU 利用率监控

工作项：
1) 引入 NVML（pynvml）采样线程
2) 每 1–5s 采集并更新 gauge
3) 无 NVML 时降级

验收：
- 指标 `egoworld_gpu_utilization{gpu}` 有数据
- 不中断运行

---

## Phase 4：告警阈值
**目标**：让异常可见、可追溯

工作项：
1) 结合 `metrics` 阈值计算
2) WARN 日志 + `alerts.jsonl`
3) 不阻断主流程

验收：
- 超阈值会触发 warn + alerts.jsonl

---

## Phase 5：性能剖析（可选）
**目标**：更细粒度诊断

工作项：
1) 按 stage 输出 histogram 的 P50/P90
2) 输出 clip 端到端 latency（提交 → 写入完成）
3) 可导出 CSV/JSON 供分析

---

## 测试计划（MVP）
- Unit: `observability/metrics.py` no-op fallback
- Unit: `qc.py` 的 empty_mask_rate 与 distribution_stats
- Integration: pipeline 运行后产出 `events.jsonl` 与 `qc_summary.json`

---

## 风险与控制
- 指标高基数：禁止把 video_id/clip_id 放入 Prometheus label
- 性能开销：QC 默认只做轻量统计
- 依赖风险：NVML / prometheus_client 均需可选降级
- 并发写冲突：events.jsonl 由 driver 单点写入

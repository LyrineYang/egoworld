# egoworld TB 级离线视频数据清洗管线 —— 编码实施计划

本文基于当前约束（离线批处理、30s–2min 视频、单机 4×H100 + 128 核 CPU、无 exactly‑once
强一致性要求）给出可落地的编码计划。目标是先做单机可运行 MVP，接口与数据契约清晰，
后续可平滑扩展到多机/流式。

## 0. 目标与非目标
- 目标
  - 产出：视频的 clip 级别结果（mask、3D/6D、hand‑action mapping 等），统一落盘为
    Parquet/Lance（优先 Parquet）。
  - 可靠性：可断点续跑、幂等写入、失败隔离、可追溯（run_id + 配置快照 + 模型版本）。
  - 性能：GPU 不空转，CPU 解码/SceneDetect 与 GPU 推理流水线并行。
- 非目标（阶段 1 不做）
  - 多机分布式部署
  - Kafka/Redis 中间件
  - exactly‑once 语义

## 1. 目录结构（建议）
```
egoworld/
  plan.md
  memory.md
  src/egoworld/
    config.py
    cli.py
    manifests/
      schema.py
      build_manifest.py
    pipeline/
      queues.py
      driver.py
      scheduler.py
      state_store.py
    operators/
      scenedetect_op.py
      sam2_op.py
      hamer_op.py
      foundationpose_op.py
      dex_retarget_op.py
    io/
      writers.py
      paths.py
    observability/
      metrics.py
      logging.py
      qc.py
    utils/
      video.py
      hashing.py
  scripts/
    run_pipeline.py
    make_manifest.py
  tests/
    test_schema.py
    test_state_store.py
    test_writer_idempotent.py
```

## 2. 数据契约与输出布局
### 2.1 Manifest Schema（建议）
- video_manifest
  - video_id, path, duration_s, fps, width, height, audio, checksum, split
- clip_manifest
  - clip_id, video_id, start_s, end_s, frame_start, frame_end, overlap_s,
    status(Pending/Running/Done/Failed), last_error, retry_count
- run_manifest
  - run_id, created_at, config_path, code_git_hash, model_versions, dataset_hash
  - parquet_params(compression,row_group_size,data_page_size)
  - coordinate_spec_version, mask_encoding, time_base

### 2.2 字段规范（必须写死）
- 时间对齐
  - 统一 time_base：`seconds` 为主，`frame_index` 为辅
  - 约束：`start_s = frame_start / fps`，`end_s = frame_end / fps`
- mask 编码
  - 默认 RLE（COCO 格式），记录 `mask_encoding=rle`
  - mask 分辨率与原视频一致，若缩放需记录 scale 与对应关系
- 坐标/参考系
  - 3D/6D 坐标系：定义右手/左手系、轴向（X/Y/Z）与单位（默认 meters）
  - 6D 姿态：`R (3x3)` 或 `quat (w,x,y,z)`，必须明确顺序与归一化
  - 物体/手的坐标系：说明是相机坐标还是世界坐标
- 时间戳与帧索引对齐
  - 所有 per‑frame 结果必须同时记录 `frame_index` 与 `timestamp_s`
  - 如果模型内部做了采样/跳帧，必须记录采样步长

### 2.3 输出布局（幂等）
```
output/
  run_id=YYYYMMDD_HHMM/
    video_id=.../clip_id=.../
      masks.parquet
      hand_pose.parquet
      object_pose.parquet
      mapping.parquet
      meta.json
```
输出文件以 video_id/clip_id 为键；重复执行可覆盖或跳过。

### 2.4 Parquet/IO 固定参数（写入 run_manifest）
- compression: zstd（默认）
- row_group_size: 256MB（初始值，可配置）
- data_page_size: 8MB（初始值，可配置）
- partition: run_id/video_id/clip_id（优先可追溯性）

## 3. 核心流水线（离线单机）
```
CPU: 读取视频 -> ffprobe 元信息 -> SceneDetect -> 生成 clip_manifest
GPU: SAM2 mask -> 3D/6D/hand 处理（CPU 或 GPU） -> Writer
```
关键原则：跨进程只传“文件路径 + 时间段”，避免搬运大帧。

## 4. 并行与反压
- GPU Actor：1 GPU 1 Actor，max_concurrency=1，避免显存冲突。
- CPU Actor/Pool：SceneDetect/解码 32–64 workers 起步。
- 队列（必须配置 + 强制执行）：
  - max_in_flight_cpu: 2 × num_gpus
  - max_in_flight_gpu: 2 × num_gpus
  - max_in_flight_write: 2 × num_gpus
  - 当队列达到上限时，阻塞上游提交（`asyncio.Queue` 或 `ray.wait`）
- Clip 策略：默认 1–2s overlap 或 warm‑start，降低时序模型边界影响。

## 5. 容错与状态管理
- 状态机：Pending → Running → Done/Failed
- 死信队列：连续失败的 clip 记录到 dead_letter 表
- 原子写入：临时文件写完后 rename/commit
- 断点续跑：仅处理非 Done clip
- 错误分类与重试策略（配置化）
  - 可重试：I/O 抖动、临时 OOM、外部依赖失败
  - 不可重试：解码损坏、格式错误、权重缺失/不匹配
  - 重试次数：默认 3 次，指数退避（5s, 15s, 45s）

## 6. 监控与质量探针
- Prometheus 指标
  - throughput、queue length、stage latency、GPU utilization
- 质量探针
  - 检测数量分布、mask 面积分布、空结果率
- 运行追溯
  - run_id + config snapshot + model hash + git hash
- 最小告警阈值（可配置）
  - GPU utilization < 60% 持续 10min
  - failure rate > 1%
  - empty mask rate > 20%

## 7. 编码实施阶段（里程碑）
### Phase A：基础框架与 Schema（MVP 1）
1) 定义 `schema.py`（Parquet schema）
2) 实现 `build_manifest.py`：
   - ffprobe 元信息
   - scenedetect 生成 clip
3) 生成 `run_manifest` 与输出路径规则
4) 固化字段规范（mask/坐标/时间）与 Parquet 参数写入

### Phase B：Ray Pipeline Skeleton（MVP 2）
1) `queues.py`：有界队列/限流封装
2) `state_store.py`：SQLite 或 Parquet 状态表
3) `driver.py`：Ray DAG 串联（CPU → GPU → Writer）
4) `writers.py`：原子写入、幂等覆盖
5) 失败分类与重试策略落地
6) SceneDetect 失败兜底（退化为整段 clip）

### Phase C：算子集成（MVP 3）
1) SAM2 Operator（GPU）
2) Hamer / FoundationPose / DexRetargeting（CPU 或 GPU）
3) 定义统一 Operator 接口

### Phase D：监控与质量（MVP 4）
1) `metrics.py`：Prometheus + DCGM exporter
2) `qc.py`：结果分布统计
3) 失败告警与日志规范
4) 阈值配置与告警规则上线

### Phase E：性能与稳定性收敛
1) Clip overlap / warm‑start 策略实验
2) GPU 利用率优化（微批/多路并发如可行）
3) IO/CPU/GPU 资源配比调参

## 8. 关键风险清单（实施时必须验证）
- Ray object store 内存压力：避免大帧常驻
- GPU0/3 监控 ERR：优先排查驱动/权限/硬件状态
- SceneDetect 质量：clip 边界对下游模型影响
- 幂等写入正确性：重跑不会污染结果

## 9. 交付物清单
- 可运行的 `scripts/run_pipeline.py`
- 完整的 manifest 与 run_id 输出
- 基础监控与 QC 报表
- 示例配置与 README

## 10. 可验收标准（建议写入 progress.md）
- MVP1：manifest 生成 + schema 固化 + run_id 输出（10–20 视频）
- MVP2：可断点续跑 + 幂等写入（重复运行结果一致）
- MVP3：GPU 利用率稳定 > 60%
- MVP4：失败率 < 1%，dead‑letter 可追溯

# 测试规范（MVP）

## 目标
- 发现断点续跑、时间对齐、输出幂等等高风险问题。
- 在 Base 环境下提供可重复的、低依赖的回归测试。
- 为后续 Phase B（模型拆环境）预留可执行的系统级测试路径。

## 分层与范围
1) 环境自检（env-smoke）
- 参考 `docs/env-policy.md` 的启动前自检。
- GPU、CUDA/Torch、FFmpeg、路径权限等必须先通过。

2) 单元/契约测试（unit）
- Manifest 生成一致性与时间对齐。
- clip_id 稳定性。
- run_manifest 字段形态与序列化。
- state_store 的 resumable 状态过滤。

3) 断点续跑与状态机（resume）
- Done 不重跑、Failed/Running/Writing 可恢复。
- 状态库与输出一致性（未来可补充）。

4) Writer 与输出一致性（io）
- 原子写入与幂等覆盖。
- Parquet 参数一致性。

5) 失败/重试策略（robustness）
- 可重试与不可重试分类正确，超过阈值进入 dead-letter。

6) 集成 smoke（pipeline-smoke）
- 端到端最小样本验证（短视频 + SAM2 + GroundingDINO）。

7) Phase B 模型环境（model-env）
- Base 调度 + 子进程 conda 环境可用性验证。
8) SAM2 集成（sam2-smoke）
- 使用真实 SAM2 + GroundingDINO 权重运行最小视频，验证加载与输出结构。

## 测试数据建议
- 2 个短视频（3-5 秒，含/不含音频）。
- 1 个损坏视频（用于错误分类测试）。
- 1 个极短视频（验证 overlap 与边界）。

## 当前已覆盖的测试文件
- `egoworld/tests/test_manifest_build.py`
- `egoworld/tests/test_run_manifest.py`
- `egoworld/tests/test_state_store.py`
- `egoworld/tests/test_writer_idempotent.py`
- `egoworld/tests/test_env_smoke.py`
- `egoworld/tests/test_pipeline_smoke.py`
- `egoworld/tests/test_sam2_integration.py`
- `egoworld/tests/test_operator_config_contract.py`
- `egoworld/tests/test_download_script_contract.py`
- `egoworld/tests/test_sam2_logic.py`

## 运行方式（Base 环境）
- 全量：`pytest -q`（在满足 GPU/Ray/PyArrow 前提下会自动运行 smoke）
- 单文件：`pytest -q egoworld/tests/test_manifest_build.py`
- 环境自检（强制）：`EGOWORLD_ENV_SMOKE=1 pytest -q egoworld/tests/test_env_smoke.py`
- Pipeline smoke（强制）：`EGOWORLD_PIPELINE_SMOKE=1 pytest -q egoworld/tests/test_pipeline_smoke.py`
- SAM2 smoke（强制）：`EGOWORLD_SAM2_SMOKE=1 pytest -q egoworld/tests/test_sam2_integration.py`
- 跳过 smoke：`EGOWORLD_ENV_SMOKE=0 EGOWORLD_PIPELINE_SMOKE=0 pytest -q`

## 服务器执行建议
1) 按 `docs/env-policy.md` 完成环境自检。
2) 先跑 unit，再跑 resume/io，最后跑 pipeline-smoke。
3) 若进入 Phase B，补充 model-env 相关测试。

## 常见问题排查
- `nvidia-smi` 不可用：驱动/权限问题，检查 GPU 驱动版本与权限。
- `torch.cuda.is_available()` 为 false：CUDA 与 torch 版本不匹配或驱动不兼容。
- `ffmpeg/ffprobe` 不在 PATH：安装系统依赖并加入 PATH。
- `pyarrow` 导入失败：Base 环境缺少依赖或版本冲突。
- `ray` 导入失败：Base 环境缺少依赖或版本不匹配。
- 模型/缓存目录不可写：确认 `EGOWORLD_MODEL_HOME` / `EGOWORLD_CACHE` 指向可写路径。
- SAM2/GD 权重缺失：设置 `EGOWORLD_SAM2_CHECKPOINT` / `EGOWORLD_SAM2_CONFIG` / `EGOWORLD_GD_CONFIG` / `EGOWORLD_GD_CHECKPOINT`。

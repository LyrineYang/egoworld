# EgoWorld

将第一视角视频批量转换为可读回的分割、手部姿态、物体姿态、机器人手关节和场景几何数据。
产品入口为命令行，面向单台 NVIDIA 服务器：从输入目录启动，中断后用同一命令恢复。

**目前已实现五阶段适配、并发处理、结构化输出和恢复机制；本地 CPU 与真实 Ray 的模型替身测试可运行。
真实模型、显存、标注效果与服务器吞吐仍待单卡验证。** Demo 暂缓。

## 能得到什么

| 阶段 | 直观输出 | 需要准备 |
| --- | --- | --- |
| SAM2.1 + GroundingDINO | 每个手/物体的 mask、类别、片段内 ID、二维轨迹 | RGB 视频、提示词、模型权重 |
| HaMeR | 左右手、21 个三维关节点、手指关节旋转、手型 | SAM2 手框、HaMeR/MANO 资产 |
| FoundationPose | 物体相对相机的位置和朝向 | 已有对齐 RGB-D、内参和带尺度网格 |
| DexRetarget | 指定机器人手各关节的角度或位移 | HaMeR 输出、机器人 YAML/URDF 和坐标约定 |
| Fast3R | 抽样相机位姿、内参、场景点云 | 多帧 RGB；默认相对尺度 |

没有 RGB-D 时关闭 FoundationPose；没有机器人资产时可关闭 DexRetarget。
本轮不接新传感器、不加复杂新模型、不做硬件控制。字段含义见[输出清单](docs/outputs.md)。

## 第一次运行

所有命令在仓库根目录执行。服务器的安装、资产路径和配置步骤见[服务器操作说明](docs/server.md)。
准备好环境后，先复制单卡分割配置：

```bash
mkdir -p config.local
cp egoworld/configs/example.json config.local/job.json

# 修改 job.json 的路径和提示词后检查
# 无 GPU 可先执行此项；不启动模型、不写运行结果
egoworld launch --config config.local/job.json --input-dir /absolute/videos --check-only

# 在服务器上执行：短样本 → 目录处理 → 查看结果
egoworld smoke --config config.local/job.json --input-video /absolute/videos/sample.mp4
egoworld launch --config config.local/job.json --input-dir /absolute/videos --run-id dataset_001
egoworld inspect --run-dir /absolute/output/run_id=dataset_001
```

中断后重新执行同一条 launch 命令、保持相同 run-id。正常 Ctrl+C 会进入清理；
外部模型 runner 由独立守护进程在 worker 退出时清理。

输出位于 `paths.output_root/run_id=<id>/`：

- `run_summary.json`：运行成功/失败、完成/跳过/失败片段数、阶段耗时。
- `video_id=*/clip_id=*/`：启用阶段的 Parquet/NPZ、`qc.json`、最终 `meta.json`。
- `events.jsonl`：失败阶段、错误与重试；分环境错误另看 `.work/<clip>/<stage>/runner.log`。

命令成功只表示执行完成；`qc.status=review` 仍需检查空结果、覆盖率和模型效果。
阶段 JSON 是提交依据，单独存在的 Parquet 可能是未完成或失效的旧结果。
改变代码、模型或输入后使用新 run-id。吞吐数字包含启动与重叠帧，不代表大规模性能已验证。

## 配置选择

| 配置 | 用途 |
| --- | --- |
| `egoworld/configs/example.json` | 单卡 SAM2.1 + GroundingDINO；第一次启动的最短路径 |
| `egoworld/configs/rgb.json` | RGB 的分割、人手、机器人手和场景；需对应模型/机器人资产 |
| `egoworld/configs/full.json` | 加入已有 RGB-D/网格的物体姿态 |
| `egoworld/configs/server.json` | 四卡固定分工模板；性能待实测 |

可分别 smoke 任一阶段，不必等待其它分支验收通过。实际依赖用 `egoworld plan --config ...` 查看，
运行时字段用 `egoworld schema` 查看。对象 ID 仅在片段内有效，各模型尚未融合为统一世界坐标。

## 无 GPU 开发与回归

Python 3.10+，系统需 FFmpeg/ffprobe；本层不安装 Torch 或模型权重。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
egoworld doctor
pytest -q -m 'not env_smoke and not pipeline_smoke and not sam2_smoke and not native_models'
```

真实 Ray 联调需另装 Ray 并开启测试开关，详见[测试说明](docs/testing.md)。

## 文档与下一步

- [服务器操作](docs/server.md)：安装、配置、预检、启动、停止、恢复与故障处理。
- [输出清单](docs/outputs.md)：模型与数据要求对应。
- [当前需求与产品推进顺序](docs/requirements.md)：验收条件、代码/测试对应和下一步。
- [测试与验证边界](docs/testing.md)：无 GPU、Ray、真模型测试与独立验证。
- [吞吐设计](docs/design.md)：当前实现、成熟方案比较与需要实测的优化。
- [项目复盘](docs/overview.md)与[原设计归档](docs/archive/plan.md)：历史和现状对照。

项目许可证尚未确定；模型与资产沿用各自上游许可。

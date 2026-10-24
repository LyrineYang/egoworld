# 从安装到处理一个数据目录

适用：Linux NVIDIA 单服务器。所有命令在仓库根目录执行，相对路径也以此为基准。
本地已覆盖真实 FFmpeg/Parquet、Ray 调度和故障回归；真实模型/显存仍待目标服务器验证。

## 1. 安装并选择处理范围

```bash
cd /absolute/path/to/egoworld
bash egoworld/scripts/setup_env.sh --weights
conda activate egoworld-base
egoworld --help
mkdir -p config.local
cp egoworld/configs/example.json config.local/job.json
```

需要 conda、NVIDIA 驱动；GroundingDINO 等原生扩展还需匹配的 CUDA toolkit/nvcc 和 C++ 编译器。
脚本安装 CLI，只下载 SAM2/GroundingDINO。第一次使用 example.json，准备成本最少。

| 要处理什么 | 配置起点 | 额外资产 |
| --- | --- | --- |
| 手/物体分割与二维轨迹 | example.json | SAM2、DINO 权重/配置，脚本可下载 |
| 加入人手、机器人手、场景 | rgb.json | HaMeR/MANO、Fast3R 快照、机器人 YAML/URDF |
| 再加入物体姿态 | full.json | 已有对齐深度、相机内参、带尺度网格 |
| 多卡分工实验 | server.json | 上述资产；调整卡数与阶段分组 |

模型环境的完整安装步骤见[环境说明](../egoworld/env/README.md)。分环境时，在该环境安装同一项目，
并给对应阶段加 Python 的绝对路径：

```json
"runner_command": ["/absolute/conda/envs/egoworld-hamer/bin/python", "-m", "egoworld.worker"]
```

不要为缺少的输入额外购买/接入传感器。没有 RGB-D 时关闭 foundationpose，没有机器人资产时关闭 dex_retarget。
各阶段可独立试跑；此处的顺序只用于尽快完成第一次使用。

## 2. 一次填写配置

修改 config.local/job.json；建议使用绝对路径。保留 num_gpus=1 先跑单卡。

| 配置位置 | 填写内容 |
| --- | --- |
| paths.output_root | 结果目录，例如 /srv/egoworld/output |
| paths.manifest_path | 自动生成清单的目录，例如 /srv/egoworld/manifests |
| paths.state_db_path | SQLite **文件**路径，例如 /srv/egoworld/state/pipeline.db |
| sam2.params | checkpoint/config；prompting 中 DINO 的配置/权重和目标词，例如 left hand . right hand . cup . |
| hamer.params | checkpoint、含 _DATA 的上游根目录 working_dir、对应 runner_command |
| fast3r.params | 完整本地 model_name_or_path（含 config.json 和权重）、runner_command |
| dex_retarget.params | 机器人 config/urdf_dir、robot_id、左右手、坐标旋转、关节单位、runner_command |
| foundationpose.params | repo_path、assets_manifest、objects 网格及尺度、runner_command |
| execution | CPU/Writer 数；小机器可各设 1，模型可见卡数须与 num_gpus 一致 |

模型字段位于 operators.<阶段>.params。示例旋转矩阵不是目标机器人的标定结果，需按真实轴向填写。
HaMeR 的工作目录不能仅含 checkpoint，还需配套 model_config.yaml 和 MANO 文件。

## 3. 预检和短样本

```bash
# 本地无 GPU 时也可执行；不加载模型、不启动 Ray、不写运行文件
egoworld launch --config config.local/job.json --input-dir /absolute/videos --check-only

# 服务器环境检查：实际调用各 runner 的 --doctor
egoworld doctor --config config.local/job.json --gpu --assets

# 真实视频短样本，默认前三秒，并自动验证再次运行不重算
egoworld smoke --config config.local/job.json --input-video /absolute/videos/sample.mp4
```

输入为恒定帧率视频。视频处理范围以视频流为准，音轨更长不会扩大片段。
预检失败时返回非零，checks 中列出失败项；它不反序列化权重，不能代替真实推理。

要分别定位模型，可给 smoke 加 --stages sam2 / hamer / fast3r / dex_retarget / foundationpose；依赖自动启用。
样本需包含目标和可靠左右手标签。没有 GPU 时继续进行[本地测试](testing.md)，真模型结果保持待测。

## 4. 处理、停止与恢复

```bash
egoworld launch --config config.local/job.json \
  --input-dir /absolute/videos --run-id dataset_001

egoworld inspect --run-dir /srv/egoworld/output/run_id=dataset_001
```

launch 自动执行环境/资产预检、生成清单、检查输入覆盖并启动处理；无需手动串接 make-manifest/run。
其它格式加 --glob '**/*.mov'。默认最长 15 秒、重叠 1 秒。

停止时在前台按 Ctrl+C，等待清理完成；正常 runner 退出、超时或 worker 被杀，都会清理该任务的外部模型进程组。
恢复时执行**完全相同的 launch 命令和 run-id**。一次 run 只启动一个 driver。

已提交且哈希/schema 有效的阶段会复用。损坏阶段及其下游会先移除提交标记，再重算；
即使重算途中再次中断，也不会将旧下游当作有效缓存。最终文件已完成但数据库未更新时，恢复会修正状态。

输入路径/大小/修改时间、切分设置、代码或模型身份改变时需新 run-id；不要手工修改提交 JSON 绕过检查。
本次更新改变了代码身份，旧代码生成的 run 请保留作记录，使用新 run-id 开始。

## 5. 判断结果与定位问题

| 看到什么 | 含义和操作 |
| --- | --- |
| 命令非零退出 / summary.status=failed | 执行失败；查 events.jsonl 中的阶段和错误，修复依赖/输入后同 run-id 重试 |
| summary.status=passed | 调度执行完成；继续查各片段 QC，不能据此判断标注准确 |
| qc.status=review | 空结果、覆盖不足或空阶段；检查样本目标、提示词、资产和结果内容 |
| qc.status=pass | 通过当前自动规则；还需抽查手型、左右手、网格投影或点云 |
| 分环境模型失败 | 查 output/run_id=<id>/.work/<clip>/<stage>/runner.log；成功片段默认清理临时目录 |
| Ray CPUs 不足 | 降低 execution.cpu_workers / write_workers / cpu_operator_workers，留系统余量 |
| CUDA OOM | 减小片段长度、Fast3R max_views/max_points 或 HaMeR batch_size；改变模型参数需新 run-id |
| 输入/模型/代码身份不符 | 新建 run-id；旧运行目录保留，不能混用结果 |

paths.manifest_path/<run-id>/ 保存 sources.json、视频/片段清单、startup.json 和全绝对路径 config.resolved.json。
output/run_id=<id>/ 保存运行清单、事件、汇总以及 video_id=*/clip_id=*/ 下的 Parquet/NPZ、提交记录和 QC/meta。
只有提交记录有效的阶段才是可复用结果；重算时旧 Parquet 可暂留，不能仅凭文件存在判断完成。

如需保存终端日志，先 set -o pipefail 再使用 tee，避免掩盖处理命令的失败退出。
完整字段见[输出清单](outputs.md)；测试范围和服务器验收证据见[测试说明](testing.md)。

## 6. 已有 RGB-D 的 FoundationPose 设置

先用同一输入获得 video_id（这个手动步骤仅用于填写 RGB-D 关联）：

```bash
egoworld make-manifest --config config.local/job.json \
  --input-dir /absolute/videos --output-dir /absolute/manifests_preview
```

复制 egoworld/configs/assets.example.json，以 video_id 填 K、depth_pattern、depth_scale_to_meters。
深度使用原视频 frame_index；相对深度路径以 assets JSON 所在目录为基准。填 objects 的网格路径和米制换算比例。
FoundationPose 源码 weights 下需保留两个版本子目录及各自的 model_best.pth/config.yml，见环境说明。
launch 检查视频 ID 覆盖和首个深度尺寸/数值，后续每帧在执行时继续校验。

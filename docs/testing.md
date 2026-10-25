# 测试与快速落地

所有命令在仓库根目录执行。默认测试自动跳过缺少硬件的模型测试；显式开启后，缺少资产/依赖应失败，不能算通过。
各分支可独立开发与测试，无需等待上一分支 GPU 验收。

当前范围已经确认，包含 SAM2/GroundingDINO；不接新传感器/复杂新模型。服务器配置和统一 launch 入口见 [server.md](server.md)。
本地启动专项：pytest -q egoworld/tests/test_server_startup.py，覆盖五阶段资产、工作目录、分环境预检协议、CLI 启动与恢复。

## 1. 无 GPU 回归

```bash
python -m pip install -e '.[dev]'
egoworld doctor
egoworld plan --config egoworld/configs/full.json
egoworld schema
pytest -q -m 'not env_smoke and not pipeline_smoke and not sam2_smoke and not native_models'
```

需要系统 FFmpeg/ffprobe。测试覆盖真实片段提取、Parquet、输入与配置校验、五分支调度、反压、
局部恢复、独立分支失败、延迟重试、坐标与点云契约、模型身份变化。模型 API 替身只证明适配器控制流。
`.github/workflows/tests.yml` 在 Linux CPU 上运行这一层与下一层。

## 2. 真实 Ray，仍不需要 GPU

```bash
python -m pip install 'ray>=2.8,<3'
EGOWORLD_RAY_TEST=1 RAY_USAGE_STATS_ENABLED=0 \
  pytest -q -m ray_integration
```

真实进程、FFmpeg、Writer、文件与恢复；显式声明的 Ray GPU 调度资源由模型替身使用，无 CUDA 推理。
同时测试 worker 崩溃、超时和替换。需要本机 socket/多进程权限；受限沙箱中可能需要在普通终端执行。

本轮审查问题的固定回归：

| 反例 | 测试位置 | 必须观察到的行为 |
| --- | --- | --- |
| 上游损坏，重算提交后再次中断 | test_pipeline_logic.py | 下游仍需重算，独立 Fast3R 复用 |
| 文件已提交，数据库仍 Running | test_pipeline_logic.py | 恢复修复 SQLite，不重新推理 |
| A OOM 后继续处理 B | test_ray_integration.py | A 在计算阶段被识别、替换，B 成功；driver 不取大结果 |
| worker 关闭/被杀，外部 runner 尚在工作 | test_ray_integration.py | runner 实际退出，心跳停止 |
| runner 正常退出留下子进程，或超时 | test_external_runner.py | 整个任务进程组清理，心跳停止 |
| 音轨比视频更长 | test_cpu_readiness.py、test_manifest_build.py | 按视频流排帧，真实 FFmpeg 准备成功 |

这些回归不需要 CUDA；进程测试使用显式心跳程序，OOM 使用明确的错误替身，不代表真实 CUDA 恢复已验收。

## 3. 真模型逐阶段测试

先按[环境说明](../egoworld/env/README.md)安装依赖并替换 full.json 的资产路径。
分环境时配置 `runner_command`，模型环境中的项目包必须来自同一份源代码。

```bash
egoworld doctor --config egoworld/configs/full.json --gpu

# 自动包含所选阶段的上游；每个命令独立产出 acceptance.json
egoworld smoke --config egoworld/configs/full.json \
  --input-video /absolute/real-video.mp4 --stages sam2
egoworld smoke --config egoworld/configs/full.json \
  --input-video /absolute/real-video.mp4 --stages hamer
egoworld smoke --config egoworld/configs/full.json \
  --input-video /absolute/real-video.mp4 --stages foundationpose
egoworld smoke --config egoworld/configs/full.json \
  --input-video /absolute/real-video.mp4 --stages dex_retarget
egoworld smoke --config egoworld/configs/full.json \
  --input-video /absolute/real-video.mp4 --stages fast3r

# 或用 pytest 逐项报告；失败不停止其余测试
EGOWORLD_NATIVE_TEST=1 EGOWORLD_NATIVE_CONFIG=egoworld/configs/full.json \
  EGOWORLD_NATIVE_VIDEO=/absolute/real-video.mp4 pytest -q -m native_models
```

16GB 服务器可在上述 pytest 命令中增加 EGOWORLD_NATIVE_PROFILE=16gb；
直接跑整图可用 smoke --profile 16gb --seconds 2。具体限制见 [16GB 操作说明](server.md#7-16gb-单卡-smoke)。
本地 test_smoke_profiles.py 用真实视频/Parquet/恢复和明确的模型替身验证 profile，不能视为已测显存。
Fast3R 的测试同时覆盖当前上游的字典返回、旧版二元组返回、device.type 和 precision 字符串，防止替身再次掩盖接口错误。

默认截取 3 秒，最多 15 秒，强制单 GPU、关闭 SAM2 编译。样本需有可见目标与可靠左右手标签。
FoundationPose 的 assets manifest 必须以 make-manifest 生成的 video_id 为键；深度编号使用原视频源帧号。
DexRetarget 默认目标右手，仅处理右手行；更换目标手需匹配 YAML/URDF。

验收要求：阶段输出非空且通过 schema/数值检查；保存环境与耗时记录；同一 run-id 再次运行不重写阶段 Parquet。
这能证明小样本推理与恢复；不能证明模型准确率、长时间稳定或大规模性能。

## 4. 完整单卡与服务器

完整单卡：省略 `--stages` 运行 `smoke --config .../full.json`。无需先通过所有单阶段测试才运行，
可并行定位不同阶段的依赖和资产问题。

服务器：替换 server.json 中的模型/资产/目录，根据卡数调整 num_gpus 和 gpu_stage_groups，执行：

```bash
egoworld run --config egoworld/configs/server.json \
  --video-manifest ./manifests/video_manifest.jsonl \
  --clip-manifest ./manifests/clip_manifest.jsonl --run-id server_trial_001
egoworld inspect --run-dir ./output/run_id=server_trial_001
```

先取 2–3 个片段确认结果；然后增加输入规模，而非先重写架构。
记录 GPU 型号/显存峰值、样本分辨率/时长、各阶段耗时、空结果率、失败原因、磁盘与实际处理速率。
`run_summary.json` 的 scheduled_frames_per_wall_second 包含启动时间与重叠帧，不能当作去重视频吞吐。
中断进程后用同一 run-id 重启，检查已提交阶段复用；用新 run-id 比较不同模型/采样设置。

当前最后门槛是：五阶段真实小样本结果 + 完整单卡运行 + 一次中断恢复。
对普通 RGB 配置，该门槛只覆盖启用的四阶段；FoundationPose 在已有 RGB-D/网格时单独加入，不要求新采集硬件。
原生 CUDA/模型故障和服务器性能必须保留实测记录；没有资源时保持待测，不阻塞代码推进。

## 5. GPT-6 时代如何让验证有效、独立

先区分两件事：**有效**是检查能发现真正重要的错误；**独立**是预期结果与判断不完全由实现者自己的假设产生。
换一个模型或新对话能减少上下文偏见，不能保证不同审查者不会犯相同错误。

### 官方 OpenAI 文档给出的依据

[Evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)建议早做任务相关评测、
使用真实分布样本、保留日志并用人工判断校准自动评分，反对凭感觉判断效果。
[Codex 开发工作流示例](https://developers.openai.com/cookbook/examples/codex/iterating-development-workflows-with-codex)
要求区分实际观察与历史报告，保留失败/跳过和验收证据，不从代码存在推断完成。
[官方 code review 说明](https://learn.chatgpt.com/docs/code-review)提供独立 reviewer 的 /review 入口，检查 diff 并报告问题，不修改工作区。
这些是官方建议/工具能力；下面的 EgoWorld 验证流程是本项目的工程建议，不是 OpenAI 的强制流程。
无需为此接入或付费调用 Evals/模型评分平台。

### 最少但有约束的工作流程

1. **先定预期。** 为本次输出/行为写 3–5 条验收条件和边界案例，明确允许误差、缺失与失败行为；
   不让实现代码反过来定义什么叫正确。开发可继续推进其它已确定部分。
2. **实现者自测。** 实现和单元测试放一起检查，快速发现局部问题；此时结果标“自测通过”。
3. **换干净上下文审查。** 只提供需求、完整候选快照、运行方式和资产说明，先不给“已经全部通过”的结论。
   审查者先从需求导出检查，再看 diff/相关文件，重点找反例；不同时修改验收条件和实现。
   新增但未跟踪文件也必须纳入快照，仅保存普通 git diff 会漏掉它们。
4. **用外部依据判断。** 数据格式用独立读取器打开；几何用手工可算例子、标定/标注或官方 demo 对照；
   原生集成使用固定上游版本和真实权重。模型生成的解释不能当数值真值。
5. **留可复验记录。** 固定代码/资产/配置身份，保存命令、exit code、原始输出、通过/跳过数量与输出文件。
   实现者可以修复错误，审查者重跑原验收；修改需求需单独说明原因。

/review 提供审查角色，但不会自动完成几何真值、资产准备或服务器性能实验。
换模型只是补充；你主要审核需求、关键样例和不可验证的限制，不必逐行阅读全部代码。

### 对本项目，哪些证据可以做“裁判”

| 验证对象 | 有效且较独立的依据 | 目前证据与还需补什么 |
| --- | --- | --- |
| 时间与索引 | 已知帧号测试视频，独立读取抽帧，逐帧对照 | 有真实 FFmpeg 测试；还需实际数据/VFR 转换验证 |
| 手关节位置/旋转 | 人工可算的坐标/旋转；官方 demo 同输入；人工检查少量手/指尖与左右手 | 有 API 替身与简单数值检查；真实裁剪、投影、手型/尺度未验证 |
| 命名手指角 | 明确定义后，用伸直、直角等已知骨架作期望，不调用实现函数生成答案 | 当前未实现角度导出，需按所选定义增加 |
| 物体姿态 | 已标定 RGB-D、已知物体位姿或可重投影的网格/坐标轴；误差定义需考虑对称物体 | 有控制流/矩阵合法性；未做真实精度验证 |
| 机器人关节 | 目标 URDF 关节限位与正向运动学、静态手势和序列状态隔离 | 有优化器替身；真实机器人配置/FK 仍待测 |
| 相机/点云 | 已知相机运动/标定、重投影；无米制真值时明确尺度对齐规则 | 有 PnP 错误和 schema 测试；没有真实重建精度证据 |
| 故障恢复 | 独立杀进程/损坏文件后，核对提交文件、最终状态和重算集合 | 部分已做；长时间任务、强杀 driver、真实 CUDA 故障待测 |
| 吞吐 | 相同输入、输出和资源的外部计时，CPU/VRAM/磁盘测量，分开启动和稳态 | 只有本机调度/IO 探针；服务器性能待测 |

骨架可按已知变换构造期望；不得用被测函数自身算 expected。
输出非空、矩阵正交或 schema 合法只证明有限条件，不能单独证明语义/精度正确。

### 如何知道测试本身不是摆设

在临时副本中注入少量有意义的错误，预期相关验收必须失败：

| 故意注入的错误 | 应当失败的检查 |
| --- | --- |
| 时间戳错一帧 | 帧/时间对齐与实际画面对照 |
| 左右手互换或关节顺序打乱 | 左右手样例与命名关节检查 |
| 米/毫米错 1,000 倍 | 标定距离/单位契约，不仅检查是否有限数 |
| 机器人不同手共用平滑状态 | 独立序列与重置检查 |
| 输出文件截断、漏点云或上游提交缺失 | 读回/恢复/依赖检查 |
| 偷偷减少采样帧数提升“吞吐” | 输出覆盖率与相同工作量对照 |

这类“故意破坏再检查是否发现”的方法叫变异测试。表中是建议的验收实验，**尚未执行一套完整变异测试**；
无需引入大型新框架，先针对最关键的 3–5 类错误做。只追求测试数量或覆盖率不能替代它。

### 模型效果的小样本集

按最终需求准备约 6–12 个短片段，覆盖左右手、遮挡、多物体、无目标、运动模糊等实际情况；
其中一部分用于调参，留一部分固定为验收样本，避免每次只在已修好的视频上宣布成功。
只抽查关键帧也可开始。人工挑选/标注的语义依据独立于实现者；验收样本一旦用于反复调参，就不能继续当未见过的证据。
该规模适合发现明显问题，不支持泛化精度或真实失败率 <1% 的统计结论。

所有结果分四栏记录：**自测已过 / 独立复验已过 / 真模型已过 / 性能已测**，可同时存在待测项。
GPU 不可用时后两栏保持待测，继续开发和验证已确定的数据流、错误处理与其它独立需求。

## 6. 服务器调度与测量回归

设计目标为 2–16 卡单服务器；16GB smoke 不代表生产吞吐配置。

```bash
pytest -q egoworld/tests/test_scheduling_resources.py egoworld/tests/test_external_runner.py
EGOWORLD_RAY_TEST=1 RAY_USAGE_STATS_ENABLED=0 pytest -q -m ray_integration
PYTHONPATH=egoworld/src python egoworld/scripts/benchmark_cpu.py \
  --output /tmp/egoworld-cpu.json --repeats 3
```

新增回归覆盖：逻辑 2/16 GPU 槽位并发、写盘重试在生产预算已满时继续排空、预算不可能满足时明确失败、
准备文件高水位、等待过久阶段让位、常驻进程跨片段复用、模型异常/超时释放、SIGTERM 清理、
profile 只读汇总和多 GPU 硬件清单。逻辑 GPU 测试使用 CPU 替身，不能作为真实多卡性能证据。

微基准应在回归结束后单独运行，避免本项目的并发测试争用 CPU；记录脚本哈希、实际源文件哈希、依赖版本、
样本类型和原始中位数。结果字节估计本身也有 CPU 成本，脚本单独测量它。
进程 CPU 时间不包含 FFmpeg 等其它子进程的 CPU 时间，RSS 是单进程生命周期高水位；
二者不能直接当整个服务器的资源需求。真实 GPU 峰值和完整输出吞吐仍需服务器短样本测量。

本机记录（2026-10-25，macOS/ARM、Python 3.13.2、Ray 2.59.0）：完整回归 **143 passed / 11 skipped，97.04 秒**。
包含真实 Ray 进程测试；11 项跳过均因缺少 GPU/真实模型条件。另用实际 CPU 集成运行验证 profile CLI，
禁止导入 Torch/Ray 后仍可读回并导出相同 JSON。Linux 目标环境及真实单/多卡仍待服务器或 CI 复核。

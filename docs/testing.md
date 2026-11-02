# 测试与快速落地

所有命令在仓库根目录执行。默认测试自动跳过缺少硬件的模型测试；显式开启后，缺少资产/依赖应失败，不能算通过。
各分支可独立开发与测试，无需等待上一分支 GPU 验收。

当前范围已经确认，包含 SAM2/GroundingDINO；不接新传感器/复杂新模型。服务器配置和统一 launch 入口见 [server.md](server.md)。
本地启动专项：pytest -q egoworld/tests/test_server_startup.py，覆盖五阶段资产、工作目录、分环境预检协议、CLI 启动与恢复。

## 2026-11-01 数据配方、腕部回放与米制世界导出

按用户新增范围实现 `egoworld data`：按数据类型/字段选择输出，补齐依赖，复用现有模型后端，
以校验后的阶段副本实现跨 run 缓存，并增加有标定依据的世界坐标派生。操作和模板见 [data-recipes.md](data-recipes.md)。
本轮未编辑 `docs/design.md` 或 `docs/handoff.md`，调度/GPU 探针仍由另一 worktree 负责。
产物根目录为 `/mnt/d/EgoWorld/runs/product-20261101`，以下均为实际运行或明确标注的数值测试。

| 检查 | 实测结果 | 证据（相对根目录） |
| --- | --- | --- |
| 上游腕部约定 | 新增可选 `input_transform: dex_wrist`；与固定版本上游参考函数同输入数值一致，未重复应用 HaMeR 朝向 | `wrist-validation.json`；同一六帧真实双手预测，右手贴边值 30/96 → 6/96，左手 20/96 → 0/96；两侧严格限位/FK 通过 |
| 动作回放 | 以真实 HaMeR 预测与 URDF FK 绘制六帧、2 fps 的左右手回放 | `wrist-fk-replay.mp4`、`wrist-fk-preview.png`；表示运动学输出，不代表接触、动力学或实机控制 |
| 新动作场景 | 再从官方固定版本预览提取炒鸡肉、压咖啡片段，每段 3 秒/30 帧；所选实例左右各六帧 HaMeR，右手机器人六帧 | `multi-scene-source.json`、`chicken-hands.png`、`aeropress-hands.png`、`native-audit.json`；两个新场景的机器人限位/FK 读回通过 |
| 增加数据类型 | 先输出框/掩码，再选择人手/机器人；两个场景分别命中 SAM2 缓存，仅计算 HaMeR/Dex | `data-runs/run_id=verified_chicken-hands/data_manifest.json` 和 `run_id=verified_aeropress-hands/data_manifest.json` |
| 更换机器人 | 最终代码先运行右手配方，再改左手目标，SAM2/HaMeR 均命中缓存，仅计算 Dex；左手限位/FK 通过 | `data-runs/run_id=release_chicken-left-robot/data_manifest.json`、`checks/chicken-left.json` |
| 米制世界真实组合 | 现有 mustard RGB-D，SAM2/FP/Fast3R 原生模型；3 视图、512 输入，在 8GB GPU 完成；导出同一 world_id 的 3 行相机、3 行物体姿态和 5,563 点 | `data-runs/run_id=verified_rgbd-world/data_manifest.json`、`world-preview.png`；独立读回未重复缩放 FP 的米制位移 |
| 标定质量门槛 | 1,527 个有效高置信度对应；3cm 内点门槛、最小内点率 50%；预留对应内点率 63.4%，其内点 RMSE 1.67cm | 对应运行的 `calibrations/*.json`、`native-audit.json`；置信度先保留有效对应的上半部分，再划分拟合/预留点；RMSE 不包含外点，不是全图或轨迹精度 |
| 仅改世界导出需求 | 三个模型 kernel 全部命中缓存，computed 为空；所有原始 Parquet/NPZ 哈希与首遍一致 | `data-runs/run_id=verified_rgbd-world-only/data_manifest.json`、`native-audit.json`；不启动 Ray/模型的路径另有拒绝调用的 CPU 反例 |
| 回归 | **182 passed / 11 deselected，76.77 秒，退出 0** | `regression.log`、`regression.xml`、`regression.exit`；开启 `EGOWORLD_RAY_TEST=1`，其余模型 pytest 标记排除；上表真模型工作没有混入这个计数 |

已查看两组手部重投影、世界散点/残差图以及 FK 回放。GroundingDINO 在新场景将袖口/较大手臂框也标为手，
因此新增显式 `object_ids` 白名单：炒鸡肉选择 1/2，压咖啡选择 1/4；手别也是人工确认输入。
双手交叠的压咖啡画面中投影高度重合，无法仅凭图片确认每只手的完整关节都正确，该例**感知质量记为待审**；
执行/数值检查通过不等于遮挡下双手分离精度通过。样例同属一个官方预览来源，不是跨主体标注基准。

米制导出绑定到首个 RGB-D 相机锚点；后续相机仍为 Fast3R 估计，不提供跨片段统一地图或轨迹真值精度结论。
原始模型表没有改写。当前 RGB 手样例没有相机系度量手关节标定，`world_hand` 分支拒绝缺测量的输入；
测量标定的正例、先验冒充米制的拒绝反例、哈希变化失效、重复缩放与跨帧误配的反例属于明确的合成几何测试。

失败探索同样保留：224 输入的全对应无法通过标定门槛；512 全对应也未通过，最终显式按模型置信度筛选，未放宽 3cm 门槛。
初次提示 `mustard bottle .` 与网格键 `bottle` 不匹配，导致 FP 空表，该试跑未记为联合通过。
诊断文件为 `alignment-trial-diagnostics.json`、`confidence-alignment-diagnostics.json` 和 `rgbd-smoke.log`。
产品入口现在将请求的空导出记为 partial 并返回非零状态。

源码与配方快照、原始证据清单保存于本轮 D 盘目录；各真实运行的 manifest 保留其实际源码身份。
`source-snapshot/` 为最终代码，`native-source-snapshot/` 保留空输出状态修正前的原生配方版本；
`release_*` 记录使用最终代码。与另一 worktree 合并后仍须做统一验收，不将本轮结果外推为合并版本通过。

## 2026-11-01 Dex 严格限位与左右手复现

本轮继续推理复现，未训练模型、迁移 WSL 或发起远端服务重启。`docs/handoff.md` 保持远端传入快照。
新增样本、配置、预测、日志和预览位于 `/mnt/d/EgoWorld/runs/dex-hands-20261031`；目录沿用本轮开始时的日期。
下节保留修复前结果，本节记录修复后的独立复验。

Dex 越界来自 dex-retargeting 0.4.6 默认 `set_joint_limit(..., epsilon=1e-3)`。
对启用限位的 YAML，适配器通过公开 API 设置 `epsilon=0.0`，随后按关节名称核对原机器人限位。
不超过 `1e-6` 的舍入误差才校正，更大越界失败；显式关闭限位的目标配置仍保留其原设置。
覆盖左右手、打乱关节顺序、舍入误差与真实越界的反例已加入 `test_model_adapters.py`。

| 检查 | 实测结果 | D 盘证据（相对上述目录） |
| --- | --- | --- |
| 固定输入的修复前后对照 | 沿用旧 RGB 链路同一份真实 HaMeR 预测；6 行 × 16 个右手关节，越界值从 30 降为 0；独立 XML 限位与 Pinocchio FK 通过，重置重复一致 | `right-fixed/comparison.json`、`right-fixed/kinematics-check.json`；优化器重新求解后的最大角度变化 0.01447 rad，最终舍入修正最大仅 5.34e-8 rad |
| 真实左手裁剪 | 3 秒 / 30 帧，原视频真实左手裁剪、无镜像；30 行非空实例，HaMeR 6 行，Allegro 左手映射 6 行 × 16，推理及同 run 复跑通过 | `samples/source.json`、`left-smoke/20261031_172248_4bfe7afa/acceptance.json`、`left-tracking.png` |
| 原画面双手 | 30 帧均有两个非空跟踪实例；HaMeR 左右各 6 行，帧索引均为 0/5/10/15/20/25；两侧各用对应 YAML/URDF 得到 6 行 × 16 关节 | `both-smoke/20261031_172518_aeb78421/acceptance.json`、`both-identity-check.json`、`both-tracking.png`；单个 Dex 阶段仍只映射配置指定的一侧，另一侧独立复用同一 HaMeR 预测 |
| 左右手机器人产物读回 | 左手裁剪、双手画面右侧、双手画面左侧，三组均为 0 越界、FK 有效；从混合输入与分离输入得到同侧结果一致，重复调用也一致 | `dex-hands-validation.json`、`left-left/kinematics-check.json`、`both-right/kinematics-check.json`、`both-left/kinematics-check.json` |
| 官方逐行数值对照 | 独立上游 checkpoint forward 和 `cam_crop_to_full` 核对全部 18 行，包括真实左手镜像、腕部偏移和双手实例关联；相机空间最大差异 7.45e-9 米，投影最大差异 4.33e-6 像素 | `hamer-official-comparison.json`；已查看 `previews/left-smoke.png` 和 `previews/both-smoke.png`，这是同输入数值复现，不是姿态真值精度 |
| 双手作业意外中断后的恢复 | 执行中断后沿用同一配置与作业恢复 HaMeR/Dex；原已提交的 4 个 SAM2 文件哈希和时间戳不变，完成后再次复跑也不改写 Parquet | `both-interruption-check.json`、`both-recovery.log`；本轮没有发起服务重启 |

本轮 GroundingDINO 使用通用提示 `hand .`，因为实测左右手文本标签不可靠，不能把检测出的 `right hand` 当作真实手别。
`handedness_by_id` 是对跟踪实例的人工确认输入：左手裁剪 `1 → left`，双手 `1 → right / 2 → left`。
已在源帧/掩码联系图中检查六个采样帧；没有把人工确认计作自动手别识别通过。
两个验证用例来自同一公开视频场景，不能代表跨场景召回、遮挡交叉或长序列稳定性。

限位合法仍不代表动作映射准确：三组映射分别有 18/96、30/96、20/96 个值距离限位不超过 `1e-6` rad。
当前 `human_to_robot_rotation` 为单位矩阵，下一步应按上游腕部坐标约定复核并做映射回放；
证据为 `limit-saturation-review.json`。未验证机器人控制、碰撞或跨模型度量融合。

复跑入口（仓库根目录，使用本机现有资产；`smoke` 会创建新的时间戳目录）：

```bash
. config.local/reproduction.env
DEX_HANDS_RUN=/mnt/d/EgoWorld/runs/dex-hands-20261031
.venv/bin/egoworld smoke --config "$DEX_HANDS_RUN/left-config.json" \
  --input-video "$DEX_HANDS_RUN/samples/epic-left-3s.mp4" --seconds 3 \
  --prompt 'hand .' --output-dir "$DEX_HANDS_RUN/left-smoke"
.venv/bin/egoworld smoke --config "$DEX_HANDS_RUN/both-config.json" \
  --input-video data/samples/epic-dough-3s.mp4 --seconds 3 \
  --prompt 'hand .' --output-dir "$DEX_HANDS_RUN/both-smoke"
# 独立复验本轮已冻结的三组机器人输出
.venv/dex/bin/python "$DEX_HANDS_RUN/verify-dex-hands.py"
```

`egoworld/scripts/verify_hamer_output.py` 在 HaMeR 环境中通过可重复的 `--acceptance` 参数检查全部保存行，
`--output` 保存数值对照，`--preview-dir` 保存重投影图。D 盘 `source-snapshot/` 保存本轮实现与验证脚本快照。

完整选定范围回归为 **166 passed / 11 deselected，74.87 秒，退出 0**，
命令与此前一致，显式开启 `EGOWORLD_RAY_TEST=1`，排除需要单独输入/资产的模型 pytest 标记。
日志、JUnit 和退出码为 `regression.log` / `regression.xml` / `regression.exit`。
上述真实 GPU/优化器验证单独记录，没有算入 166 项。

## 2026-10-31 HaMeR 与 RGB 四阶段复现

按用户要求继续原推理复现任务；新增 checkpoint 与本轮运行产物使用 D 盘，源码/环境保留原工作路径。
从已核验的 45 个分片直接读取压缩流，在 `/mnt/d/EgoWorld/models/hamer` 提取三项必要资产，
未重新下载或创建完整合并大包。checkpoint 为 2,689,536,166 bytes，SHA-256
`e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53`。

本节证据根目录为 `/mnt/d/EgoWorld/runs/reproduction-20261031`，均为本机实际运行产物：

| 检查 | 实测结果 | 证据与边界 |
| --- | --- | --- |
| HaMeR 分环境预检 | `doctor --gpu --assets` 退出 0 | `doctor-hamer-final.log` / `.exit`；不以预检替代模型推理 |
| SAM2/DINO → HaMeR | 3 秒 / 30 源帧，stride 5，batch 1；输出 6 行 × 21 个手关节；同 run 复跑恢复通过 | `hamer-smoke/20261031_094155_2be78a2f/acceptance.json`；仅检测到的右手，Torch 峰值保留显存 5,641,338,880 bytes |
| 官方同输入对照 | 独立调用上游 checkpoint、ViTDetDataset 和 `cam_crop_to_full`，相机空间最大差异 7.45e-9 米、投影最大差异 4.26e-6 像素 | `hamer-official-comparison.json`；核对同一个真实准备帧和手框，这是数值复现一致性，不是真值精度 |
| 手姿态产物读回 | 6 行数值有限，腕部归零，16 个旋转矩阵合法；最大正交误差 2.43e-7；已查看实际关节重投影 | `hamer-output-check.json`、`hamer-preview.png`；全图焦距是模型估计，不能把相机平移当作场景标定距离；另一只手未覆盖 |
| RGB 四阶段组合 | SAM2/DINO → HaMeR → Dex，加上 Fast3R，同一 8GB GPU 及独立 Dex CPU 环境；推理与复跑恢复通过 | `rgb-chain-smoke/20261031_094401_5401cd54/acceptance.json`；阶段分别 150 / 6 / 6 / 2 行，Fast3R 有 16,726 点；首遍流水线 56.169 秒，不是服务器吞吐基准 |
| 视频 → 机器人关节 | HaMeR 真实视频预测作为输入，得到 6 行 × 16 Allegro 右手关节，FK 数值与旋转有效 | `rgb-chain-kinematics.json`；严格限位检查为 review / 退出 1，30 个关节值越界，最大约 0.000999986 rad；保留上游原输出，不裁剪或声称机器人控制合格 |
| 四阶段真实中断恢复 | 首个 HaMeR 提交后只向本次处理 driver 发送 SIGINT，同 run-id 恢复；最终 3 个一秒片段、每个阶段 3 个提交均完成 | `rgb-chain-interruption-check.json`；中断前已提交的 9 个文件 SHA-256 和 mtime 全部不变；未重启 Codex/WSL 服务。报告 `rgb-chain-interruption-report.html` |

当前是普通 RGB 四阶段整图，加上此前独立 FoundationPose RGB-D/网格样例，五个处理阶段均有真模型证据。
没有声称同一视频同时跑过五阶段、跨模型米制融合、双手召回合格或基准准确率。
离线报告为 `rgb-chain-report.html`。机器人限位复核与流水线执行验收分别记录，不能将 QC pass 当作物理关节合规。

本机复跑命令（在仓库根目录，使用已准备好的本地资产）：

```bash
. config.local/reproduction.env
.venv/bin/egoworld smoke --config config.local/reproduce-rgb-chain.json \
  --input-video data/samples/epic-dough-3s.mp4 --seconds 3 \
  --prompt 'left hand . right hand . dough .' \
  --output-dir /mnt/d/EgoWorld/runs/reproduction-20261031/rgb-chain-smoke
```

本轮未改生产代码，沿用此前完整 CPU/Ray 回归的 **160 passed / 11 deselected**；
上表模型 smoke、官方对照和读回是另行执行的真实验证，不计入 160 项。
本轮模型及恢复所需的本地配置快照、代码差异和环境导出保存在 D 盘证据根目录的 `source-snapshot/` 与 `*-pip-freeze.txt`；
模型二进制未纳入 Git。后续优先处理真实试用发现的输入/输出问题；双手覆盖、严格机器人限位、几何真值精度和长序列稳定性仍需单独验证。

## 2026-10-29 Linux/WSL2 复核

本节为修复前的初次审查记录。源码 `e0638a23553d8be935c334248bc3fc539d918ba0`；该次仅同步、审查和更新文档，未修生产代码。
Linux `6.18.54.1-microsoft-standard-WSL2`，Python 3.12.3、pytest 8.4.2、Ray 2.59.0、
NumPy 2.5.4、PyArrow 26.0.0、OpenCV 4.14.0，静态 FFmpeg/ffprobe 7.0.2。
环境为 `/tmp/egoworld-audit-20261029`，工具在 `/tmp/egoworld-ffmpeg-static`；未复用其它项目环境。
本机 GPU 为 RTX 4060 Laptop 8188 MiB；本轮仅使用逻辑 GPU 调度资源，无真实 CUDA 推理。

实际命令（仓库根目录，临时取消了不可用的 HTTP 代理）：

```bash
PATH=/tmp/egoworld-audit-20261029/bin:/tmp/egoworld-ffmpeg-static:$PATH \
EGOWORLD_RAY_TEST=1 RAY_USAGE_STATS_ENABLED=0 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest -q \
  -m 'not env_smoke and not pipeline_smoke and not sam2_smoke and not native_models' \
  --junitxml=output/review-20261029/pytest.xml
```

结果：**149 passed / 3 failed / 11 deselected，62.62 秒，退出码 1**。
其中非 Ray 的 146 项通过；6 项 Ray 测试中 3 项通过、3 项失败。排除项不计为通过。

| 检查 | 本次观察 / 下一步 |
| --- | --- |
| `test_real_ray_all_branches_and_resume` | 失败：runner 日志报 `ModuleNotFoundError: numpy`。原配置的 `/tmp/egoworld-audit-20261029/bin/python` 经 `config.py` 的 `Path.resolve()` 变成 `/usr/bin/python3.12`，脱离 venv；需保留解释器符号链接语义并补子进程反例 |
| `test_external_process_group_stops_when_actor_exits[close]` / `[kill]` | 均失败：5 秒观察期后外部 runner 仍是活进程，报 `External runner survived actor exit`；测试 finally 清理了该进程组。需修复 Linux actor/guard/runner 生命周期，不能放宽存活断言 |
| 非 Ray 外部进程专项、常驻复用与 Ray worker 替换 | 已在上述通过项内；这些通过不能覆盖 actor 退出后的两个失败路径 |
| 隐式资产身份探针 | 临时合成 HaMeR checkpoint、模型配置和 MANO 文件；修改 MANO 和 `model_config.yaml` 后指纹均未变，哈希资产只含 checkpoint。需显式收集实际依赖，并验证同 run 拒绝旧结果 |
| CLI / 静态检查 | CPU doctor、full plan、schema 成功；77 个 Python 文件 AST、5 个 JSON 配置和 3 个 shell 脚本语法检查通过 |
| 默认分割资产检查 | `doctor --assets` 退出 1：缺 SAM2/DINO 权重与两份模型配置；无真实输入样本与模型环境，不将此计作模型 smoke 失败或通过 |

已核对 [主分支 CI 40721327295](https://github.com/LyrineYang/egoworld/actions/runs/40721327295)：
源码同为 `e0638a2`，安装步骤成功，pytest 步骤失败。公开 API 提供了状态/步骤；日志接口返回 403，
本机 GitHub CLI 凭据失效，未取得完整失败用例，不能从本地结果推断托管 CI 的精确根因。

原始证据保存于 `output/review-20261029/`：`pytest.log`、`pytest.xml`、`pytest.exit`、`pip-freeze.txt`、
`doctor-cpu.json`、`doctor-assets.json`、`plan.json`、`schema.json`、`implicit-assets.json` 和 CI API 快照。
该目录被 Git 忽略，临时环境会随 `/tmp` 清理而失效；后续修复须建立持久独立环境，并再次运行原失败反例与完整选定范围。
该次没有下载真实模型、运行 CUDA、测量服务器吞吐或修改既有验收标准；后续本机推进结果见下一节。

## 2026-10-29 本机推理复现

当前任务是既有模型推理复现，不训练、不开发调度算法。远端传入的 `handoff.md` 保持原样。
使用 WSL2 / RTX 4060 Laptop 8188 MiB，项目持久环境 `.venv`，独立环境 `.venv/dex` 与 `.venv/foundationpose`；
真实产物、原始命令/日志及各环境版本在 `output/reproduction-20261029/`（Git 忽略）。

修复了 runner Python 符号链接丢失虚拟环境、Ray 先杀 guard 导致外部 runner 泄漏，以及默认 HaMeR 隐式资产未进入指纹。
在加入 Python 头文件、固定 NumPy 1.26.4 / OpenCV 4.11 / PyArrow 19.0.1 后，完整 CPU/Ray 范围为
**157 passed / 11 deselected，69.04 秒**。日志 `regression-fixed-env.log` / XML；先前不兼容 PyArrow 26 的失败日志也保留。
新增直接系统 SIGKILL actor 的回归；未放宽原进程存活断言。托管 CI 未在本轮重新运行。
随后真实 FoundationPose 暴露分环境缺少压缩 COCO RLE 解码器，补入项目依赖和原生预检，并增加缺失依赖反例。
最终同一 CPU/Ray 范围 **158 passed / 11 deselected，72.05 秒，退出码 0**，见 `regression-final.log` / XML / `.exit`。
11 项真模型测试未纳入这个 pytest 命令；下表为独立实际执行的模型 smoke，不能混作 158 项的覆盖。

| 路径 | 实际结果 | 证据与边界 |
| --- | --- | --- |
| SAM2.1 small + GroundingDINO Swin-T | 3 秒 / 30 帧，60 条非空实例；mask/实例/轨迹合计 150 行；同 run 恢复通过 | `sam2-smoke/*/acceptance.json`；QC 帧覆盖率 1.0；PyTorch 峰值保留显存 2,057,306,112 bytes；预览 `sam2-preview.png` 显示只检出一只手和一块面团，另一只手漏检 |
| Fast3R | 224×224 两视图，2 行重建相机，16,726 个有限数值点；同 run 恢复通过 | `fast3r-smoke/*/acceptance.json`；相对尺度 `unscaled`；PyTorch 峰值保留显存 4,148,166,656 bytes；点云预览 `fast3r-pointcloud.png`，非几何精度基准 |
| 两阶段组合 | 同一 GPU 顺序执行原生模型进程，两个启用阶段均通过；复跑不重写结果 | `combined-smoke/*/acceptance.json`；只有 SAM2/DINO 与 Fast3R，不是五阶段全部通过 |
| 真正中断恢复 | 三个 1 秒片段，首个阶段提交后给 driver SIGINT，再用同 run-id 启动；最终三个片段完成 | `interruption-check.json`、`interrupt-first.log` / `interrupt-resume.log`；中断前 4 个提交文件的哈希和 mtime 全部保持；报告 `available-rgb-report.html` |
| DexRetarget | 官方手关节样例经真实优化器得到 30 行 × 16 个 Allegro 右手关节（radians），重置后重复结果一致 | `dex-sample/acceptance.json` / `mapping.parquet`；使用上游 `human_joint_right.pkl`，不是 HaMeR 预测或 RGB 整链通过；源数据未提供时间，因此输入采用显式的逻辑 1 Hz 索引 |
| FoundationPose | 官方 mustard 的 3 组 RGB-D 与米制网格，输出 3 行 `object_to_camera` 姿态；复跑恢复通过 | `foundationpose-smoke/20261029_120339_aafab7f8/acceptance.json`；PyTorch 峰值保留显存 4,475,322,368 bytes；`foundationpose-preview.png` 为实际网格包围框重投影 |
| 三 GPU 阶段组合 | SAM2/DINO → FoundationPose，另加 Fast3R，同一 8GB GPU 顺序执行；推理及复跑恢复通过 | `available-three-smoke/20261029_121017_09b34cf9/acceptance.json`；3 帧，阶段分别 9 / 3 / 2 行，首遍流水线 43.763 秒；不是五阶段整链或吞吐基准 |

RGB 样本来自 [EPIC-KITCHENS/VISOR 官方公开预览](https://github.com/epic-kitchens/VISOR)，固定提交
`fce55699745b9abcee20f7b3366ff039c82660a6`，仅裁出底部中间的连续视角，保留原始画面并采样为 10 fps。
此预览不提供 benchmark ground truth；未采用已经覆盖彩色掩码的演示片。
`data/samples/sample-source.json` 保存源 URL、SHA-256、裁剪参数及产物哈希；可重新运行：

```bash
PATH="$PWD/.venv/tools:$PATH" .venv/bin/python egoworld/scripts/fetch_reproduction_sample.py
PATH="$PWD/.venv/bin:$PWD/.venv/tools:$PATH" \
  .venv/bin/egoworld smoke --config config.local/reproduce-sam2.json \
  --input-video data/samples/epic-dough-3s.mp4 --seconds 3 --stages sam2 \
  --prompt 'left hand . right hand . dough .' --output-dir output/reproduction-20261029/sam2-smoke
```

本机配置在 `config.local/reproduce-sam2.json`、`reproduce-fast3r.json` 和 `reproduce-rgb-available.json`；
模型源码与资产固定到实际 git/Hugging Face revision，BERT 使用本地离线快照。
`.venv` 使用 Torch 2.5.1+cu124 / torchvision 0.20.1；CUDA 扩展实际使用已有只读 nvcc 12.8 编译器和项目内解包的 Python 3.12 头文件，
构建日志、`pip-freeze.txt` 为本机环境证据。其它机器须匹配自己的 `CUDA_HOME` 和开发头文件，不能照搬本机绝对路径。

FoundationPose 的 PyTorch3D 0.7.9、NVDiffRast 0.4.0 与 mycpp 均在本机完成原生构建。该环境使用 NumPy 2，
不与 RGB/Dex 的 NumPy 1.26.4 环境混装。`model-identities.json` 保存源码提交及 FoundationPose 权重 SHA-256；
`foundationpose-pip-freeze.txt`、构建与失败/修复日志均保留。
公开 mustard 数据来源及原帧名见 `data/samples/mustard/source.json`；三组原始对齐 RGB-D 编码为名义 10 fps，
不是原始采集时间。深度毫米转米系数 0.001，网格单位为米。第一帧 SAM2 掩码对官方 demo 掩码 IoU 为 0.96796，
这是单帧分割重合度，不是姿态精度。没有位姿真值，不报告 6D 姿态误差；Fast3R 相对尺度仍独立保存。

本机三阶段重跑（只适用于当前已准备的本地环境和资产；`config.local` 不入 Git）：

```bash
source config.local/reproduction.env
env -u HTTPS_PROXY -u HTTP_PROXY -u ALL_PROXY -u https_proxy -u http_proxy -u all_proxy \
  .venv/bin/egoworld smoke --config config.local/reproduce-available-three.json \
  --input-video data/samples/mustard/mustard-3frames.mp4 --seconds 0.3 \
  --prompt 'bottle .' --output-dir output/reproduction-20261029/available-three-smoke
```

环境文件设置离线模型缓存、项目内动态库、FFmpeg 和线程上限，不依赖其它项目环境的运行库。
代理取消仅用于绕开本机失效代理；其它机器按自身网络配置处理。通用安装见环境说明，RGB 兼容约束见
`egoworld/env/reproduction-rgb-constraints.txt`，不是五阶段统一 lock。

HaMeR 的 `MANO_RIGHT.pkl` 按[上游安装说明](https://github.com/geopavlakos/hamer#installation)需注册取得。
此前按用户要求暂时跳过该分支，停止过公共大包下载。2026-10-30 用户注册完成并授权在 WSL 登录获取后，
已从官方认证下载页取得 `mano_v1_2.zip`（175,200,815 bytes），只提取 `MANO_RIGHT.pkl`（3,821,356 bytes）
到 `external/hamer/_DATA/data/mano/`，未修改模型内容。无需经聊天传文件；Windows 已下载的文件也可从 `/mnt/c/Users/Oscar/Downloads/` 直接读取。
MANO SHA-256 为 `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767`，
来源记录在 `models/mano/source.json`；账号密码和会话 Cookie 未写入项目文件。

实际 MANO 初始化暴露 Chumpy 缺失；从官方 `mattloper/chumpy` 提交
`580566eafc9ac68b2614b64d6f7aaa84eebb70da` 构建 0.71，无需修改旧模型或使用全局兼容补丁。
真实 HaMeR MANO 包装器已完成中性姿态前向，输出 778 个顶点和 21 个关节，数值有限；
证据为 `output/reproduction-20261030/mano-runtime-check.json` / `.log`，仅证明资产可加载，不是 HaMeR checkpoint 推理通过。
公开 checkpoint/配置/均值参数大包正在从上游大学镜像分段续传下载；资产齐备后再验证手型/左右手和 HaMeR → Dex 视频整链。

本次真实 MANO 还暴露了手根坐标问题：中性姿态腕部相对 MANO 模型原点约为
`[0.09567, 0.00638, 0.00619]` 米。原适配器从关节减去腕部，却未给相机平移加回该偏移，
因此不能直接用导出的 `joints_3d + camera_translation` 还原官方相机空间结果。现已修复，左手偏移使用相同的 x 镜像；
左右手解析反例检查相机空间点保持不变。Chumpy 与均值参数缺失也纳入预检。
最新完整 CPU/Ray 回归 **160 passed / 11 deselected，78.47 秒**，
记录在 `output/reproduction-20261030/regression-hamer-wrist.log` / XML / `.exit`；该数量不包含真模型 smoke。

### 2026-10-31 中断后的保全与恢复点

用户负责处理宿主磁盘空间和 WSL 异常，本轮只核对、备份和准备恢复，不启动大文件提取或模型推理。
此前观察到 C 盘剩余 0 bytes，WSL 根挂载进入 `emergency_ro` 并报 I/O 错误；重新连接时根挂载已恢复正常，
C 盘约有 3.5 GiB 可用。未修改系统挂载、WSL 配置或清理已有代码/模型；没有本任务残留下载或自动推理进程。

最新未提交代码、未跟踪脚本与选定本地配置已保存到 Windows 工作区的 `egoworld-code-backup-20261031.zip`，
ZIP 完整性检查通过，约 104KB；备份不含模型二进制或凭据。`handoff.md` 仍与 HEAD 相同。
原始 MANO 的 SHA-256 重新核对通过，数值与上文一致。

对 45 个下载分片进行流式读取：**6,037,554,929 bytes，gzip CRC 通过**；
完整包 SHA-256 为 `ccfb70abd672b64c3ea90891c808d4499cc36a37dd6cf86c561a665113aef11e`。
三项必要资产的字节数及 SHA-256 已从完整流读出，见
`output/reproduction-20261030/hamer-parts-integrity.json` / `.log` / `.exit`，无需重新下载。

合并后的 `models/hamer/hamer_demo_data.tar.gz` 实际只有 5,397,086,208 bytes；已提取的 `hamer.ckpt` 只有
629,145,600 bytes，完整 checkpoint 应为 2,689,536,166 bytes，因此两者均不可当作完整资产使用。
模型配置和均值参数已提取，所有现有文件均保留。待宿主磁盘/WSL 恢复后，从分片直接提取，不再创建另一份约 6GB 合并包：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  output/reproduction-20261030/recover-hamer-from-parts.py --extract
source config.local/reproduction.env
.venv/bin/python -u output/reproduction-20261030/run-hamer-reproduction.py
```

恢复脚本会先校验临时资产，再把原有不完整 checkpoint 改名保留，最后写资产就绪标记；当前只完成脚本语法检查，未执行提取。
后续工作依次为 HaMeR 原生预检、真模型 smoke、同输入的官方相机投影对照、RGB 四阶段 smoke 与 Dex 限位读回。
首次恢复需额外写约 2.7GB checkpoint，推理还需要运行余量；不按 WSL 虚拟磁盘显示的空闲量替代宿主剩余空间。

### 2026-10-31 D 盘存储规划与迁移准备

用户要求优先使用 D 盘。盘点时项目 `.venv` 约 26 GiB、`models` 约 16 GiB、`external` 约 2.2 GiB；
Ubuntu 的 `ext4.vhdx` 位于 C 盘，文件长度约 69.08 GiB。本机 WSL 3.0.3 的 `--help` 确认支持 `--manage --move`。
推荐将整个 Ubuntu 磁盘迁到 D，保留当前 Linux 路径、环境解释器、权限和源码引用；
[微软文件系统指南](https://learn.microsoft.com/en-us/windows/wsl/filesystems)也建议 Linux 工具链的工作文件保留在 Linux 文件系统。

| 物理位置 / 用途 | 规划与当前状态 |
| --- | --- |
| `D:\WSL\Ubuntu\ext4.vhdx` | 规划中的 Ubuntu 目标；项目仍从 `/home/lyrine/projects/egoworld` 访问，模型、环境、数据、输出都随磁盘迁移；尚未执行 |
| `D:\WSL\swap.vhdx` | WSL 交换文件目标；已生成保留原有配置的草案，尚未修改生效配置 |
| `D:\EgoWorld\backups` | 已复制并核验代码 ZIP、全部 45 个 HaMeR 分片和原始 MANO；C 盘原文件均保留 |
| `D:\EgoWorld\exports` | 已复制当前三阶段离线报告及实际分割/点云/物体姿态预览 |
| `D:\EgoWorld\tools` | `migration-plan.json`、`Move-UbuntuToD.ps1` 与交换文件配置草案 |
| `D:\EgoWorld\logs` | 迁移预检、模型备份校验和迁移前环境检查结果；实际迁移后继续保存日志 |

D 盘模型备份的串联 SHA-256 与此前完整大包一致，MANO SHA-256 也一致；记录在
`D:\EgoWorld\backups\hamer-assets-20261031\backup-verified.json`。
迁移脚本 PowerShell 语法分析与不带 `-Execute` 的只读预检均通过；D 盘当次空闲量足以容纳完整 VHDX 备份和移动后的磁盘。
迁移前基线检查也通过：解释器前缀、原始 MANO、45 个分片、handoff 原样以及 GPU 可见性均正常。

实际迁移会停止整个 WSL，影响 Ubuntu 中其它项目及当前连接，因此等待用户确认整体迁移范围。
脚本只在显式 `-Execute` 时运行；先在 D 盘备份整个 VHDX 并比较 SHA-256，再调用原生移动命令，随后检查路径与资产。
不使用注销发行版的命令。空间与配置身份在执行前重新检查；退出失败时保留备份供恢复。
已有轻量备份不是完整 VHDX 备份，后者将在停机迁移步骤中创建；当前不能宣称已经整体迁移或释放了 C 盘虚拟磁盘空间。

用户随后要求先审查连接影响、官方依据和迁移风险，尚未授权停机迁移。
核对 [OpenAI WSL 指南](https://learn.chatgpt.com/docs/windows/wsl)与微软文件系统指南后，
官方建议针对 Linux 工作文件保留在 Linux 文件系统，没有发现专门偏好 C 盘的指南；
[微软 FAQ](https://learn.microsoft.com/en-us/windows/wsl/faq#how-can-i-move-my-wsl-distribution-to-a-different-drive-or-location)
直接以 D 盘为跨盘迁移示例。整体迁往 D 是本机空间约束下的工程建议，本项目生成的 PowerShell 脚本不具有官方背书。

审查确认 C/D 都是物理 Disk 0 上的固定 NTFS 分区；D 备份不覆盖整块物理磁盘故障。
现有脚本使用 `wsl --shutdown`，会停止所有 WSL 发行版及共享 VM，影响范围大于当前 Ubuntu 会话。
当前 Codex 的 WSL/SSH 执行连接会中断，自动重连与任务自动续跑没有验证保证；应从独立 Windows PowerShell 发起维护。
脚本已调整为完整 VHDX 备份 SHA-256 比较通过后再改交换文件配置，明确停机范围，拦截 WSL UNC 工作目录中的执行，
并取消已迁移位置直接返回的分支，避免跳过验证。修改前版本保留；语法、只读预检、目录防护表达式及静态操作顺序检查通过。
尚未执行真实迁移或恢复演练；出错后保留备份并停止，需要人工审查恢复，没有自动回滚保证。
完整审查报告保存在 `D:\EgoWorld\migration-review-20261031.txt`。

### 2026-10-30 收尾复核

整理了本机运行环境 `config.local/reproduction.env`，三阶段 `doctor --gpu --assets` 退出码 0，
记录在 `doctor-available-three-final.json`。从实际三阶段运行导出 `available-three-report.html`；
未修改远端交接文件。修正本地 mustard 配置中从 RGB 配置继承的样本来源标签，历史运行快照保留原样。
按上述环境文件再次执行三阶段 smoke，实际结果为 **passed / resume_verified=true**，阶段行数仍为 9 / 3 / 2，
首遍流水线 53.335 秒。新结果在 `output/reproduction-20261030/available-three-smoke/20261030_122513_8ec18b97/acceptance.json`，
本次离线报告在 `output/reproduction-20261030/available-three-report.html`。
基础、Dex、FoundationPose 三环境的 `pip check` 均通过；后二者用基础 pip 的 `--python` 选项检查目标环境，
无需往隔离环境额外安装 pip。代码与本地配置快照保存于 `output/reproduction-20261030/source-snapshot/`。

独立读取 `mapping.parquet` 和原始 Allegro URDF，不调用优化器，使用 Pinocchio 检查全部 30 帧的四指尖 FK：
所有位置有限、旋转合法。但严格 URDF 限位检查为 **review / 退出码 1**：前 11 帧的 `joint_12.0` 和 `joint_13.0`
共 22 个值越过下限，最大约 0.0010000055 rad。已核实上游 `Optimizer.set_joint_limit(..., epsilon=1e-3)`
主动向外扩展边界；保留原始结果，未裁剪或放宽严格检查。这不推翻真实优化器已运行的事实，也不能据此宣称机器人关节严格合规。
结果在 `dex-sample/kinematics-check.json` / `.log` / `.exit`，可复查：

```bash
source config.local/reproduction.env
.venv/dex/bin/python egoworld/scripts/verify_dex_output.py \
  --mapping output/reproduction-20261029/dex-sample/mapping.parquet \
  --urdf external/dex-retargeting/assets/robots/hands/allegro_hand/allegro_hand_right.urdf \
  --output output/reproduction-20261029/dex-sample/kinematics-check.json
# 本样例预期退出 1：保留严格限位差异，FK 计算仍已完成。
```

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

2026-10-28 可视化改动验证（macOS/ARM、Python 3.12、pytest 8.4.2）：
CPU 回归命令额外排除 `ray_integration`：**146 passed / 17 deselected，15.19 秒**。覆盖报告从真实提交/损坏提交/部分结果导出、
上游失效传播、失败日志、恢复后空计时、导出上限，以及禁用 Torch/Ray 后的 CLI 和 JSON 文本转义。
可选 Prometheus 缺失时的监控降级、wheel 中的报告导出也已检查。
离线 DOM 检查另覆盖搜索、状态筛选、分页、详情/关闭、空结果及恶意文本；未完成浏览器截图/视觉验收。
这些检查不运行真实模型，也不提供 CUDA、Ray 或目标服务器的新验收证据。

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

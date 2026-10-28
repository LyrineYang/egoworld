# EgoWorld 开发交接

更新：2026-10-28（Asia/Shanghai）。面向接手开发和远端验证的人员；这是交接快照，不代替运行记录。
本轮整理起点为 `e94e7cc`（此前调度实现基线为 `6cd5a95`），本次交付另含离线报告、监控降级修复和展示材料更新。
本文区分已实现能力与研究候选。开始工作时用 `git log` / `git status` 核对实际版本，不将文档中的历史提交号当作当前 HEAD。

## 先读结论

EgoWorld 是把第一视角视频转换为**手、物体、机器人手和场景结构化数据**的离线 CLI。
当前五阶段已有适配、调度、写出、恢复和测试入口。2026-10-25 的完整回归为 **143 passed / 11 skipped**；
2026-10-28 工作区 CPU 回归为 **146 passed / 17 deselected**，本次排除了 Ray/原生模型/硬件测试，两组结果不能直接比较或相加。
真实模型权重推理、Linux 进程清理、真实多卡吞吐仍需目标环境验证，不能称作五模型已经验收的成品。

下一步应当完成：**远端 Linux 回归 → 已有模型短样本与恢复 → 按真实失败简化安装和操作 → 独立 GPU 探针 → 实测驱动吞吐优化**。
不同分支可以独立推进，缺少某分支资产或显存不应阻止其它分支的开发与验证。
近期不再扩展模型清单，也不重写第二套执行后端。

**已选重点研究方向：调度。** 在有限 GPU、多模型切换开销、阶段依赖、片段内时序约束和阶段提交边界下，
研究怎样提高同等输出要求下的有效数据产出。已有模型偏好/等待保护/预算是起点；实测代价驱动的选择仍是候选，尚无新算法或吞吐提升结论。
主要缺口与解决方法、对齐顺序见本文后续对应章节；详细实验定义见 [调度研究](design.md#11-调度作为突破点研究问题与验证方法)。

## 用户的需求与工作方式

| 已确认的要求 | 对开发的约束 |
| --- | --- |
| 快速上线，使项目实际可用并有专业展示 | 优先交付可照做的 CLI 主流程、真实样本、恢复证据；README 保持专业产品主页风格 |
| 用户久未推进，对项目并不熟悉 | 用直观的数据名称解释能力，给出模型与输出对应；说明状态和下一步，不要求用户理解全部实现细节 |
| 可以大规模重构，但工作时间和算力有限 | 选择最小完整路径，控制代码冗余；不因架构偏好重写已可用部分 |
| 已有分支先全部推进到可测试，再逐阶段修复 | 不将上一模型通过验收作为开发下一模型的前提；测试失败需要定位和修复，不能以此停止所有工作 |
| 无卡时推进到本地可验证边界 | 本地验证契约、适配控制流、真实视频/Parquet、Ray 调度和故障；硬件结果明确待测 |
| 生产面向单服务器 2–16 张 GPU | 16 GB 单卡只用于 smoke；不要围绕 16 GB 限制设计整套生产架构 |
| 不接复杂新模型、传感器与额外硬件 | I2 力/触觉/扭矩、硬件采集不做；FoundationPose 仅使用已有 RGB-D/网格 |
| Demo 暂缓 | 先交付处理工具；已有 Replay 构思不是当前待实施任务 |
| 当前可视化重点是 README、架构图和创新点说明 | 首图保持简洁，用图片解释每步提取的内容，模型名作为次要标注；调度细节留在 design.md |
| 保持现有文件结构，内容按主题分工 | 不把细节全挤进交接文档；交接说明每个文件的内容、阅读顺序及当前状态 |
| 把调度作为重点突破方向，并保留缺口与解决方法 | 先补测量和可比基线，比较代价驱动策略；对齐/质量问题进入交接，候选能力不能写成已实现 |
| 需求与实现一致，可独立验证 | 输出由代码契约定义；需求有对应代码和测试；模型替身通过不等于真模型可用 |
| 允许自主创建本地 commit | **每次 push 前必须取得新的明确同意**；准备交接不代表同意推送或发布 |

产品表述可使用“面向大规模数据集的可恢复手—物体—场景结构化处理框架”。
只有单卡/小样本证据时，只能描述可扩展机制和覆盖范围，不能声称已实测 TB 级吞吐、精度或长期稳定性。

## 项目演进：原设计与本轮实际变化

原设计针对单台 4×H100 / 128 核 CPU，设想处理 TB 级第一视角视频，输出分割、人手姿态、物体位姿、机器人映射等数据。
原文还讨论对象存储、预热 actor 池、GPU 利用率 >60%、失败率 <1%、Prometheus/DCGM 等。
这些是历史设想或待测指标，不能作为当前实现与测试结论。
原始资料保留于 [archive](archive/)，详细复盘见 [overview.md](overview.md)。

本轮工作前已有 CLI、清单、Ray、SQLite、Parquet 和 SAM2 接线；其它四算子仍有空结果占位。
原 driver 已能跨片段/多卡并发，但在一个 actor 内依次执行整片段模型、将大结果取回 driver，并使用阻塞退避。
准确问题是职责集中和流水重叠不足，而非全系统完全串行。

之后补齐了四模型 API 适配、schema 3、阶段依赖调度、阶段级提交/恢复、故障隔离、非阻塞重试、环境/资产预检、统一 launch 和 smoke。
随后修复真模型接口、显存隔离、损坏下游失效、提交与数据库不一致、外部进程退出等问题，并补充反例测试。
最新实现增加共享 GPU 池的模型偏好/等待保护、常驻隔离 runner、估计字节预算、CPU 读取优化与资源观测。
代码完成度的提升仍需要真模型验证支撑。

## 模型与输出：当前交付范围

| 阶段 | 用户拿到什么 | 必需条件 | 主要文件 |
| --- | --- | --- | --- |
| SAM2.1 + GroundingDINO | 手/物体类别、独立掩码、片段内对象 ID、二维框/中心轨迹、帧号和时间 | RGB 视频、文本提示、两套权重/配置 | masks.parquet、instances.parquet、tracks.parquet |
| HaMeR | 左右手、21 个三维关节点、手根朝向、15 个手指关节旋转、手型、估计相机平移 | SAM2 手框、可靠左右手、HaMeR 配置/权重和 MANO | hand_pose.parquet |
| FoundationPose | 物体相对相机的位置和朝向，以及随时间变化的姿态 | SAM2 实例、已有对齐 RGB-D、内参、带尺度网格及权重 | object_pose.parquet |
| DexRetarget | 指定机器人手各关节角度或位移、关节名称及单位 | HaMeR 关节点、真实机器人 YAML/URDF、左右手与坐标约定 | mapping.parquet |
| Fast3R | 抽样帧相机位姿、内参、有界场景点云 | 多帧 RGB、完整模型快照；需核对 BF16/API 条件 | reconstruction.parquet、reconstruction-pointcloud.npz |

SAM2 是核心需求，GroundingDINO 提供文本检测框，SAM2 做视频分割传播；当前不是 SAM3。
二维轨迹是实例中心序列，不等于三维追踪或跨片段全局 ID。
HaMeR 的 21×3 是关节点位置，15×3×3 是旋转矩阵，不可称为 135 个关节角度。
命名的手指弯曲角、接触/动作标签、速度等派生量暂不作为启动验收要求。
HaMeR 关节点采用腕部相对表示及模型先验尺度，Fast3R 默认片段内相对尺度；各阶段尚未融合成统一米制世界坐标。
不可把未经标定的 Fast3R 点云直接当 FoundationPose 的米制深度输入。

字段、类型、参考系和校验以 [contracts.py](../egoworld/src/egoworld/contracts.py) 为准，命令 `egoworld schema` 直接输出实际契约。
易读说明见 [outputs.md](outputs.md)，需求 R01–R11 的代码/测试映射见 [requirements.md](requirements.md)。
更改数据要求时同步修改契约、适配/读写/恢复、测试及说明；不要只改文档或用同义字段掩盖语义变化。

## 当前 pipeline 怎样运行

阶段依赖如下；各模型的时序状态仍在片段内部按顺序处理。

```text
视频目录 → 视频/片段清单 → CPU prepare
                            ├─ SAM2/DINO ─┬─ HaMeR ─ DexRetarget（CPU）
                            │            └─ FoundationPose（已有 RGB-D/网格）
                            └─ Fast3R
每阶段完成 → Writer 提交结果 → 解锁下游 → 片段 QC/meta
```

driver 按完成事件推进；多个片段、独立分支、CPU 和磁盘写入可以重叠。
prepare 当前仍会转码片段，后续算子各自读取；它尚未准备好所有模型所需的帧和裁剪。

| 部分 | 已实现的行为与保留的边界 |
| --- | --- |
| GPU 池 | 默认每 GPU 一个串行 ModelActor，共享启用的 GPU 阶段；两张卡也能轮流执行四个 GPU 阶段；可显式设置 gpu_stage_groups |
| 模型偏好 | 空闲槽位优先复用已加载阶段；model_residency_tasks 默认 8；就绪任务等待超过 max_stage_wait_s 默认 60 秒时优先；不抢占当前任务，也不保证完成时限 |
| 分环境运行 | auto 识别内置 python -m egoworld.worker，使用串行常驻进程跨片段复用权重；换阶段、错误、超时和退出时释放；每片段/对象时序状态独立 |
| 兼容模式 | per_task 每任务重启；自定义命令在 auto 下仍使用一次性协议；显式 persistent 要支持 --serve 协议；16gb smoke 强制 per_task |
| 数据搬运 | 模型调用返回小状态与大结果两个引用；driver 读小状态，Writer 直接消费大结果 ObjectRef；这不等于全链路零拷贝 |
| 数量限制 | max_active_clips 与 CPU/模型/Writer 在途任务上限；max_in_flight_gpu 当前也约束 CPU retarget，名称有局限 |
| 字节预算 | 计算前预留估计字节，完成后按结果包估计修正，提交后释放；写盘重试不被生产侧满队列阻塞；不可能满足的预留明确失败 |
| 文件压力 | max_prepared_bytes 限制活动片段转码文件高水位；已启动准备可超出；不是主存或硬盘硬限额 |
| 恢复 | 有效提交记录、schema/哈希和运行身份共同判断复用；损坏阶段及依赖下游失效，独立分支保留；文件已提交而 SQLite 未更新时修正状态 |
| 故障 | 到期重试队列、超时、worker 替换、外部进程组清理；一 run 一个 driver；没有多 driver/exactly-once 保证 |

预算不是总 RAM/显存配额，不覆盖全部权重、私有堆、子进程、页缓存或 CUDA 分配。
模型 OOM 仍可能发生；不要把 admission 设置写成防 OOM 保证。
server.json 是四卡起始模板，支持调整到 2–16 卡；CPU、Writer 和系统余量也要一起检查。

已做的 CPU 优化：HaMeR/Dex 仅读上游必要列；FoundationPose 同帧 RGB-D 与同片段网格复用，但 tracker、网格及可变输入保持隔离；
worker_threads 默认 1，限制 OpenCV、Arrow、OMP/MKL/OpenBLAS，FFmpeg 转码也用单线程。第三方库仍可能有额外线程。

## 文件结构与阅读地图

保留现有目录布局，按问题查对应文件。交接文档负责状态、入口和下一步，详细内容由下表所列文件维护。

| 文件 / 目录 | 内容 | 何时阅读 |
| --- | --- | --- |
| [README.md](../README.md) | 项目定位、简洁输出示意、能力、快速启动与验证边界 | 第一次了解项目 |
| [architecture.svg](assets/architecture.svg)、[pipeline.svg](assets/pipeline.svg)、[绘图说明](assets/pipeline.prompt.md) | 首图按输出命名并配掩码/骨架/位姿/点云图片；另有详细模型输出图；均为示意 | 修改展示素材；不把调度细节重新塞进首图 |
| [server.md](server.md)、[env/README.md](../egoworld/env/README.md) | 安装、资产、配置、启动、停止恢复、离线报告及排错 | 在目标环境实际使用 |
| [outputs.md](outputs.md) | 五阶段输出、字段含义、坐标/单位/尺度，以及逐层对齐方案 | 读取结果、处理几何对齐 |
| [requirements.md](requirements.md) | 需求与代码/测试映射、主要缺口、解决路径和验收标准 | 判断还缺什么、怎样算完成 |
| [design.md](design.md) | 当前调度实现、资源/IO风险、CPU 实验、调度研究基线与消融 | 研究和修改调度；区分已有能力与候选策略 |
| [testing.md](testing.md)、[tests/](../egoworld/tests/) | 测试命令、已观察结果、反例及硬件验证边界 | 复验改动和检查证据 |
| [overview.md](overview.md) | 项目历史复盘、工程贡献与 insight | 理解设计取舍与项目演进 |
| [archive/](archive/) | 历史计划和讨论 | 查原始动机；不能据此认定功能已实现 |
| [src/egoworld/](../egoworld/src/egoworld/) | 实际 CLI、契约、调度、算子、提交及报告代码 | 开发；具体职责见下一节 |

## 接手时先找哪些代码

以下路径相对 `egoworld/src/egoworld/`：

| 责任 | 入口 |
| --- | --- |
| 命令、用户启动、预检与验收 | cli.py、launch.py、preflight.py、acceptance.py |
| 配置、输出契约与输入清单 | config.py、contracts.py、manifests/ |
| 运行协调、依赖、队列与恢复 | pipeline/driver.py、engine.py、scheduling.py、state_store.py |
| Ray worker、模型生命周期和隔离环境 | pipeline/backend.py、stages.py、external.py、worker.py |
| 模型适配 | operators/ 下五阶段文件及 groundingdino_op.py |
| 提交与追溯 | pipeline/artifacts.py、io/writers.py、utils/provenance.py |
| 观测与离线报告 | utils/performance.py、profiling.py、observability/ |
| 离线 HTML 检查报告 | reporting.py、report_template.py；CLI report；不加载模型、不做标注视频回放 |
| 外部进程守护 | utils/process_guard.py |

配置模板在 [configs](../egoworld/configs/)，测试在 [tests](../egoworld/tests/)，CI 在 [tests.yml](../.github/workflows/tests.yml)。
保持一个 CLI、一套契约和一个生产后端，避免将测试逻辑复制成第二套 pipeline。

## 已有验证证据与未完成事项

| 证据 | 观察到什么 | 不能证明什么 |
| --- | --- | --- |
| 2026-10-25 macOS/ARM 完整回归 | 143 passed / 11 skipped，97.04 秒；Python 3.13.2、Ray 2.59.0，包含真实 Ray 进程测试 | 真实 CUDA/权重、Linux 和多 GPU 性能；这是历史运行记录，本次文档整理未重跑 |
| 2/16 逻辑 GPU 槽位测试 | CPU fixture 验证任务并发、预算、队列排空、等待保护 | 没有物理 GPU，不证明 2/16 卡扩展效率 |
| 模型 API 替身 | 控制流、字段、左右手、FP 隔离、Fast3R 错误路径等 | 上游安装、实际权重/API 兼容、预测质量 |
| 外部 runner 故障测试 | 常驻复用、错误重启、超时、退出后的杀死与回收 | 最新修复在 Linux 上仍需复测；此前 Linux CI 曾有进程清理失败 |
| profile CLI | 在禁用 Torch/Ray 导入时仍可读已有运行并导出报告 | 不运行推理、不自动规划配置，不是独立 GPU 探针 |
| 2026-10-28 工作区 CPU 回归 | 146 passed / 17 deselected，15.19 秒；Python 3.12、pytest 8.4.2；报告/可选监控专项、离线 DOM 交互及 wheel 报告导出已检查 | 未重跑 Ray、CUDA、Linux 或真模型；该次没有浏览器截图/报告布局验收 |

历史运行的 11 项跳过不能统计为模型通过；本次 17 项 deselected 是按测试范围主动排除，也不能解释为通过。
本机微基准原始数据见 [cpu-2026-10-25.json](benchmarks/cpu-2026-10-25.json)：
512 行合成实例表全列读取 14.79ms、投影读取 2.48ms；六次零推理 fixture 的一次性 runner 2.754s、常驻 0.503s。
这些只证明对应 CPU 路径有收益，不是端到端、真实权重或服务器吞吐结果。

独立验证的实际要求：固定输入/预期与测试规则，读回产物，保存原始命令、退出码、环境/代码/模型身份；
检查失败反例，使用未参与实现的检查视角复核关键风险。先看实际产物，再判断是否满足要求，不能拿实现说明当证据。
schema 非空仍需人工抽查左右手、投影、尺度与轨迹；不能为了通过而放宽断言。
详细流程及既有 OpenAI 官方建议见 [testing.md](testing.md)。

## 主要缺口、解决路径与验收

八类缺口的完整表放在 [requirements.md](requirements.md#主要缺口解决路径与验收)：真模型/目标环境、调度代价、CPU/GPU 职责、
资源预算、对齐、QC、资产身份与首次使用。每项包含解决路径、依赖和验收条件；这些路径尚待执行，不等于问题已解决。

## 对齐问题的处理顺序

完整方案放在 [outputs.md](outputs.md#对齐问题的处理顺序)：源帧与预处理 → 实例身份/左右手 → 坐标方向 → 尺度/参考系 → 机器人映射。
该文件同时解释现有字段、单位与尺度。世界坐标融合仍是候选研究，不新增传感器、不静默改变现有输出语义。

## 调度突破点：交接时保留的研究约束

当前已实现完成事件推进、共享 GPU 池、模型复用、等待保护、数量/估计字节准入与阶段提交恢复。
研究候选是联合实测切换代价、阶段时间和中间态压力选择任务及空闲 worker；片段时序、输出质量和提交恢复边界保持一致。
问题定义、基线、指标、消融与故障反例统一放在 [design.md](design.md#11-调度作为突破点研究问题与验证方法)，不能将候选策略写成已实现能力。

## 推荐开发方向与完成条件

| 顺序 | 具体交付 | 完成条件与继续工作的方式 |
| --- | --- | --- |
| 1. Linux 可运行基线 | 独立 EgoWorld 环境，CPU/Ray 与进程清理回归 | 固定源码；FFmpeg/Parquet/CLI 可用；退出、超时和孤儿子进程测试通过；修复 Linux 差异，保存环境导出 |
| 2. 现有模型真实可用 | 有限真实短样本、各启用阶段推理及整图、停止恢复 | 保存 acceptance.json、实际输出/日志、版本及抽查；缺资产分支注明待测，其余继续；无需逐模型通过才尝试整图 |
| 3. 产品首次使用收尾 | 清楚的安装/资产列表、失败提示、可照做文档 | 从干净环境按 server.md 重走；用户不读源码也能配置启动并定位缺项；独立读取器可读结果 |
| 4. 独立 GPU 探针 | 冷/热测量、报告缓存、可解释建议 | 不改输出语义，探测失败可诊断；报告绑定环境/模型/输入规格；先建议参数，不自动应用 |
| 5. 调度与服务器吞吐 | 同负载下当前启发式、静态分组及代价驱动候选的对照 | 先测加载、解码、裁剪、推理、后处理、写盘、CPU/RAM/VRAM；保持输入/采样/输出一致，补消融与故障恢复；有条件再验证 2→4→更多卡 |
| 6. 发布材料 | 已验证配置、样本产物与专业 README | 清楚标出覆盖阶段、硬件与限制；模型/资产许可及样本分享条件核对；推送和发布按用户明确授权执行 |

资源不足期间，可继续完善失败提示、依赖诊断、探针与测试代码；不能据此跳过待测项或宣布交付全部完成。
不为速度静默减少输出要求；Fast3R 减视图、降分辨率/帧数改变工作量，应单独记录。

## GPU 预检：已决定的实现路线

Data-Juicer 的可借鉴点是先测量、再估算资源、复用执行池；本项目已选择保留 Ray Core，暂不迁移到 Data-Juicer/Ray Data。
已有比较、源码研究链接和测量方法集中在 [design.md](design.md)，本交接不将研究建议标成已实现功能。

| 能力 | 当前状态 | 成本与推荐 |
| --- | --- | --- |
| 硬件/API/资产检查 | doctor 已实现，列出可见 GPU 名称/容量/架构并检查分环境原生 API | 低成本，不实际加载全部权重推理，不能证明模型装得下 |
| 随实际运行观测 | 加载/墙钟/CPU、排队、缓存命中、结果字节、RSS、可选 Torch 显存；profile 汇总 | 已实现；不是纯 GPU 前向计时，RSS 是进程生命周期高水位，Torch 不覆盖全部设备分配 |
| 独立冷/热探针 | 待开发 | 中等开发成本：复用 runner、上游已提交产物、超时及报告机制；建议真实模型可用后优先补 |
| 自动 actor 数/GPU 共驻规划 | 待开发，后置 | 高成本；需要 NVML/全设备观测、CPU/IO竞争与多卡负载证据，短样本不能提供安全共驻保证 |

探针建议从“1 次冷加载 + 1 次预热 + 3 次测量”开始，运行成本近似各阶段加载加四次样本处理，另计未缓存上游生成。
这是候选设计，尚无 probe 命令。相同硬件类别可复用匹配报告，异构卡分别测；默认顺序探测，减少共享存储/主存竞争。
缓存键至少包含代码、环境、模型/资产身份、GPU 架构/容量、精度和输入长度/分辨率/采样及实例数；条件变化重新测。
HaMeR 可比较保持相同输入集合的微批 1/2/4；不自动降低精度或 Fast3R 视图数。

性能候选风险包括：prepare 无损重编码、各分支重复读取、SAM 提示/掩码 CPU 工作、HaMeR 裁剪、Fast3R JPEG/loader/PnP、
Python 大结果序列化、小文件/SQLite 与 20ms 文件信箱轮询。它们是代码可见的成本，尚无真实服务器瓶颈排名。
先选实测占比最高且可独立验证的一项；共享帧缓存必须有字节上限和生命周期，不先上无限缓存或复杂自动调度器。

## 远端接手：先做什么

2026-10-28 已读的远端聊天报告：`zhen` 是 Ubuntu 24.04 WSL2，RTX 4060 Laptop **8188 MiB**，WSL 内存约 15 GiB。
该记录来自 humanoid-demo 环境，不是 EgoWorld 验收；当时该聊天仍在安装 humanoid-r1 依赖。
不要修改/复用正在安装的 humanoid-r1 环境，也不要假定其已经证明 EgoWorld CUDA 或模型可用；接手后重新读取实际状态。

1. 确认当前聊天执行位置、项目根目录、HEAD、未提交文件和运行中任务；避免重复提交测试或与其它 GPU 工作争用资源。
2. 为 EgoWorld 使用独立 conda 环境。基础安装见 [环境说明](../egoworld/env/README.md)，HaMeR/Fast3R/FP 可分环境，通过 runner_command 接回同一代码版本。
3. 先跑下面的 Linux CPU/Ray 检查，再按可用资产运行各模型短样本。8 GB 只代表现有测试资源，不是生产设计目标。
4. 当前只有 default/16gb 两种 smoke profile，没有 8gb profile。16gb 是参数预设而非显存保证；可复用其隔离思路，但不能据此承诺 8 GB 全图可跑。
5. 显存不够时记录具体阶段、输入与峰值；其它分支继续。不要先实现 CPU offload/量化或为通过而更换模型。

以下命令在仓库根目录、已激活独立环境时执行。FFmpeg/ffprobe 需可用；配置/资产路径先按 server.md 填写。

```bash
git status --short --branch
git log -3 --oneline
python -m pip install -e '.[dev,pipeline]'
egoworld doctor
egoworld plan --config egoworld/configs/full.json
egoworld schema
EGOWORLD_RAY_TEST=1 RAY_USAGE_STATS_ENABLED=0 \
  pytest -q -m 'not env_smoke and not pipeline_smoke and not sam2_smoke and not native_models'

# 已配置本机路径与实际可用资产后；失败需保留退出码和报告
egoworld launch --config config.local/job.json --input-dir /absolute/videos --check-only
egoworld doctor --config config.local/job.json --gpu --assets
egoworld smoke --config config.local/job.json --input-video /absolute/videos/sample.mp4 \
  --stages sam2 --seconds 2 --profile 16gb
```

真实模型准备从 example.json（分割）起，rgb.json 添加人手/机器人手/场景，已有 RGB-D/网格才加入 full.json 的 FoundationPose。
HaMeR 的 MANO、机器人 YAML/URDF、FP 的深度/内参/网格均不能用虚构文件代替；缺少时明确关闭或待输入。
完成启用阶段整图后，以相同 launch 命令和 run-id 测停止恢复；更改影响运行身份的代码/模型/输入/配置时使用新 run-id。
收集完整命令、退出码、resolved config、清单、acceptance/summary/QC、环境导出、实际产物和日志位置。
结果放运行目录，交接文档仅更新结论与位置，不不断追加冗长日志。

## Git、交接范围和文档维护

历史交接整理时，本地 main 为 `6cd5a95`，相对当时本地 origin/main 领先两个 commit（未重新联网核验远端）。
本轮整理起点为 `e94e7cc`，另含交接文档提交；后续交付版本和未提交改动以 `git log` / `git status` 为准。下表保留原记录：

| 提交 | 内容 |
| --- | --- |
| 6cd5a95 | 服务器调度、常驻 runner、预算、观测、测试和 CPU 基准 |
| 26cf6c8 | 结构化可编辑 pipeline.svg 与生成 prompt；图为示意，不是真模型预测 |
| 6348557 | 上次已推送的 README 产品主页版本 |

当时前两个提交尚未推送；该历史记录不能证明今天远端包含哪些提交。准备远端项目时保留相同 origin，
先核对实际分支差异和未提交文件；handoff 后核对完整候选快照是否实际带到目标 worktree。
此前已核对的 handoff 前置条件是：目标主机在线，并在 Codex 保存同一 Git 仓库的项目；若项目是仓库子目录，两端须保持相同相对位置。
当时远端仅登记 humanoid-demo，尚无 egoworld；实际迁移前重查项目/主机列表。可通过聊天底部切换位置，或在另一个聊天请求迁移，当前聊天不能用工具迁移自身。
模型权重、external 源码、config.local 与运行目录多被 Git 忽略，不能假定随 Git/chat 自动迁移；部署时另行准备。
工作区另有 `.DS_Store` 修改和与 EgoWorld 无关的个人 PDF/DOCX，勿提交、清理或作为项目资产同步。
迁移前检查这些文件的传输范围。本轮不执行迁移或远端环境修改；用户已明确授权完成本轮项目修改后提交并推送到现有 origin/main。
授权范围不包含个人 PDF/DOCX、.DS_Store、模型权重、运行产物或其它项目文件；后续新的推送仍按用户当轮授权处理。

相关上下文为聊天“报告远程测试项目与环境”（`01a181c4-049a-7823-9890-8f3cdd112550`，当时 host 为 `remote-control:env_e_6ae1ab32b07c8331990e55255681166e`）
和“介绍 Codex 远程协作与 handoff”（`01a17702-7576-7512-8614-9740a54e5aad`，local）。需要依赖后续进展时先重新 read_thread，不能假定聊天仍停留在本文记录的状态。
其它项目对话里的推送或跨聊天操作授权，不自动扩展成 EgoWorld 的授权。

继续维护现有文档分工：README 是产品入口；server.md 是用户操作；outputs.md/requirements.md 是需求范围；
design.md 是实现/研究/性能边界；testing.md 是验证方法与记录；overview.md 是历史复盘。
本文只维护必要的交接快照，旧设计放 archive，不把临时进度写进 AGENTS.md，也不额外创建平行 PLAN/SPEC。

接手者的第一份有效交付应是：**Linux 回归实际结果、可用/缺失资产列表、首个真模型 smoke 的产物或明确失败证据，以及下一项可继续的修复。**

# 相同输入与输出约束的具身数据实验

核心目标是：在具身数据上，以相同输入、模型行为、输出要求和清洗判定，获得比 Data-Juicer 更高的有效产出和更低的资源成本。是否优于它是待验证的研究假设。不能用不同模型、少采样、减少输出字段，或者把低质量结果计入产量来证明优势。

## 结果应回答什么

分开报告两个问题：

1. **处理效率**：同一工作负载、同一保留集合和输出契约，完成一次任务需要多少时间、内存、显存和恢复工作。
2. **清洗完整性与质量**：能否识别真实不可用数据、解释拒绝原因，并保留对下游有用的样本。规则多、输出类型多，本身不能证明更完善；需要标注的错误集合和误拒绝率。

主要结果表应是：数据集与任务、固定输入数、固定采样覆盖、有效输出数、墙钟时间、有效产出/秒、内存/显存峰值、输出一致性、失败/OOM/重试、误拒绝率。每行关联原始运行记录，失败运行不能从分母中消失。

“有效”必须有层次：本次 pilot 的 `accepted` 只表示满足手部输出契约；人工确认准确、几何上可用、可供机器人学习，是更强且不同的标准。不得把契约有效行数描述为高质量训练样本数。

## 对照分层

| 层次 | 固定工作 | 实现与结论边界 |
| --- | --- | --- |
| A：标注后的 CPU 清洗 | 相同已生成手部标注、契约、逐条拒绝原因、完整保留字段 | 本次实现真实 Data-Juicer DefaultExecutor + 自定义 Mapper，与流式实验参考执行器对照；不包含视频解码、推理或产品 recipe 调度 |
| B：共同模型处理 | 相同 SAM2/DINO、HaMeR、FoundationPose、Fast3R/Dex 权重/资产与 runner，帧、crop、精度、随机种子、输出一致 | 已有可选生产阶段 Mapper 及真实执行器验收入口；逐 GPU 模型、批处理和完整依赖链仍待验证，不把框架开销与模型差异混在一起 |
| C：完整产品任务 | 输入视频 → 过滤/分支选择 → 模型 → QC/导出 → 最终提交 | 清洗/缓存和执行代码已集成；待接入 Data-Juicer 对应产品任务。包括启动、身份检查、数据搬运、导出和恢复成本 |
| D：各自原生能力 | 各自推荐模型/算子与输出 | 独立能力和质量/成本研究；例如 HaWoR/MegaSaM/MoGe 与 HaMeR/Fast3R 不等价，不能纳入同模型效率比值 |

Data-Juicer 本次固定源码为 `09178e1e0ac76cfbfe11232d66244e1ccc5add6f`。使用其正式 [自定义算子接口](https://github.com/datajuicer/data-juicer/blob/09178e1e0ac76cfbfe11232d66244e1ccc5add6f/docs/DeveloperGuide.md) 和 [DefaultExecutor](https://github.com/datajuicer/data-juicer/blob/09178e1e0ac76cfbfe11232d66244e1ccc5add6f/data_juicer/core/executor/default_executor.py)，并保留原始导出。此前源码调研的 revision 不同，应分别引用，不能合并成一组固定版本结果。

## 已实现的 A 层入口

代码独立放在 `egoworld/experiments/equal_work/`，通过 `egoworld/scripts/benchmark_equal_work.py` 调用。它不改变生产 recipe、调度或资源配置。

- `freeze` 从原始 `hand_pose.parquet` 创建一次性实验语料。记录视频、原始表、clip metadata、运行 provenance 的 SHA-256，以及实验代码与现有 `contracts.py` 的文件哈希。保留完整行，不降低精度、不裁字段。用磁盘索引拒绝重复帧/对象身份。
- 两个执行器调用同一个 `evaluate`，再调用生产 `validate_rows("hand_pose", ...)`。检查源帧范围/FPS 对齐、手侧、坐标/单位/尺度、数组长度、有限数值；不添加未经验证的速度、骨长或置信度阈值。
- 每条输入都写入 `decisions.jsonl`，含输入 payload 哈希、判定和原因；保留行写入 `accepted.jsonl`。Data-Juicer 原始导出、配置、日志也保留。
- 每轮交替执行顺序；串行启动全新进程，两个路径都限制数学库线程为 1，Data-Juicer `np=1`，关闭结果 cache/checkpoint 并使用独立临时目录。OS 页缓存未清空；这种模式称为 **fresh-process artifact cleaning**，不能称为 GPU 冷/热测试。
- 主指标从子进程创建到退出，包含导入、语料验证、执行和标准化写出，排除实验语料制作、源文件准入核验与事后审计。库内阶段计时仅用于归因。
- 20 ms 采样进程树 RSS，并记录 worker 自身高水位。前者可能漏过短峰值、共享页可能重复计数；本 pilot 无 GPU 推理，不报告 VRAM 结果。
- 审计按 sample ID 检查所有判定、拒绝原因和全部保留字段，要求 canonical JSON 精确一致，忽略输出顺序。磁盘 SQLite 索引避免随记录数持有整张 Python 表。
- 只有计划中的每次运行都成功、通过审计，并且运行前后解释器/包版本快照一致，才生成墙钟比值。测量设置 UV/PIP/HF 离线，禁止把网络依赖准备当作正常执行；缓存内的补装若改变环境，也会使对比无效。少一次、失败一次、额外/重复/遗漏输出、任意数值变化都会阻止比值。原始耗时仍保留。

安装独立环境（Linux；源码包构建需要 C/C++ 编译器和对应 Python 开发头文件）：

```bash
python -m venv .venv/datajuicer-bench
.venv/datajuicer-bench/bin/python -m pip install \
  'https://github.com/datajuicer/data-juicer/archive/09178e1e0ac76cfbfe11232d66244e1ccc5add6f.zip'
```

实际接入发现 DefaultExecutor 的导入/资源检查路径还会触发 Ray 和 Torch；测量前应显式装好，CPU pilot 使用 CPU Torch，避免首次运行下载依赖：

```bash
.venv/datajuicer-bench/bin/python -m pip install 'ray==2.59.0'
.venv/datajuicer-bench/bin/python -m pip install 'torch==2.15.0+cpu' \
  --index-url https://download.pytorch.org/whl/cpu
```

不需要安装其视频模型或模型权重来执行 A 层。实际包版本由每组运行的 `environment-*.json` 保留。优先两个路径使用同一个解释器，避免依赖/NumPy 版本造成混淆。不要升级共享模型环境。要复现本次包版本，在安装上述 CPU Torch 后，使用 [环境约束](../egoworld/experiments/equal_work/requirements-pilot.txt)：`python -m pip install -r egoworld/experiments/equal_work/requirements-pilot.txt`。

```bash
.venv/datajuicer-bench/bin/python egoworld/scripts/benchmark_equal_work.py freeze \
  --hand-pose /absolute/run/video_id=X/clip_id=Y/hand_pose.parquet \
  --meta /absolute/run/video_id=X/clip_id=Y/meta.json \
  --provenance /absolute/run/run_manifest.json \
  --output /absolute/benchmark/corpus-v1

.venv/datajuicer-bench/bin/python egoworld/scripts/benchmark_equal_work.py run \
  --corpus /absolute/benchmark/corpus-v1 \
  --reference-python "$PWD/.venv/datajuicer-bench/bin/python" \
  --datajuicer-python "$PWD/.venv/datajuicer-bench/bin/python" \
  --repeats 5 --output /absolute/benchmark/trials-v1

.venv/datajuicer-bench/bin/python egoworld/scripts/benchmark_equal_work.py audit \
  --corpus /absolute/benchmark/corpus-v1 \
  --trial /absolute/benchmark/trials-v1/000-datajuicer
```

输出目录必须不存在。源文件或规则变化需要新语料/新目录，不覆盖旧报告。冻结目录同时保存实验代码和契约源码快照。`freeze --fault-fixtures` 可以额外加入四条明确标记的合成错误行，用于真实执行器的拒绝分支检查；它们计入 `synthetic_fault_rows`，不增加真实样本规模。现阶段只支持一次冻结一个 clip，不能重复复制它后宣称获得独立视频规模。冻结需要可序列化的规范 JSON；坏文件、重复身份导致冻结失败，失败不属于测量集。数值/单位等逐行拒绝可在规则中解释，但这些不覆盖输入视频层清洗。

`stream_reference` 是实验中的流式参考，不是生产 EgoWorld 执行器。其与 Data-Juicer 的启动时间比不能命名为“EgoWorld 加速比”。这个接口目前证明实验可追溯与输出判定可比较，不构成核心系统优越性证据。

## 2026-11-02 实际接入记录

[机器可读报告](benchmarks/equal-work-pilot-2026-11-02.json) 实验文件哈希对应提交 `25a3b4d`，包含输入/源码哈希、包版本、全部逐次记录、worker 分项计时、环境与失败说明。原始目录为 `output/equal-work-20261102/`；源码快照随语料保留。生产包源码哈希仍是此前 GPU 验证的 `e94d1111…`，本次没有重跑 GPU 推理。

- 原始语料来自既有真实 HaMeR 产物：1 个视频、40 个源帧（200–395）、2 个手部对象、80 行；不包含合成错误。重复运行不增加独立样本规模。
- 5 轮交替、共 10 次运行全部成功；每次 80 行判定和 80 行完整保留输出精确匹配，运行前后包版本一致。
- 另有 1 轮实际执行器检查：80 条原生行 + 4 条标记的合成错误，两端都保留 80、拒绝 4；其数值不混入原始语料的 5 轮统计。
- 初次接入自动补装依赖的运行被终止，报告标为无效并保留；随后在测量外完成 Ray/CPU Torch 安装，正式 pilot 使用离线与环境稳定性门槛。

| CPU artifact cleaning，fresh process | 次数 | 墙钟中位数（范围） | 进程树采样 RSS 最大值 |
| --- | --- | --- | --- |
| 实验流式参考 | 5 | 1.541 s（1.518–2.541） | 56.96 MiB |
| Data-Juicer DefaultExecutor | 5 | 65.572 s（62.977–76.339） | 568.48 MiB |

**这不是生产 EgoWorld 与 Data-Juicer 的完整处理对照。** 机器为 Ryzen 9 7945HX / WSL2，解释器与工作区位于 `/mnt/c` 的 9p 挂载盘；小语料的 fresh-process 结果深受导入、框架初始化和文件系统成本影响，不代表稳态清洗速度。此处只验证接口、审计与记录流程，不能据此声称完整具身处理加速或提高标注质量。更大语料、持久进程和 Linux 本地盘需要独立成组，完整模型与产品流程仍待 B/C 层接入。

## B/C 层：下一步的主要实验

集成后的可执行阶段接入检查、具体交付与结果表要求见 [比较推进计划](comparison-next.md)。它补充本协议，不将小样例阶段验证当作生产整链或性能结论。

**固定数据和语义。** 建立独立来源的 10–60 秒 RGB 视频集，分层覆盖单手、双手、遮挡、快速运动、多物体、无手但有场景价值、坏输入；另设 RGB-D/mesh 组。冻结来源/许可、内容哈希、帧范围/FPS、分辨率、prompts、手侧复核、深度单位、mesh、请求输出、模型权重与预处理版本。训练和阈值选择用开发集；正式报告用固定留出集。短 smoke 和重复同一片段只作工程验证。

**冻结等价判定。** mask 解码后像素和覆盖一致；帧/对象 key 集一致；手部关节、旋转、相机内外参、物体变换、机器人关节、点云使用预先确定的尺度/坐标与数值容差。容差来自重复原生推理噪声及任务标准，并在实验前确定，不能看到差异后放宽。空输出和非有限值单独计数；输出接近不等于正确，还需独立质量评审。

**固定模型适配器。** Data-Juicer 自定义 Mapper 调用同一现有 runner/Writer，使输入准备、模型权重、后处理和最终文件契约可核对。先顺序单 GPU 做逐模型原生输出验证，再接完整依赖链。适配器不得用离线预测替代实时模型、丢失依赖或隐藏重试。数据集加载、进程启动和导出都计入端到端时间。

**充分的对手。** 同时保留 Data-Juicer 默认与经公开参数调优的基线，考察其批处理、算子融合、缓存和资源选项；两端拥有相同 CPU/GPU/内存预算和相同调参预算。记录所有候选，独立开发集选参。不能只给 EgoWorld 复用缓存、给另一端强制冷启动。

**运行矩阵。** fresh runner/model、warm model/result miss、跨运行完整命中、部分失效、输入缓存开关、异常恢复分别成组。每组至少 5 次完整运行，交替/随机平衡顺序；记录竞争负载。GPU 预检分离冷调用、预热和热测量，不把微基准与完整 job wall time 混算。按独立视频汇总并给出分布/置信区间，不能把同一视频的帧当作独立样本扩大显著性。

**消融。** 在同一执行语义下逐项关闭身份检查复用、共享采样帧/crop、mask 搬运复用、HaMeR 合批、模型驻留窗口、CPU/GPU 阶段拆分、实测成本排序和压力约束。先比较单项，再比较组合，检查互相掩盖的收益与负收益。

**清洗效果。** 对已知坏数据和随机保留/拒绝样本分别人工审查，报告误拒绝率、漏检率、按场景/任务分组的覆盖。无手视频对 hand 分支可能无效，对 scene 分支可能有用，不能整条视频全局丢弃。用于资源筛选的 cheap gates 必须与最终质量策略分开，且保留拒绝理由与阈值版本。

## 与设计文档第三部分的关系

第三部分的调度/资源优化应当作为上述同工作量实验的可解释变量。已有有限准入、模型驻留、阶段拆分、观测/成本排序和可追溯阶段提交，为这项研究提供执行基础。

最有价值的问题不是“哪个调度器单次最快”，而是：在固定模型和产出下，减少多少重复准备与换模，CPU/GPU 重叠是否带来真实 job 收益，长尾等待和峰值内存是否可控，失败后要重做多少工作。已有单卡研究中，成本策略尚未稳定优于简单驻留窗口；需要完整对照和置信区间决定是否值得增加复杂度。

完成 B/C 层、留出集质量评估及恢复实验之后，才适合形成“同输入同输出下优于 Data-Juicer”的核心结果报告。现有 [单卡阶段与调度记录](throughput-validation.md) 是内部设计证据，可帮助挑选实验因素，不能替代外部框架对照。

## V100 服务器对照与两卡长测

2026-11-09 在同一台服务器完成 CPU 清洗、单卡同模型和两卡分档对照。GPU 两端锁定输入顺序、采样、模型资产、清洗、QC 和导出契约，关闭跨运行结果缓存。Data-Juicer 1.6.1 使用固定版本的两个 `DefaultExecutor`，统一筛选后分配互斥的 clip。EgoWorld 常驻原生阶段 worker 通过已提交的 receipt 连接准备、推理和写出。两卡实测包含冷启动、就绪检查和最终提交。

| 实测组 | 冻结正例 | EgoWorld 墙钟 | Data-Juicer 墙钟 | 达标产出比 |
| --- | ---: | ---: | ---: | ---: |
| CPU 清洗，五轮中位数 | 12 段 | 33.04 s | 38.71 s | 1.17× |
| 单卡同模型，五轮中位数 | 12 张 | 137.76 s | 138.62 s | 1.01× |
| 原生双 worker 对照，试跑 | 12 张 | 129.29 s | 129.85 s | 1.00× |
| 两卡常驻阶段，小批量 | 12 张 | 52.81 s | 99.56 s | 1.89× |
| 两卡常驻阶段，大批量 | 96 张 | 109.09 s | 178.26 s | 1.63× |
| 两卡长测 | 864 张 | 624.38 s | 921.29 s | 1.48× |

CPU 使用带官方相机与手部标签的 EgoDex 视频，逐段核对标签长度与解码帧数。五轮判定逐条一致，误拒 0/12，漏拒 4/16，均为像素相同的改封装副本。包级去重原型在三轮交替对照中将中位耗时从 31.27 s 降到 28.97 s，并拦下全部四个副本，保留全部 12 段正例。[EgoDex 数据与条款](https://github.com/apple-aiml-research/ml-egodex) · [CPU 原始记录](benchmarks/v100-2026-11-09/cpu.json)

GPU 使用 FreiHAND 官方评估标签。每张图像编码为 3 帧，评分取中间帧。达标手要求手侧正确且腕部相对 MPJPE 不高于 20 mm。单卡 12/12 达标，独立留出集 10/12 达标。正式两卡大批量两端均为 93/96 达标，MPJPE 12.12 mm，阶段表、数组、QC 和导出逐值相同。含外部 CUDA 作业的试跑保留在诊断记录。每个有效两卡档位的比值来自一对运行，适用于冻结的静态姿态负载。[Zimmermann 等，ICCV 2019，FreiHAND 数据与条款](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) · [冻结样本与来源哈希](benchmarks/v100-2026-11-09/corpora.json) · [两卡逐轮记录](benchmarks/v100-2026-11-09/two-gpu.json)

大批量的模型流水线用时 85.59 s 对 154.21 s，贡献约 69 s 总差距。按内容复用源码导入解析后，两端导出分别用时 8.24 s 和 8.13 s。SAM2 是 EgoWorld 当前排队最集中的阶段，中位等待 4.07 s，HaMeR 和 DexRetarget 中位等待约 1 ms。可信资产登记适用于已核验且由实验方控制的固定权重，首次登记成本单列。[单卡与留出记录](benchmarks/v100-2026-11-09/single-gpu.json) · [Nsight 与阶段分解](benchmarks/v100-2026-11-09/profiling.json)

长测合计预算窗口为 **1631 s**，两端各跑一次。两端均有 **785/864** 达标，MPJPE 为 **12.57 mm**，清洗误拒 0/864、漏拒 0/6。EgoWorld 与 Data-Juicer 分别产出 **1.257** 和 **0.852** 只达标手每秒。逐值审计覆盖 864 个 clip。两臂采样期间均未观察到外部 CUDA 作业。 [长测机器可读结果](benchmarks/v100-2026-11-09/endurance.json)

### 复现入口

使用相同模型环境、两张空闲 V100 和本地冻结 recipe。配置中 `num_gpus` 设为 2，模型 runner 指向各自的原生解释器。按发布的种子、排除集合和样本 SHA-256 准备 FreiHAND 语料。小批量使用 `stream-checked-pinned`，较大批次使用 `stream-checked-dual`。下列命令假设在包含 Python、FFmpeg、CUDA、NVML 和模型依赖的环境中从仓库根目录运行。

```bash
python egoworld/scripts/benchmark_two_gpu.py --study output/two-gpu-rerun freeze \
  --recipe config.local/two-gpu-recipe.json --corpus output/freihand-frozen/corpus.json \
  --devices 1,2
timeout --signal=INT --kill-after=15s 1780s \
  python egoworld/scripts/benchmark_two_gpu.py --study output/two-gpu-rerun run \
    --coordinator .venv/sam2/bin/python --dj-python .venv/dj-jammy/bin/python \
    --layout stream-checked-dual --budget-seconds 1765
```

发布记录包含配置、语料身份、每轮指标、分支判定、导出哈希、匿名化 GPU 采样和 Nsight 聚合表。原始图像、视频、关节坐标、权重、完整 profiler 二进制和机器日志保存在服务器实验目录。[GPU 采样](benchmarks/v100-2026-11-09/gpu-observations.jsonl.gz) · [长测预算和逐项审计](benchmarks/v100-2026-11-09/endurance.json)

# 可观测的阶段内部执行

本实现延续 schema 3、逻辑阶段依赖、受监护的 runner、最终 Writer 提交和按运行保留历史。
新增执行模式与代价策略均须显式启用；不会修改生产配置、worker 数、精度或输出采样。

## 配置与执行边界

```yaml
execution:
  phase_profiling: true
  split_execution: true
  split_stages: [sam2, hamer, foundationpose, fast3r]
  phase_workers: 2
  split_block_bytes: 67108864
  split_stage_bytes: 536870912
  split_total_bytes: 2147483648
  split_directory: /absolute/local/nvme/egoworld-phases
  scheduling_policy: heuristic
  model_residency_tasks: 8
  max_stage_wait_s: 60
```

这是一份需要按机器资源评估的示例，不是建议直接复制的最佳配置。默认 `phase_profiling`、
`split_execution` 都为 false。`phase_workers` 占用 Ray CPU，与 prepare、GPU actor 和 Writer 一起检查总额度。
CPU 阶段的内置 runner 使用原模型 Python 环境，但隐藏 CUDA 设备；GPU 阶段继续复用原受监护进程。
`split_stages` 可只选择有收益的模型，其他模型保持整阶段调用。
自定义 runner 未实现此协议时，配置校验会拒绝拆分模式。

| 模型 | CPU 输入 | GPU 步骤 | CPU 输出 |
| --- | --- | --- | --- |
| HaMeR | 读取实际需要的帧、官方 ViTDetDataset crop、有界跨帧合批 | 原精度、原模型前向，批量搬回预测 | 按帧/对象/左右手映射回填、保留镜像与腕部坐标修正 |
| SAM2 | 有限提示帧预取 | 原视频状态与时序传播；每帧阈值化 mask 一次 D2H，打包暂存 | 同一份 mask 生成实例与 union RLE、框、面积及轨迹 |
| Fast3R | 原 JPEG 编码与官方 loader | 原多视图前向及官方传输，导出预测点图 | CPU PnP、点选择、可选世界坐标像素对应 |
| FoundationPose | 按需要准备 RGB-D，校验内参、深度单位与对齐 | 片段内独立 tracker，按对象时间顺序注册和跟踪 | 原物体姿态表交付 Writer |

HaMeR 不自动扩大 batch：沿用 `batch_size`，并可通过 `max_batch_crops` 和 `max_crop_bytes` 限制数量与字节。
跨帧合批会改变批次组成，浮点结果应做数值对照。尾批必须执行。Fast3R 仍在 PnP 失败时明确报错；
不会用单位矩阵伪造输出。SAM2/FP 不在片段之间复用时序状态。

本版每次 Ray 调用处理一个逻辑阶段的一个内部步骤，步骤内部流式消费有界块。
它实现**跨片段** CPU 输入、GPU 推理、CPU 后处理重叠；没有承诺同一片段的三个步骤同时执行。
SAM2 官方视频状态初始化仍包含 GPU worker 内的解码/准备工作，FP mesh 构造仍在 GPU 步骤内。
它们由分项计时显示，不能把整个 GPU 步骤的墙钟称作纯 GPU 核函数时间。

## 预算、临时数据与恢复

- 在提交 CPU 输入前，同时预留原结果额度与一个 `split_stage_bytes` 临时盘额度。
  所有活动拆分阶段的固定预留不得超过 `split_total_bytes`。
- 临时块是不可变连续数组数据文件＋JSON，含大小和 SHA-256；无 pickle。写入前检查单块和阶段总预算，超限明确失败。
  可用 `split_directory` 指定本地盘；按 work_dir 哈希和 stage 隔离，清理只删除本任务命名空间。默认在原 work_dir 内。
  读取以只读 memmap 校验并复制，随后关闭映射；映射和可变 tracker 不跨 worker 共享。
- CPU 准备不会先把整段视频展开进内存。HaMeR 有限 crop、FP 一帧 RGB-D、SAM2 有界多帧打包 mask（默认最多 16 帧、8MiB）
  分块交接；Fast3R 的多视图联合前向仍由 `max_views` 与单块上限约束。
- GPU 完成后即释放其槽位，CPU 后处理继续持有数据与预算。Writer 无须新的生产额度即可排空。
  仅 Writer 提交成功才将逻辑阶段置为 Done、解锁下游并释放阶段预算。
- 输入/推理/后处理失败从 CPU 输入重新执行该逻辑阶段；已校验的上游提交继续复用。
  Writer 临时失败保留最终结果，单独重试写盘。临时块不构成恢复提交。
- 正常完成、阶段终止失败和 driver 最终关闭后清理临时块；先关闭 worker，再删除其文件。
  中断时保留已提交产物及运行历史，恢复时重新构建未提交阶段。

这些预算不等于 OS RAM 限额。模型权重、Torch/NumPy 临时副本、Ray 对象库、页缓存、转码与 debug
目录仍有独立占用。固定阶段预留偏保守；超大视频/视图需要显式评估预算，系统不偷偷减少输出。
输入缓存的 run 级 LRU 预算与这些短生命周期交接块是两个独立层次，见 [input-cache.md](input-cache.md)。

## 分项计时与追溯

`phase_profiling` 保留 wall、进程 CPU 核秒及已初始化 CUDA 当前 stream 上的事件时间。
不会为计时初始化 CUDA，不在每个小操作后同步；在请求边界解析事件。一次请求最多保存 2048 条详细事件，
聚合计数继续累计，超出/未观测 CUDA 次数明确记录。CUDA events 测的是同一 stream 两个事件之间的经过时间，
也可能包含 CPU 提交间隙，并非 kernel 忙碌时间之和；不能用于推算 GPU 利用率。嵌套区间和 CPU/GPU 区间可能重叠，不能直接求和。

记录关联 invocation、clip、stage、operation、worker PID、trial、attempt。阶段产物新增可校验的
`phase_<stage>.json`；失败区间的详细 trace 写入 work/phase-failures。生产运行重试成功且未选择保留 work 时，
这些临时 trace 随 work 清理，持久的 Retry/Failed 事件与错误文本仍留在 invocation 历史。
逐次预检另保留每步 request、response、运行环境与错误，不用成功记录覆盖失败尝试。
计时覆盖准备转码、读取依赖、取帧/seek/decode、crop、加载、HaMeR H2D/forward/D2H、SAM2 传播与 mask 编码、
Fast3R 官方前向含传输及 PnP、FP 注册/跟踪含渲染、临时块 IO、序列化、Writer 压缩/哈希与队列等待。
FP 内部渲染尚未独立于注册/跟踪测量，Fast3R 官方 API 的传输也未冒充独立核函数计时。
阶段 commit 自己的耗时只能在提交后观测，记录在 invocation 事件，不能递归写进其自身不可变收据。

`gpu-probe` 在拆分模式下仍使用现有 runner，分别保留 CPU/GPU/CPU 观测。
GPU step 时间与完整步骤时间分开；CPU runner RSS 和 GPU runner RSS 分开记录，不把峰值之和冒充同时峰值。
`--seed 0` 是显式诊断选项，固定 Python/NumPy/Torch/OpenCV RNG；不保证所有第三方 GPU 算子严格确定性。

## 显式代价调度

```bash
python -m egoworld.cli schedule-calibrate --reports REPORT_A REPORT_B --output costs.json
```

每阶段选择一份成功、确认模型复用、无其他计算 PID 的冷/热报告，保留原报告 SHA、输入规格、设备身份、资产与参数身份、
冷加载成本、热服务时间样本和输出大小。选择 `scheduling_policy: cost` 并指定 `scheduling_profile` 后才使用。
加载项优先采用 `model_load_seconds`；它没有完整建模外部进程启动、环境导入与每次协议开销，
也不预测 CPU 池的排队。因此这是可解释的首版成本估计，不是端到端完成时间预测器。
启动时检查 EgoWorld 包源码、阶段参数与版本、显式/已收录隐式资产；不匹配明确拒绝。
上游运行环境身份保留在源报告中，仍须按同一设备、库版本与输入规格解释测量。每次运行只对每个资产路径验证一次。

调度先满足依赖、资源槽位、数量和字节预算。超出最大等待的候选优先，之后比较可用 worker 的模型加载成本、
按工作量线性估计的服务时间，以及片段完成后退出活动 prepared 准入统计的字节。释放项的优先级奖励有上限。
当 `keep_prepared: false` 时，这也对应片段收尾清理的准备文件；设为 true 时，文件继续保留在盘上，
只能把该项解释为活动额度退出，不能称作物理磁盘空间释放。它不估计所有结果包、模型堆或页缓存。
每次选择把候选值、选中 worker、校准哈希与理由写入 events.jsonl。
`cost_terms: [load, service, release]` 可以显式移除一项做消融。没有匹配数据时不自动猜校准值。

工作量估计不是所有分辨率/对象数下的性能模型。GPU 步骤开始前，拆分模式会报告实际 crop/对象/视图数量；
同模型不同分辨率仍可能偏离线性估计。成本策略只改变顺序，不做扩缩容、抢占或多模型显存共驻。
默认仍使用已有驻留窗口与等待保护的启发式策略。

## 验证入口

`egoworld/scripts/benchmark_throughput.py` 生成独立的 baseline/profile/split 配置，保留每次冷调用、预热与热调用。
`egoworld/scripts/compare_throughput.py` 比较全部热调用，包含基线自身波动和明确的回归容差；不据单个最快值判断收益。
`egoworld/scripts/benchmark_schedule.py` 对显式片段窗口运行驻留窗口和代价项消融；实验快照固定使用
常驻 runner，避免继承 smoke 的 `per_task` 而使驻留对照失效。它记录每次独立 run、队列、加载、
进程树 RSS、整设备 VRAM 与本 run 临时文件，并检查每个请求片段完成；失败会返回非零退出状态。
单元测试检查搬运次数、RLE 像素、左右手/尾批映射、memmap 关闭、预算与成本公平性。
真实 Ray 测试检查跨资源重叠、内部步骤/Writer 重试、预算释放和恢复。
实际单卡结果与多片段/消融记录另见验收报告；不得将逻辑多 GPU 资源测试当作物理多卡扩展结论。

本轮核心与真实 Ray 回归命令（已配置项目测试环境、FFmpeg 和 Ray）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=egoworld/src:egoworld/tests EGOWORLD_RAY_TEST=1 \
  python -m pytest -q egoworld/tests -p no:cacheprovider \
  -m 'not env_smoke and not pipeline_smoke and not sam2_smoke and not native_models'
```

原生模型对照和上述回归应分时运行，避免 CPU/Ray 测试负载污染性能测量。
最终结果和版本说明见 [单卡验收](throughput-validation.md)。

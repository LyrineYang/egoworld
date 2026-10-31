# 可追溯阶段回放与冷/热 GPU 预检

本功能沿用现有 `RunnerSession`、`egoworld.worker`、算子和阶段 Writer。
每次诊断创建独立目录；不修改源 run、源配置、采样频率、精度、batch size 或生产 worker 数。
`doctor` 仍负责环境/资产检查，`profile` 读取已有记录；`replay` 和 `gpu-probe` **会实际运行模型**。

## 回放一个阶段

```bash
egoworld replay \
  --config /absolute/config.json \
  --source-run /absolute/output/run_id=source \
  --clip-id ORIGINAL_CLIP_ID --stage hamer \
  --output-dir /absolute/replay-results
```

回放只执行指定阶段。其直接与传递上游（例如 Dex 的 HaMeR、SAM2）从原 run 的提交文件读取：

1. 检查源 run 的 schema 与原始视频/片段清单哈希。
2. 检查所选视频的完整 SHA-256，以及源帧/fps/时间对应关系。
3. 检查上游提交指纹、文件大小、SHA-256、Parquet schema 和行契约。
4. 将清单及上游文件复制到本次 `inputs/`，复制后重新校验；读取目录以本次快照为准。
5. 在本次目录准备 clip，并记录准备文件哈希、分辨率和帧数。FoundationPose 另复制实际需要的深度帧。
6. 通过隔离 runner 执行并用现有 Writer 提交到本次 `trials/`，不接受源 run 中已存在的目标阶段作为缓存。

只要上游已提交，源 run 可以尚未完成；不要求最终 `meta.json` 存在。
当前需要同一 schema 版本，源文件及模型资产仍须可读。清单缺少视频 checksum 时拒绝回放。
本功能是**显式固定上游结果的阶段实验**，不实现生产调度的跨 run 自动缓存或改变后的下游自动失效。

新流水线调用将清单快照保存在 `invocations/`。旧版 smoke 的 `videos.jsonl` / `clips.jsonl` 和
launch 清单位置会自动尝试；其它旧作业可显式传入：

```bash
  --video-manifest /absolute/video_manifest.jsonl \
  --clip-manifest /absolute/clip_manifest.jsonl
```

这两个文件必须匹配源 run 已记录的哈希，不能换一份手工编辑过的清单绕过检查。
目标模型采用本次 `--config`，可以和源 run 不同；报告同时保存源 run 身份和当前模型身份。

## 冷启动、预热和热调用

```bash
egoworld gpu-probe \
  --config /absolute/config.json \
  --source-run /absolute/output/run_id=source \
  --clip-id ORIGINAL_CLIP_ID --stage sam2 \
  --output-dir /absolute/gpu-probes \
  --gpu-device 0 --warmup 1 --repeats 3 --timeout-s 180
```

默认执行 **一次冷调用 + 一次预热 + 三次热测量**，每次使用同一准备输入与固定上游快照，分别提交结果。
冷调用从新的模型进程开始；后续调用复用该进程及模型权重，算子原有的片段/对象状态隔离保持不变。
“冷”不表示清空操作系统页缓存、驱动/kernel 缓存或将 GPU 恢复到某种温度/功耗状态。
热测量必须报告 `model_cache_hit=1`，否则整次探针失败，不报告未经验证的热模型收益。
`replay` 则只执行一次冷调用。Dex 也可作为 CPU 阶段单独诊断，报告不将其算作 GPU 工作。

`--gpu-device` 是当前 `CUDA_VISIBLE_DEVICES` 内的可见序号，默认 0；只向 runner 暴露该设备，
并在诊断配置副本中将 CUDA 设备映射为 `cuda:0`。生产配置文件保持原样。
不同阶段/卡型应分别调用；不会自动遍历多卡、并行加载模型、自动调整采样或进行显存共驻。
请在设备空闲时采样：全设备观测会包含其它进程和桌面显存。

内置 `python -m egoworld.worker` runner 自动使用已有 `--serve` 协议；自定义 runner 必须在配置中
显式选择 `external_runner_mode=persistent` 并实现该协议及模型缓存统计。一次性自定义 runner 的生产行为不变，
但本版诊断不尝试把它转换成热模型服务。

## 每次诊断的产物

```text
<output-dir>/<invocation-id>/
├── report.json                  # 状态、身份、所有试验及资源建议
├── config.supplied.json         # 用户配置的解析快照
├── config.execution.json        # 仅供诊断的设备/观测/输出路径覆盖
├── inputs/                      # 原清单、源 run manifest、已校验上游及可选深度
├── prepared/clip.mp4
├── logs/                        # 现有进程守护/runner 日志
└── trials/000-cold/...           # 以及 001-warmup、002-hot 等
    ├── request.json / response.json
    ├── record.json              # 单次计时、运行环境、结果文件及错误
    └── outputs/stage_<name>.json
```

每次 CLI 调用均生成新 ID；失败和超时不会覆盖先前成功记录。CLI 成功退出 0，诊断失败退出 1，
用户中断退出 130。超时/失败关闭现有 runner 会话及其进程组；没有自动减配重试。
硬杀协调进程或机器掉电可能留下 `running` 记录，已写入的请求、阶段提交与逐次记录仍可检查；
本功能不提供断电持久性或自动续接诊断会话。

报告记录：项目代码哈希、配置、模型版本及资产哈希、上游源码提交/改动、输入规格、实际 runner 的 Python/包版本、已导入模型包的 Python 源码哈希、
GPU 名称/容量/架构（可用时）、设备 UUID 与驱动（可用时），以及上下游身份关系。
当前只保留报告，不自动命中报告缓存。代码、环境、资产或输入规格变化后，应重新测量。

## 时间和资源的解释

- `runner_roundtrip_seconds`：从协调端发起到返回完整结果，包括冷进程启动、协议和结果序列化。
- `synchronized_stage_wall_seconds`：模型进程内阶段调用墙钟；已有 CUDA 工作在边界显式同步。
  它包含加载、解码、预处理、GPU 运算和后处理，**不是纯 GPU 前向时间**。当前尚未对各模型内部逐项插桩。
- `model_load_seconds`、`model_cache_hit`、`process_cpu_seconds`、`result_bytes`：复用现有算子/runner 统计。
- `write_seconds`：本次独立 Writer 的写出时间；固定输入准备时间单独记录，不混入热模型时间。
- `process_tree_peak_rss_bytes`：Linux `/proc` 采样的模型进程及后代 RSS 总和；共享页可能重复计数，短峰值可能漏掉。
- `torch_peak_*`：PyTorch allocator 观测；不涵盖全部 CUDA/渲染器分配。
- `device_used_peak_bytes`、`device_free_min_bytes`：可选 `pynvml` 设备级采样。安装提供此模块的 `nvidia-ml-py`
  可启用该路径；未安装、驱动不支持或无法匹配实际设备时记录原因，并把设备容量建议标为 `unknown`。
  数字设备序号不直接当作 NVML 序号，优先用实际 CUDA 设备 UUID 匹配。
- `device_memory_samples` 与 `device_memory_coverage`：区分从调用开始观测、CUDA 初始化后才匹配设备和不可用。
  UUID 可见性允许在 CUDA 初始化前记录基线；数字序号不会被擅自当作 NVML index。
- `device_gpu_utilization_percent_*`、`device_power_watts_peak`、`device_compute_pids_observed`：设备级辅助信号。
  WSL/MIG 不支持某项时单独记录错误，不填零，也不因此抹掉有效显存观测。利用率均值是样本均值，不是模型 GPU 时间占比。
  同时记录 runner PID、非 runner 计算 PID、线程环境和动态库路径；其它 PID 可能是并行作业或 runner 子进程，
  需要人工核查。未发现其它计算 PID 仍不能排除桌面图形负载。
- `retained_artifact_bytes`：本次结束时的保留文件大小，不是运行过程的磁盘峰值。

采样间隔默认 0.1 秒，记录采样次数；显式同步和采样本身有开销。
冷启动和热调用必须分开比较，不能拿原有短 smoke 的冷启动墙钟当服务器稳态吞吐。
还应人工确认目标非空、样本长度/分辨率/对象数量具有代表性；固定样本多次重复不能覆盖所有长尾工作量。

NVML 绑定可用 `pip install '.[diagnostics]'` 安装到**实际模型 runner** 环境；只安装到协调端不足以观测隔离模型进程。
也可以安装到独立 `--target` 目录并显式加入 runner 继承的 `PYTHONPATH`。报告记录实际绑定版本与驱动版本。
显存采样失败后的状态为 `partial`，资源余量建议保持 `unknown`。

若显式启用 [共享输入缓存](input-cache.md)，同一探针的冷调用构建输入缓存，预热/热调用可复用；
因此必须将“仅模型热”和“模型与输入都热”两种实验分开。用两份显式实验配置做开/关对照，生产配置不自动改变。
每次调用保留命中/淘汰/构建/读写统计，退出清除缓存数据并记录 `released_input_cache_bytes`，其它可追溯产物保留。

## 建议的边界

`recommendations` 只给建议，并明确 `automatic_resource_plan=false`、`configuration_modified=false`：

- 保持每卡一个串行 GPU worker；不根据 Torch 峰值推算多模型共驻数。
- 给出观测结果 payload 的最低预留建议（最大值 × 1.25），不把它当完整 RAM 工作集或自动写入 `stage_memory_bytes`。
- 完整设备观测可用时，以 `max(512 MiB, 15% 容量)` 的明确启发式检查余量，标记 `low` 或 `observed_for_this_sample`。
  缺失观测或任一试验失败时为 `unknown`；这些状态均不保证不会 OOM。
- 记录热调用中位数、缓存复用证据及空输出次数。不会用少算帧、减少视图或更改精度制造提速。

## 保存生产流水线的每次性能记录

`run` / `launch` / `smoke` 使用的 driver 现在为每次调用保留独立目录：

```text
run_id=…/
├── run_summary.json             # 最新调用，兼容旧读取入口
├── events.jsonl                 # 累计事件，每行带 invocation_id
└── invocations/<invocation-id>/
    ├── run_manifest.json
    ├── videos.jsonl / clips.jsonl  # 或原 Parquet 格式
    ├── events.jsonl
    └── run_summary.json
```

恢复运行会产生新的 invocation，原首次推理的时间不再被覆盖。
失败或用户中断时也保留已提交阶段统计。旧目录迁入时，最后一份旧 summary 保存为 legacy 记录；
此前已被旧程序覆盖的历史无法追补。配置/清单校验在执行前失败不算一次已启动的流水线调用。

```bash
egoworld profile --run-dir /absolute/output/run_id=source --history
egoworld profile --run-dir /absolute/output/run_id=source --invocation INVOCATION_ID
```

不带选择参数时仍查看最新调用。`gpu-probe` 自身的完整记录直接查看其 `report.json`。

## 验证

```bash
pytest -q egoworld/tests/test_diagnostics.py egoworld/tests/test_invocation_history.py
```

专项测试使用真实 FFmpeg 视频、Parquet 提交和隔离子进程，模型是明确的 CPU 替身。
覆盖：冷/热复用、不跳过已有目标提交、源输入篡改与缺失提交、传递依赖、固定输入副本、模型身份变化、
失败/超时/中断清理、未知设备观测、实际 UUID 匹配、CLI 状态和恢复后的历史保留。
真实 GPU 测量另外记录，不能用逻辑 GPU 或替身测试声称显存与吞吐达标。

## 2026-11-01 本机真实预检

[原始指标摘要](benchmarks/gpu-probe-2026-11-01.json)：RTX 4060 Laptop 8GB，既有 3 秒/30 帧 RGB 样本，SAM2/DINO。
一次冷调用、一次预热、三次热调用均通过，缓存命中为 0/1/1/1/1；冷阶段同步墙钟 10.31 秒，热调用中位数 4.81 秒。
五次输出均为 masks 30 行、instances 60 行、tracks 60 行，三个 Parquet 文件的 SHA-256 在五次调用间一致。
实际模型环境缺少 `pynvml`，设备级余量保持 `unknown`；PyTorch 分配器数据已保存。
这验证了该样本的真实模型驻留、逐次提交与诊断闭环，不证明长序列容量、多 GPU 扩展或整条流水线提速。

本轮集成验证：将本功能与主开发目录已有的模型适配、虚拟环境和 Linux Ray 清理修复组合，
在独立临时快照执行 CPU / 真实 Ray 回归，结果 **192 passed / 11 deselected**。
功能分支自身的旧提交基线尚不包含 `backend.py` 中关闭 Ray 自动杀子进程的已有修复；
该修复是 Linux Ray 退出清理回归通过的前提，合并时应一并保留。以上为上一轮验证；
本轮新增缓存接入涉及少量共用文件，需要三方合并与集成回归，不能仅靠 worktree 判断无合并冲突。

## NVML 与三模型补测、缓存对照

2026-11-01 续测：[逐次指标与身份摘要](benchmarks/nvml-cache-2026-11-01.json)。
使用包含另一开发任务模型适配修正的隔离集成快照；三份实验源码归档均核对了报告中的 `source_tree_hash`。
实际 runner 通过隔离依赖目录加载 `nvidia-ml-py 13.615.71`，驱动 617.42，GPU UUID 匹配。
原始失败、受竞争影响的初轮记录和后续复测全部保留；下面采用复测时未发现其它计算进程的结果。
桌面图形负载仍可能计入全设备观测。

每项为一次冷调用、一次预热、三次热调用，全部成功并验证权重复用 0/1/1/1/1；以下**关闭输入缓存**：

| 阶段/输入 | 冷阶段墙钟 | 热阶段中位数 | NVML 全设备峰值 | 模型进程树 RSS 采样峰值 |
| --- | ---: | ---: | ---: | ---: |
| SAM2/DINO：3 秒、30 帧、426×206 | 11.578 s | 4.331 s | 3.47 GiB | 3.20 GiB |
| HaMeR：同片段，6 个采样手框，batch=1 | 21.789 s | 0.360 s | 6.79 GiB | 3.47 GiB |
| FoundationPose：640×480 RGB-D、3 帧、1 对象 | 18.227 s | 8.886 s | 6.40 GiB | 9.53 GiB |
| Fast3R：同 RGB 片段，2 views、224、BF16 | 12.722 s | 2.489 s | 5.41 GiB | 5.32 GiB |

SAM2 每次 masks/instances/tracks 分别 30/60/60 行；HaMeR 6 行、FP 3 行、Fast3R 2 个视图。
NVML 显存观测每次从调用开始有效。FoundationPose 最初缺 `libusb` 的动态库路径，补入已有本地库路径后通过；
没有替换权重、调低精度、缩减采样或自动降低 batch。
HaMeR 最小设备余量约 1.20 GiB，接近当前 15% 的启发式余量；FP 的主机工作集明显大于结果 payload。
这些数据支持继续采用单卡单串行 worker，并要求整机 RAM 另覆盖 Ray/准备/Writer/系统；不能从短样本推算多模型共驻。

缓存对照的热阶段中位数：

| 场景 | 关闭缓存 | 开启缓存 | 观测 |
| --- | ---: | ---: | --- |
| HaMeR，缓存与报告都在 `/mnt/c` | 0.360 s | 0.669 s | 热调用每次 6 帧＋6 crop 命中，但挂载盘读回/校验与锁开销抵消收益 |
| FP，缓存与报告都在 `/mnt/c` | 8.886 s | 9.680 s | 每次命中 3 帧；没有据此得到提速结论 |
| Fast3R，缓存与报告都在 `/mnt/c` | 2.489 s | 2.497 s | 每次命中 2 帧，整体差异很小 |
| HaMeR，显式 `directory=/tmp/egoworld-validation-input-cache` | 0.358 s | 0.335 s | 同一新快照上的本地存储对照；只有三个热样本，仍需长负载重复确认 |

因此缓存保持**默认关闭、用户显式选择位置和预算**。本次测量证明了正确复用与可控生命周期，
没有证明对所有模型/存储默认开启会提高吞吐。
HaMeR 开关缓存的五次 Parquet SHA 均一致；HaMeR 官方 crop、Fast3R 官方 loader 和 FP RGB 预处理均做了字节对照。
FP 不同进程输出存在小量差异；Fast3R 在重复基线中已存在点云、相机和 PnP 数值波动。
摘要保留开关对照与重复基线的数值差异，未将它们自动判定为新的模型质量验收通过。

另运行真实 SAM2→HaMeR 与 Fast3R 流水线并验证恢复：HaMeR 命中了 Fast3R 已缓存的一个共同帧，
缓存高水位约 6.26 MiB，片段/流水线结束后数据缓存文件为零，阶段结果及性能历史保留。
不同 OpenCV 构建的共享边界见 [输入缓存](input-cache.md)。

最终验证：本轮缓存专项 20 项通过；与另一任务完成后的最新改动三方合并无冲突，
完整 CPU/真实 Ray 回归 **231 passed / 11 deselected**。源码仍保留在独立 worktree，未覆盖主开发目录。
第三部分的下一步详见 [吞吐推进详解](throughput-next.md)。

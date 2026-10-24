# 调度与高吞吐：研究结论和下一步设计

复核日期：2026-10-24。原始目标与本阶段交付见 [复盘](overview.md)，输出含义先看 [输出清单](outputs.md)。
本文明确区分 **当前代码** 与 **推荐、尚未实现的改进**。服务器性能没有实测结论。

## 1. 用一个例子理解并发

片段 A 正在 GPU 上分割时，CPU 可以准备片段 B，Writer 可以保存已结束的片段 C；
若有第二张卡，还可处理已完成分割的片段 D 的手姿态。一个片段中的视频传播/物体跟踪仍按时间顺序。

| 资源 | 同一时刻可做的工作 |
| --- | --- |
| CPU 预处理 worker | 读视频、准备下一个任务所需输入 |
| GPU worker | 已加载的模型前向与片段内时序计算 |
| CPU 后处理/retarget worker | 编码 mask、整理结果、机器人运动学 |
| Writer | 压缩表、提交文件、返回小型确认记录 |
| driver | 决定谁已就绪、哪里还有容量、哪个阶段需恢复 |

任务的依赖关系是：分割 → 人手姿态 → 机器人映射；分割 → 物体姿态；场景重建可以独立执行。
可选分支由最终输出要求决定，不应因为模型已接入就全部启用。

## 2. 当前代码究竟做到了什么

- PrepareActor 在 CPU 上用 FFmpeg 提取并重新编码 clip.mp4，随后检查帧数/fps。**它没有把模型需要的帧/裁剪张量全部提前准备好。**
- ModelActor 占有一张 GPU，执行该阶段的整个 run，包括不少 CPU 工作；默认仅缓存一个模型，空闲时偏好同模型任务。
- 不同阶段经 Writer 提交结果后再解锁依赖，支持局部恢复。Writer 消费 ObjectRef，driver 不取回大批结果。
- 活动片段数、准备数、计算数、未提交结果数均有限制，但不是按内存字节限制。
- server.json 的四卡固定分工是可运行图的模板，不是根据阶段速度求得的最优分配。
- GPU 模型阶段之外还设有 CPU retarget worker；当前 max_in_flight_gpu 实际也限制它，命名不够准确。

恢复规则：进入重算前先删除失效阶段、依赖下游和最终 meta/QC 的提交标记，保留独立有效阶段。
Ray 模型调用返回两份引用：小型完成状态与大结果；driver 只读取前者，错误隔离后才复用槽位，结果直接给 Writer。
分环境 runner 有独立守护进程，通过父 actor 持有的管道感知退出；管道关闭、正常结束或超时都会清理模型进程组。
这不改变五阶段输出 schema，也不提供多 driver/exactly-once 或断电后的文件系统持久性保证。

当前实现入口：[engine.py](../egoworld/src/egoworld/pipeline/engine.py)、
[backend.py](../egoworld/src/egoworld/pipeline/backend.py)、[stages.py](../egoworld/src/egoworld/pipeline/stages.py)。

## 3. 代码复核发现的吞吐风险

| 位置 | 目前发生的工作 | 为什么可能卡住 | 推荐处理 |
| --- | --- | --- | --- |
| 视频准备 | 解码后 libx264 无损重编码，再写盘 | CPU 编码和中间文件 IO；加 worker 可能扩大磁盘压力 | 比较直接读源片段、现有转码、共享采样帧；按实测选 |
| SAM2 | CPU 收集提示帧，再由 predictor 加载视频；模型内做 mask 编码 | GPU actor 在等解码、CPU 编码和数据搬运 | 提示帧预取、掩码后处理拆出；保留官方时序状态管理 |
| SAM2 掩码 | union 与实例路径分别取 CPU mask | 重复 GPU→CPU 搬运与遍历 | 尽量只搬一次，GPU 上先阈值化，CPU 编码复用结果 |
| HaMeR | 同 GPU actor 内寻帧、裁剪、加载 dataset、小批前向 | 跨帧手框没聚成批；裁剪可能让 GPU 间歇等待 | CPU 提前裁剪；按输出频率收集有限跨帧 crop batch |
| FoundationPose | 逐对象读深度、读 RGB、加载 mesh，再做跟踪 | 同帧多对象重复 IO；几何加载与时序推理混合 | 按帧复用 RGB-D；缓存不可变网格；每对象 tracker 保持独立 |
| Fast3R | 寻帧、写 JPEG、官方 loader 再读、前向、CPU PnP | 临时图片与 CPU PnP 占用 GPU worker 时间 | CPU 采样/预处理；前向与几何后处理分别计时和调度 |
| 依赖表读取 | HaMeR/FP 读取 SAM2 整表；Dex 读取手姿态整表 | 不需要的 RLE/MANO 列也被反序列化 | 按所需列读取；保留阶段文件作为恢复依据 |
| Python 结果 | 大量 dict/list/string 经 Ray 序列化 | ObjectRef 省去 driver 拷贝，但不等于 worker 零拷贝 | 热路径逐步用列式/紧凑数组，按 CPU profile 判断 |
| 状态/小文件 | 逐事件 SQLite 连接/提交，日志 flush，多份 Parquet/JSON | 大量短任务时产生控制与小文件开销 | 先测；必要时复用连接、合并普通事件写入，提交记录仍可靠 |
| 资源预算 | GPU actor 申请 1 个 CPU，Writer 申请 0.5 | 实际 OpenCV/FFmpeg/Torch/PnP 线程和内存未受这个数硬限制 | 显式线程上限、内存预算与系统余量 |

以上是代码路径可见的工作，**不是已经测得的瓶颈排名**。当前 inference_seconds 包含模型加载、CPU 预处理、
前向和后处理，不能据此认定 GPU 计算时间占比；必须拆分计时再决定优先级。

本机 CPU 探针（合成 testsrc2，4 秒、1280×720、30 fps，3 次重复的中位数，macOS/ARM，Python 3.13.2 / OpenCV 4.12）：
源文件约 1.53 MB，当前 prepare 约 0.550 秒，产物约 9.43 MB；对产物四次全帧解码合计约 0.459 秒。
直接顺序解码源文件一次约 0.112 秒。该实验仅显示转码与重复解码的成本，**没有测共享缓存构建、模型耗时或优化后全流程**，
不能把两组时间相除宣传提速，也不能外推服务器配置。

## 4. 成熟开源方案比较

| 方案 | 能借鉴什么 | 本项目适配代价/边界 | 判断 |
| --- | --- | --- | --- |
| **Ray Data** | 流式块执行、下游承载控制、常驻 actor 批推理、运行统计 | 阶段级提交、分支依赖和时序输入仍需适配；一行塞整段大视频也可能过大；单卡多模型常驻池需专门安排 | 最值得做小范围后端对照的通用方案 |
| **Cosmos Curator / Xenna** | 视频阶段拆分、按阶段吞吐平衡 worker、模型环境隔离与预取 | 接管现有 stage/配置/恢复有迁移成本；不能照搬大 HGX 的资源默认值 | 最接近本项目负载的设计参考 |
| **Data-Juicer** | 配方与算子边界、减少重复处理、兼容算子融合、采样试跑 | 当前重点是有状态几何模型；完整引入大量算子/抽象未必省代码，融合和恢复也有配置约束 | 借鉴复用与融合原则，暂不整框架迁入 |
| **Triton** | 无状态请求动态批处理、有状态序列路由 | 把时序状态、前后处理和文件提交接成服务需要额外工程 | HaMeR 未来服务化候选；不是当前整个 driver 的替代品 |
| **DALI / NVDEC** | GPU 硬件解码、视频预取与批处理 | 不解决模型依赖、恢复与 worker 分配；编码格式/卡型/主机拷贝会影响收益 | CPU 解码已证实为瓶颈时试验 |

依据与边界：

- [Ray Data 内部机制](https://docs.ray.io/en/latest/data/data-internals.html)描述了流式执行与内存/队列控制；
  [批推理](https://docs.ray.io/en/latest/data/batch_inference.html)给出模型常驻 actor 的用法。它不能自动消除模型函数内部 CPU 工作。
- [Cosmos 架构](https://github.com/NVIDIA/cosmos-curator/blob/main/docs/curator/reference/architecture.md)与
  [流水线设计](https://github.com/NVIDIA/cosmos-curator/blob/main/docs/curator/guides/pipeline-design.md)强调预处理分离和阶段平衡。
  [当前变更记录](https://github.com/NVIDIA/cosmos-curator/blob/main/CHANGELOG.md)也包含 Ray Data 路径，不能将所有 Curator 功能都归为同一个 Xenna 引擎。
- [Data-Juicer 处理文档](https://datajuicer.github.io/data-juicer/en/main/docs/ProcessData.html)说明了兼容算子融合；
  default executor 的 checkpoint 与 op_fusion 不能简单同时打开，不宜照搬为“融合与恢复天然兼得”。
- [Triton batcher](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md)区分无状态动态批与有状态序列批。
  SAM2/FP 的 Python 状态不能直接通过打开 dynamic_batching 配置解决。
- [DALI 视频 reader](https://docs.nvidia.com/deeplearning/dali/archives/dali_1_48_0/user-guide/operations/nvidia.dali.fn.experimental.readers.video.html)
  支持 FFmpeg/NVDEC 解码；此处只确认机制，未验证本项目收益。

## 5. 推荐的设计方向（待按输出需求实施）

**近期保留一个 Ray Core 后端和现有阶段提交协议，先减少 CPU 重复工作、限制内存，并改善模型驻留安排。**
引入成熟项目的少数机制，不先编写完整自动扩缩容系统。Ray Data 只做一条代表路径的对照实验；
若后来迁入，替换现后端，避免长期维护两套生产调度器。

### A. 将“逻辑输出阶段”与“执行工作”分开

逻辑阶段仍是分割/人手/物体/机器人/场景，方便追溯、恢复与需求对照。
每阶段内部按需要拆成：CPU 输入准备 → GPU 前向或时序推理 → CPU 后处理与 Writer 提交。
并非为每个小函数创建一个 Ray task；一次任务处理有界帧块/一批手框，避免调度开销超过计算。
HaMeR 的裁剪可以跨帧合批；SAM2 与 FP 维持每片段/对象的顺序，优先跨片段并行。

### B. 数据复用采用有上限的缓存

- 原始视频保持不可变；优先复用相同源帧的采样/裁剪结果，模型专用归一化各自完成。
- 只为实际消费者缓存需要的分辨率和帧，不能把所有视频全量展开到内存。
- 粗估：15 秒 × 30 fps × 1920×1080 × RGB uint8 约 **2.61 GiB/片段**，转 float32 还会更大。
- 完整原分辨率帧缓存可用短生命周期 memmap/本地 NVMe，按 RAM/磁盘预算释放；小型 crop/采样张量可走内存。
- SAM2 原生 loader 未改造前，允许其保留一条专用解码路径；先消除 HaMeR/FP/Fast3R 重复读，不承诺立即做到全局只解码一次。
- 下游优先读所需列；默认省去可从实例 mask 重算的重复结果，是否保留 union mask 由兼容需求决定。

### C. 单卡按模型处理一批片段

默认显存不足以确认四个大模型共存，因此单卡的稳妥吞吐方案是：
同一模型连续处理一批已就绪片段，再切换到下一模型；CPU 同时准备输入/做 retarget，Writer 同时落盘。
批次大小 W 由内存/临时盘预算决定，先比较 W=1/4/8，而不是盲目增加活动片段。

单卡一个片段需要的 GPU 工作量基本要相加：t_seg + t_hand + t_object + t_scene。
W 越大，加载成本越能摊薄，近似为每片段增加 Σ模型加载时间 / W；但排队延迟和中间态占用也会上升。
此处是容量估算，实际阶段可能含 CPU 等待，不应直接套用当前 inference_seconds。

如果模型只能放在不同环境：一个阶段的模型进程处理整个批次，再释放 GPU 并切换。
**不能在一张卡上同时启动四个各申请 num_gpus=1 的常驻池**，它们会竞争同一逻辑资源。
当前 runner_command 是每任务重启进程，仅作兼容性通路；长期方案应将启动成本按批次摊薄。

### D. 多卡按工作量分配，不固定一模型一卡

固定分工适合稳定吞吐，但前提是每个阶段都够忙。
例如假设每片段四阶段分别需 8/2/6/4 GPU 秒（纯示意），四卡各一模型的吞吐上限受分割阶段 1/8 片段/秒约束，
其他卡会有空闲；总 GPU 工作量为 20 秒/片段，四卡的理想总算力上限才是 4/20，实际还受 CPU、磁盘与依赖影响。

先使用可解释的人工阶段分组或按批次切换，把更多卡/驻留时间给慢阶段，资源调整发生在批次边界。
初期不做频繁自动迁移；每次迁移会重载模型，还要检查显存和时序状态是否已结束。
分支不要求即时完成时，可选择整批模型遍历，以更长结果等待换较少切换成本。

### E. 并发同时受任务数、字节数与物理资源约束

当前上限只计任务数；推荐再加入准备缓存字节、未提交结果字节、临时盘占用、阶段自身的峰值工作内存估计。
每次发任务前预留额度，提交/释放后归还；估计误差靠实测峰值与安全余量修正。
Ray 逻辑资源只是调度额度，不是操作系统对线程/RAM 的强制限额；模型库内线程还需显式控制。
这一点见 [Ray 资源约定](https://docs.ray.io/en/latest/ray-core/scheduling/resources.html)。

预处理 worker、模型 worker、PnP、Writer 共用 CPU 总预算。保留 driver/Ray/系统余量，限制 Torch/OpenMP/BLAS/OpenCV 和 FFmpeg 线程；
逐步调整解码、预处理与 Writer 数，不从旧文档直接复制 32–64 workers。
优先完成接近写出的下游任务以释放内存，同时保留模型复用偏好和等待时间上限，避免一个分支长时间饿死。

Writer 确认仍是恢复边界。先保持阶段 Parquet；小文件确实影响吞吐后，再为已结束 run 做一次离线合并，
不为尚未证实的问题同时加入 Lance、数据库和第二套结果索引。

恢复与调试还需区分“同一 run 的损坏恢复”和“更换模型/代码后的跨 run 复用”：当前全局指纹改变要求新 run，
不保证自动复用其它未变化分支。下一步可先增加固定上游结果的阶段回放入口，减少定位一个模型时重复跑其上游。
DexRetarget 本身可以在 CPU 上测试，但现有 native smoke 会为生成输入先运行 HaMeR；提供可追溯手姿态样本后可单独验证它。
每阶段独立身份/跨 run 缓存涉及更多失效规则，不作为本轮已完成能力。

## 6. 如何算 CPU 与 GPU 配比

将同一输出配置下每片段的平均工作量记为：CPU c 核秒、GPU g 卡秒、磁盘 d 字节。
若可用 CPU 为 C 核、GPU 为 G 卡、实测有效磁盘带宽为 D 字节/秒，则稳态吞吐上限近似：

```text
片段/秒 ≤ min(C/c, G/g, D/d, 各固定阶段的容量)
```

这需要固定分辨率、采样频率、对象数与输出集合；这些一变，c/g/d 都会变。
例：4 张卡希望每秒处理 1 个片段，若 CPU 每片段需 12 核秒，预处理/后处理合计至少要能供应 12 核的实际工作，
再留运行余量；多开 actor 并不会凭空增加 CPU 吞吐。

第一个要回答的问题是“GPU 在等输入、在算，还是在等结果编码/写盘”，而不是“worker 数是不是够多”。

## 7. 小实验如何决定方案

| 实验 | 对照 | 保持相同 | 要记录/决定什么 |
| --- | --- | --- | --- |
| CPU 输入路径 | 现有转码 / 直接读源范围 / 共享采样缓存 | 视频、帧范围、分辨率与输出 | CPU 核秒、墙钟、临时盘大小、解码/IO；选择是否保留转码 |
| 线程与预取 | CPU worker 1/2/4，库线程 1/2，预取 1/2 块 | 总机器资源 | GPU 等输入时间、RSS/缓存占用、吞吐；发现过量并发 |
| 单卡驻留 | 逐片段切换 / W=4/8 同模型批次 | 同一模型权重与输出 | load 时间、显存峰值、结果等待与实际吞吐 |
| 手框批处理 | 同帧 batch / 有限跨帧 crop batch | 同样手框、采样与精度设置 | crops/s 与输出数值一致性；不能靠少算帧获得假提速 |
| CPU 后处理 | GPU actor 内编码 / CPU 后处理池 | 同一 mask 与字段 | GPU 空闲时间、搬运量、CPU 峰值、完整提交速率 |
| 多卡分组 | 一模型一卡 / 按阶段速度分组或按批次切换 | 总 GPU/CPU 与同一输入 | 各阶段积压、GPU 忙/等时间、端到端吞吐 |
| 框架对照 | 当前后端 / 一条 Ray Data 原型路径 | 同一模型包装、预处理与输出提交 | 代码量、CPU/内存、恢复行为；确认迁移能否真省工程 |
| 可靠性 | 同一负载中断重启、杀 worker、损坏结果 | 固定 run/资产 | 不丢已提交结果、只重跑必要阶段、状态与最终文件一致 |

无 GPU 时先跑 CPU 路径、合同/故障测试和容量估计；真实单卡分别跑原生模型，再用 2–3 个片段联调，
随后用足够任务排除启动影响、观察多个稳态窗口。实验一次只改变一项，预热与稳定阶段分别记录。
小样本可判断明显瓶颈，不能证明 TB 级可靠性或低于 1% 的真实失败率。

计时至少拆为 load / decode / preprocess / queue / GPU / device-to-host / postprocess / write。
另记录源视频去重秒数、成功覆盖帧数、所选输出、分辨率、对象/手框数、峰值 RAM/VRAM、spill/临时盘、重试与失败。
当前 scheduled_frames_per_wall_second 计入重叠帧与启动时间，不是去重视频吞吐；NVML 利用率也不能替代这些指标。

## 8. 当前机器格式索引

所有逐帧结果带源 frame_index 与 timestamp_s，ID 为片段内 ID。schema 3 的实际字段用 egoworld schema 查看。

| 文件 | 直观含义 | 编码约定 |
| --- | --- | --- |
| masks / instances / tracks.parquet | 分割、独立对象与二维轨迹 | COCO RLE；exclusive xyxy 框；轨迹是框中心，不是专门跟踪模型 |
| hand_pose.parquet | 人手关节点、关节旋转与手型 | OpenPose21；腕部相对位置；米制模型先验；MANO 右手模板旋转，左手需镜像解释 |
| object_pose.parquet | 物体相对相机的位置与朝向 | 行优先 object_to_camera 4×4 变换；平移米；RGB-D/网格有条件 |
| mapping.parquet | 目标机器人关节状态 | 名称、顺序、radians/meters 与源尺度；不是力或动作语义 |
| reconstruction.parquet + NPZ | 抽样相机与点云 | camera_to_world；首视图参考；K 对应模型处理后图像；默认 relative/unscaled |

阶段文件写完再提交 stage_<name>.json，最终 meta.json 与 qc.json 描述片段状态。
完整单卡配置、四卡模板和恢复命令见 [测试说明](testing.md)。

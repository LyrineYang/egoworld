# EgoWorld 与 Data-Juicer：具身视频效率调研

日期：2026-11-01。范围是复现、处理速度、内存和清洗效率，不增加训练任务。

本轮交付是源码核查、真实产物 CPU 实验、差距判断和实施/对照计划；没有改变生产模型、精度、采样或调度。
尚未运行 Data-Juicer 的模型整链，不能声称 EgoWorld 比它更快、更省内存。

## 结论和产品边界

同意现有 13 种类型足够支撑当前产品。用户选择数据、配置参数、自动补齐依赖、复用已有计算即可。
暂不把通用插件系统、任意 DAG、多模型 provider、GUI 或更多数据名称列为必做。
**类型覆盖充分，不等于每一种都可在任意输入上无条件取得。** 世界手需要真实手部度量对应；物体姿态需要 RGB-D/网格；世界物体需要同帧相机。

| 配置名称 | 实际提供的数据 | 使用前提或边界 |
| --- | --- | --- |
| `masks` | 每个实例的 RLE 掩码、标签、像素面积 | SAM2/DINO；图像坐标 |
| `boxes` | 每个实例的二维框、标签、检测分数 | 与 masks 共用同一 SAM2 阶段；不是独立廉价 detector |
| `tracks` | 片段内实例中心、面积、标签的二维时序轨迹 | 不是跨视频身份追踪 |
| `hand_joints` | 21 个腕部相对关节、估计相机平移、左右手 | HaMeR；MANO 先验尺度不等于测量得到的真实尺度 |
| `hand_mano` | MANO 全局旋转、15 指关节旋转、形状参数 | 与 hand_joints 共用一次 HaMeR；不额外导出完整网格 |
| `object_poses` | 物体到相机的 4×4 米制姿态 | FoundationPose；RGB-D、内参、米制网格及匹配标签 |
| `robot_joints` | 机器人关节名、数值、单位、手别和机器人 ID | Dex；当前一个 kernel 配置一侧手/机器人，双手可用两份配方复用人手结果 |
| `camera_poses` | 采样帧相机外参、内参和尺寸 | Fast3R；无标定时是相对尺度，非每帧轨迹 |
| `scene_points` | 场景三维点、置信度、视图编号 | Fast3R；有点数预算，默认非米制 |
| `world_camera` | 显式世界锚点下的米制相机姿态 | 场景尺度标定；只覆盖已有相机帧 |
| `world_scene` | 标定并过滤后的米制世界点云 | 变换和筛选，不是 TSDF/全局地图优化 |
| `world_object` | 米制物体到世界的姿态 | 对象和相机严格同源帧匹配，缺帧为 partial |
| `world_hand` | 米制世界手部 21 关节 | 额外需要逐观测的真实相机空间手关节对应；不能只凭场景尺度生成 |

现状以 [registry.py](../egoworld/src/egoworld/product/registry.py)、[recipes.py](../egoworld/src/egoworld/product/recipes.py)
和 [用户配方指南](data-recipes.md) 为准。字段选择发生在导出时，不减少网络内部工作量。

## 调研版本和证据

Data-Juicer 固定源码 `fc0314b17209e5a2c15a2735d26cfa69d7e3f08d`，2026-10-30 的 main；避免未来文档变更影响结论。
EgoWorld 主工作区 HEAD 为 `e0638a23553d8be935c334248bc3fc539d918ba0`，包含未提交的产品/复现实现。
实验时主包 Python 内容哈希为 `be64881882ed631fe7883f9405ab7c5bc586cc745f748018000b25eb28130fd5`。

调度任务另一个工作区的提交为 `edc72cdbe34e41ea9a70d7242dd53c0b603336f5`，分支 `codex/stage-replay-gpu-probe`。
本轮已读取该任务，并检查其代码、文档和原始测量报告哈希；本轮没有合入它，也没有修改其工作区。
所以“共享输入缓存未实现”不再适合描述整个项目：**独立分支已实现，本工作区尚未集成**。

原始实验、来源快照、逐项时间和哈希均保存在 `/mnt/d/EgoWorld/runs/research-20261101/`：

- `datajuicer-sources/`：固定提交官方文档/实现与来源索引。
- `cache-formats/report.json`：20 个 blob 压缩组合和 8 个 Parquet 组合，全部无损回读通过。
- `input-costs.json`：3 次身份准备复测、4 个指纹变化对照、4 个解码对照。
- `other-branch-measurements.json`：已验证原始报告 SHA 的独立分支性能摘要。
- `identity-cost.json`：早期 2 次探索测量；首次与压缩实验并行，不作为主要时间结论。

## Data-Juicer 能提供哪些我们当前没有的数据

它已经覆盖具身视频，不应只把它看作文本清洗框架。
官方 [算子目录](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/docs/Operators.md)
和 [VLA 示例配方](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/demos/ego_hand_action_annotation/configs/vla_pipeline.yaml)
已经串联帧提取、MoGe 标定、HaWoR 手重建、MegaSaM 相机和动作/LeRobot 导出。

| 它的现成能力 | EgoWorld 当前差别 | 是否补齐及优先级 |
| --- | --- | --- |
| 视频时长、分辨率、运动分数、重复检测等统计和过滤 | 有 ffprobe、内容 SHA、SAM 空掩码 QC；没有完整的可配置前置清洗链 | **P0** 借鉴便宜统计、精确去重和原因记录；先减掉无效 GPU 工作 |
| MoGe/DeepCalib/DroidCalib 内参估计、畸变校正 | 主要使用已知内参；Fast3R 有重建视图内参，非同等校准流程 | **P2，输入需要时补**；估计内参不能替代真实标定精度证明 |
| MegaSaM/MoGe 时序相机和深度、HaWoR 时序手部结果 | 当前 Fast3R 是采样视图，HaMeR 是单帧手估计，自动时序世界手路径缺失 | **P2 可行性验证**；长视频/连续世界手确有需求时升优先级，先验证 8GB 和质量，不为补目录引入重模型 |
| 每帧 8 维 state 和 7 维 action | 我们有几何和机器人关节，没有这种 VLA 状态/增量动作导出 | **P2 按下游需要**；可从已有输出派生，但必须先对齐时间、坐标、尺度、姿态和手别 |
| 世界轨迹平滑、异常替换、动作重算 | 有可视化和数值验证，没有通用时序清洗输出 | **P1 先做异常标记，P2 再提供可选平滑**；保留原数据和修改日志，避免抹掉真实快速动作 |
| 重叠片段轨迹重组、按速度极小值切原子动作 | 有场景切片，没有相同动作切片与世界轨迹重组 | **P2**；只有消费 episode/action 时才值得做，不等同语义动作识别 |
| LeRobot v2/LIBERO 风格数据集和轨迹覆盖帧 | 我们是 Parquet/NPZ/清单、复现离线报告 | **P2**；出现明确消费者后做导出适配，比换底层更便宜 |

动作算子的 [文档及实现](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/data_juicer/ops/mapper/video_hand_action_compute_mapper.py)
把夹爪开合从 MANO 手指弯曲推断出来；这是代理信号，不是采集的机器人控制命令。
其“world”输出也不能据此证明绝对物理尺度。HaWoR 的官方配置还需要左右 MANO 授权资产；当前不能假定已有完整许可文件。
我们的 FoundationPose 米制物体姿态、Dex 特定机器人重定向及严格 URDF/FK 验证仍有独立价值；没有必要为了接近其目录而替换这条复现路径。

## 缓存机制：借鉴原则，按数据层实现

用户关注的原则是合理的：`same input + same operator configuration → same fingerprint → cache hit`。
这里的 input 应包括上游真实内容，configuration 应包括会影响结果的代码、模型、预处理和尺度约定。
不能缩减成“同一路径和模型名称”，也不应把 run ID、日志目录、无关模型版本混入语义身份。

### Data-Juicer 实际做了什么

官方 [Cache.md](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/docs/Cache.md)
描述 Hugging Face dataset 指纹缓存、`use_cache`、目录、`cache_compress: gzip/zstd/lz4` 及上下文控制 API。
进一步核查 [fingerprint_utils.py](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/data_juicer/utils/fingerprint_utils.py)
和 [base_op.py](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/data_juicer/ops/base_op.py)：

- 指纹递归连接输入 dataset 指纹、transform 和参数；对 OP 有专用序列化规则，排除 `work_dir` 等逐 run 属性。
- 无法稳定序列化时使用随机指纹并提示，不能保证命中。
- 这些代码不等于对所有视频外链、模型权重和外部文件自动逐字节做 SHA 校验；是否完整覆盖依赖仍取决于数据和 OP。
- 确认的自动指纹/压缩路径位于 HF `NestedDataset` 的 map/filter 和 default executor；**不能把它直接推广为所有 Ray 执行器拥有同一种持久缓存**。Ray 性能实验需分别确认实际命中和 checkpoint 策略。

[compress.py](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/data_juicer/utils/compress.py)
压缩上一步已不用的 Arrow 缓存文件，移除原文件；复用时解压，再交给 dataset 读取。不是在压缩文件中直接 mmap。
支持多进程压缩和文件锁。它节约磁盘，同时引入压缩/解压、临时空间与 CPU 竞争。
default executor 的 `use_checkpoint` 会关闭这种缓存，且与 `op_fusion` 互斥；Ray partition checkpoint 另有配置，比较时须注明执行器。

### EgoWorld 已有机制和本轮发现

当前阶段缓存已有上游文件哈希、模型内容、环境、参数、源码、输入范围及损坏校验，同 run 恢复和跨 run 复用可以共存。
但当前 key 偏保守，实际函数的受控对照确认：

| 只改变一项 | SAM2 key 是否变化 | 判断 |
| --- | --- | --- |
| Dex 普通算子参数 | 不变 | 正确隔离，可复用 |
| Dex model_versions 版本项 | 变化 | 无关失效，值得优化 |
| 主包整体源码哈希，包括无关模块修改 | 变化 | 保守正确但影响迭代速度 |
| 视频和 clip ID 改名，视频 SHA/帧范围不变 | 变化 | 内容重复处理仍可能 miss |

这些是 `stage_identity` 函数实验，不冒充 GPU 重跑对照。
读取时会验证并复制产物；校验、复制、重复发布检查及导出也有成本。世界标定/世界导出当前仍会重新执行。

**最直接的成本是命中查询前的身份准备。** 真实 chicken 配方读取约 3,595,843,059 字节的不同本地资产，
独立三次 `provenance()` 耗时 **12.87、13.08、13.10 秒**。这是身份准备的墙钟，主要路径是资产内容读取/哈希，不是 GPU 推理；源码处理也包含在内。
环境身份扫描约 0.10、0.09、0.07 秒，当前小得多。资产哈希按块读取，不表示同时占用 3.6GB 内存。
`run_recipe` 会调用 provenance，缺阶段进入 `run_pipeline` 时还会再次调用；不能用 driver 的 elapsed 代替用户入口总耗时。

优化建议：同 invocation 共用一次验证结果；只为阶段收集真实依赖；再建设经首次校验、不可变的内容寻址资产登记。
跨 run 不能只根据 size/mtime 猜内容相同：保留 strict 全内容校验模式，可信不可变存储模式必须说明信任和重验规则。
不直接删除权重、MANO、深度或源码身份来提高命中率。

建议的新键形态（**计划，非已实现接口**）：

```text
K_stage = hash(stage_semantic_version, schema_version,
               input_content_digest + frame_range + timebase,
               relevant_model_and_asset_digests,
               effective_preprocessing_and_model_parameters,
               relevant_runtime_identity,
               upstream_semantic_artifact_digests)
```

stage 模块版本必须覆盖真正的共享代码依赖；不能只 hash 一个入口文件。
去除路径/ID 前应验证并重绑定来源元数据，不能让缓存产物错误标注为另一个视频。
随机种子和精度继续属于结果语义；batch 是否能从键中剔除需先证明数值/质量允许，不能凭其“看起来只是调度”排除。
压缩算法属于存储表示版本，可与语义键分层，但压缩文件及还原后内容都需完整性验证。

### 实测压缩：现有产物并非未压缩

EgoWorld 原始/导出 Parquet 已用 Zstd，NPZ 已用 ZIP/Deflate；当前 cache 复制的是这些已压缩文件。
本轮对真实 chicken 的 30 帧和 mustard 点云做 5 次重复，逐项无损回读。下面是中位 CPU 编解码时间：

| 数据和表示 | 保存大小 | 编码 / 解码 | 结论 |
| --- | --- | --- | --- |
| 30 帧 BGR 的原始 NPY | 7.90 MB | 基准容器已准备，表中 codec none 不含 NPY 构建 | 大数组可压缩，但会失去直接 mmap |
| 同一帧 NPY + LZ4 frame | 6.78 MB，原来的 85.9% | 7.98 / 1.68 ms | 低 CPU 成本，空间收益有限 |
| 同一帧 NPY + Zstd | 6.55 MB，82.9% | 14.00 / 4.37 ms | 比 LZ4 小一些，编解码更贵 |
| 同一帧 NPY + gzip-9 | 5.56 MB，70.4% | 170.48 / 27.19 ms | 更省盘，不适合默认热输入路径 |
| 点云无压缩 NPZ | 389,330 B | 基准 | 可选分层格式的参照 |
| 点云无压缩 NPZ + Zstd | 280,105 B | 0.40 / 0.22 ms | 可作为冷存储候选，仍需大规模实验 |
| 现有 Deflate 点云 NPZ | 249,583 B | 现有实现 | 这里的样本比上述 Zstd blob 更小 |
| 已压缩点云 NPZ 再套 Zstd | 249,598 B | 0.11 / 0.02 ms | 几乎没有空间收益，略增大 |
| instances Parquet：无压缩 → Zstd | 34,465 → 16,066 B | Zstd 0.32 / 0.40 ms | 当前默认已吃到主要收益 |
| hand_pose Parquet：无压缩 → Zstd | 28,841 → 19,609 B | Zstd 0.47 / 0.48 ms | 小产物无需优先折腾 codec |

对已压缩 Parquet 再包 Zstd，样本还能省约 11–12%，但牺牲直接列读/映射，文件很小，通常不值得默认开启。
本轮全流程实验进程 RSS 高水位约 186 MiB，**不是隔离 codec 或生产 pipeline 的内存峰值**。
测量使用 Arrow 自带 Zstd/LZ4 与 Python gzip，不是运行 Data-Juicer 的压缩实现；不能把数值当成其性能。
D 盘 I/O 是暖 OS 页缓存、buffered write、不含 fsync，顺序测量且样本很小；表中比率不外推到任意视频。

结论：**热帧默认不加压缩，继续设预算；Parquet 保持 Zstd；不对 NPZ 默认二次压缩；冷缓存压缩列 P2。**
后续按 `读取节省时间 > 解压时间` 判断命中路径，再把首次编码、写盘、临时空间、复用次数计入总收益。

## 中间计算与内存：针对这五类模型优化

### 已有输入缓存分支揭示的边界

独立调度分支已提供按需共享 BGR 帧、HaMeR 官方 crop、只读映射后复制、SHA 校验、LRU、字节/条目预算、超限旁路、损坏重建和退出清理，默认关闭。
不是零拷贝；FoundationPose 与 HaMeR/Fast3R 的 OpenCV 构建不同，帧缓存保守隔离；SAM2 仍用官方 loader。
这些内容不是本轮新实现，合入仍需正式验证产品缓存和世界输出。

其固定短样本中，HaMeR 热调用在 `/mnt/c` 缓存 **0.360 → 0.669 秒**，变慢；Linux 本地缓存 **0.358 → 0.335 秒**，仅约 6% 改善，样本不足以默认开启。
Fast3R 热调用约 2.489 → 2.497 秒，FP 8.886 → 9.680 秒，不能宣称输入缓存普遍提速。
独立分支观察到 HaMeR 设备显存峰值约 6.79 GiB、FP 进程树 RSS 约 9.53 GiB；输入缓存预算不约束模型堆、复制出的数组、Ray、CUDA 或页缓存。
NVML 数值包括整卡桌面占用，不能和 PyTorch allocator 峰值混用。

大结果/长期缓存继续放 D 盘。用户偏好 D 盘不意味着必须把每个热路径都落到挂载盘；若实测有收益且可用空间允许，再显式选小预算本地 scratch。
不迁移 WSL、不重启服务、不自动更改盘布局；本轮新增实验结果全部在 D 盘。

### 本轮新增解码对照

对真实 30 帧 prepared clip，HaMeR 的 `[0,5,10,15,20,25]` 取帧：
当前随机 seek 中位 **79.60 ms**；从头顺序读到第 25 帧、只保留所需帧为 **26.52 ms**。
Fast3R 两端点 `[0,29]` 为 **33.03 → 25.84 ms**。原视频也验证同方向。
每种方法 7 次重复，所有选中帧的原始像素 SHA 完全一致；这是解码子步骤，**不是整链 3 倍提速**。
长间隔/长视频时，全部顺序扫描可能反而慢，需按采样密度、GOP 和实测选择，而非无条件换成顺序扫描。

### 应优先做的专用优化

| 模块/数据 | 发现与具体优化 | 正确性与资源验收 |
| --- | --- | --- |
| 资产/阶段缓存 | 去重单 run 哈希；按阶段依赖构键；明确 miss 原因；派生 world/导出缓存分离 | 权重内容变更仍失效；无关 Dex 版本不再使 SAM miss；同内容换路径需正确重绑定来源 |
| SAM2 掩码、框、轨迹 | 当前 union 与 instance 路径分别搬运 logits/掩码；一次阈值化/搬运派生所有视图；再测 CPU RLE 外移 | bitwise 掩码/RLE 解码结果、框、面积和实例帧键一致；保留时序 state，不跨 clip 共享 tracker |
| HaMeR 手关节/MANO | 有限 CPU crop 预取；同 clip 跨相邻采样帧凑小批次；按 frame/object/handedness 回填 | 官方 crop 张量和相机元数据完全一致；数值与官方基线对照；限制 crop 数/字节和等待，不减帧 |
| 稀疏 RGB 消费 | 先评估顺序解码/短段 seek；复用采样集合交集；保留解码器身份 | 分环境像素 SHA；内存只保留 bounded ring，不把整视频展开到 RAM |
| Fast3R 相机/点云 | 单独测 JPEG 写读、官方 loader、GPU 和 CPU PnP；证明等价后才换内存 loader | JPEG 路径有损语义；不能直接用原 BGR 替代后宣称同输入；不得只降 views/分辨率冒充优化 |
| FoundationPose | 分离 depth/RGB、首次 register、后续 track、渲染时间；共享只读几何/预测网络，tracker 独立 | 对象互不污染；保留按帧复用、必要副本及真实深度单位；先测再删 copy |
| 表格/点云/世界输出 | 以列投影/iter_batches 代替大表 to_pylist；限制在途包；避免点云和 correspondence 不必要重复装载 | 维持 FP 时间顺序及分手隔离；全部原帧键/单位/校准来源保留；峰值进程树/队列/临时盘分别测 |
| 模型驻留和队列 | 在单卡串行基础上测同模型 W=1/2/4/8 驻留窗口、等待保护和反压 | 比较整链吞吐及尾等待；不能只证明单阶段快，也不能让某个模型饥饿 |

Data-Juicer 的 [OP fusion 实现](https://github.com/datajuicer/data-juicer/blob/fc0314b17209e5a2c15a2735d26cfa69d7e3f08d/data_juicer/ops/op_fusion.py)
值得借鉴的是共享 loaded video/采样帧上下文、合并兼容 CPU 清洗和减少 stage 交接。
其自动 GPU mapper fusion 也要求显式 opt-in、独立输出、相同 runtime_env、显存估计及总预算；不是任意模型相加。
我们的 HaMeR/FP 分环境，且单模型显存已接近 8GB 上限，优先借鉴**共享输入 + CPU 小算子融合 + 模型驻留**，不把五个模型同时装进 GPU。
Data-Juicer 的自适应 microbatch 对兼容 mapper 有参考价值，默认也非普遍启用；不能把 OOM 后缩 batch 直接用于有状态跟踪器。

1080p BGR 一帧约 5.93 MiB，30fps 一分钟约 10.43 GiB，尚不包括 tensor/copy。
因此按用户所选输出算采样需求并限制 CPU 预取，比“全视频先解码，所有模型随取”更适合当前机器。
总 RAM 准入应涵盖 worker 模型堆、解码/crop、FP/SLAM 状态、在途 packet、Ray object store、压缩 scratch；单独 512MiB 输入缓存不是总内存保证。

## 高效清洗与使用体验计划

清洗先从低成本、可解释的必要条件入手，避免为了丢掉数据先跑更多 GPU 网络。

1. **入场前 CPU 检查**：复用已有 ffprobe/内容 SHA；标记坏文件、时长/尺寸/时间基不满足、精确重复。第一次按内容去重即可，不先上感知近重复或 RAFT。
2. **按分支过滤**：SAM 后记录有效手帧覆盖、面积/轨迹连续、明显重复框和手别待确认；FP 前记录 mask 内有效深度比例、网格/标签匹配。没有手只跳手分支，不删仍有重建价值的视频。
3. **结果质量标签**：finite、关节限位、缺帧、尺度是否测量、轨迹跳变和标定内点率。优先输出 flag/reason；原始数据保留。
4. **可选清洗配方**：用户显式选择 reject/保留待审/平滑。平滑需要记录参数、修改帧、原值与重算动作；慢动作和小目标不能被统一 motion/area 门槛误删。

每个 gate 记录输入/保留数量、拒绝原因、耗时、估计节省的下游调用，并抽查误拒。
衡量“有效输出/秒”和坏样本漏过率，不能用“丢掉更多数据”获得虚假的高吞吐。

CLI 足够作为近期产品入口，改进顺序为：

- **P0** `data plan` 展示模型依赖、度量前提、采样覆盖及哪些参数只影响导出；执行 manifest 增加入口总墙钟、各项成本、缓存 miss 原因。
- **P1** 支持可解释的 cache inspect/stats/受保护 gc，区分 stage/input/model 缓存；提供明确的清洗预览，列出会跳过的分支及原因。
- **P1** 沿用 13 类型，提供手/机器人、RGB-D 物体、场景三组少量模板；检查手别、网格标签和世界帧覆盖，让错误在加载大模型前出现。
- **P2** 允许显式 `faithful` 与 `throughput` 运行配置（拟议名，当前没有该接口），差异逐项展示。任何降分辨率、降采样、换模型都进入质量/效率权衡实验，不计作同质量纯性能优化。
- 暂不建设通用插件商店或 GUI。先使一次运行“会算什么、复用什么、缺什么、花多久、产物是否合格”可解释。

## 推荐推进顺序和完成门槛

| 顺序 | 交付 | 价值依据 | 完成门槛 |
| --- | --- | --- | --- |
| P0-A | 固定对照配方与分项计时；接入已有 NVML/回放/输入缓存分支 | 先区分模型加载、CPU 等待和重复工作；已有分支避免重做 | 产品入口/冷热/全命中各计时；整链结果及恢复回归；不把分支测试数当主分支结果 |
| P0-B | 单 run 身份验证去重、内容资产登记设计与 cache miss 解释 | 当前每次身份准备约 13s，是直接可测的交互成本 | 真实同输入缓存命中；严格模式仍识别同路径权重变更；报告身份 I/O 字节和时间 |
| P0-C | CPU 元信息/精确重复 gate + SAM 后分支 gate | 降低无效 GPU 调用，通常比增加清洗模型便宜 | 固定保留集/拒绝原因；误拒抽查；有效输出吞吐及内存不回退 |
| P1-A | 取帧策略、单次掩码搬运、有限 crop 预取/合批，逐项 A/B | 有源码重复和像素一致的解码实验证据 | 固定模型/帧/精度；bitwise 预处理与数值质量门槛；整链速度、峰值 RAM/VRAM、尾等待同时测 |
| P1-B | 阶段语义键、派生结果缓存、列式流读、模板和缓存管理 | 改善换配置反复处理和长视频内存；不扩大类型数 | 相关改变失效/无关改变复用；损坏/并发/回收边界；长输入内存有界 |
| P2 | 冷缓存 codec、VLA state/action/平滑/LeRobot、连续相机/手模型可行性 | 与下游消费或大规模结果量有关，当前不是已证瓶颈 | 单独需求/代表性质量评估；明确 8GB 可行性和授权资产；不挤占 P0/P1 |

这是研究后给出的排序，不是宣称上述优化已经实现，也不承诺尚无测量的加速倍数。

## 为与 Data-Juicer 公平比较做准备

必须区分 **相同算法的执行效率** 与 **不同算法的能力/成本**。
拿 HaMeR/Fast3R 和 HaWoR/MegaSaM 直接比 FPS，无法把差异归因到框架。

### 三层实验

| 层级 | 可比较对象 | 当前准备状态 |
| --- | --- | --- |
| A：共同 CPU 清洗 | 同一视频元信息/内容去重/同设置运动统计，固定保留集合 | 可优先适配；DJ 现有 video dedup 按编码视频 packet 的 MD5，我们按整文件 SHA，不能直接当成完全同一语义 |
| B：匹配模型阶段与整链 | 给 DJ mapper 包装现有官方适配器，传相同准备 clip、框/掩码、权重、环境和采样 | **尚未实现 DJ wrapper，也未跑吞吐对照**；需要它才能评价调度/交接差异 |
| C：产品能力/质量成本 | 两边各自推荐的具身流程，比较有效轨迹、世界覆盖、质量、运行成本 | 不要求同模型；单列质量与数据种类，不能宣传为纯框架提速 |

主对照建议先跑 B 的手链 `SAM2/DINO → HaMeR → Dex`，然后 RGB-D `SAM2 → FoundationPose`，最后独立/联合 Fast3R。
世界标定、世界导出和清洗也必须在两侧同时计入，或明确都排除。DJ 示例的 keyframe 采样不直接等价于我们的 stride/max_views。

### 工作负载与测量矩阵

- 现有 chicken/aeropress/双手/mustard 只作为短回归夹具，包含人工实例手别和来自同一官方预览的裁剪，不能作为独立场景性能/准确率基准。
- 下一批需真实 10–60s 连续片段，覆盖单手、双手、遮挡、多实例、无手/坏样本及不同分辨率；RGB-D 单列深度/网格条件。不通过循环同一短视频制造“更多独立数据”。
- 输入清单固定 SHA、来源/许可、帧率/尺寸/范围、深度尺度、目标对象、人工标注/审查状态；手别和实例筛选两边一致。
- 模型/采样/精度/线程数/环境一致，单卡串行起步。固定模型冷加载、模型热但阶段结果 miss、跨 run 结果 hit、input-cache on/off、codec on/off、单项参数变更和故障恢复模式。
- “冷”指新 runner/model；不声称 OS/驱动缓存已冲刷，不重启服务或系统。warm 模式和磁盘页缓存单独注明。
- 短阶段建议至少 1 次冷启动、2 次预热、5 次热重复；A/B 交替次序，整链至少多次完整 invocation。长负载按实际资源扩充样本，报告原始分布和样本数；少量短测试不推算大规模置信区间。
- 分别记录从产品命令开始到最终清单的总耗时，以及 manifest/provenance、cache lookup/verify/copy、prepare/decode/crop、H2D/GPU forward/D2H、CPU postprocess/PnP、序列化/写盘、QC/世界导出、queue wait。
- CUDA 计时使用专门 profiling 模式或 events；CPU 计时不能冒充纯 GPU 时间。阶段可重叠，不能把分项简单相加成整链墙钟。
- 统一报告视频秒/墙钟秒、有效手帧/秒、对象帧/秒、处理像素/秒与有效输出率；补充启动延迟、队列尾等待、失败/OOM/重试、缓存 hit/miss/失效原因。
- 内存分别报 NVML 整卡/相关计算进程、Torch allocator、进程树 RSS（共享页可能重复，能采集时另报 PSS）、Ray/object store、在途字节、输入缓存映射、临时盘高水位和 I/O 字节。不能只报 result packet。
- 两边输出帧键、行数、手别、单位、坐标、相机/点云覆盖必须相同；mask/crop 做像素对照；姿态随机路径用重复基线分布和预先确定的质量门槛。拒绝样本数/质量不一致时，不比较“同质量速度”。
- default、ray、ray_partitioned 分别列实验，注明 fusion/checkpoint/cache 的兼容条件及预检成本；不把一个执行器的缓存收益移植到另一个的数字上。

优先优化有测量证据的重复工作，并同时监控质量、内存和有效输出。以此构成 EgoWorld 针对具身视频的优势依据，而不是以功能数或未经验证的预计加速倍数作结论。

## 重现实验命令

在项目根目录，使用已有环境，无需安装额外模型：

```bash
. config.local/reproduction.env
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python egoworld/scripts/research_cache_formats.py \
  --root /mnt/d/EgoWorld/runs/product-20261101 \
  --output /mnt/d/EgoWorld/runs/research-20261101/cache-formats --repeats 5
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python egoworld/scripts/research_input_costs.py \
  --root /mnt/d/EgoWorld/runs/product-20261101 \
  --output /mnt/d/EgoWorld/runs/research-20261101
```

复跑依赖原始冻结夹具仍存在；脚本只生成研究产物，不更新生产缓存/模型参数。

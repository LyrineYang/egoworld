# 服务器迁移与产品比较交接

日期：2026-11-02。接收基线为 main `8ced276`；本次工作分支 `codex/server-c1-comparison`。本文件是新交接，远端传入的 `docs/handoff.md` 保留原样。项目以推理复现和具身数据清洗处理为目标，不包含训练。

## 目标环境与结果口径

**主要环境是 Linux 单服务器 2–16 张 NVIDIA GPU**，不是把本机 4060 的参数当服务器设计上限。RTX 4060 Laptop 8 GB 是额外的开发/可用性环境；当前所有真实 GPU 证据都来自它，不能推断服务器扩展效率。多机分布式、DDP 和模型训练不是当前实现范围。

核心报告比较相同输入、模型、完整输出与质量要求下的有效产出、入口到最终提交时间、资源及恢复成本。用户同时要求 Data-Juicer 按原生设置运行：原生算子选择 CPU 时保留 CPU，不改为 EgoWorld runner，不修改其设备决策来制造对等。原生 WiLoR/HaWoR/MegaSaM 与 HaMeR/Fast3R 不同，这一组属于产品能力/有效产出观察；不提供同模型框架加速比。

## 已有能力与基线验收

- 13 类输出、模型前清洗、分支准入、严格阶段及派生缓存、资产登记、恢复、校准世界接口。
- Ray 每张 GPU 一个串行模型 actor；单模型不跨卡拆分，跨片段/阶段并发不能合并单卡显存；默认所有 GPU actor 可执行所有启用 GPU 阶段，模型驻留窗口减少换模。CPU prepare、Dex、phase 和 Writer 是独立资源池。
- 有预算的输入/中间块、共同帧/crop、HaMeR 合批、mask 搬运复用、可选 CPU/GPU 拆分和成本调度。预算不是整个进程树或机器 RAM 硬上限。
- 基线适用回归 297 项通过；新比较准备 31 项针对性检查通过；真实左右手 Dex 在 Data-Juicer 自定义阶段接口中各 6 行精确一致。这是先前 B 层接口检查，不是本轮原生 Data-Juicer 产品对照。
- 本机已复现真实 RGB 手部/机器人/场景与独立 RGB-D/mesh 世界样例。米制世界需要深度/标定证据；没有自动米制手、全帧相机或跨视频全局地图的结论。

## 迁移资产与环境

当前代码：`/home/lyrine/projects/egoworld`；新增大产物与下载：`/mnt/d/EgoWorld`。保留本地源视频、已下载权重与许可文件，不迁移 WSL、不重启 Codex/远端服务。

服务器建议分开存放源码、版本化模型、原始输入、运行目录、结果缓存、临时块及环境，例如 `/srv/egoworld/{repo,models,data,jobs,cache,scratch,envs}`。这是可替换的部署约定，不依赖 Windows 盘符。共享模型可只读；中间块和 SQLite/写盘尽量使用服务器本地磁盘。一次 run 只允许一个 driver；不要让两台机器共享同一 SQLite 运行库。网络盘上的锁和缓存一致性须实测。

需要转移或按相同 revision 重新获取：SAM2/DINO、完整 HaMeR checkpoint 和 model_config、许可取得的 MANO/均值、Fast3R 完整快照、FoundationPose 源码/权重、机器人 YAML/URDF/mesh，以及 RGB-D 深度/内参/单位/物体网格。MANO 按原许可获取，不随公共代码发布。不能只复制一个 checkpoint 就视为 HaMeR 环境齐备。

不要直接搬运 WSL venv：在服务器重建隔离环境并安装同一版 EgoWorld，编译 DINO/FoundationPose 等原生扩展，保存 Python、Torch/CUDA、驱动、GPU 架构、原生库版本、上游 revision、完整权重哈希。安装路径改变后重写 `runner_command`；HaMeR 工作目录与 checkpoint 可以分开。RGB-D assets JSON 内部路径需要同步改写，保留原始文件名/内容和 video identity。

可信资产登记须在目标服务器重新生成，不能把本机 inode/mtime/路径登记当作服务器的信任证据。搬运旧结果缓存后仍需通过当前配置、运行时、源码与输出契约校验，允许正常失效重算；临时 memmap/spool 不作为交接资产。服务器也应重新标定调度成本，不能沿用 4060 的时延表。

本轮 Data-Juicer 的原生模型资产和环境与 EgoWorld 分开准备，CPU Torch 保持 CPU。安装、下载、首次依赖修复发生在测量之外，所有失败准备日志保留。不得在计时期间下载或修改解释器依赖。

本机原生参考解释器为 `.venv/datajuicer-native/bin/python`，环境锁见 [CPU 参考依赖](../egoworld/experiments/requirements-native-c1.txt)。它不替代服务器 CUDA 环境。Data-Juicer 的源码安装本身也需要 Python 开发头文件和 C++ 构建工具。旧 Windows 挂载盘 pilot 环境只保留历史用途，不作为本轮完整模型环境。

可直接交接的私有运行资料还包括 `/mnt/d/EgoWorld/runs/c1-comparison-20261102/server-asset-inventory.json`（真实模型资产哈希、目录要求和上游 revision）、`source-patches/`（需随对应上游版本保存的源补丁）、`host.json`、`final-pair/source-snapshot/` 与 `final-robot-fk.json`。原生 WiLoR 的源码未改，未跟踪的 pycache 是运行缓存。正式三行机器人产物的独立 URDF/Pinocchio 验证通过，严格越界为零。

## 可执行服务器配置准备

```bash
python egoworld/scripts/prepare_server_configs.py \
  --config /absolute/config.local/job.json \
  --output /absolute/deployment/server-profiles \
  --map /home/lyrine/projects/egoworld=/srv/egoworld/repo \
  --map /home/lyrine/projects/egoworld/models=/srv/egoworld/models \
  --map /home/lyrine/projects/egoworld/data=/srv/egoworld/data \
  --map /mnt/d/EgoWorld=/srv/egoworld \
  --deployment-root /srv/egoworld/jobs \
  --environment-root /srv/egoworld/envs \
  --gpus 2,4,8,16 --cpus 64 --ram-gib 256
```

64 CPU/256 GiB 只是明确的规划示例，必须替换成实际服务器可用预算。生成器对绝对路径要求显式映射，重写模型解释器和运行路径，检查 actor CPU 预约能否满足，并保留模型精度、采样及输出语义。不能映射的路径或不足的 CPU 会报错。输出 `deployment.json` 标记 `hardware_validated=false`；文件生成与结构验证不证明实机可运行，也不重写 RGB-D JSON 的内部依赖。

本机 RGB 和 RGB-D 示例分别位于 `/mnt/d/EgoWorld/runs/c1-comparison-20261102/server-profiles-auto/` 与 `server-world-profiles-auto/`。默认 `--runner-mode auto` 保持内置 worker 常驻；`--runner-mode per_task` 用于隔离诊断，不把 4060 保守试跑设置当服务器推荐值。先使用少量 GPU 跑通，再按 2/4/8/16 卡测同一冻结数据；不能为了适配某卡改变采样/精度后沿用同一公平对照名称。

## 服务器执行与验收顺序

1. 记录 `nvidia-smi`、CPU/RAM/磁盘与软件环境，核对所有资产，执行 `launch --check-only` 和 `doctor --gpu --assets`。
2. 逐张卡用 `CUDA_VISIBLE_DEVICES=<物理卡>`、单 GPU 配置跑 native smoke，尤其检查 BF16/attention 与原生扩展。不要仅凭显存大小承诺兼容。
3. 先跑 2 卡短数据集，验证每个 GPU actor 的可见卡、实际多卡并行、CPU/GPU 拆分与子 runner 生命周期；采样 NVML 和进程树内存。至少有足够独立片段填满 worker，单片段不能证明扩展效率。
4. 再跑 4/8/16 卡，保持输入、采样、模型/权重、精度、阈值和输出一致，重新实测服务器成本 profile。4060 的时间/显存记录只作参考，不直接当服务器调度成本。
5. 冻结冷 runner、热模型但结果 miss、跨 run 命中、部分失效、恢复组；保留所有失败/OOM/重试和输入/输出审计。至少 5 次平衡顺序重复后再报告分布和置信区间。

服务器验收记录应包含：实际卡型/数量/利用率、GPU 与 CPU 资源分配、入口墙钟、有效样本/秒、阶段加载/推理/搬运/写出、排队长尾、进程树 RAM 与显存观测口径、磁盘/网络、产出覆盖、错误与恢复重做量。阶段时间可能重叠，不能相加当 job wall time。

## 本轮 C1 执行与结果

`egoworld/scripts/benchmark_products.py` 使用独立配置启动真实 EgoWorld recipe 或原生 Data-Juicer DefaultExecutor，保存输入/配置/源码哈希、前后包快照、默认设置、全程日志、20 ms 进程树 RSS、全设备 NVML 观测和最终文件审计。NVML 包含其它进程，不能当作模型独占显存。

```bash
.venv/bin/python egoworld/scripts/benchmark_products.py run \
  --spec /absolute/experiment/spec.json \
  --output /absolute/new-trials
```

本轮原始目录：`/mnt/d/EgoWorld/runs/c1-comparison-20261102/`。[本轮报告](c1-native-comparison-20261102.md) 与 [机器记录](benchmarks/c1-native-pilot-2026-11-02.json) 已补齐。最终双 pipeline 均通过：相同三帧输入各有 6 条有限手姿态；EgoWorld auto 为 100.77 秒，原生 Data-Juicer CPU 为 61.18 秒。EgoWorld 另有 3 条机器人映射，Data-Juicer 另有 6 个网格和 3 张可视化。模型和输出范围不同，不能计算同模型/同输出加速比。原生 Data-Juicer 缺少相同输出、执行失败或输出为空时，必须计入产品完成情况；成功退出本身不表示有效产出。

后续接手者先读取本轮机器可读报告和失败日志，区分 A 层 CPU 清洗、B 层阶段接口、C1 产品观察与尚未完成的公平对照。不能把本机原生能力实验升级成 2–16 卡结果，也不能把不同模型/阈值的耗时比写成框架加速比。

已完成的是 C1 层次的原生产品试跑；严格同模型/完整输出 C1、对应三组跨框架矩阵、B2 五轮重复、C2 恢复与独立质量留出集尚未完成。原生 WiLoR 声明 CUDA 支持，本轮 CPU Torch 使其自然走 CPU；服务器需另设其标准 CUDA 环境组，并保留 CPU 清洗算子的原生选择。

服务器多任务应使用调度器或互斥的 CUDA_VISIBLE_DEVICES 分配 GPU；不要让多个独立 Ray driver 同时声明同一批卡。检查模型 actor 可见卡数与 num_gpus 一致。原生 WiLoR 按视频 basename 写帧/网格，正式数据应冻结唯一 basename 与 input_id 映射，避免输出混写；本样例的手侧 ID 映射也不能推广到全数据集。

本轮清洗发现两个值得推进的缺口：EgoWorld 按文件字节去重漏掉相同码流的改封装，原生默认运动阈值丢掉本夹具中的静止纹理。清洗报告同时保留双方不利结果。下一步先补任务感知的流级/附属资产联合身份，再推进服务器上的等工作量、稳态和恢复比较。

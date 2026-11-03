# 统一输出质量与固定模型效率：两条比较口径

核心产品问题调整为：**同一输入、同一数据类型与输出契约下，哪条产品管线得到更多达到质量要求的数据，付出多少时间与资源。模型可以不同。** 固定同一模型的比较继续保留，用来解释框架执行成本。二者分别报告，模型不同不再阻止产品质量比较，也不能据此声称框架自身更快。

## 本次实际结果

两条完整管线都成功执行，配置/输入/生产源码/环境前后校验通过。主结果按事前固定的 12 个源帧评分，不把漏帧改成检出；[机器记录](benchmarks/output-quality-2026-11-03.json) 保存原始路径与哈希。

| 预先固定的精确源帧评分 | EgoWorld | 原生 Data-Juicer CPU |
| --- | --- | --- |
| 目标覆盖 | 12/12 | 11/12 |
| 单目标任务的额外预测 | 6 | 1 |
| 检出项腕部相对 MPJPE | 11.21 mm（12 项） | 12.51 mm（11 项） |
| 检出项 PA-MPJPE | 6.46 mm | 7.19 mm |
| 标注判定 MPJPE≤20 mm 且手侧符合 | 12/12 | 10/12 |
| 达标手/秒（本次完整运行） | 0.02283 | 0.06043 |
| 入口到整链退出 | 525.57 秒 | 165.48 秒 |
| 采样进程树峰值 RSS | 8.10 GiB | 6.42 GiB |

每图只有一个标注目标，额外预测按该单目标任务计误检，不推广为穷尽标注下的通用检测精度。达标数由独立标注判定，不表示系统已自动排除多余/低质量预测；EgoWorld 的 6 条额外预测仍需处理。条件误差覆盖集合不同，不能仅按均值判胜。RSS 是进程求和采样值，可能重复计算共享页。

原始 36 帧中 EgoWorld 导出 54 条手部记录，DJ 导出 39 条；只在预定位置评分，重复帧不增加样本量。GPU 全设备显存峰值约 6.14/0.78 GiB，包含约 0.77/0.78 GiB 其它进程基线，不是模型独占显存。CPU-only Torch 原生回退 CPU 不代表 DJ 不支持 GPU。

本次发现两项需审计的原生行为：WiLoR 按 `frame_0, frame_1, frame_10, ...` 字典序处理；该固定版本 uniform 采样实际 PTS 对应 `0,2,3,...,35,35`，不是 `0..35`，漏掉预定评分帧 1。绑定器重放**原生抽帧且不运行模型**，读取 PTS 并逐文件核验解码图像，恢复来源；原生源码/输出未修改。实际覆盖不能只从 `frame_num` 推断。

另列**事后辅助的相同图像内容评分**：逐像素验证每组三帧完全相同后，允许用同组实际抽出的等内容帧评分，每图仍只选一次，选择不使用 3D 真值误差。DJ 此组覆盖 12/12、额外 1 条、MPJPE 12.45 mm、PA-MPJPE 7.31 mm、20 mm 达标 11/12；EgoWorld 不变。它不是预注册主结果，不掩盖精确帧缺口。

本轮仅 12 图：EgoWorld 该门槛下覆盖较好但额外预测更多、成本更高；DJ 更快且额外预测较少。不能宣布任一产品全面更优或误差差异显著。固定模型完整效率组、独立具身留出集和服务器扩展实验仍未完成。

前次运行被外部环境中断，原因未确定，最后观测为 138.37 秒；原始阶段记录保留。成功轮时间不含该尝试，EgoWorld 所有已记录尝试的运行暴露至少 663.95 秒；未知终止时间和停机时长未补成零。最初帧绑定拒绝错误的“序号即源帧”假设，失败目录也保留，未产生可用准确率报告。

**32 项针对性检查通过**，其中 13 项覆盖质量/帧绑定。生产源码保持基线，未将旧 297 项回归称为本轮重跑。新增产物在 D 盘，远端传入的 `docs/handoff.md` 未修改，没有重启服务。

## 能保证的“一致”

统一的是来源、采样帧、字段、关节顺序、单位、坐标原点/轴向、有效覆盖及质量规则。预测数值允许不同，差异需要独立真值评估；不能承诺不同模型同样准确，也不能把均为 21×3 数组视为坐标含义已经一致。

手部共同契约为 `openpose21`、腕部原点、相机轴向 x 右/y 下/z 前、米、显式帧和手侧。HaMeR/WiLoR 的尺度均来自模型先验，这不是测量标定后的米制世界坐标。EgoWorld 已减去腕部，原生 WiLoR 仍保留 MANO 原点偏移，适配器显式减腕；WiLoR 左手 x 已由原生算子镜像，不能再翻一次。

原生 WiLoR `keypoints` 字段投影的是网格顶点，不是这 21 个关节。适配器从 `joints`、原生相机平移和模型实际使用的焦距重新投影关节，不用真值内参修正模型预测。适配不重跑/替换模型，也不改写原生产物；原文件和哈希均保留。

## 两条口径

| 口径 | 固定内容 | 允许变化 | 结论范围 |
| --- | --- | --- | --- |
| 产品：统一输出与质量 | 同一冻结输入、采样、共同契约、独立真值、评分政策 | 模型、检测器、内部准备和原生资源策略 | 覆盖、误检/漏检、姿态误差、达标产量及每秒达标产量；整链成本 |
| 执行：固定模型与工作量 | 模型/权重、预处理、采样、完整输出、质量门槛、资源预算 | 框架、缓存与调度的受控配置 | 输出等价后，比较准备、加载、推理、搬运、提交、复用与恢复成本 |

产品组保留 Data-Juicer 原生 CPU/GPU 选择，不为加速把 CPU 算子改到 GPU。配置采样数量、选择所需算子和输出位置是明确的任务选项，必须披露，不能称作“所有参数均为默认”。之前的原生默认链试验保留，新配置不覆盖旧结果。

执行组已有真实 Data-Juicer DefaultExecutor + 同一 runner 的左右手 Dex 接入检查，每侧 6 条机器人行逐字段数值一致。这只证明一阶段接口等价；完整模型/recipe 的五轮效率比较尚未完成，见 [既有证据](benchmarks/comparison-prep-2026-11-02.json)。

## 按数据类型保证契约

| 类型 | 共同契约与独立质量参考 | 当前边界 |
| --- | --- | --- |
| 手部关节 | 21 关节、手侧、帧、单位/轴向；3D/2D 标注 | 本轮实现原生 HaMeR/WiLoR 适配和真值评分 |
| 分割/框/轨迹 | 同像素空间、类别、实例对应；人工 mask/框/身份 | 双方有算子；YOLOE/SAM2 与 DINO/SAM2 跨端质量实跑待补 |
| MANO/网格 | MANO 版本、参数约定/拓扑和相机；关节/网格真值 | 保留原生额外网格，未验收完整参数导出等价 |
| 相机/场景 | 姿态方向、尺度/锚点、帧；轨迹/深度/几何参考 | DJ 有 MegaSaM/MoGe 算子，适配和实跑待补，不能记作缺少能力 |
| 世界手/物体 | 同一世界锚点、真实尺度和时间；测量/多视角参考 | DJ 有 HaWoR、手动作/世界变换分支，未验收；场景标定不自动证明手尺度准确 |
| 已知物体 6D | 同一 mesh、对象原点、单位/对称性；物体姿态真值 | EgoWorld RGB-D FoundationPose 已复现；DJ 等价产品任务未运行 |
| 机器人关节 | 同机器人、URDF、关节名/单位/限位；末端任务参考 | EgoWorld 有 Dex；DJ 手动作与 LeRobot 导出不自动等价于 Allegro 16 关节重定向。共享后处理需另列并披露 |

未适配、未运行和失败分别记录，不强凑相近字段，不把选定算子链当作整个库的能力上限。共同任务可以分别成立，不要求所有类型同时具备。

## 真值试验与预先规则

从 [官方 FreiHAND 评估集](https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html) 的 3960 张图中，用 Python `Random(20261102).sample` 不放回抽取 12 张，在任何推理前冻结。官方现已公开评估标注；数据只在 D 盘保存，按研究用途条款使用，不随代码再分发。下载器按 HTTP Range 读取 ZIP 中两个标注和 12 张图，校验范围/ETag、ZIP CRC 和成员 SHA，仅下载约 5 MB，没有声称获得整个 724 MB ZIP 的 SHA。

每张图重复三帧，制成同一 1fps/36 帧视频。EgoWorld 每三个帧一段；原生 WiLoR 显式配置 `frame_num=36`。双方只评分各组中间帧。12 张源图不是 36 个独立样本，也不能推断独立受试者数量。这是静态姿态的端到端工程质量试验，不是连续视频时序、动作或具身泛化评测；模型训练成员未独立审计。

DJ 选择原生时长/分辨率/手重建链，不用会排除本任务静态图的运动过滤；设备、并发、缓存、容错、batch=16、网格和可视化策略保持原生。单视频布局避免多行数据同时复制 CPU 模型；没有修改资源策略。EgoWorld 只请求 `hand_joints`，不额外运行 Dex；双方仍有额外工作差异，耗时不是框架速度比。

EgoWorld 获得固定“右手”任务先验，没有输入真值框/关节/相机；WiLoR 自行判断手侧。这是当前产品输入需求的区别，不能据此公平比较自动手侧识别。后续需双方相同先验组与真实左右手/双手独立标注。

匹配采用预先固定的图像腕部距离（最大为对角线 25%），按距离确定对应，不选择与真值 3D 最接近的预测。额外预测算误检，无输出标注算漏检。这是小试验的固定匹配政策，不冒充完整 AP。

- 腕部相对 MPJPE：保留尺度和方向误差，毫米；检出项条件均值必须配合漏检阅读。
- PA-MPJPE：允许缩放/旋转/平移，禁止反射；只诊断形状，不能证明世界位置/尺度准确。
- PCK：20/50 mm，分母为所有标注的 21 关节，漏检不移出分母。
- 达标手数/比例：MPJPE≤20/50 mm 且手侧符合任务要求；分母为全部 12 标注，额外误检另报。
- 达标手数/秒：达标手数除以入口到整链退出时间；不是只保留成功帧后的吞吐，也不是框架加速比。

结构错误拒绝评分，完整但部分缺失的产品保留 `partial` 并计入漏检，不改写为全通过。拒绝、空输出、身份/坐标错误和损坏文件保留记录；模型间分歧不能充当真值。

## 复现与交接

```bash
python egoworld/scripts/prepare_freihand_quality.py \
  --output /absolute/quality/images --samples 12 --seed 20261102
python egoworld/scripts/prepare_freihand_quality.py \
  --from-corpus /absolute/quality/images/corpus.json --output /absolute/quality/montage
python egoworld/scripts/benchmark_products.py run \
  --spec /absolute/quality/trial-spec.json --output /absolute/quality/fresh-trials
/absolute/envs/datajuicer/bin/python egoworld/scripts/native_quality_sample_times.py \
  --video /absolute/quality/montage/freihand-pilot-12.mp4 --frame-num 36 \
  --output /absolute/quality/native-times.json
python egoworld/scripts/bind_native_quality_frames.py \
  --corpus /absolute/quality/montage/corpus.json --frames-root /absolute/native/frames \
  --frame-times /absolute/quality/native-times.json --native-export /absolute/native/datajuicer.jsonl \
  --output /absolute/quality/bound-corpus.json
python egoworld/scripts/compare_output_quality.py \
  --spec /absolute/quality/quality-spec.json --output /absolute/quality/fresh-scores
```

可复制 [质量 spec 示例](../egoworld/experiments/quality-spec.example.json)，修改 `corpus`、`policy`、两个 `pipelines` 的名称、`kind=egoworld|datajuicer`、原始 `trial`、`focal_ratio`，以及 `manifest` 或 `export`。本轮二模型焦距比均为 5000/256，来自当前权重配置；换模型须重新核验顺序、轴向与投影。

评分 spec 的 `corpus` 使用绑定后的新文件，DJ `export` 使用其 `native_frame_binding.score_view_export` 指向的独立评分视图。原生产物原样保存。默认 `--matching exact` 保留漏帧；另加 `--matching identical-image` 只能生成事后辅助视图，必须验证重复源帧像素完全相同，不能通用于运动视频。两种评分使用不同目录。

原始目录 `/mnt/d/EgoWorld/runs/output-quality-20261102/`。服务器仍以 2–16 GPU 为主、4060 为补充，按 [服务器交接](handoff-server-comparison-20261102.md) 迁移。复制允许使用的真值/图像，重建环境并重定向绝对路径；先验、任务配置、来源哈希、评分政策和失败记录一并交接。

下一轮先处理 EgoWorld 同目标额外手预测，分离重复加载/准备/提交成本，在服务器用多卡驻留验证；不为赢对照修改 DJ 原生设备策略。随后扩展具身带标注左右手/双手/遮挡片段，再接相机/场景共同输出。开发集选阈值和正式留出集分开；五轮平衡重复和恢复比较仍待完成。

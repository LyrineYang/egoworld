# 视频清洗、计算复用与效率验收

新数据建议使用 `egoworld data`。清洗在模型环境检查与 GPU 加载前执行，被拒绝的视频不进入作业清单；原文件保留。
13 种输出和模型算法保持原有约定，详见 [配方指南](data-recipes.md)。

## 先预览，或直接运行

```bash
egoworld data plan --recipe my.yaml --screen
egoworld data screen --recipe my.yaml --output /mnt/d/EgoWorld/reports/quality.json
egoworld data run --recipe my.yaml
```

`plan --screen` 和 `screen` 只运行 CPU。普通 `plan` 展示模型依赖、采样和度量前提，不解码视频。
`scene.yaml` 显式提供 224/bf16/两视图的小显存例子，`metric-world.yaml` 是 512/bf16/三视图；这些是可编辑示例，不会覆盖用户已有配置。
RGB-D 示例的提示标签必须与自己的 `foundationpose.objects` 网格键一致；先配置资产，不能直接把通用 full.json 的 cup 网格当作 mustard/bottle。
`data run` 默认排除打不开、元信息无效、解码不完整、解码尺寸变化、时间基准不一致和统计采样点全部为纯色空白的视频，并按完整文件 SHA-256 去重。
重复内容只处理第一份合格输入，报告记录 `duplicate_of`；如果第一份被质量策略拒绝，后续副本也会按同一策略判断。
全部被拒绝时返回 `status: rejected`、CLI 非零退出，不启动模型环境检查或 Ray。

## 清洗配置

```yaml
cleaning:
  enabled: true
  deduplicate: true
  reject_blank: true
  decode: full
  sample_fps: 2
  max_samples: 64
  max_side: 320
  motion_method: farneback
  min_duration_s: 0
  min_width: 1
  min_height: 1
  # 以下按数据集校准后启用，不对正常慢动作设统一默认门槛：
  # max_duration_s: 120
  # min_blur: 8
  # min_brightness: 0.03
  # max_brightness: 0.98
  # max_dark_fraction: 0.95
  # min_motion: 0.0001
  # max_motion: 0.3
branch_cleaning:
  min_hand_coverage: 0
  min_hand_area_px: 0
  min_depth_fraction: 0
```

- `decode: full` 顺序验证所有声明帧并检查 EOF 边界，内存只保留当前图像和上一张统计灰度图；不是将全视频装入内存。
- `decode: sample` 只验证均匀采样点（含首尾），能降低扫描成本，但不能证明中间帧全部可解码。报告明确区分覆盖。
- `reject_blank` 默认启用，拒绝所有统计采样灰度图像素极差为零的片段；可显式关闭。它不按“运动为零”拒绝静止场景，不声称采样能找到长视频中的每个短暂有效事件。
- 质量统计在保持宽高比、最长边至多 `max_side` 的灰度图上计算；不改变送给模型的图像。
- brightness 为平均灰度/255，dark_fraction 为灰度 ≤8 的占比，blur 为 Laplacian 方差。改变统计分辨率会改变模糊门槛的解释。
- Farneback motion 为相邻采样图平均光流长度/图像对角线；单位是“每对采样图的相对位移”，不是米/秒。`difference` 为平均绝对灰度差/255。
- 采样密度、分辨率、光流方法和 OpenCV 构建参与统计缓存身份；阈值不参与统计身份，修改阈值会复用统计并重新判定。
- `enabled: false` 关闭解码/画质统计，仍做元信息检查；`deduplicate: false` 单独关闭精确重复过滤。
- 本版检查的是解码器可读性，不声称发现解码器自动修复/隐藏的所有编码损伤，也不替代标注质量评估。

清洗报告位于 `run_id=…/cleaning.json`，含每个输入的哈希、元信息、统计、接受/拒绝原因、重复来源、缓存命中和耗时。
`data_manifest.json` 给出计数。排除样本不等于提高模型精度；对新门槛应抽查误拒和坏样本漏过。

## 模型分支过滤

配方执行在调用 HaMeR、FP、Dex 前，使用已提交上游数据做 CPU 准入：

- HaMeR 仅消费所选 ID、采样帧、合格面积且手别已知的手实例，可按有效采样帧覆盖率拒绝整条手分支。
- FP 仅消费匹配网格标签、有掩码且含有效深度的物体实例，可提高 mask 内有效深度比例门槛。
- Dex 没有对应侧的人手结果时跳过，保留其它分支。

被排除的分支不加载权重。普通和拆分执行均保持 Writer 提交协议，空结果有 filter_reason 和独立过滤记录。
过滤只修改当前消费者的私有输入视图，不修改 SAM 原始产物；无手视频仍可生成场景数据。
请求的某类结果为空时仍明确标记 partial，不把它伪装成完整数据。
旧 `run` 入口可通过模型配置 `extra.branch_cleaning` 显式启用同一准入机制；旧清单不自动改写。

每个导出片段附带 `quality.json`：记录分支过滤、手部先验尺度、腕部运动诊断等。原始观测不平滑、不覆盖。
模型先验尺度下的速度不能当作真实物理速度。

## 缓存身份和校验

```yaml
cache:
  root: /mnt/d/EgoWorld/cache/stages
  read: true
  write: true
  verification: strict
```

默认 `strict` 每次 invocation 验证实际文件内容，同一次调用链通过受限生命周期的哈希会话复用已验证摘要，避免产品层和 driver 重复读取 GB 权重。
每次复用均检查文件身份/大小/纳秒修改时间和状态变更时间；会话结束即丢弃 memo，默认没有跨 run 的 mtime-only 快捷路径。
哈希期间文件改变会拒绝，不发布不稳定身份。

阶段键使用该适配器及静态导入依赖的源码身份、当前阶段模型版本/资产、参数、输入内容/范围/时间基和上游产物内容。
运行目录和视频文件名不再单独使相同内容失效；恢复后重新绑定当前 run 的提交和来源。
无关 Dex 版本不会使 SAM 失效；相关模型、预处理、共享代码、深度、MANO、机器人资产变更仍会失效。
运行内恢复仍保留全局 fingerprint 的严格一致性，新配置应使用新 run ID。
阶段键从 v1 升至 v2，旧缓存不自动猜测迁移；首次会重算。

### 显式可信资产登记

对由用户控制、预期不修改的本机模型目录，可选择跨 run 快捷验证：

```bash
egoworld data assets register \
  --files /absolute/model.ckpt /absolute/model_config.yaml /absolute/MANO_RIGHT.pkl \
  --output /mnt/d/EgoWorld/models/trusted-assets.json
```

```yaml
cache:
  root: /mnt/d/EgoWorld/cache/stages
  verification: registered
  asset_registry: /mnt/d/EgoWorld/models/trusted-assets.json
```

登记时完整读取文件并记录 SHA/文件身份。registered 模式只对显式登记且 stamp 未变的文件省去重读；其它文件、已变文件退回内容哈希。
这是一项**明确的本机信任选择**：模型和 registry 必须由用户控制，不用于不可信目录或对抗性修改检测。
它不是签名、不可变存储或密码学防篡改；需要逐字节保证时使用 strict。不会自动改写配置或登记模型。

### 派生结果与压缩

字段投影和世界标定/导出使用单独派生缓存；键包含输出选择、源码、源产物和标定证据，RGB-D 的实际锚点深度也参与。
命中后重绑定当前运行路径；校准所属 video/clip 身份不跨片段猜测复用。
源数据、深度、校准或派生文件损坏会使缓存失效。Parquet 按列、按批导出，不为选择少量列读取整个 Python 行表。
Parquet 保持 Zstd，NPZ 保持现有 Deflate；不对热帧或 NPZ 默认再加压缩。

## 缓存解释、预算和回收

```bash
egoworld data cache inspect --root /mnt/d/EgoWorld/cache/stages
egoworld data cache stats --root /mnt/d/EgoWorld/cache/stages
egoworld data cache gc --root /mnt/d/EgoWorld/cache/stages --max-bytes 21474836480
# 确认策略后显式执行；默认仅预览
egoworld data cache gc --root /mnt/d/EgoWorld/cache/stages --max-bytes 21474836480 --delete
```

回收只处理带管理标记根目录内、认可格式和摘要名称的 stage/derived/quality 条目，忽略符号链接和未知文件夹。
不删除模型、原视频、run 输出或其它项目文件；配合文件锁避免与本项目缓存读回/发布竞争。
`max-age-seconds` 可按年龄选择，`max-bytes` 是受管理条目的逻辑文件长度预算，不是全盘空间保证。
阶段缓存命中解释位于 `data_manifest.cache_decisions`：hit、not_found、corrupt_entry、identity_changed、read_disabled；变更时列出身份维度。
只剩 Dex 时不申请 GPU，也不重新提取/编码视频。资源准入后缓存依赖又改变，会明确失败要求重跑，而不是无 worker 等待。
input cache、模型常驻和本节跨 run 缓存是不同层，统计和预算不可混算。

## 分项计时和公平对比

`data_manifest.timings` 覆盖清洗、清单、provenance、缓存恢复、pipeline 和发布/导出，并给出产品入口墙钟。
`hashing` 报告实际读取字节、哈希次数、会话复用和登记命中；computed 只指模型阶段，不能代替总耗时。

```bash
egoworld data compare --runs /absolute/run_id=A /absolute/run_id=B
```

比较前检查输入内容、输出选择、过滤策略/保留数量、阶段身份和产物覆盖。不同工作负载会明确列出差别，不给“同质量加速”结论。
结果缓存命中和模型热调用必须分开解释；阶段 RSS 是进程生命周期峰值，不是整机并发内存。
NVML、split execution、输入缓存及冷/热回放由独立吞吐分支提供，合入后沿用其显式开关和边界。

可选 `product/datajuicer_adapter.py` 提供 `egoworld_stage_mapper`，供 Data-Juicer 的 `custom_operator_paths` 加载。
它复用相同 ModelActor/runner/Writer，行输入需包含 `egoworld_clip`、`egoworld_prepared`、`egoworld_inputs`、`egoworld_fingerprint`。
每个 OP 指定相同 config_path/stage 和独立 output_root，输出仍只返回 receipt，不把大数组放入数据集行。
这是匹配算法的对照入口；并未据此宣称完成 Data-Juicer 吞吐实验，也未证明所有 Data-Juicer 版本/执行器的兼容性。

相关研究与验证协议见 [效率研究](datajuicer-efficiency-research-20261101.md)。

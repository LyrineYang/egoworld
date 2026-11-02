# 按数据需求组合 EgoWorld

视频坏样本过滤、运动/画质统计、缓存身份与管理的新选项见 [清洗与效率指南](cleaning-and-cache.md)。

先选需要的数据，再由配方补齐计算依赖。用户不必为了取手关节而打开场景重建，也不必为了换机器人而重复做人手推理。
现有 `run/launch/smoke` 仍可使用；新入口是 `egoworld data`，执行仍复用同一套模型、runner、Writer 和恢复协议。

这一组织方式借鉴 Data-Juicer 的 [OP 与配方](https://github.com/datajuicer/data-juicer/blob/main/docs/tutorial/QuickStart.md)、
[缓存与 checkpoint](https://github.com/datajuicer/data-juicer/blob/main/docs/ProcessData.md)。
Data-Juicer 使用 mapper/filter 等算子抽象；这里的 kernel 指一个可复用的计算模块，不是 CUDA kernel。
不引入其 executor，也不把有依赖、有时序的模型当作可以任意打乱的列表。

## 1. 选择数据

```bash
egoworld data catalog
egoworld data plan --recipe egoworld/recipes/hand-joints.yaml
egoworld data run --recipe /absolute/path/to/my-recipe.yaml
```

`catalog` 列出数据类型及可选字段；`plan` 不运行 CUDA，展示自动加入的依赖和缺少的标定条件。
模板不包含模型资产；先按 [server.md](server.md) 配置真实模型路径和隔离环境，再修改配方的 config 与 inputs。
本机复现实验使用 `config.local/reproduction.env` 及 D 盘已准备的模型。

| 所需数据 | 配方中的名称 | 自动计算的模型 |
| --- | --- | --- |
| 实例掩码 / 框 / 二维轨迹 | `masks` / `boxes` / `tracks` | SAM2 + GroundingDINO |
| 人手 21 关节 / MANO 参数 | `hand_joints` / `hand_mano` | 分割 → HaMeR |
| 机器人关节 | `robot_joints` | 分割 → HaMeR → Dex |
| 物体姿态 | `object_poses` | 分割 → FoundationPose；需 RGB-D/网格 |
| 相机 / 场景点云 | `camera_poses` / `scene_points` | Fast3R |
| 米制世界相机 / 点云 | `world_camera` / `world_scene` | Fast3R + 显式标定 |
| 米制世界物体 / 手 | `world_object` / `world_hand` | 对应姿态模型 + Fast3R + 显式标定；手另需度量对应点 |

选择 `hand_joints` 和 `hand_mano` 只运行一次 HaMeR；选择 `boxes` 和 `masks` 只运行一次 SAM2。
相互无关的数据可以自由组合，计算顺序由依赖决定。当前内置五种模型 kernel 和世界坐标派生模块；
新增类型可在 `product/registry.py` 注册已有 kernel 的输出。新增模型还须实现原有模型与契约接口，不能仅凭一个名字执行任意函数。

## 2. 配方与字段

```yaml
version: 1
config: /absolute/path/to/model-config.json
inputs:
  - /absolute/path/to/video.mp4
outputs:
  - boxes
  - type: hand_joints
    fields: [joints_3d, camera_translation]
  - hand_mano
output_dir: /mnt/d/EgoWorld/runs/data
cache:
  root: /mnt/d/EgoWorld/cache/stages
  read: true
  write: true
operators:
  sam2:
    prompting:
      prompt_text: 'hand .'
  hamer:
    object_ids: [1, 2]
    handedness_by_id: {'1': right, '2': left}
```

配方内的 config、inputs、output_dir、cache 和标定文件路径相对配方目录解析；
模型 config 的已有参数路径语义保持不变，相对执行目录解析，因此建议本机资产使用绝对路径。
`operators` 对所选模型参数做递归覆盖，不启用无关阶段。

`fields` 控制导出列，不会少算帧、改变精度或改变模型内部工作量。
帧号、时间、实例身份、左右手、坐标/尺度说明，以及机器人关节名称/单位会自动保留，避免导出无法解释的数值。
完整原始阶段文件保留在 `model-output/`，精简结果在 `exports/`，运行与数据来源记录在 `data_manifest.json`。

`object_ids` 是可选的显式实例白名单，用于排除把袖口等误检为手的实例；编号仅在当前片段内有效。
`handedness_by_id` 也是用户输入，不代表自动手别识别通过。示例中的编号不能原样用于任意视频；应先输出 boxes/masks 检查。
若选择的手不存在或左右手不明，明确失败，不默认猜右手。
FoundationPose 的检测标签须与 `objects` 的网格键匹配，例如提示 `bottle .` 对应 `objects.bottle`。

## 3. 缓存与恢复

这是三个不同的概念：

- **模型缓存**：保留模型文件或常驻权重，由现有环境和 runner 管理。
- **同 run 恢复**：输入、代码和配置不变时继续未完成阶段；仍走原有全局指纹与提交检查。
- **跨 run 阶段缓存**：新配方在运行前验证并复制兼容的已提交阶段，剩余工作交给原 pipeline。

例如，先取 boxes/masks，再增加 hand_joints：复用 SAM2，只计算 HaMeR；
之后更换机器人配置：复用 SAM2/HaMeR，只计算 Dex。导出字段变化也可以复用模型产物。

缓存键包含源码身份、模型与实际资产、环境/包版本和上游模型源码、视频内容和片段范围、该 kernel 参数以及上游产物哈希。
Dex 的机器人资产、FoundationPose 的实际深度帧、HaMeR 的隐式 MANO/均值参数也参与身份。
阶段 v2 身份按模型适配器、静态导入和共享执行代码收集依赖，并只包含该阶段的模型版本；同 run 恢复仍保留主包整体身份。
依赖收集保持保守，不承诺任意代码/环境修改后的最小重算；仅改 CLI/报告等无关模块通常可复用模型结果。
每次读缓存都检查文件大小、SHA-256、Parquet schema 和提交标记；损坏条目不复用。
缓存恢复使用副本，不用可写硬链接；修改某次运行的文件不会污染缓存和其它运行。

全命中时不启动 Ray 或模型，字段投影和世界派生也有独立缓存。
部分命中仅为剩余计算配置模型 worker；只剩 Dex 时不申请 GPU，也不重新提取视频。
缓存文件保留在配置的目录，没有后台自动删改数据。关闭读取或写入分别用 `cache.read: false` / `cache.write: false`。

```bash
# 相同配方和输入：恢复同一次作业
egoworld data run --recipe my.yaml --run-id task_A
# 修改输出需求或模型：使用新 ID，仍可命中未变化的阶段缓存
egoworld data run --recipe my-new.yaml --run-id task_B
```

## 4. 机器人腕部约定

Dex 默认保留原 `configured_rotation` 行为。需要复现上游参考手部坐标时启用：

```yaml
operators:
  dex_retarget:
    input_transform: dex_wrist
```

`dex_wrist` 从腕、食指 MCP、中指 MCP 建立随手运动的坐标轴，再使用上游左右手 `OPERATOR2MANO` 约定。
最后才应用配置的 `human_to_robot_rotation` 和 scale_factor。不会再应用 HaMeR global_orient，避免重复旋转。
退化/共线手掌明确失败。限位仍使用严格 URDF 边界。
每个 Dex kernel 配置一个机器人和一侧手；双手或多个机器人可用不同配方复用同一批人手结果。

## 5. 米制世界坐标

原 schema 3 的模型输出不覆盖、不改单位。派生的 `world_*` 使用独立 v1 清单，记录 world_id、尺度依据、输入哈希和校验结果。
默认世界原点是一个片段的首个标定相机，不是房间/GPS 坐标；不同 world_id 不能直接拼在一起。

有已有对齐 RGB-D、内参和真实深度单位时：

```yaml
outputs: [world_camera, world_scene, world_object]
operators:
  fast3r:
    image_size: 512
    max_views: 3
calibration:
  method: rgbd
  assets_manifest: /absolute/path/to/rgbd-assets.json
  threshold_m: 0.03
  min_inlier_fraction: 0.5
  confidence_quantile: 0.5
```

Fast3R 额外保存所选点对应的原图像素。由原图内参和深度反投影，估计正尺度相似变换，
保留高置信度有效深度对应，使用确定性的 RANSAC 排外点，并以预留的 20% 对应检查门槛。
`confidence_quantile: 0.5` 表示先保留有效对应中模型置信度的上半部分；不会按拟合残差挑选预留点。
报告同时保存有效对数、置信度门槛、训练/预留点内点率和预留内点 RMSE，不能只看最后一个误差数字。
world_scene 也按同一置信度门槛筛选；这是有度量锚点的估计，不是全场景精度保证。

约定 `T_A_from_B` 将 B 系点变到 A 系。相机中心和点云接受估计的尺度，物体的米制位移不再次缩放：

```text
object_to_world = T_world_from_camera × object_to_camera
```

只匹配同一源帧的相机姿态，不借用最近帧。缺帧会写入 skipped_frames、状态为 partial，并返回非零退出码。
可提高 max_views 覆盖所需帧，或提供有依据的 `camera_overrides`，不自动编造相机运动。
无真实尺度依据、退化对应、门槛不通过或输入哈希变化时明确失败。

已有外部标定可写 `calibration: {path: /absolute/calibration.json}`；
先用以下命令生成可检查、可绑定输入的标定文件和导出：

```bash
egoworld data calibrate --clip-dir /absolute/completed-clip \
  --rgbd-assets /absolute/rgbd-assets.json --output /mnt/d/EgoWorld/calibration.json
egoworld data fuse --clip-dir /absolute/completed-clip \
  --calibration /mnt/d/EgoWorld/calibration.json \
  --outputs world_camera,world_scene,world_object --output-dir /mnt/d/EgoWorld/world
```

### 人手需要额外的度量对应

HaMeR 的 model_prior 相机深度不能直接拼到真实物体旁。`world_hand` 要求与当前预测绑定的相机系度量手关节：

```json
{
  "version": 1, "video_id": "实际视频ID", "clip_id": "实际片段ID",
  "joint_order": "openpose21", "length_unit": "meters",
  "samples": [{
    "frame_index": 0, "object_id": 1,
    "joints_camera_meters": ["替换为按 OpenPose21 排列的 63 个真实度量坐标"],
    "evidence": "测量或数据集标定来源；不能填写 HaMeR 自己的预测"
  }]
}
```

```bash
egoworld data calibrate-hand --clip-dir /absolute/completed-clip \
  --calibration /absolute/scene-calibration.json --measurements /absolute/measured-hands.json \
  --output /mnt/d/EgoWorld/hand-world-calibration.json
```

每个观测独立检查对应残差，结果绑定 hand_pose 文件哈希；先修正手部相机空间，再转换到相同 world_id。
缺少匹配测量时拒绝导出 calibrated 手关节。当前公开视频样例缺这种度量手标定，因此手部这一分支有几何/拒绝反例测试，
不能将它记为已经完成真实米制手精度验证。未新增传感器、训练模型或实现 SLAM/跨片段闭环。

## 6. 开发与验证范围

数据目录/字段与推理参数分别位于 `product/registry.py` 和原模型 config；
`product/recipes.py` 负责组合和导出，`product/cache.py` 负责复用，`product/world.py` 负责有标定依据的派生坐标。
不修改调度优先级、GPU 探针、阶段回放或性能记录接口。新增 CLI 集中在 `egoworld data` 子命令，方便与另一 worktree 合并。
真实模型记录、失败试跑与数值测试分开保存，见 [测试记录](testing.md)。

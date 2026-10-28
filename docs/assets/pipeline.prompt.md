# EgoWorld 流程图绘制 prompt

当前有两张可编辑矢量图，沿用现有素材目录：

- `architecture.svg`：README 默认首图，1600×900。按输出内容命名，用图片解释每步提取什么；模型名作为次要标注。
- `pipeline.svg`：展开后查看的详细输出示意，1600×1160。手、杯子、骨架和点云均为示意，不代表真实模型预测。

默认首图保持简洁：输入视频 → 掩码/轨迹 → 三维人手 → 机器人关节；分割另接物体位姿，视频另接独立场景分支。
每项配相应图片：原始手/杯场景、独立着色的 mask 与框、21 点手骨架、机械关节、带坐标轴的物体、点云与相机。
图片沿用 `pipeline.svg` 的可编辑矢量元素，不伪造模型效果；保留 `data-stage` 和连接线的 `data-from`/`data-to` 便于核对依赖。
共同数据卡标明来自每个启用阶段，以小型表格示意 Parquet/NPZ，避免看成单一模型输出。
不把执行池、状态反馈、字节预算或待研究策略挤进首图；调度细节在 `docs/design.md`，文件阅读地图在 `docs/handoff.md`。
必需资产和坐标/尺度限制放简短脚注与输出文档。依赖箭头不表示所有阶段顺序执行或模型共用统一世界坐标。
在 960 像素宽度检查可读性，不加入未经实测的速度、规模、精度或显存数字。

以下 prompt 用于复刻详细的 `pipeline.svg` 输出示意：

```text
Design a polished product overview diagram for EgoWorld, a command-line
library that turns egocentric video into structured hand, object, robotic
hand, and scene annotations. The diagram must explain both the workflow
and what the resulting data looks like.

FORMAT AND ART DIRECTION
- A 1600 × 1160 landscape editorial diagram for a GitHub README.
- Prefer an editable SVG. Use a restrained off-white background, dark navy
  typography, generous spacing, precise alignment, and thin orthogonal
  connectors. No decorative 3D blocks, stock icons, glows, or floating cards.
- Give each modality one consistent accent: teal for segmentation, violet
  for human hands, blue-gray for robot joints, amber for object pose,
  and blue for scene geometry.
- Use readable English typography. Titles 25–40 px, primary descriptions
  21–23 px, secondary labels at least 16 px at the original canvas size.
- Hierarchy: source and conditional inputs → actual model dependencies
  with visual output examples → shared dataset and run capabilities.

HEADER
Title: “From first-person video to usable annotations.”
Subtitle: “Hands, objects, robot joints and scene geometry — choose the
outputs your data supports.”

INPUTS
Show a simple illustrated first-person view of a hand reaching for a cup
on a tabletop. Label it “Egocentric RGB video + text prompts”. Reuse the
same hand-and-cup scene in the segmentation example for an intuitive
before-and-after comparison. Do not present this illustration as a video
capture or model prediction.

The RGB input feeds “Clip preparation”: video clips and source frame/time
mapping. RGB is available to every visual stage. Do not imply that every
model shares one decoded frame cache; that optimization is not implemented.

Separate conditional inputs from the main RGB flow:
- Aligned RGB-D, camera intrinsics K, and a metric object mesh for
  FoundationPose.
- A robot configuration and URDF for DexRetarget.
Use dashed amber connectors for these assets.

EXACT MODEL DEPENDENCIES — DO NOT CHANGE
- Prepared RGB clips → SAM2.1 + GroundingDINO.
- Prepared RGB clips → Fast3R, as an independent branch.
- SAM2 hand boxes + RGB → HaMeR.
- HaMeR 3D joints → DexRetarget.
- SAM2 object masks + RGB → FoundationPose, together with its RGB-D,
  camera intrinsics, and mesh assets.
- Do not put Fast3R after SAM2, HaMeR, or FoundationPose.
- Do not draw DexRetarget after FoundationPose.
- Do not imply that all stages are mandatory or execute simultaneously
  on one GPU. These connectors show data dependencies, not a time schedule.

MODEL CARDS — INCLUDE VISUAL EXAMPLES AND REAL OUTPUT NAMES
SAM2.1 + GroundingDINO:
  Show separate translucent masks on the illustrated hand and cup, two
  bounding boxes, stable instance colors, and a short center trajectory.
  Human-readable outputs: instance masks, clip-local IDs, 2D trajectories.
  Files: masks.parquet / instances.parquet / tracks.parquet.

HaMeR:
  Show a coherent five-finger skeleton with 21 joints.
  Outputs: 21 hand joints, joint rotations, hand shape.
  File: hand_pose.parquet.
  Do not label the rotations as named finger flexion angles. Hand joints
  use a hand-root reference frame and model-prior scale, not world fusion.

DexRetarget:
  Show an articulated robotic hand, visibly distinct from the human hand.
  Outputs: robot joint states, named joints, radians/meters as appropriate.
  File: mapping.parquet.
  This is kinematic retargeting, not hardware control or force estimation.

FoundationPose:
  Show the same cup with a projected 3D bounding box and XYZ axes.
  Outputs: object position, object orientation, camera reference frame,
  translation in meters when the required metric assets are supplied.
  File: object_pose.parquet.

Fast3R:
  Show a schematic tabletop/cup point cloud and multiple camera frustums.
  Outputs: sampled-view camera poses and K, scene point cloud, relative
  scale by default. Do not suggest a dense, globally metric SLAM map.
  Files: reconstruction.parquet + reconstruction-pointcloud.npz.

COMMON OUTPUTS AND OPERATIONS
Use a visually distinct shared-output panel labeled “Structured dataset —
from every enabled stage”. Show Parquet + NPZ, source frames/timestamps,
QC, metadata, and stage commit records. It must not appear to be an output
of only one branch.

An understated footer may mention CPU preparation, GPU inference, CPU
retargeting, asynchronous writes, bounded scheduling, and per-stage recovery.
Do not claim zero-copy decoding, optimized throughput, benchmarks,
production certification, or verified 16 GB compatibility.

HONESTY AND LEGEND
Solid arrows = video/stage inputs. Dashed arrows = additional assets.
Explicitly label the examples “Illustrations, not model predictions.”
Define K as camera intrinsics. State that model coordinate frames remain
separate. No invented accuracy, throughput, dataset scale, or real demo UI.
```

结构依据：[`STAGE_DEPENDENCIES` 与输出契约](../../egoworld/src/egoworld/contracts.py)、[输出说明](../outputs.md)、[当前调度实现与设计边界](../design.md)。

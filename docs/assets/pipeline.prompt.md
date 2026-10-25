# EgoWorld 流程图绘制 prompt

用于图像生成或设计工具复刻。当前 `pipeline.svg` 是按以下信息结构绘制的可编辑矢量图；图内的手、杯子、骨架和点云均为示意，不代表真实模型预测。

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

文件描述： 技术栈、版本与依赖环境。

内容：

核心语言与框架:

Python 3.10+ (类型提示强依赖)

Ray: 分布式计算核心 (Actor 调度与 Shared Memory)

PyTorch 2.x (CUDA 12.x，具体以兼容矩阵为准)

核心算法模型栈 (Core Model Stack):

检测与分割 (Detection & Segmentation):

SAM2.1 (small): 默认骨干，负责检测、分割与视频跟踪一体化。

SAM2.1 (concept=hand): 专门配置用于手部区域的高精度分割与提取。

Grounding DINO 1.5: 开放词汇检测 (Open-vocabulary Detection)，用于为 SAM2.1 提供语义提示。

3D 重建与 SLAM (3D & SLAM):

DROID-Splat: 端到端 SLAM + 3DGS (3D Gaussian Splatting)，实现“视频一条龙”转 3D 场景与位姿。

视频处理与切分 (Video Structure):

TransNet V2: 镜头边界检测 (Shot Boundary Detection)，替代传统算法进行高精度场景切分。

精细化跟踪 (Fine-grained Tracking):

CoTracker3: SOTA 点轨迹跟踪 (Point Trajectory Tracking)，用于捕获物体或手部的精细运动流。

数据与存储:

存储后端: S3 Compatible Object Storage (MinIO / AWS S3)

中间态数据: Apache Parquet (用于存储 Metadata, Mask RLE, Trajectories), LanceDB

序列化: PyArrow (Zero-copy 传输)

监控与观测:

Prometheus, Grafana, Ray Dashboard

环境与兼容矩阵:

见 `docs/env-policy.md` 与 `docs/env-matrix.md`（在线 H100 单机离线环境）。

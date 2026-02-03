# 环境兼容矩阵（H100 离线服务器）

状态：Draft（用于审核）
最后更新：2026-02-02
范围：仅覆盖在线 H100 单机离线批处理服务器（Linux x86_64）。

## 1. 硬件基线
- GPU：4× NVIDIA H100 80GB HBM3
- CPU：128C（目标配置）
- 内存：TBD
- 存储：本地磁盘（离线批处理）

## 2. 系统与驱动
- OS：Rocky Linux 8.8 (Green Obsidian)
- Kernel：4.18.0-477.27.1.el8_8.x86_64
- NVIDIA Driver（最低版本）：570.86.15
- NVIDIA Driver（当前观测）：570.86.15（观测于 2026-01-31）
- CUDA Driver Runtime：12.8（`nvidia-smi` 报告）
- cuDNN：未安装（系统未发现包/库/头文件）
- NVIDIA Container Toolkit：未安装（nvidia-container-cli/runtime 不存在）

说明：实际 PyTorch CUDA 版本以锁文件为准，必须与驱动兼容。

## 3. 运行时组件（Base）
- Python：3.10.x（锁文件固定）
- PyTorch：2.x（锁文件固定）
- Ray：锁文件固定
- FFmpeg/ffprobe：系统安装，需在 PATH 中可用
- opencv：锁文件固定（SAM2 依赖）
- pycocotools：锁文件固定（可选，有纯 Python 兜底）

## 4. 关键扩展库（需与 Torch/CUDA 对齐）
- torchvision：锁文件固定
- xformers：锁文件固定（如使用）
- triton：锁文件固定（如使用）

## 5. 系统工具链与 ABI
- glibc：2.28
- gcc/g++ 或 clang：gcc 8.5.0
- cmake：未安装（仅在编译扩展时必需）
- ninja：未安装（仅在编译扩展时必需）
- CUDA Toolkit / nvcc：12.8（V12.8.61）
- NCCL：未安装（系统未发现包/库/头文件）

说明：cmake/ninja 仅在编译 C++/CUDA 扩展时必需；MVP skeleton 可不安装。

## 6. Model 环境差异（按需补充）
| Model Env | 关键依赖 | 备注 |
| --- | --- | --- |
| egoworld-sam2 | sam2.1 small, groundingdino, opencv, pycocotools | 当前已接入（Phase A 暂在 Base 环境统一） |
| egoworld-groundingdino | TBD | Grounding DINO 1.5 |
| egoworld-hamer | TBD | Hand pose |
| egoworld-foundationpose | TBD | 6D pose |
| egoworld-dex-retarget | TBD | Retargeting |
| egoworld-droid-splat | TBD | 3D/SLAM |
| egoworld-tracking | TBD | CoTracker |
| egoworld-transnet | TBD | Shot boundary |

## 7. 更新规则
- 环境锁文件变更必须同步更新本矩阵。
- 记录变更日期与影响范围（Base / Model Env）。

# egoworld 环境管理规范（离线 H100 服务器）

状态：Draft（用于审核）
范围：仅覆盖在线 H100 单机离线批处理服务器（Linux x86_64）。不考虑本地开发机或多机集群。

## 1. 目标与原则
- 复现性：环境可重建、依赖可追溯、版本可核验。
- 低耦合：调度/IO 与模型推理解耦，避免模型依赖冲突拖垮全局。
- 渐进式：先统一、再拆分、最终容器化；每一步都有清晰触发条件。

## 2. 分阶段策略（A → B → C）
- Phase A（当前 MVP）：统一环境（Base）
  - 目标：跑通 pipeline skeleton 与核心 IO/调度。
  - 规则：单一环境，依赖严格锁定。
- Phase B（模型陆续接入/冲突出现）：按模型拆环境（Model Envs）
  - 目标：将推理解耦为独立环境，降低冲突与排障成本。
  - 规则：出现依赖冲突即拆环境。
- Phase C（稳定量产）：容器化（Runtime Image）
  - 目标：固定 CUDA runtime 与依赖，提升可复现性与可移植性。
  - 前置门禁：宿主机驱动版本必须满足兼容矩阵的最低版本要求。

## 3. 环境分层（最小但够用）
### 3.1 Base 环境（统一）
职责：调度、IO、manifest、状态管理、写入与监控。
最低依赖（以锁文件为准）：
- Python 3.10+
- Ray
- PyTorch GPU 版（匹配 CUDA 12.x）
- PyArrow、scenedetect
- SAM2.1 small + GroundingDINO（当前已接入模型，Phase A 先统一在 Base 环境）
- opencv、pycocotools（SAM2 所需；pycocotools 可选，存在纯 Python 兜底）
- prometheus_client（如启用指标采集）
- FFmpeg/ffprobe（系统依赖）

### 3.2 Model 环境（按模型拆分）
每个模型一个独立环境，名称建议：
- egoworld-sam2（SAM2.1 small 相关）
- egoworld-groundingdino
- egoworld-hamer
- egoworld-foundationpose
- egoworld-dex-retarget
- egoworld-droid-splat
- egoworld-tracking（CoTracker）
- egoworld-transnet

规则：
- 以“冲突即拆环境”为唯一触发条件。
- Model 环境只包含推理所需依赖，不承载 pipeline 调度逻辑。

## 4. Phase B 的实现方式（推荐）
**推荐方案：Base 环境调度 + 子进程调用 Model 环境**
- Base 环境负责任务调度/IO/写入。
- Model 环境通过子进程执行推理（`conda run -n <env> ...`）。
- 每个模型提供一个“Runner CLI”入口，输入为路径 + clip 范围，输出为规范化结果文件。
- GPU 隔离：调度端分配 GPU id，子进程通过 `CUDA_VISIBLE_DEVICES` 继承；默认 1 进程独占 1 GPU。

理由：
- 与 Ray Actor/调度解耦，排障最直接。
- 依赖冲突最小化，不会污染 Base。
- 在离线批处理场景中，子进程启动开销可接受。

**备选方案：Ray runtime_env（conda）**
- 适用于“模型常驻显存 + 高频调用”场景。
- 代价是环境创建与调试复杂度提升。
- 若子进程开销成为瓶颈，再切换到该方案。

## 5. 依赖锁定与文件结构
建议目录：
```
/
  docs/
    env-policy.md
    env-matrix.md
  egoworld/
    env/
      base.yml
      models/
        sam2.yml
        hamer.yml
        foundationpose.yml
        dex_retarget.yml
        droid_splat.yml
        tracking.yml
        transnet.yml
      locks/
        linux-64/
          base.lock
          sam2.lock
          ...
```
规则：
- 仅维护 linux-64 锁文件（服务器唯一目标）。
- `conda-lock` 为唯一锁定来源；pip 依赖必须受 `constraints.txt` 约束。
- 明确 conda channel 顺序（如 `pytorch` / `nvidia` / `conda-forge`），禁止随意新增 channel。
- PyTorch 必须来自 CUDA 发行渠道（例如 `pytorch-cuda` 或官方 CUDA wheels），禁止 CPU-only wheel 混入。
- Git 依赖在生成锁文件前必须 pin 到具体 commit（POC 允许临时使用 main，但不得生成锁文件）。
- 禁止“无锁直接升级依赖”。

## 6. 系统级工具链与 ABI 依赖
以下依赖由系统提供，必须记录在兼容矩阵中：
- glibc 版本
- gcc/g++（或 clang）版本
- cmake、ninja
- CUDA Toolkit / nvcc 版本（如需编译扩展）
- cuDNN / NCCL 版本（如使用）

说明：
- 仅在编译 C++/CUDA 扩展时需要（例如某些模型依赖）。
- MVP skeleton 可不安装，但建议服务器预装以降低模型接入成本。

## 7. 兼容矩阵
兼容矩阵为唯一权威来源：`docs/env/matrix.md`。
变更要求：
- 更新环境文件时同步更新矩阵。
- 记录变更日期与影响范围。

## 8. 权重与缓存路径规范
- 权重目录：`EGOWORLD_MODEL_HOME`（默认 `./models/`）
- 缓存目录：`EGOWORLD_CACHE`（默认 `./cache/`）
- 配置中继续使用 `extra.model_paths` 作为显式覆盖。

## 9. 启动前自检（仅规范，不写代码）
- GPU：`nvidia-smi` 可用；驱动版本符合矩阵。
- CUDA/Torch：`torch.cuda.is_available()` 为 true。
- FFmpeg：`ffmpeg`/`ffprobe` 在 PATH。
- 权重目录：可读写、空间充足、权重校验通过。
- 输出目录：可写、磁盘剩余满足批处理阈值。
- Ray：object store 预算合理，避免大帧常驻。

## 10. 何时从 A 切到 B / C
- 出现依赖冲突或 C++/CUDA 扩展编译不稳定 → 切到 B。
- 进入长期离线批处理/稳定生产 → 切到 C。

## 11. 非目标环境说明
本规范不覆盖本地开发与 CI。若需要，仅允许 CPU-only 的轻量 smoke test，且不作为支持目标。

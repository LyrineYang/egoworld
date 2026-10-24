# 模型环境安装与资产

先在 [服务器操作说明](../../docs/server.md)选择配置。五阶段已接入；安装命令参考官方环境，
原生 CUDA/编译环境仍需在目标服务器验证。各环境安装**同一份项目**，不要把互相冲突的依赖强行合并。

## 1. 基础环境：SAM2.1 + GroundingDINO

在仓库根目录执行：

```bash
bash egoworld/scripts/setup_env.sh --weights
conda activate egoworld-base
python -m pip install -e '.[dev]'
egoworld doctor --config egoworld/configs/example.json --gpu --assets
```

base.yml 固定 Torch 2.5.1 / CUDA 12.4 与两个上游提交。需要 conda；GroundingDINO 扩展还需要匹配的 CUDA toolkit 和编译器。
脚本安装 CLI，仅下载 SAM2/DINO。相对 pip constraints 在 env 目录执行，权重落在仓库根 models。
模型 pins 不是经过服务器验证的完整 lock。

## 2. HaMeR 环境

下列环境名和目录要与 job.json 中的 runner_command / working_dir 一致。
Torch 2.2.0/cu118 是[上游 Docker 配置](https://github.com/geopavlakos/hamer/blob/main/docker/hamer-dev.Dockerfile)的一个安装起点，
不要对该独立环境误用 SAM2 的 Torch>=2.5.1 检查；项目预检已按阶段区分。

```bash
export EGOWORLD_REPO=/absolute/path/to/egoworld
mkdir -p "$EGOWORLD_REPO/external"
git clone --recursive https://github.com/geopavlakos/hamer.git "$EGOWORLD_REPO/external/hamer"
conda create -n egoworld-hamer python=3.10 pip -y
conda activate egoworld-hamer
python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
cd "$EGOWORLD_REPO/external/hamer"
python -m pip install --no-build-isolation -e '.[all]' 'numpy<2' 'opencv-python<4.12'
python -m pip install -e "$EGOWORLD_REPO" 'numpy<2' 'opencv-python<4.12'
bash fetch_demo_data.sh
git rev-parse HEAD
```

按[官方说明](https://github.com/geopavlakos/hamer#installation)取得 MANO_RIGHT.pkl，放入上游根目录的 _DATA/data/mano。
本适配器使用已有 SAM2 手框，不使用 demo 中的 ViTPose 手部检测路径。
配置 checkpoint 指向 _DATA/hamer_ckpts/checkpoints/hamer.ckpt，model_config.yaml 保留在其上两级目录。
working_dir 指向 external/hamer，不能仅复制一个 checkpoint 后忽略其余资产。

## 3. Fast3R 环境

[官方安装说明](https://github.com/facebookresearch/fast3r#installation)使用 Python 3.11 与 CUDA 12.4。
不要安装 DUSt3R 的 cuROPE 扩展。

```bash
export EGOWORLD_REPO=/absolute/path/to/egoworld
git clone https://github.com/facebookresearch/fast3r.git "$EGOWORLD_REPO/external/fast3r"
conda create -n egoworld-fast3r python=3.11 cmake=3.14.0 pip -y
conda activate egoworld-fast3r
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 nvidia/label/cuda-12.4.0::cuda-toolkit -c pytorch -c nvidia
cd "$EGOWORLD_REPO/external/fast3r"
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip install -e "$EGOWORLD_REPO"
git rev-parse HEAD
```

下载完整本地快照（不是只下载 README/config）：

```bash
python - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id="jedyang97/Fast3R_ViT_Large_512",
                  local_dir=os.path.join(os.environ["EGOWORLD_REPO"], "models/fast3r"))
PY
```

job.json 的 model_name_or_path 使用该绝对目录，runner_command 使用该环境 Python。

## 4. DexRetarget 环境

仅使用已有手姿态作运动学映射，不增加传感器或手部检测模型。
[官方安装](https://github.com/dexsuite/dex-retargeting#installation)提供 pip 包：

```bash
conda create -n egoworld-dex python=3.10 pip -y
conda activate egoworld-dex
python -m pip install dex_retargeting
python -m pip install -e /absolute/path/to/egoworld
```

准备真实机器人 retarget YAML、对应 URDF 与网格，填写 urdf_dir、robot_id、handedness 和 human_to_robot_rotation。
纯转动关节可写 joint_units=radians；带滑动/浮动基座的目标按顺序给 radians/meters 数组。
包新旧版本的 NumPy 条件不同，优先用隔离环境；预检和真实优化器测试确认实际版本。

## 5. FoundationPose 环境（仅已有 RGB-D/网格时）

没有已有 RGB-D/网格则保持该分支关闭，不为此增加硬件。
[上游 conda 构建说明](https://github.com/NVlabs/FoundationPose#env-setup-option-2-conda-local)需要编译器、nvcc、PyTorch3D、NVDiffRast 与 mycpp。

```bash
export EGOWORLD_REPO=/absolute/path/to/egoworld
git clone --recursive https://github.com/NVlabs/FoundationPose.git "$EGOWORLD_REPO/external/FoundationPose"
cd "$EGOWORLD_REPO/external/FoundationPose"
conda env create -f environment.yml
conda activate foundationpose
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
export CUDA_HOME=/absolute/path/to/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/pytorch3d.git'
python -m pip install --no-build-isolation 'git+https://github.com/NVlabs/nvdiffrast.git'
python -m pip install -r requirements.txt
bash build_all_conda.sh
python -m pip install -e "$EGOWORLD_REPO"
git rev-parse HEAD
```

模型式路径不需要额外开发 NeRF/BundleSDF。按上游准备权重，保留在仓库 weights/2023-10-28-18-33-37 和 weights/2024-01-11-20-02-45，
每个目录含 model_best.pth 与 config.yml。job.json 的 repo_path 指向这份源码；RGB-D/网格和深度尺度按服务器文档填写。

## 6. 将环境接回同一流水线

各模型 params 中填写该环境的 Python（无需在模型环境安装 Ray）：

```json
"runner_command": ["/absolute/conda/envs/egoworld-hamer/bin/python", "-m", "egoworld.worker"]
```

返回 egoworld-base，进入项目根目录，执行 egoworld doctor --config config.local/job.json --gpu --assets。
预检会进入对应环境检查原生 API 导入与 CUDA，并返回 JSON 子报告；模型资产由协调环境静态检查。
不要配置任意不支持 --doctor 协议的自定义命令。推理失败日志在 .work/<clip>/<stage>/runner.log。
分环境每任务重新加载模型，是兼容性起点；常驻与批次优化不作为本轮新增需求。

将各上游 git commit、下载快照 revision 和环境版本填入 model_versions/部署记录。实际通过后保存 pip freeze 和 conda env export，
不要把未经目标服务器执行的模板标成已验证 lock。最后执行 [单阶段和完整 smoke](../../docs/testing.md)。

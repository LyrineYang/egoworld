Environment setup (H100 server)

Base environment (Phase A: SAM2.1 small integrated)
1) Create from YAML (unlocked):
   conda env create -f egoworld/env/base.yml
   conda activate egoworld-base

2) Lock and install (recommended):
   pip install conda-lock
   conda-lock lock -f egoworld/env/base.yml -p linux-64 -o egoworld/env/locks/linux-64/base.lock
   conda-lock install --name egoworld-base egoworld/env/locks/linux-64/base.lock

One-command setup (recommended)
  bash egoworld/scripts/setup_env.sh --weights --smoke

SAM2.1 small model environment (template)
1) requirements already points to official repos (main); pin commits when stable.
2) Create the env:
   conda env create -f egoworld/env/models/sam2.yml

Notes
- All pip installs must use constraints.txt (see env-policy).
- Pin git dependencies before generating lock files.
- Use CUDA_VISIBLE_DEVICES to bind one process to one GPU.

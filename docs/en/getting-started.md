# Getting started

Start with a short segmentation run. Add hand, robot, object, or scene branches after validating their model environments and assets.

## 1. Install

Real inference targets Linux with NVIDIA CUDA and Conda. There is no universal minimum VRAM guarantee: input size, batch size, models, and branches change memory requirements. The [server guide](../server.md) provides hardware and environment details.

```bash
git clone https://github.com/LyrineYang/egoworld.git
cd egoworld
bash egoworld/scripts/setup_env.sh --weights
conda activate egoworld-base
mkdir -p config.local
cp egoworld/configs/example.json config.local/job.json
```

The setup script prepares the base environment and segmentation weights. Other models have separate dependencies, weights, and sometimes licensed assets. See the [environment guide](../../egoworld/env/README.md). The base installer does not prepare all five stages.

## 2. Choose a configuration

| Template | Starting point |
| --- | --- |
| [example.json](../../egoworld/configs/example.json) | Segmentation and 2D tracking |
| [rgb.json](../../egoworld/configs/rgb.json) | Also human hands, robot joints, and scene geometry |
| [full.json](../../egoworld/configs/full.json) | Also object pose from RGB-D and metric mesh assets |
| [server.json](../../egoworld/configs/server.json) | Shared GPU pool; values need hardware validation |

Edit your local copy for model/environment paths, text prompts, output paths, and enabled branches. Object pose needs matched depth, intrinsics, and meshes; retargeting needs robot configuration and URDF. Keep machine-specific paths in the ignored `config.local/` directory.

## 3. Check before processing

```bash
egoworld plan --config config.local/job.json
egoworld launch --config config.local/job.json --input-dir /absolute/videos --check-only
egoworld doctor --config config.local/job.json --gpu --assets
egoworld smoke --config config.local/job.json --input-video /absolute/videos/sample.mp4
```

Replace absolute paths with existing files. `plan` describes dependencies; `--check-only` performs launch preflight; `doctor` checks readiness. Smoke runs enabled stages and checks reuse on a second invocation. Readiness alone does not prove inference quality.

## 4. Run and recover

```bash
egoworld launch --config config.local/job.json --input-dir /absolute/videos --run-id first_run
```

If interrupted, repeat with unchanged inputs, configuration, models, and run ID. Valid stage commits can be reused. Use a new run ID when processing identity changes. Avoid modifying output tables in place; commit validation relies on identity and contracts.

## 5. Inspect

Use the directory under your configured `paths.output_root`:

```bash
egoworld inspect --run-dir /absolute/output/run_id=first_run
egoworld report --run-dir /absolute/output/run_id=first_run
egoworld schema
```

The generated offline `report.html` contains timing, searchable clip QC, failures, and row counts. Its default detail limit is 1,000 clips; `--max-clips` changes it. Partial coverage is marked. It inspects execution records, not physical pose accuracy.

Review visual quality, source-frame coverage, handedness, coordinate conventions, scale, and asset provenance before downstream use. The [architecture guide](architecture.md) explains why timestamps alone are insufficient.

## Development recipes

An output-oriented prototype exists in the development workspace: `data catalog` → `data plan` → `data run`, with shared model stages and verified cross-run caching. This interface is not in the public baseline described here; check [status](status.md) before following development instructions. The configuration commands above are the published entry point.

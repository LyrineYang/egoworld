# Documentation Map

Start here for usage and configuration: `docs/index.md`.

## Document roles (single source of truth)
- `docs/index.md`: quickstart, config surface, operator matrix, output layout.
- `docs/env/policy.md`: environment strategy, phases, and rules.
- `docs/env/matrix.md`: hardware/driver compatibility baseline.
- `egoworld/env/README.md`: env creation and locking commands.
- `egoworld/tests/TEST_SPEC.md`: test matrix and required smoke tests.
- `docs/ops/progress.md`: milestone tracking.
- `docs/ops/runlog.md`: implementation log.
- `docs/design/tech-context.md`: model/stack inventory.
- `docs/design/memory.md`: background notes and decisions.
- `plan.md`: implementation plan and milestones.

## Documentation rules (efficiency)
- One topic, one owner doc. Do not repeat the same rule in multiple files.
- Update the owner doc first; other docs should only link back.
- When changing config fields, update `docs/index.md` and `egoworld/configs/example.json` together.
- When changing environment dependencies, update `docs/env/policy.md`, `docs/env/matrix.md`, and `egoworld/env/*` together.
- Always keep model baselines consistent (current default: SAM2.1 small).

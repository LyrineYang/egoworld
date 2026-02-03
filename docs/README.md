# Documentation Map

Start here for usage and configuration: `egoworld/README.md`.

## Document roles (single source of truth)
- `egoworld/README.md`: quickstart, config surface, operator matrix, output layout.
- `docs/env-policy.md`: environment strategy, phases, and rules.
- `docs/env-matrix.md`: hardware/driver compatibility baseline.
- `egoworld/env/README.md`: env creation and locking commands.
- `egoworld/tests/TEST_SPEC.md`: test matrix and required smoke tests.
- `egoworld/progress.md`: milestone tracking.
- `egoworld/runlog.md`: implementation log.
- `egoworld/techContext.md`: model/stack inventory.
- `egoworld/memory.md`: background notes and decisions.

## Documentation rules (efficiency)
- One topic, one owner doc. Do not repeat the same rule in multiple files.
- Update the owner doc first; other docs should only link back.
- When changing config fields, update `egoworld/README.md` and `configs/example.json` together.
- When changing environment dependencies, update `docs/env-policy.md`, `docs/env-matrix.md`, and `egoworld/env/*` together.
- Always keep model baselines consistent (current default: SAM2.1 small).

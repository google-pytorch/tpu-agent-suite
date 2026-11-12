# ICI mesh sharding (`ici-mesh-sharding`)

This dataset tests whether an agent with the `ici-mesh-sharding` skill uses
it to attribute collective time to mesh axes and to rank mesh proposals. Its
one task runs on CPU from exported XProf JSON files: no TPU and no compute
backend.

## Tasks

| Task | Workload | Slice |
| :--- | :--- | :--- |
| [`ici-mesh-collective-portfolio`](./ici-mesh-collective-portfolio/) | MoE training step, baseline mesh `data=1, fsdp=4, tensor=2, sequence=1, expert=8` | v5p `4x4x4`, 64 chips |

The mesh, slice and chip count are the fixture's, from
`environment/work/workload_config.json`.

## Scoring

`tests/test.sh` gives 1.0 when every check passes and 0.0 otherwise:

- `/workspace/work/portfolio_recommendation.json` exists and is a JSON
  object. A JSON block in the agent's messages is reported
  (`message_json_found`) but not scored.
- The file names `expert` as the bottleneck axis, an MoE routing role, a
  collective share within 1.5 points of 57% (the share of the fixture's
  `op_profile.json` that `cli.py breakdown` reports), and
  `baseline_requires_split_physical_axes: true`.
- It lists 3 to 7 distinct legal meshes for the fixture's 64 devices, at
  least one of which has `expert < 8` (the baseline's `expert=8`) and fits
  `4x4x4` without splitting a physical axis.
- The agent's ATIF trajectory shows a shell call of the skill's
  `cli.py rank-portfolio` whose result was not an error and holds that
  command's own output: the report's title and proposal-table header, or the
  JSON keys `hypotheses` and `estimated_gain_pct`. An echoed command does not
  count. The answer can be worked out by hand from the five ops in the
  fixture's `op_profile.json`, so the file alone is not enough.
- The trajectory shows no `compute run`, `compute submit` or `kubectl`
  apply/create call.

Without a readable trajectory the last two checks fail, so the reward is 0.
`eval/tests/test_ici_mesh_portfolio_verifier.py` runs the verifier on
synthetic trials, including a right answer with no run of the skill and an
echoed command.

## Arms

`ici-mesh-sharding` ships only in `package.experimental.json`, so the
agent-suite arm runs with `PACKAGE_MANIFEST=package.experimental.json`. See
[`eval/configs/README.md`](../../configs/README.md).

| Config | Installs | Role |
| :--- | :--- | :--- |
| `claude-code/agent-suite` or `agy/agent-suite`, with `PACKAGE_MANIFEST=package.experimental.json` | the experimental package | the agent-suite arm |
| `claude-code/baseline` or `agy/baseline` | nothing | the control |

The control has no `cli.py` to run, so it scores 0 by construction. The task
measures whether an agent with the skill uses it correctly.

## Running

```bash
# agent-suite arm
PACKAGE_MANIFEST=package.experimental.json python3 eval/run.py -c claude-code/agent-suite -d ici-mesh-sharding

# control arm
python3 eval/run.py -c claude-code/baseline -d ici-mesh-sharding
```

# close-gaps CPU gates (`close-gaps-gates`)

This dataset tests whether an agent with the experimental `close-gaps` skill
uses its serving hill-climb gates (`hillclimb_cli.py`) to decide which
vLLM-TorchTPU serving arms to launch and whether to stop a running one. Its
one task runs on CPU from per-leg server metrics, a stub `vllm-torchtpu`
checkout, candidate configs and `aiperf` telemetry: no TPU and no compute
backend.

## Tasks

| Task | Workload | Slice |
| :--- | :--- | :--- |
| [`serving-hillclimb-disagg-gates`](./serving-hillclimb-disagg-gates/) | `zai-org/GLM-5.2-FP8` 1P1D at `c=48`, four candidate arms and one in-flight arm | two tpu7x 8-chip slices, 16 chips |

## Scoring

`tests/test.sh` gives 1.0 when every check passes and 0.0 otherwise:

- `binding_leg` is `decode`: in the fixture metrics, the prefill waiting queue
  has a p50 and p75 of 0 while decode runs at its cap (15.8 of 16 running,
  waiting p50 of 4).
- The prefill-only arm and the symmetric `DCP_SIZE=8` arm are rejected, the
  80-token decode step arm is flagged for a kernel extension (or rejected),
  and the asymmetric DCP arm is launched and selected. Each reason has to name
  the mechanism: a decode-limited deployment, a halved prefill KV cache, or 80
  tokens over the 64-token `ROW_BUCKETS` cap.
- The in-flight arm is stopped, and the reason cites its equal-time
  tok/s/chip ratio, which the fixture telemetry puts at 0.752x (-24.8%) at
  10:30.
- The agent's ATIF trajectory shows shell calls that run
  `hillclimb_cli.py preflight` or `legs`, and `hillclimb_cli.py early-verdict`,
  whose results hold the CLI's own output for these fixtures: a preflight check
  name (`off_leg`, `dcp_size_prefill_kv_halved` or `decode_tp_kernel`),
  `decode_limited` from `legs`, and the 0.752 ratio at 630 s from
  `early-verdict`. The answer can be worked out by hand, so `answer.json`
  alone is not enough, and echoed commands or markers do not count.
- The trajectory shows no `compute submit`, `compute run` (with or without
  flags before the verb) or `kubectl apply` call.
- A missing or malformed trajectory scores 0.

`eval/tests/test_serving_hillclimb_disagg_gates_verifier.py` runs the verifier
on synthetic trials built from real CLI calls, including a right answer with
no run of the CLI, right decisions with empty or wrong reasons, forged CLI
output, compute jobs and malformed trajectories.

## Arms

`close-gaps` and its `hillclimb_cli.py` ship only in
`package.experimental.json`, so the agent-suite arm runs with
`PACKAGE_MANIFEST=package.experimental.json`. See
[`eval/configs/README.md`](../../configs/README.md).

| Config | Installs | Role |
| :--- | :--- | :--- |
| `claude-code/agent-suite` or `agy/agent-suite`, with `PACKAGE_MANIFEST=package.experimental.json` | the experimental package | the agent-suite arm |
| `claude-code/baseline` or `agy/baseline` | nothing | the control |

The control has no `hillclimb_cli.py` to run, so it scores 0 by design.
The task measures whether an agent with the skill uses its gates.

## Running

```bash
# agent-suite arm
PACKAGE_MANIFEST=package.experimental.json python3 eval/run.py -c claude-code/agent-suite -d close-gaps-gates

# control arm
python3 eval/run.py -c claude-code/baseline -d close-gaps-gates
```

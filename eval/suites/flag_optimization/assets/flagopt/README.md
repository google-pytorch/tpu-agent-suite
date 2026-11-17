# flagopt

`flagopt` runs, measures and compares MaxText training runs of this task's
workload on its TPU slice. It is one Python file,
`/opt/flagopt/flagopt.py`, and reads this task's spec (`$FLAGOPT_TASK`) from
`/opt/flagopt/tasks.json`: the fixed settings, the tunable knobs with their
allowed values, the baseline config, the slice and the measurement.
`flagopt spec --json` prints the resolved spec.

## Commands

| command | what it does |
|---|---|
| `flagopt spec [--json]` | the task: model, slice, batch, time budget, fixed settings, tunable knobs with their baseline and allowed values |
| `flagopt example > cfg.json` | the current (baseline) config in the deliverable format, a starting point for edits |
| `flagopt validate cfg.json` | checks a config and prints the exact MaxText overrides and `LIBTPU_INIT_ARGS` |
| `flagopt aot cfg.json` | ahead-of-time compiles the config for the task's slice **on your CPU** (no TPU): catches configs that do not compile and reports peak HBM per device against the slice's HBM |
| `flagopt render cfg.json` | prints the Kubernetes manifests a run would submit, without submitting |
| `flagopt run cfg.json --label NAME` | submits one TPU Job (one pod per host), waits for it, collects logs and metrics, deletes the Job; the Job name is the label plus a unique suffix, so repeating a label never overwrites an earlier run |
| `flagopt run baseline --label base` | the same for the current config |
| `flagopt compare RUN_A RUN_B` | speedup of B over A and the accuracy check between two finished runs |
| `flagopt logs JOB [--tail N]`, `flagopt list`, `flagopt delete JOB` | inspect and clean up Jobs |
| `flagopt timer` | time used and remaining in the time budget |

`flagopt run` options:

| option | meaning |
|---|---|
| `--steps N` | training steps (default: the measurement's step count) |
| `--weights random` | random init instead of loading the model checkpoint: starts faster, same step time |
| `--profile` | captures an XProf trace of steps 2..4 (before the measured steps) into the leader pod's `/tmp/flagopt/out` |
| `--hold-sec N` | keeps the leader pod up for N seconds after training, e.g. to `kubectl cp <pod>:/tmp/flagopt/out ./prof` and open the trace with `xprof` / TensorBoard |
| `--keep` | does not delete the Job when it finishes (delete it yourself with `flagopt delete`) |
| `--queue-timeout S`, `--run-timeout S` | override how long to wait for capacity and for the run |
| `--runs-dir DIR` | where results go (default `/workspace/runs`) |

## Config files

A config is a JSON object:

```json
{
  "maxtext_overrides": {"<knob>": <value>},
  "xla_flags": ["--xla_flag=value"],
  "notes": "optional free text"
}
```

Only the tunable knobs of `flagopt spec` may appear in `maxtext_overrides`;
a knob left out keeps its baseline value. Integers are JSON integers,
booleans `true` / `false`, enums strings. `xla_flags` entries are passed to
the TPU runtime as `LIBTPU_INIT_ARGS` and must be empty when the task does
not allow XLA flags.

## How a run is measured

- A run trains for 26 steps. Its step time is the **median over steps
  5..25**; the earlier steps, which include compilation, are skipped.
  `flagopt spec --json` shows these numbers under `measurement`.
- A run counts as successful when every host finished, the leader exited 0
  and every step it was asked for logged metrics (with a step time past the
  skipped steps).
- Results go to `/workspace/runs/<job>/` (`flagopt run` prints the Job name): `result.json` (status, per-step
  loss / gradient norm / step time, p50 / p90 / CV step time, tokens/s),
  `leader.log`, `job.json` and `pod_script.sh`. Every submit and delete is
  appended to `/workspace/runs/job_log.md`.
- `flagopt compare` reports the speedup (baseline median step time /
  candidate median step time) and checks that, over the first 15 steps, the
  candidate's per-step loss matches the other run's within 0.1% (relative)
  and its gradient norm within 5%; a step missing either value in either run
  fails the check. Performance knobs should not change the math; a config
  that fails this check is not a valid result.
- A profiled run (`--profile`) traces steps 2..4, outside the measured steps.

## Waiting for capacity

The cluster is shared. A Job waiting for capacity stays queued (Kueue) and
`flagopt run` waits for it, within its queue timeout; a full cluster is not a
failure. Once running, a Job is stopped at its run timeout. Use only these
bounded waits.

## Time budget

`flagopt timer` counts from when your container started and compares it with
the task's time budget.

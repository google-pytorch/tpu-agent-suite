---
title: "Profiling a vLLM server on TPU with the phased profiler"
type: howto
tags: [vllm, vllm-torchtpu, xprof, profiling, phased-profiler]
created: 2026-10-09
updated: 2026-10-12
sources: 0
human_reviewed: false
---

# Profiling a vLLM server on TPU with the phased profiler

The phased profiler captures **one short XProf trace per inference phase**
(prefill-only, prefill-heavy, balanced, decode-heavy, decode-only) instead of
one long trace of everything. Each phase gets its own directory, so you can
look at prefill and decode separately, and each capture is only a few steps
long, so the traces stay small.

It is part of `vllm_torchtpu`, not upstream vLLM. Checked against vLLM 0.29.0
and `vllm_torchtpu` in `torchtpu-vllm-dev` images from 2026-10.

## The example

Two rules decide everything else:

1. **The profiler has to be armed when the server starts.** A server started
   without it cannot be profiled, and restarting a served model on a shared or
   preemptible slice can cost you the slice.
2. **Every rank writes its own trace.** Profile the smallest deployment that
   fits the model, not the full slice.

### 1. Start the server armed

**Inside this package, start the server with `serve_vllm.sh`**
(`scripts/common/serve_vllm.sh`), as every onboarding recipe does, and do not
run `vllm serve` yourself. It arms the phased profiler by default — setting
(A) `USE_PHASED_PROFILER=true`, (D), (E), (F) `max_iterations=5`, (H), (I)
`PYTHON_TRACER_LEVEL=0`, and (J) `--xla_xprof_register_llo_debug_info=true` —
and serves the trace directory on its own port (`8790`) where `tpu-diagnostics`
(the single profiling entry point in this package, using
`scripts/tpu-diagnostics/capture_vllm_profile.py`) collects and validates
`<phase>/plugins/profile/`. The command below is the phased setup when starting
a server any other way, and the list of flags to check when a trace comes out
wrong.

The values are an example; the labels are what matters. It runs as pasted:
the comments sit in a bash array, outside the JSON, which cannot hold them.

```bash
PROFILE_DIR=/logs/artifacts/xprof/raw   # local disk; copied to GCS in step 4
export USE_PHASED_PROFILER=true         # (A) the phased profiler, not the plain torch one
export PYTHON_TRACER_LEVEL=0            # (I) default 1 floods the trace and drops the TPU timeline
# (J) when the question is inside a Pallas kernel; see "Kernels" below
# export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_xprof_register_llo_debug_info=true"

args=(
  --served-model-name "$MODEL_ID"
  --tensor-parallel-size 1              # (C) the smallest TP that fits the model; no DP
  --max-model-len 9216
  --no-enable-prefix-caching            # random prompts must not hit the prefix cache
  # (D) profiler, (E) trace dir, (F) steps per phase, (G) decode-heavy skip, (H) workers only
  --profiler-config "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_DIR\", \"max_iterations\": 5, \"delay_iterations\": 0, \"ignore_frontend\": true}"
)
vllm serve "$MODEL_PATH" "${args[@]}"
```

(E) is the only directory setting. `VLLM_TORCH_PROFILER_DIR` is no longer
read: the worker and the phased runner both take
`profiler_config.torch_profiler_dir`.

| Flag | What happens without it |
| --- | --- |
| (A) `USE_PHASED_PROFILER=true` | You get the plain torch profiler: one long trace of the whole profiled window, much larger. |
| (I) `PYTHON_TRACER_LEVEL=0` | On a large model the Python tracer can fill the 1,000,000-event buffer first. The server log says `Hit maximum event limit of 1000000, truncating`, and the traces come out **host-only**: no TPU timeline. Observed with gemma-4-31B on tpu7x; Qwen3-0.6B on v6e (512/256, c=8) kept its TPU planes at level 1, with traces 2-12% larger. Level 0 is the safe default either way, but it does not stop a long plain capture from hitting the limit: the same Qwen3 run with the plain profiler hit it at level 0 too (146 MB, vs 202 MB at level 1). Short phased captures stayed under it (24-44 MB each). |
| (C) small TP, no DP | One trace per rank, so trace size grows with the deployment. One observed example: gemma-4-31B on a tpu7x 2x2x1 slice wrote 600-780 MB per benchmark cell, 3 GB for four cells. Keep DP out of the profiling deployment, too: replicas can collide on trace names. |
| (D)+(E) `profiler` + `torch_profiler_dir` | With (A) set, the server **refuses to start**: `USE_PHASED_PROFILER is set but there is nowhere to write traces`. Without `--profiler-config` at all, vLLM ≥ 0.22 does not register `/start_profile`, so the server cannot be profiled. |
| (F) `max_iterations` | 0 falls back to 15 steps per phase. 5 is enough to see the steady-state step. |
| (G) `delay_iterations` | Decode steps skipped before the **decode-heavy** capture only (it becomes `num_decode_steps_to_skip`). The decode-only capture ignores it; move that one with the `*_kv_len_threshold` keys (see Capturing long-context steps). |
| (J) `LIBTPU_INIT_ARGS=--xla_xprof_register_llo_debug_info=true` | Pallas kernels show as opaque `tpu_custom_call`s: no LLO ops inside them, and the roofline reads `0.0 GFLOP/s`, `bound_by: Unknown`. Optional; see Kernels. |
| (H) `ignore_frontend` | The API server process gets traced too: more bytes, nothing about the TPU. |

The legacy `additional_config["phased_profiling_dir"]` is rejected at startup.
Use (A) + (E) instead.

### 2. Warm up, then capture

```bash
bench=(
  --model "$MODEL_ID" --tokenizer "$MODEL_PATH"
  --dataset-name random --random-input-len 1024 --random-output-len 1024
  --ignore-eos            # every request decodes its full output length
  --max-concurrency 8
  --num-prompts 32        # at least 4x the concurrency, so the batch is steady
  --num-warmups 16        # warmup runs BEFORE /start_profile, so it is not in the trace
  --profile               # POSTs /start_profile before the measured run and /stop_profile after
)
vllm bench serve "${bench[@]}"
```

`--profile` is all that triggers the capture. Without `vllm bench serve`, do it
by hand: `curl -X POST http://localhost:8000/start_profile`, send the load,
then `curl -X POST http://localhost:8000/stop_profile`. Arm it **before** the
load starts: a phase is captured the first time it runs, so arming mid-run
misses the prefill phases and returns `decode_only` alone.

**Numbers from a profiled pass are not reportable.** Profiling slows the
steps it records. Run the same command again without `--profile` and report
that one.

### 3. Validate the trace before going further

Do this after the **first** capture, before spending hours on a sweep. A
trace with no TPU device plane is a host-only trace. It tells you nothing
about the TPU, and a benchmark that cannot produce one is not a usable
benchmark.

```bash
# Traces only: the directories also hold hlo_proto.pb and op_stats_v2.pb.
find "$PROFILE_DIR" -name '*.xplane.pb' -printf '%s\t%p\n' | sort -k2

# Device data present? Device planes are named /device:TPU:<n>.
for f in $(find "$PROFILE_DIR" -name '*.xplane.pb'); do
  printf '%s\t%s\n' "$(grep -a -c '/device:TPU:' "$f")" "$f"
done                                         # 0 on every file of a phase = no TPU data

# The cause, when it is the tracer:
grep -c 'Hit maximum event limit' vllm_server.log   # must be 0
```

Healthy output: at least one file per captured phase with a non-zero count,
and those files much larger than the host-only ones. Host-only files beside
them are normal: not every rank owns a TPU device. (Observed on tpu7x 2x2x1
with vLLM 0.29: even ranks carried `/device:TPU:0` and `:1`, one chip's two
TensorCores, about 50 MB each at 15 steps; odd ranks were host-only, about
0.5 MB.) Then ask `tpu-diagnostics` to read it.

What bad traces looked like in a real run, all of which "existed":

| Symptom | Cause |
| --- | --- |
| 34-40 MB per rank, no `/device:TPU` plane | `PYTHON_TRACER_LEVEL` left at 1 (I) |
| ~38 KB per rank | the capture window opened and closed before any request arrived |
| only `decode_only/` | profiler armed after the prefill phases had already run |

### 4. Move it to ML Diagnostics and record the link

Local traces disappear with the pod. Copy the directory to a GCS bucket that
ML Diagnostics reads, register it, and put the hosted XProf link in the
benchmark table row it belongs to. How to register a profile:
`mldiag-cli`, `mldiag-gke`, `gcp-mldiagnostics-non-sdk`.

## Many captures from one server

You don't restart a server to profile it again. Each `/start_profile`
starts a capture that stops by itself after its steps, but the profiler stays
armed, recording batch statistics every step, until `/stop_profile`. A second
`/start_profile` before a stop is ignored ("already running"), so always pair
them; `vllm bench serve --profile` does. A server started with the profiler
configured but never asked to profile costs one `None` check per scheduler
step, so arm every server you might want to profile.

On vLLM 0.29.0 and 0.31.0 (the version vllm-torchtpu pins at `9725110e`)
`/start_profile` takes no arguments, so every capture lands in the same
directory. vllm-torchtpu's `docs/developers_guide/profiling/server.md`
(`index.py get vllm-torchtpu-server-profiling`) shows a JSON body with
`profile_prefix` and profiler options; the plugin's workers accept them, but
vLLM's router passes a body on only from v0.32.0, and then only
`profile_prefix`, `delay_iterations` and `max_iterations` (other keys are
dropped). Offline, `LLM.start_profile(profile_prefix=...)` works on 0.31.0.
Until the image's vLLM passes the body, keep one profile per benchmark point
by moving each capture out before starting the next:

```
for each point:
  clean pass          vllm bench serve ...                (the reportable numbers)
  profile pass        vllm bench serve ... --profile      (short: warmup c, about 2 waves)
  wait                until the trace files stop growing  (the writer flushes after stop)
  move                $PROFILE_DIR -> <point>/xprof/
  validate            /device:TPU plane present (step 3)
```

Inside this package, `serving_benchmark.py` runs only the clean benchmark
passes, and `tpu-diagnostics` (with
`scripts/tpu-diagnostics/capture_vllm_profile.py`) owns the profile capture,
trace-flush settle wait, download/move of new `<phase>/plugins/profile/<timestamp>/`
directories, and `/device:TPU:` plane validation.

## What the directory looks like

```
$PROFILE_DIR/
  <phase>/                                  prefill_only, prefill_heavy, balanced, decode_heavy, decode_only
    plugins/profile/<timestamp>/rank<N>_<host>.xplane.pb   the XProf trace, one per chip (rank<N>, no underscore)
    rank_<N>/batch_composition_stats_<ts>.json             per-step batch composition (rank_<N>, with underscore)
    rank_<N>/<host>_<pid>.<id>.pt.trace.json.gz            torch trace of the same window
```

- **Glob `<phase>/plugins/profile/*/*.xplane.pb`.** One phase can span two
  `<timestamp>` directories (ranks split between them). Pick one and you may
  see only the host-only ranks.
- **Match `*.xplane.pb`, never `*.pb`.**
- **A missing phase is not an error.** A phase is captured the first time a
  step of that kind runs. A step's phase comes from its prefill-token ratio:
  1.0 prefill-only, ≥ 0.9 prefill-heavy, 0.4-0.6 balanced, ≤ 0.2
  decode-heavy, 0.0 decode-only. Anything else is ambiguous and never
  captured. A 1k/1k run at low concurrency may never produce a balanced step.

## Kernels (Pallas custom calls)

`vllm_torchtpu` runs attention and other hot ops as Pallas kernels, which a
default trace shows only as opaque `tpu_custom_call`s. To see inside them,
set (J) in the server's environment **before it starts**: libtpu reads
`LIBTPU_INIT_ARGS` once, when it initializes, and a later change has no
effect. Append to any value the image already sets rather than replacing it.
Which flags to use, what they cost and how to read the result:
`index.py get custom-call-profiling` and `index.py get kernel-profiling`.

## Capturing long-context steps

The decode-only capture normally lands on the first decode steps, where the
KV cache is short. To capture steady-state long context, set a minimum KV
length through `--additional-config`. These keys stay out of the compile
cache key, so setting them does not trigger a recompile:

```bash
--additional-config '{"phased_profiler_decode_only_kv_len_threshold": 4096,
                      "phased_profiler_prefill_only_kv_len_threshold": 4096}'
```

## Source

Read from the installed packages, not from upstream docs:
`vllm_torchtpu/platforms/tpu_platform.py` (`_validate_phased_profiling_config`),
`vllm_torchtpu/runner/tpu_runner.py` (`start_phased_profiling`),
`vllm_torchtpu/runner/utils.py` (`PhaseBasedProfiler`, the phase thresholds),
`vllm/benchmarks/serve.py` (warmup before `/start_profile`),
`vllm_torchtpu/worker/tpu_worker.py` (`profiler_config.torch_profiler_dir`).
Directory layout and trace sizes are from a gemma-4-31B-it campaign on tpu7x
2x2x1, 2026-10-09.

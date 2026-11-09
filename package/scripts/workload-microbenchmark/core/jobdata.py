# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Read a finished serving job into a compact, checked-in *snapshot*.

A snapshot is everything the derivation needs, and nothing it does not: the served
command and env, the per-request records of the measured window, the server's counters
split warmup vs profiling, and (optionally) step costs read from the server log. It is
small enough to commit next to a reference config, which is what lets `selftest`
re-derive every reference offline.

Input is a job directory, `<jobs root>/<job-id>/`. Where the server's settings and the client's
results come from are two separate choices, each made in one place:
- server (`read_recipe`): for a server this package started (`compute submit --serve`), vLLM's
  own log. An AgentX job's JobSet manifest, `recipe.yaml`, is read by `jobdata_agentx`, which
  builds on this module and passes its own readers to `snapshot`.
- client (`read_requests`): aiperf's per-request records (`profile_export.jsonl`, with the server
  counters and the step slices beside them); else the cells of the package's own
  `serving_benchmark.py` (`summary.json`), from which the requests are modelled.
Benchmark and server outputs are under `outputs/`, server container logs optionally under
`logs/`. The root is `DEFAULT_ROOT` (`cli.py --jobs`, default `<work>/jobs`). This module fetches
nothing: the agent fetches the job's files first (`compute job artifacts`), then points here.
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
import statistics
from typing import Any

DEFAULT_ROOT = os.path.join(os.getcwd(), ".work", "jobs")

# ---------------------------------------------------------------- recipe


def read_recipe(job_dir: str) -> dict[str, Any]:
    """The served command and env, from the vLLM log of a server this package started."""
    log = _serve_log(job_dir)
    if not log:
        raise FileNotFoundError(
            f"{job_dir}: no vLLM server log with a `non-default args` line")
    return _recipe_from_vllm_log(job_dir, log)


# ---------------------------------------------------------------- aiperf records


def _find(job_dir: str, pattern: str) -> str | None:
    hits = sorted(
        glob.glob(os.path.join(job_dir, "outputs", "**", pattern),
                  recursive=True))
    return hits[0] if hits else None


def read_requests(job_dir: str) -> list[dict[str, Any]]:
    """The client's per-request records: aiperf's `profile_export.jsonl`, else requests modelled
    from the cells of a `serving_benchmark.py` run (`summary.json`)."""
    path = _find(job_dir, "profile_export.jsonl")
    if not path:
        cells = read_summary_cells(job_dir)
        if cells:
            return _summary_requests(cells)
        raise ValueError(
            f"{job_dir}: no profile_export.jsonl (aiperf) and no serving_benchmark.py "
            "summary.json under outputs/")
    rows = []
    for line in open(path):
        d = json.loads(line)
        m, x = d["metadata"], d.get("metrics", {})

        def g(k):
            return (x.get(k) or {}).get("value")

        rows.append({
            "conv": m.get("conversation_id"),
            "turn": m.get("turn_index"),
            "kind": m.get("source_kind"),
            "phase": m.get("benchmark_phase"),
            "t0": m.get("request_start_ns", 0) / 1e9,
            "t1": (m.get("request_end_ns") or 0) / 1e9 or None,
            "isl": g("input_sequence_length") or 0,
            "osl": g("output_sequence_length") or 0,
            "ttft_ms": g("time_to_first_token"),
            "error": d.get("error") is not None,
        })
    rows.sort(key=lambda r: r["t0"])
    t0 = rows[0]["t0"] if rows else 0.0
    for r in rows:
        r["t0"] = round(r["t0"] - t0, 3)
        if r["t1"]:
            r["t1"] = round(r["t1"] - t0, 3)
    return rows


# ---------------------------------------------------------------- serving_benchmark.py summary


def read_summary_cells(job_dir: str) -> list[dict[str, Any]] | None:
    """The cells of a run of the package's own client (scripts/common/serving_benchmark.py,
    `summary.json`): fixed input / output lengths at a fixed concurrency, every prompt new. It keeps
    no per-request records, only each replicate's totals."""
    path = _find(job_dir, "summary.json")
    if not path:
        return None
    d = json.load(open(path))
    if not isinstance(d, dict) or "cells" not in d or "settings" not in d:
        return None
    out = []
    for c in d["cells"]:
        reps = [
            r for r in c.get("replicates", [])
            if r.get("completed") and r.get("duration_s")
        ]
        if c.get("status") != "ok" or not reps:
            continue
        ttft = [
            r["ttft_mean_ms"] for r in reps
            if r.get("ttft_mean_ms") is not None
        ]
        out.append({
            "cell": c["cell"],
            "isl": c["isl"],
            "osl": c["osl"],
            "concurrency": c["concurrency"],
            "requests": int(sum(r["completed"] for r in reps)),
            "duration_s": sum(r["duration_s"] for r in reps),
            "ttft_ms": statistics.mean(ttft) if ttft else None,
        })
    return out or None


def _summary_requests(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Requests MODELLED from a summary's cells: each cell's prompts back to back on
    `concurrency` slots, each lasting the cell's duration x concurrency / completed (a closed
    loop), cells one after another. Every prompt is new, so none extends another."""
    rows, t = [], 0.0
    for c in cells:
        n, conc = c["requests"], c["concurrency"]
        dur = c["duration_s"] * conc / n
        for i in range(n):
            t0 = t + (i // conc) * dur
            rows.append({
                "conv":
                f"{c['cell']}-{i}",
                "turn":
                0,
                "kind":
                "modelled from summary.json",
                "phase":
                "profiling",
                "t0":
                round(t0, 3),
                "t1":
                round(t0 + dur, 3),
                "isl":
                c["isl"],
                "osl":
                c["osl"],
                "ttft_ms":
                c["ttft_ms"] if c["ttft_ms"] is not None else 0.0,
                "error":
                False,
            })
        t += math.ceil(n / conc) * dur
    return rows


def prefix_chain(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per request (warmup included, so it seeds the chain): the trace-level prefix it
    extends. No cache rule here, only lengths:

      reusable  longest earlier (prompt + output) of the same conversation that fits in this
                prompt, taken to be its prefix; new = isl - reusable
      src_turn  the turn that prefix came from
      gap       the previous turn's prompt + output is LONGER than this prompt, so the chain
                fell back to an older turn (e.g. reasoning stripped from the history)
    """
    seen: dict[str, list[tuple[int, Any]]] = {}
    out = []
    for r in requests:
        if not r.get("isl"):
            continue
        hist = seen.setdefault(r["conv"], [])
        prior = [(e, t) for e, t in hist if e <= r["isl"]]
        base, src = max(prior, key=lambda z: z[0]) if prior else (0, None)
        prev = hist[-1][0] if hist else None
        hist.append((r["isl"] + (r.get("osl") or 0), r.get("turn")))
        out.append({
            **r, "reusable": base,
            "src_turn": src,
            "new": r["isl"] - base,
            "prev_len": prev,
            "gap": bool(prev and prev > r["isl"])
        })
    return out


def concurrency(requests: list[dict[str, Any]],
                phase: str = "profiling") -> dict | None:
    """Requests in flight from the client's start/end times, time-weighted over `phase`."""
    ev = []
    for r in requests:
        if r.get("phase") == phase and r.get("t1"):
            ev += [(r["t0"], 1), (r["t1"], -1)]
    if not ev:
        return None
    ev.sort()
    dur: dict[int, float] = {}
    cur = 0
    for (t, dx), (t2, _) in zip(ev, ev[1:]):
        cur += dx
        dur[cur] = dur.get(cur, 0.0) + (t2 - t)
    tot = sum(dur.values())

    def q(p: float) -> int:
        acc = 0.0
        for c in sorted(dur):
            acc += dur[c]
            if acc >= p * tot:
                return c
        return max(dur)

    return {
        "p50": q(0.5),
        "p90": q(0.9),
        "max": max(dur),
        "mean": round(sum(c * t for c, t in dur.items()) / tot, 2),
        "window_s": round(tot, 1),
        "time_share": {
            c: round(dur[c] / tot, 4)
            for c in sorted(dur)
        }
    }


# ---------------------------------------------------------------- server metrics


def _series(block: dict, name: str, **labels) -> dict | None:
    for key in (name, "vllm:" + name):
        if key in block:
            for s in block[key]["series"]:
                if all(
                        s.get("labels", {}).get(k) == v
                        for k, v in labels.items()):
                    return s["stats"]
    return None


def read_server_metrics(job_dir: str) -> dict[str, Any]:
    """Counters as (profiling - warmup) deltas; aiperf's profiling block is cumulative."""
    path = _find(job_dir, "server_metrics_export.json")
    if not path:
        return {}
    d = json.load(open(path))
    W, P = d.get("warmup_metrics", {}), d.get("metrics", {})

    def delta_total(name, **lab):
        p, w = _series(P, name, **lab), _series(W, name, **lab)
        if p is None:
            return None
        return (p.get("total", 0.0) or 0.0) - (
            (w or {}).get("total", 0.0) or 0.0)

    def delta_count(name):
        p, w = _series(P, name), _series(W, name)
        if p is None:
            return None
        return (p.get("count", 0) or 0) - ((w or {}).get("count", 0) or 0)

    def pct(name,
            keys=("p10_estimate", "p50_estimate", "p90_estimate",
                  "p95_estimate", "p99_estimate", "avg")):
        s = _series(P, name) or {}
        return {k.replace("_estimate", ""): s.get(k) for k in keys if k in s}

    def gauge(name):
        s = _series(P, name) or {}
        return {
            k: s.get(k)
            for k in ("avg", "p10", "p25", "p50", "p75", "p90", "max")
        }

    def buckets(name):
        """Per-bin counts (not cumulative), profiling - warmup, keyed by upper bound."""
        for key in (name, "vllm:" + name):
            if key in P:
                p = P[key]["series"][0].get("buckets") or {}
                w = (W.get(key) or {
                    "series": [{}]
                })["series"][0].get("buckets") or {}
                cum = [(b, p[b] - w.get(b, 0)) for b in p]
                return {
                    b: c - (cum[i - 1][1] if i else 0)
                    for i, (b, c) in enumerate(cum)
                }
        return None

    ranges = d.get("summary", {}).get("phase_time_ranges", {})
    prof = ranges.get("profiling", {})
    accepted_per_pos = []
    for pos in range(8):
        v = delta_total("spec_decode_num_accepted_tokens_per_pos",
                        position=str(pos))
        if v is None:
            break
        accepted_per_pos.append(v)
    return {
        "profiling_duration_s":
        (prof.get("end_ns", 0) - prof.get("start_ns", 0)) / 1e9,
        "prompt_tokens":
        delta_total("prompt_tokens"),
        "prompt_tokens_cached":
        delta_total("prompt_tokens_by_source", source="local_cache_hit"),
        "prompt_tokens_computed":
        delta_total("prompt_tokens_by_source", source="local_compute"),
        "generation_tokens":
        delta_total("generation_tokens"),
        "iterations":
        delta_count("iteration_tokens_total"),
        "spec_drafts":
        delta_total("spec_decode_num_drafts"),
        "spec_draft_tokens":
        delta_total("spec_decode_num_draft_tokens"),
        "spec_accepted":
        delta_total("spec_decode_num_accepted_tokens"),
        "spec_accepted_per_pos":
        accepted_per_pos,
        "preemptions":
        delta_total("num_preemptions"),
        "itl_s":
        pct("inter_token_latency_seconds"),
        "running":
        gauge("num_requests_running"),
        "waiting":
        gauge("num_requests_waiting"),
        "kv_usage":
        gauge("kv_cache_usage_perc"),
        # tokens per engine iteration = computed prompt + generated. vLLM books a request's
        # whole computed prefill in the iteration it completes, so bins above the largest
        # decode-only step are (approximately) per-request computed prompt tokens.
        "iteration_tokens_buckets":
        buckets("iteration_tokens_total"),
    }


#: `vllm:iteration_tokens_total` upper bounds kept per slice: <=1, <=8, <=16, <=32 (decode
#: under MTP: 4 x D tokens), <=512 (a small chunk rides the step), >512 (a prefill chunk).
SLICE_EDGES = ("1.0", "8.0", "16.0", "32.0", "512.0", "+Inf")


def read_step_slices(job_dir: str) -> dict[str, Any] | None:
    """aiperf's 1-s scrape of the step-token histogram and the running gauge, over the
    measured window, in the requests' time origin (`read_requests`). Compact enough to keep
    in the snapshot; `core/mix.py` turns it into the step mix.

    One row per second: [t, steps per SLICE_EDGES bin (6), running max, running min,
    running avg]. The histogram is observed when a step ENDS, so a second with no steps and
    requests running is inside a long step."""
    path, rq = _find(job_dir, "server_metrics_export.json"), _find(
        job_dir, "profile_export.jsonl")
    if not (path and rq):
        return None
    origin = min(
        json.loads(line)["metadata"].get("request_start_ns", 0)
        for line in open(rq)) / 1e9
    d = json.load(open(path))
    M = d.get("metrics", {})
    prof = d.get("summary", {}).get("phase_time_ranges",
                                    {}).get("profiling", {})
    if "vllm:iteration_tokens_total" not in M or not prof:
        return None
    lo, hi = prof["start_ns"] / 1e9, prof["end_ns"] / 1e9
    run = {
        t["start_ns"]: t
        for t in M["vllm:num_requests_running"]["series"][0]["timeslices"]
    }
    rows, last = [], 0.0
    for t in M["vllm:iteration_tokens_total"]["series"][0]["timeslices"]:
        t0 = t["start_ns"] / 1e9
        if not lo <= t0 < hi:
            continue
        cum = [t["buckets"].get(e, 0) for e in SLICE_EDGES]
        n = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
        r = run.get(t["start_ns"])
        rmax, rmin, ravg = (r["max"], r["min"],
                            r["avg"]) if r else (last, last, last)
        last = rmax
        rows.append([round(t0 - origin, 3), *n, rmax, rmin, round(ravg, 3)])
    return {
        "edges":
        list(SLICE_EDGES),
        "window": [round(lo - origin, 3),
                   round(hi - origin, 3)],
        "columns": [
            "t", *(f"n_le{e}" for e in SLICE_EDGES), "run_max", "run_min",
            "run_avg"
        ],
        "rows":
        rows
    }


# ---------------------------------------------------------------- server log (optional)

_SCHED = re.compile(
    r"(\d\d):(\d\d):(\d\d) .*\[vLLM Scheduler\] Step: (\d+) prefill tokens.*"
    r"running_reqs=(\d+)")

#: Startup facts read from the server log. The TPU runtime resolves a hybrid (GDN / mamba)
#: model's page size itself (`--block-size` is only a floor), and logs it with the block count;
#: every model's page is also logged by vllm_torchtpu as "Using KV cache block size: N"
#: (platforms/tpu_platform.py), e.g. 256 under the batched-RPA CUSTOM backend.
_LOG_FACTS = {
    "kv_pool_tokens":
    (re.compile(r"GPU KV cache size: ([\d,]+) tokens"), "GPU KV cache size"),
    "block_size_resolved":
    (re.compile(r"(?:TPU block_size derivation path: .*final_block_size=|"
                r"Using KV cache block size: )(\d+)"),
     ("final_block_size=", "Using KV cache block size: ")),
    "kv_num_blocks":
    (re.compile(r"\bnum_blocks=(\d+) num_groups=\d+"), "num_blocks="),
}

#: vLLM logs the sampling defaults it took from the checkpoint's generation_config.json once at
#: startup (vllm/config/model.py): "Default vLLM sampling parameters have been overridden by the
#: model's `generation_config.json`: `{'temperature': 0.6, 'top_k': 20, 'top_p': 0.95}`".
_SAMPLING_CHEAP = "sampling parameters have been overridden"
_SAMPLING_RX = re.compile(r"(\{[^{}]*\})")


def parse_sampling_line(line: str) -> dict[str, Any] | None:
    """The sampling defaults from the server's startup line, else None (pure, unit-tested)."""
    import ast
    if _SAMPLING_CHEAP not in line:
        return None
    m = _SAMPLING_RX.search(line)
    if not m:
        return None
    try:
        d = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    return {
        k: d[k]
        for k in ("temperature", "top_k", "top_p", "min_p",
                  "repetition_penalty") if k in d
    } if isinstance(d, dict) else None


def read_sampling_defaults(job_dir: str,
                           max_lines: int = 2_000_000
                           ) -> dict[str, Any] | None:
    """The served sampling defaults (`parse_sampling_line`) from the job's server log (the API
    server logs it after engine init: line 338,608 on j-9fb52480), with the log
    it came from; None when the server logged no override (vLLM's own defaults then apply,
    unless the client sent per-request parameters)."""
    import gzip
    for path in _server_logs(job_dir):
        gz = path.endswith(".gz")
        with (gzip.open(path, "rt", errors="replace")
              if gz else open(path, errors="replace")) as f:
            for i, line in enumerate(f):
                d = parse_sampling_line(line)
                if d is not None:
                    return {
                        **d, "source":
                        f"server log {os.path.relpath(path, job_dir)}: "
                        "'Default vLLM sampling parameters have been overridden'"
                    }
                if i > max_lines:
                    break
    return None


def _server_logs(job_dir: str) -> list[str]:
    """`vllm.log` in the outputs, else the server container's own log as the job platform saved it
    (`logs/<job>-vllm-server-0/<pod>/vllm-server.log.gz`), largest first: a restarted job keeps
    the short log of its failed attempt beside the real one."""
    path = _find(job_dir, "vllm.log")
    if path:
        return [path]
    hits = glob.glob(
        os.path.join(job_dir, "logs", "*", "*", "vllm-server.log*"))
    if not hits and _serve_log(job_dir):
        return [_serve_log(job_dir)]
    return sorted(hits, key=os.path.getsize, reverse=True)


def read_log(job_dir: str, scan_steps: bool) -> dict[str, Any]:
    """Startup facts always (KV pool size, resolved page size, block count; all near the top);
    prefill step cost only if asked and the log is plain text, since it can be gigabytes. Step
    cost = mean gap between consecutive full-budget prefill steps, the only gap the 1 s
    timestamps can resolve (async scheduling blurs the rest)."""
    import gzip
    out: dict[str, Any] = {}
    logs = _server_logs(job_dir)
    for path in logs:
        gz = path.endswith(".gz")
        with (gzip.open(path, "rt", errors="replace")
              if gz else open(path, errors="replace")) as f:
            for i, line in enumerate(f):
                for key, (rx, cheap) in _LOG_FACTS.items():
                    if key not in out and any(c in line for c in (
                            cheap if isinstance(cheap, tuple) else (cheap, ))):
                        m = rx.search(line)
                        if m:
                            out[key] = int(m.group(1).replace(",", ""))
                if len(out) == len(_LOG_FACTS) or i > 2_000_000:
                    break
        if "kv_pool_tokens" in out:
            out["log_path"] = os.path.relpath(path, job_dir)
            break
    if logs and "log_path" not in out:
        out["log_path"] = os.path.relpath(logs[0], job_dir)
    if not scan_steps or not out.get("log_path") or out["log_path"].endswith(
            ".gz"):
        return out
    with open(os.path.join(job_dir, out["log_path"]), errors="replace") as f:
        f.seek(0)
        rows = []
        for line in f:
            if "[vLLM Scheduler] Step:" not in line:
                continue
            m = _SCHED.search(line)
            if m:
                h, mi, s, p, r = map(int, m.groups())
                rows.append((h * 3600 + mi * 60 + s, p, r))
    budget = max((p for _, p, _ in rows), default=0)
    gaps = [
        b[0] - a[0] for a, b in zip(rows, rows[1:]) if a[1] >= budget -
        64 and b[1] >= budget - 64 and 0 < b[0] - a[0] <= 20
    ]
    if gaps:
        out["prefill_step_s"] = round(statistics.mean(gaps), 3)
        out["prefill_step_pairs"] = len(gaps)
        out["prefill_budget_tokens"] = budget
    return out


# ---------------------------------------------------------------- vLLM server log

#: A job with no `recipe.yaml` (a server started through the package's launcher) is read from
#: vLLM's own log: the API server logs the arguments it was given that differ from the defaults
#: (vllm/entrypoints/utils.py: "non-default args: {...}"), and the engine logs its resolved config
#: ("Initializing a V1 LLM engine ... with config: ..."). Neither records the environment.
ARGS_MARK = "non-default args: "
ENGINE_MARK = "with config: "
#: `non-default args` keys that are not `vllm serve` flags.
_ARGS_SKIP = {"model_tag", "model", "served_model_name", "dispatch_function"}


def _open_log(path: str):
    import gzip
    return (gzip.open(path, "rt", errors="replace")
            if path.endswith(".gz") else open(path, errors="replace"))


def _serve_log(job_dir: str) -> str | None:
    """The vLLM server log (the one holding the `non-default args` line), anywhere under the job
    directory, largest first."""
    for path in sorted(glob.glob(os.path.join(job_dir, "**", "*.log*"),
                                 recursive=True),
                       key=os.path.getsize,
                       reverse=True):
        with _open_log(path) as f:
            for i, line in enumerate(f):
                if ARGS_MARK in line:
                    return path
                if i > 100_000:
                    break
    return None


def parse_args_line(line: str) -> dict[str, Any] | None:
    """The dict of a `non-default args: {...}` line (pure, unit-tested). Values that are not
    Python literals (config objects such as `ProfilerConfig(...)`) are kept as their source
    text, so one such value does not lose the rest."""
    import ast
    if ARGS_MARK not in line:
        return None
    text = line.split(ARGS_MARK, 1)[1].strip()
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Dict):
        return None
    out: dict[str, Any] = {}
    for k, v in zip(node.keys, node.values):
        try:
            key = ast.literal_eval(k)
        except (ValueError, SyntaxError):
            continue
        try:
            out[key] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[key] = ast.unparse(v)
    return out


def _engine_flag(line: str, name: str) -> str | None:
    m = re.search(rf"\b{name}=([^,\s]+)", line.split(ENGINE_MARK, 1)[-1])
    return m.group(1) if m else None


def _recipe_from_vllm_log(job_dir: str, log: str) -> dict[str, Any]:
    """`read_recipe` for a server with no recipe.yaml: the args from vLLM's `non-default args`
    line, as `vllm serve` flags (`tensor_parallel_size` -> `tensor-parallel-size`), and prefix
    caching from the engine config line when the flag was left at vLLM's default. vLLM does not
    log the environment, so `env` is empty: env the model needs comes from its model.py
    `TWIN_ENV`, or `--override env.NAME=...`. The pod cannot see its image: `image` is None, and
    `forward-cmd` then needs `--image` (container backends) or `--interpreter` (host backends)."""
    raw: dict[str, Any] = {}
    prefix = None
    with _open_log(log) as f:
        for line in f:
            if not raw and ARGS_MARK in line:
                raw = parse_args_line(line) or {}
            elif prefix is None and ENGINE_MARK in line and "LLM engine" in line:
                prefix = _engine_flag(line, "enable_prefix_caching")
            if raw and prefix is not None:
                break
    model = raw.get("model") or raw.get("model_tag")
    args = {
        k.replace("_", "-"): v
        for k, v in raw.items() if k not in _ARGS_SKIP
    }
    if "enable-prefix-caching" not in args and prefix in ("True", "False"):
        args["enable-prefix-caching"] = prefix == "True"
    return {
        "job_id": os.path.basename(os.path.normpath(job_dir)),
        "owner": None,
        "image": None,
        "model_path": model,
        "server_args": args,
        "env": {},
        "env_container": {},
        "env_from_script": {},
        "serve_parse": "vllm-log",
        "bench_env": {},
        "slice_workers": None,
        "launch_log": os.path.relpath(log, job_dir),
    }


def read_checkpoint_config(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The checkpoint's config.json for a server read from its vLLM log: the model path the
    server was given, when it is a directory on this host, else this host's Hugging Face cache
    for that model id. None when neither has it."""
    model = rec.get("model_path")
    if not model:
        return None
    path = os.path.join(model, "config.json")
    if not os.path.isfile(path):
        try:
            from huggingface_hub import try_to_load_from_cache
            path = try_to_load_from_cache(
                model,
                "config.json",
                revision=rec["server_args"].get("revision"))
        except Exception:  # noqa: BLE001  (not installed, or not a repo id)
            path = None
    if not isinstance(path, str) or not os.path.isfile(path):
        return None
    with open(path) as f:
        return {
            **json.load(f), "_source":
            f"{path} (model path of the server's args)"
        }


# ---------------------------------------------------------------- snapshot

#: Snapshot line width. A request record (~250 characters) fits on one line.
SNAPSHOT_WIDTH = 260


def dumps_snapshot(snap: Any, width: int = SNAPSHOT_WIDTH) -> str:
    """A snapshot as JSON with one record per line rather than one field per line.

    An object or list that fits in `width` stays on one line; a longer one is
    expanded one entry per line. A list of short flat entries (the step-slice
    rows) is packed several to a line. `json.loads` reads it back unchanged.
    """

    def compact(v: Any) -> str:
        return json.dumps(v, separators=(", ", ": "))

    def flat(v: Any) -> bool:
        return not isinstance(v, (dict, list)) or (isinstance(
            v, list) and not any(isinstance(x, (dict, list)) for x in v))

    def fmt(v: Any, ind: int) -> str:
        pad = " " * ind
        one_line = compact(v)
        if not isinstance(
                v, (dict, list)) or not v or len(one_line) + ind <= width:
            return one_line
        if isinstance(v, dict):
            entries = [
                f"{pad} {json.dumps(k)}: {fmt(x, ind + 1)}"
                for k, x in v.items()
            ]
        else:
            parts = [compact(x) for x in v]
            if all(flat(x) for x in v) and max(map(len, parts)) <= width // 4:
                entries, line = [], ""
                for part in parts:
                    if line and len(line) + len(part) + ind + 3 > width:
                        entries.append(f"{pad} {line}")
                        line = part
                    else:
                        line = f"{line}, {part}" if line else part
                entries.append(f"{pad} {line}")
            else:
                entries = [f"{pad} {fmt(x, ind + 1)}" for x in v]
        open_, close = ("{", "}") if isinstance(v, dict) else ("[", "]")
        return f"{open_}\n" + ",\n".join(entries) + f"\n{pad}{close}"

    return fmt(snap, 0) + "\n"


def snapshot(job_id: str,
             root: str | None = None,
             scan_log: bool = True,
             read_recipe=read_recipe,
             read_checkpoint_config=read_checkpoint_config) -> dict[str, Any]:
    """The job's snapshot. `read_recipe` and `read_checkpoint_config` read the server's side;
    the defaults read a server this package started, and `jobdata_agentx.snapshot` passes the
    ones that also read an AgentX job's manifest."""
    job_dir = os.path.join(root or DEFAULT_ROOT, job_id)
    if not os.path.isdir(job_dir):
        raise FileNotFoundError(
            f"{job_dir} missing. Fetch the job's files into it first "
            f"(`compute job artifacts --job-id {job_id} --out-dir {job_dir}`)")
    rec = read_recipe(job_dir)
    ckpt = read_checkpoint_config(rec)
    # a serving_benchmark.py run: its cells, and the note that `requests` are modelled
    aiperf = _find(job_dir, "profile_export.jsonl")
    cells = None if aiperf else read_summary_cells(job_dir)
    return {
        "_what":
        "workload-microbenchmark job snapshot: the inputs a derivation reads",
        "job_id": job_id,
        "recipe": rec,
        "checkpoint_config": ckpt,
        "requests": read_requests(job_dir),
        "summary_cells": cells,
        "server": read_server_metrics(job_dir),
        "step_slices": read_step_slices(job_dir),
        "log": read_log(job_dir, scan_log),
        "sampling_defaults": read_sampling_defaults(job_dir),
    }

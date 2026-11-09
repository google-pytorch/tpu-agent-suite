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
"""Model-agnostic online serving benchmark for an already-running server.

Sweeps input/output length x concurrency cells with replicates, and writes one
`summary.json` whose flat metric names (`ttft_p50_ms`, `output_tokens_per_s`,
...) are the names a study protocol's `primary_metric` and
`admissibility_gates` refer to, and the names an onboarding report quotes.

Each run is driven by the engine's client:

-   `--engine vllm` (default): `vllm bench serve` -- the client
    vllm-torchtpu's own perf CI uses -- against the OpenAI-compatible server.
    The served name, the path the server loaded (used as the tokenizer) and
    `max_model_len` are read from `/v1/models`, so the same invocation
    measures any model, and the baseline and every arm of a study are
    measured by an identical client.
-   `--engine custom_fastapi`: `benchmark_fastapi_server.py`, beside this
    script, against a server built from torch-inference-coder's FastAPI
    recipe. It writes vLLM's result keys, so the summary has the same names;
    the recipe does not stream, so TTFT, TPOT and ITL are absent. That server
    has no `/v1/models`, so `--model` is required.

`--workload` selects the requests that the client sends. It is independent
of `--engine`.

-   `--workload random` (default): random prompts of fixed length, one cell
    for each ISL:OSL pair and concurrency, as described above.
-   `--workload agentx`: the AgentX agentic-coding benchmark by
    SemiAnalysis. AgentX replays recorded Claude Code sessions, in which
    each turn resends the growing conversation. The client runs AgentX once
    for each `--concurrency` value, for a fixed duration.

    The client does not reimplement AgentX. It downloads the InferenceX
    repository at a pinned commit (`INFERENCEX_SHA`) and calls the functions
    in `benchmarks/benchmark_lib.sh` that the published AgentX results use.
    These functions run aiperf, the load generator, and compute the result.
    The client then maps the result to the metric names of `summary.json`.

    The client needs git and network access to GitHub, PyPI and the Hugging
    Face Hub. The server needs prefix caching enabled and a `max_model_len`
    long enough for the recorded sessions; see `agentx_corpus`.

The client holds no chips, and on a pod of the torchtpu image without them
the plain `vllm` CLI does not start. So by default this script runs vllm as
itself, `serving_benchmark.py --as-vllm-cli <vllm CLI args>`, which keeps
torch_tpu and libtpu out of the process; see `run_as_vllm_cli`.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import json
import os
import pathlib
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

Path = pathlib.Path

PERCENTILES = (50, 90, 99)
LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")

# vllm bench serve result key -> summary metric name.
THROUGHPUT_KEYS = {
    "output_throughput": "output_tokens_per_s",
    "total_token_throughput": "total_tokens_per_s",
    "request_throughput": "requests_per_s",
    "duration": "duration_s",
    "completed": "completed",
    "failed": "failed",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_output_tokens",
}

ENGINES = ("vllm", "custom_fastapi")
FASTAPI_TASKS = ("generate", "embed", "vqa")
FASTAPI_CLIENT = Path(__file__).resolve().with_name(
    "benchmark_fastapi_server.py")
# First argument that makes this script run the vllm CLI; see run_as_vllm_cli.
AS_VLLM_CLI = "--as-vllm-cli"

ENDPOINTS = {
    "completions": ("vllm", "/v1/completions"),
    "chat": ("openai-chat", "/v1/chat/completions"),
}

SGLANG_BACKENDS = {
    "completions": ("sglang-oai", "/v1/completions"),
    "chat": ("sglang-oai-chat", "/v1/chat/completions"),
}

BENCH_CLIENTS = ("vllm", "sglang")

EXIT_OK = 0
EXIT_CELL_FAILED = 1
EXIT_SETUP = 2

GCS_SCHEME = "gs://"

WORKLOADS = ("random", "agentx")

# The InferenceX commit that the AgentX client uses. It is the commit pinned
# by SemiAnalysis's inferencex-agentx client image. At this commit, the aiperf
# submodule is at 754356e9. Both repositories are public.
INFERENCEX_REPO = "https://github.com/SemiAnalysisAI/InferenceX.git"
INFERENCEX_SHA = "5ecf6fdb01d3854c127356342e5c36b69d07b4de"
AIPERF_SUBMODULE = "utils/aiperf"
# The two sets of recorded sessions that AgentX can replay. The full set has
# sessions of up to about 1M tokens. The other set is capped at 256k tokens.
# In both cases, aiperf skips any session longer than the server's
# max_model_len. AGENTX_FULL_CONTEXT is the max_model_len from which the full
# set is used. Below AGENTX_MIN_CONTEXT, the client warns that many sessions
# are skipped.
AGENTX_CORPUS_FULL = "semianalysis_cc_traces_weka_062126"
AGENTX_CORPUS_256K = "semianalysis_cc_traces_weka_062126_256k"
AGENTX_FULL_CONTEXT = 1_000_000
AGENTX_MIN_CONTEXT = 262_144
# The standard length of the measured phase, in seconds. AgentX marks a run
# with a measured phase under AGENTX_MIN_DURATION_S as not valid for
# submission.
AGENTX_DURATION_S = 1800
AGENTX_MIN_DURATION_S = 900
# The default run timeout is the measured phase plus this many seconds. It
# covers loading the recorded sessions (up to 30 minutes), waiting for the
# warmup requests to finish (up to 30 minutes) and aiperf's shutdown.
AGENTX_TIMEOUT_SLACK_S = 7200
# AgentX fails a run when latency measurements cover less than a set share of
# the measured phase. The share is set in AGENTX_SCENARIO_FILE, on the line
# that starts with AGENTX_COVERAGE_KEY. --agentx-coverage-ratio changes that
# line in the client's copy of InferenceX.
AGENTX_SCENARIO_FILE = (
    "utils/aiperf/src/aiperf/common/scenario/inferencex_agentx_mvp.py")
AGENTX_COVERAGE_KEY = "minimum_profile_metric_coverage_ratio="
# The script for one AgentX run. It runs in the InferenceX directory and calls
# the same benchmark_lib.sh functions as the AgentX scripts in InferenceX.
# prepare_agentx installs aiperf once per sweep, so AIPERF_DEPS_READY=1 tells
# benchmark_lib.sh not to reinstall it for each run.
AGENTX_REPLAY_SCRIPT = """set -eo pipefail
source benchmarks/benchmark_lib.sh
AIPERF_DEPS_READY=1
resolve_trace_source
mkdir -p "$RESULT_DIR"
build_replay_cmd "$RESULT_DIR"
echo "REPLAY_CMD=$REPLAY_CMD"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
"""
# The script that installs aiperf in its own Python environment, with the
# installer in benchmark_lib.sh.
AGENTX_INSTALL_SCRIPT = """set -eo pipefail
source benchmarks/benchmark_lib.sh
install_agentic_deps
"""


def local_tokenizer(name: str) -> str:
    """A local copy of a gs:// checkpoint folder's tokenizer and config files
    (no weights), made by fetch_weights.py.

    In a job, fetch_weights.py is shipped beside this script; in the package
    it is in the tpu-model-onboarding scripts.
    """
    here = Path(__file__).resolve().parent
    for folder in (here, here.parent / "tpu-model-onboarding"):
        if ((folder / "fetch_weights.py").is_file()
                and str(folder) not in sys.path):
            sys.path.insert(0, str(folder))
    try:
        import fetch_weights  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            f"{name} is a gs:// folder: ship fetch_weights.py beside this "
            "script (--input <fetch_weights.py>:fetch_weights.py)") from e
    # Its progress lines go to stderr: stdout is this script's JSON.
    with contextlib.redirect_stdout(sys.stderr):
        return fetch_weights.fetch_gcs(name, small_only=True)


def parse_cells(spec: str) -> list[tuple[int, int]]:
    """Parses "1024:128,2048:256" into [(1024, 128), (2048, 256)]."""
    cells = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        isl, sep, osl = item.partition(":")
        if not sep:
            raise ValueError(f"expected ISL:OSL, got {item!r}")
        cells.append((int(isl), int(osl)))
    if not cells:
        raise ValueError("no ISL:OSL cells given")
    return cells


def parse_ints(spec: str) -> list[int]:
    values = [int(v) for v in spec.split(",") if v.strip()]
    if not values or min(values) < 1:
        raise ValueError(f"expected positive integers, got {spec!r}")
    return values


def parse_avoid_rules(
        spec: str) -> list[tuple[int | None, int | None, int | None, str]]:
    """Parses comma-separated 'isl:osl:c' patterns ('*' matches any value)."""
    rules: list[tuple[int | None, int | None, int | None, str]] = []
    for raw in (spec or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"--avoid rule must be isl:osl:c (use * for any), got {raw!r}")
        parsed: list[int | None] = []
        for p in parts:
            p = p.strip()
            if p == "*":
                parsed.append(None)
            else:
                val = int(p)
                if val < 1:
                    raise ValueError(
                        f"expected positive integer or *, got {p!r} in {raw!r}"
                    )
                parsed.append(val)
        rules.append((parsed[0], parsed[1], parsed[2], raw))
    return rules


def avoid_rule_for(
    isl: int,
    osl: int,
    concurrency: int,
    rules: list[tuple[int | None, int | None, int | None, str]],
) -> str | None:
    """Returns the first matching --avoid rule string, or None."""
    for r_isl, r_osl, r_c, raw in rules:
        if ((r_isl is None or r_isl == isl) and (r_osl is None or r_osl == osl)
                and (r_c is None or r_c == concurrency)):
            return raw
    return None


def skipped_cell(isl: int, osl: int, concurrency: int,
                 error: str) -> dict[str, Any]:
    return {
        "cell": cell_name(isl, osl, concurrency),
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "status": "skipped",
        "error": error,
        "replicates": [],
    }


def filter_c_range(
    isl: int,
    osl: int,
    c_range: list[int],
    *,
    max_model_len: int | None = None,
    range_ratio: float = 0.0,
    max_concurrency_limit: int | None = None,
    avoid_rules: list[tuple[int | None, int | None, int | None, str]] = (),
) -> tuple[list[int], list[dict[str, Any]]]:
    """Static filter: adjusts c_range for (isl, osl) before running.

    Returns (kept_c_range, skipped_cells).
    """
    needed = (max_sampled_len(isl, range_ratio) +
              max_sampled_len(osl, range_ratio))
    if max_model_len and needed > max_model_len:
        err = (f"ISL+OSL up to {needed} tokens exceeds the server's "
               f"max_model_len {max_model_len}")
        return [], [skipped_cell(isl, osl, c, err) for c in c_range]

    kept: list[int] = []
    skipped: list[dict[str, Any]] = []
    for c in c_range:
        if max_concurrency_limit is not None and c > max_concurrency_limit:
            skipped.append(
                skipped_cell(
                    isl,
                    osl,
                    c,
                    f"concurrency {c} > max_concurrency_limit "
                    f"{max_concurrency_limit}",
                ))
            continue
        rule = avoid_rule_for(isl, osl, c, avoid_rules)
        if rule is not None:
            skipped.append(
                skipped_cell(isl, osl, c, f"avoided by --avoid rule {rule}"))
            continue
        kept.append(c)
    return kept, skipped


def _cell_tps(cell: dict[str, Any]) -> float | None:
    return cell.get("metrics", {}).get("output_tokens_per_s", {}).get("mean")


def check_early_stop(
    ok_cells: list[dict[str, Any]],
    curr_cell: dict[str, Any],
    *,
    stop_below_gain_pct: float | None = None,
    stop_on_failure: bool = False,
) -> str | None:
    """Dynamic early-stopping policy evaluated after each concurrency point.

    Returns the stop error string if remaining higher concurrencies in c_range
    for this (isl, osl) shape should be skipped, or None to continue.
    """
    if curr_cell["status"] == "failed":
        if stop_on_failure:
            return f"earlier concurrency c={curr_cell['concurrency']} failed"
        return None
    if stop_below_gain_pct is not None and len(ok_cells) >= 2:
        tps = [
            t for t in (_cell_tps(c) for c in [*ok_cells, curr_cell])
            if t is not None
        ]
        if len(tps) >= 3 and min(tps[:-1]) > 0:
            best_prev = max(tps[:-2])
            best_curr = max(tps[:-1])
            gain_prev = 100.0 * (tps[-2] - best_prev) / best_prev
            gain_curr = 100.0 * (tps[-1] - best_curr) / best_curr
            if (gain_prev < stop_below_gain_pct
                    and gain_curr < stop_below_gain_pct):
                return (
                    f"throughput saturated at c={curr_cell['concurrency']} "
                    f"({gain_curr:+.1f}% < {stop_below_gain_pct:.1f}% vs best "
                    f"{best_curr:.1f} for 2 steps)")
    return None


def shape_ok(shape_cells: list[dict[str, Any]], stop_on_failure: bool) -> bool:
    """Whether one (isl, osl) shape's cells pass.

    Some cell measured, and none failed unless `--stop-on-failure` made a
    failure the end of the shape. Cells skipped by the static filter or by
    early stopping do not fail it. serving_sweep.py judges each shape of a
    merged sweep by this rule.
    """
    return any(c["status"] == "ok"
               for c in shape_cells) and (stop_on_failure
                                          or not any(c["status"] == "failed"
                                                     for c in shape_cells))


def http_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url: str, timeout_s: float) -> None:
    """Waits for /health to answer 200, raising TimeoutError at the deadline."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"{base_url}/health did not answer within {timeout_s:.0f}s")
        time.sleep(5)


def server_alive(base_url: str) -> bool:
    """Whether /health answers 200 right now."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def discover_model(base_url: str) -> dict[str, Any]:
    """Reads the served model's name, loaded path and max_model_len."""
    data = http_json(f"{base_url}/v1/models").get("data") or []
    if not data:
        raise RuntimeError(f"{base_url}/v1/models lists no models")
    entry = data[0]
    max_model_len = entry.get("max_model_len")
    if max_model_len is None:
        for info_path in ("/get_server_info", "/get_model_info"):
            try:
                info = http_json(f"{base_url}{info_path}")
                if isinstance(info, dict):
                    max_model_len = (info.get("context_length")
                                     or info.get("max_total_tokens")
                                     or info.get("max_model_len"))
                    if max_model_len is not None:
                        break
            except (urllib.error.URLError, OSError, ValueError):
                continue
    return {
        "id": entry["id"],
        "root": entry.get("root") or entry["id"],
        "max_model_len": max_model_len,
    }


def reset_prefix_cache(base_url: str, attempts: int = 10) -> bool | None:
    """Empties the server's prefix cache, so a run cannot hit the prompts an
    earlier run of the sweep left in it.

    Tries vLLM's `/reset_prefix_cache` first and falls back to SGLang's
    `/flush_cache`. Returns None when the server has neither endpoint, else
    whether the reset took. A reset fails while blocks are still held, e.g. by
    requests finishing, so it is retried.
    """
    for route in ("/reset_prefix_cache", "/flush_cache"):
        request = urllib.request.Request(f"{base_url}{route}",
                                         data=b"",
                                         method="POST")
        route_missing = False
        for attempt in range(attempts):
            if attempt:
                time.sleep(1)
            try:
                with urllib.request.urlopen(request, timeout=30) as resp:
                    body = resp.read()
            except urllib.error.HTTPError as e:
                if e.code in (404, 405):
                    e.close()  # It holds the response open.
                    route_missing = True
                    break
                raise
            try:
                if json.loads(body).get("success", True):
                    return True
            except (json.JSONDecodeError, AttributeError):
                return True  # An older server or SGLang answers 200 with text.
        if not route_missing:
            return False
    return None


def cell_name(isl: int, osl: int, concurrency: int) -> str:
    return f"isl{isl}_osl{osl}_c{concurrency}"


def max_sampled_len(length: int, range_ratio: float) -> int:
    # vllm bench serve samples lengths in [(1-r)*len, (1+r)*len].
    return int(length * (1 + range_ratio))


def client_name(args: argparse.Namespace) -> str:
    if getattr(args, "workload", "random") == "agentx":
        return "aiperf (InferenceX AgentX)"
    if getattr(args, "bench_client", "vllm") == "sglang":
        return "sglang.bench_serving"
    return ("vllm bench serve" if getattr(args, "engine", "vllm") == "vllm"
            else FASTAPI_CLIENT.name)


def sweep_mode(args: argparse.Namespace) -> str:
    """`exploratory` when any flag may skip a point, else `fixed`: only a
    fixed sweep is comparable across a study's arms."""
    return ("exploratory" if args.stop_below_gain_pct is not None
            or args.stop_on_failure or args.avoid_rules else "fixed")


def prompts_for(args: argparse.Namespace, concurrency: int) -> int:
    """Prompts in one clean run at `concurrency`: `--num-prompts`, else
    `--prompts-per-concurrency` waves of it, at least `--min-prompts`.
    serving_sweep.py sizes its shapes by the same rule."""
    return args.num_prompts or max(args.min_prompts,
                                   args.prompts_per_concurrency * concurrency)


def build_command(args: argparse.Namespace, model: dict[str, Any], isl: int,
                  osl: int, concurrency: int, result_dir: Path,
                  result_filename: str, seed: int) -> list[str]:
    num_prompts = prompts_for(args, concurrency)
    num_warmups = (args.num_warmups
                   if args.num_warmups is not None else concurrency)
    if getattr(args, "engine", "vllm") == "custom_fastapi":
        return build_fastapi_command(args, model, isl, osl, concurrency,
                                     num_prompts, num_warmups, result_dir,
                                     result_filename, seed)
    if getattr(args, "bench_client", "vllm") == "sglang":
        backend, _ = SGLANG_BACKENDS[args.endpoint]
        # SGLang random dataset uses range_ratio=1.0 for fixed length.
        sglang_ratio = max(0.01, 1.0 - args.range_ratio)
        cmd = list(args.sglang_cmd) + [
            "--backend",
            backend,
            "--base-url",
            args.base_url,
            "--model",
            model["id"],
            "--tokenizer",
            args.tokenizer or model["root"],
            "--dataset-name",
            "random",
            "--random-input-len",
            str(isl),
            "--random-output-len",
            str(osl),
            "--random-range-ratio",
            str(sglang_ratio),
            "--num-prompts",
            str(num_prompts),
            "--max-concurrency",
            str(concurrency),
            "--request-rate",
            str(args.request_rate),
            "--warmup-requests",
            str(num_warmups),
            "--seed",
            str(seed),
            "--output-file",
            str(result_dir / result_filename),
        ]
        if args.no_ignore_eos:
            cmd.append("--disable-ignore-eos")
        if args.trust_remote_code:
            cmd.append("--trust-remote-code")
        cmd += shlex.split(args.extra_bench_args or "")
        return cmd

    backend, endpoint = ENDPOINTS[args.endpoint]
    cmd = list(args.vllm_cmd) + [
        "bench",
        "serve",
        "--backend",
        backend,
        "--base-url",
        args.base_url,
        "--endpoint",
        endpoint,
        "--model",
        model["id"],
        "--tokenizer",
        args.tokenizer or model["root"],
        "--dataset-name",
        "random",
        "--random-input-len",
        str(isl),
        "--random-output-len",
        str(osl),
        "--random-range-ratio",
        str(args.range_ratio),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--request-rate",
        str(args.request_rate),
        "--num-warmups",
        str(num_warmups),
        "--percentile-metrics",
        ",".join(LATENCY_METRICS),
        "--metric-percentiles",
        ",".join(str(p) for p in PERCENTILES),
        "--seed",
        str(seed),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if not args.no_ignore_eos:
        cmd.append("--ignore-eos")
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd += shlex.split(args.extra_bench_args or "")
    return cmd


def build_fastapi_command(args: argparse.Namespace, model: dict[str, Any],
                          isl: int, osl: int, concurrency: int,
                          num_prompts: int, num_warmups: int, result_dir: Path,
                          result_filename: str, seed: int) -> list[str]:
    """The FastAPI client, given the flags vllm bench serve would get.

    It is always greedy and has no endpoint or request-rate choice: the recipe
    serves one path per task and the client sends as fast as concurrency
    allows.
    """
    cmd = [
        sys.executable,
        str(FASTAPI_CLIENT),
        "--base-url",
        args.base_url,
        "--task",
        args.task,
        "--model",
        model["id"],
        "--input-len",
        str(isl),
        "--output-len",
        str(osl),
        "--num-prompts",
        str(num_prompts),
        "--max-concurrency",
        str(concurrency),
        "--num-warmups",
        str(num_warmups),
        "--metric-percentiles",
        ",".join(str(p) for p in PERCENTILES),
        "--seed",
        str(seed),
        "--request-timeout",
        str(args.run_timeout or 1800.0),
        "--save-result",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if args.tokenizer:
        cmd += ["--tokenizer", args.tokenizer]
    if args.image_url:
        cmd += ["--image-url", args.image_url]
    if not args.no_ignore_eos:
        cmd.append("--ignore-eos")
    cmd += shlex.split(args.extra_bench_args or "")
    return cmd


def load_raw_result(result_path: Path) -> dict[str, Any]:
    """Reads a benchmark result JSON or SGLang JSONL file."""
    text = result_path.read_text(encoding="utf-8").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise
        return json.loads(lines[-1])


def normalize(raw: dict[str, Any]) -> dict[str, float]:
    """Maps one vllm bench serve or sglang.bench_serving result to flat summary metric names."""
    metrics: dict[str, float] = {}
    for key, name in THROUGHPUT_KEYS.items():
        if raw.get(key) is not None:
            metrics[name] = raw[key]
    if "total_tokens_per_s" not in metrics and raw.get(
            "total_throughput") is not None:
        metrics["total_tokens_per_s"] = raw["total_throughput"]
    if ("failed" not in metrics and raw.get("num_prompts") is not None
            and raw.get("completed") is not None):
        metrics["failed"] = max(
            0,
            int(raw["num_prompts"]) - int(raw["completed"]))
    for m in LATENCY_METRICS:
        raw_aliases = (m, "e2e_latency") if m == "e2el" else (m, )
        for alias in raw_aliases:
            if f"{m}_mean_ms" not in metrics and raw.get(
                    f"mean_{alias}_ms") is not None:
                metrics[f"{m}_mean_ms"] = raw[f"mean_{alias}_ms"]
            if f"{m}_p50_ms" not in metrics and raw.get(
                    f"median_{alias}_ms") is not None:
                metrics[f"{m}_p50_ms"] = raw[f"median_{alias}_ms"]
            for p in PERCENTILES:
                value = raw.get(f"p{p}_{alias}_ms")
                if f"{m}_p{p}_ms" not in metrics and value is not None:
                    metrics[f"{m}_p{p}_ms"] = value
    tpot = metrics.get("tpot_mean_ms")
    if tpot:
        # Decode speed one user sees, as opposed to the aggregate rate.
        metrics["output_tokens_per_s_per_user"] = 1000.0 / tpot
    return metrics


def aggregate(replicates: list[dict[str, float]]) -> dict[str, dict]:
    """Mean, stdev and coefficient of variation per metric across replicates."""
    out: dict[str, dict] = {}
    names = sorted({k for r in replicates for k in r})
    for name in names:
        values = [r[name] for r in replicates if name in r]
        mean = statistics.fmean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        out[name] = {
            "mean": mean,
            "stdev": stdev,
            "cv_pct": (100.0 * stdev / mean) if mean else 0.0,
            "values": values,
        }
    return out


def failure_reason(raw: dict[str, Any], max_failed_pct: float) -> str | None:
    # vllm bench serve and sglang.bench_serving exit 0 even when every request
    # fails, so the result file is the only place a failed run shows.
    completed = int(raw.get("completed") or 0)
    if raw.get("failed") is not None:
        failed = int(raw["failed"])
    elif raw.get("num_prompts") is not None:
        failed = max(0, int(raw["num_prompts"]) - completed)
    else:
        failed = 0
    total = completed + failed
    if completed == 0:
        errors = [e for e in raw.get("errors") or [] if e]
        first = f": {errors[0][:300]}" if errors else ""
        return f"no request completed{first}"
    if total and 100.0 * failed / total > max_failed_pct:
        return f"{failed}/{total} requests failed (limit {max_failed_pct}%)"
    return None


def _run_client(
    args: argparse.Namespace,
    cmd: list[str],
    log_path: Path,
    reset: bool,
    label: str,
    allowed: tuple[int, ...] = (0, ),
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> str | None:
    """Runs one client command and writes its output to `log_path`.

    Args:
        args: The parsed command line.
        cmd: The client command.
        log_path: Where the client's stdout and stderr are written.
        reset: Whether to reset the server's prefix cache first.
        label: Printed to stderr before the run.
        allowed: The exit codes that count as success.
        env: The client's environment. None uses this process's.
        cwd: The client's working directory. None uses this process's.

    Returns:
        An error message, or None if the run succeeded.
    """
    if reset and not reset_prefix_cache(args.base_url):
        return "the server did not reset its prefix cache"
    print(f"[serving_benchmark] {label}", file=sys.stderr, flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd,
                                  stdout=log,
                                  stderr=subprocess.STDOUT,
                                  check=False,
                                  timeout=args.run_timeout or None,
                                  env=env,
                                  cwd=cwd)
        except subprocess.TimeoutExpired:
            # A server that stops producing tokens mid-run would otherwise
            # hold the client for as long as its own HTTP timeout.
            return (f"{client_name(args)} did not finish within "
                    f"{args.run_timeout:.0f}s; see {log_path}")
    if proc.returncode not in allowed:
        return f"{client_name(args)} exited {proc.returncode}; see {log_path}"
    return None


def compute_pareto_frontier(
        points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Returns non-dominated points maximizing (output_tokens_per_s, output_tokens_per_s_per_user)."""
    frontier = []
    for p in sorted(points, key=lambda x: x["concurrency"]):
        tput = p.get("output_tokens_per_s")
        inter = p.get("output_tokens_per_s_per_user")
        if tput is None or inter is None:
            continue
        dominated = False
        for q in points:
            if q is p:
                continue
            q_tput = q.get("output_tokens_per_s")
            q_inter = q.get("output_tokens_per_s_per_user")
            if q_tput is None or q_inter is None:
                continue
            if (q_tput >= tput and q_inter >= inter
                    and (q_tput > tput or q_inter > inter)):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def build_curves(cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Groups completed cells by ISL:OSL and computes their Pareto frontier."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    shape_dims: dict[str, tuple[int, int]] = {}
    for cell in cells:
        if cell.get("status") != "ok" or not cell.get("metrics"):
            continue
        if cell.get("isl") is None or cell.get("osl") is None:
            continue
        isl, osl = int(cell["isl"]), int(cell["osl"])
        key = f"{isl}:{osl}"
        shape_dims[key] = (isl, osl)
        metrics = cell["metrics"]
        point: dict[str, Any] = {
            "cell": cell["cell"],
            "concurrency": int(cell["concurrency"]),
        }
        for m_name in (
                "output_tokens_per_s",
                "total_tokens_per_s",
                "output_tokens_per_s_per_user",
                "requests_per_s",
                "ttft_p50_ms",
                "ttft_p99_ms",
                "tpot_p50_ms",
                "itl_p50_ms",
                "e2el_p50_ms",
        ):
            if m_name in metrics and "mean" in metrics[m_name]:
                point[m_name] = metrics[m_name]["mean"]
        grouped.setdefault(key, []).append(point)

    curves: dict[str, dict[str, Any]] = {}
    for key, pts in grouped.items():
        pts_sorted = sorted(pts, key=lambda x: x["concurrency"])
        frontier = compute_pareto_frontier(pts_sorted)
        tputs = [
            p["output_tokens_per_s"] for p in pts_sorted
            if "output_tokens_per_s" in p
        ]
        inters = [
            p["output_tokens_per_s_per_user"] for p in pts_sorted
            if "output_tokens_per_s_per_user" in p
        ]
        isl, osl = shape_dims[key]
        curves[key] = {
            "isl":
            isl,
            "osl":
            osl,
            "points":
            pts_sorted,
            "pareto_frontier":
            frontier,
            "peak_throughput_tokens_per_s":
            max(tputs) if tputs else None,
            "peak_interactivity_tokens_per_s_per_user":
            max(inters) if inters else None,
        }
    return curves


def evaluate_pareto_reproduction(
    curves: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    tolerance_pct: float = 5.0,
) -> dict[str, Any]:
    """Compares measured curves against a reference Pareto curve for BKC promotion.

    Accepts `reference` as a full `summary.json` (with `curves` or `cells`), or
    a `curves` mapping (`{"1024:128": {"pareto_frontier": [...]}}` or
    `{"1024:128": {"points": [...]}}`). A sweep is consistent (and promoted to
    Best Known Config) when every matched reference Pareto point is reproduced
    within `-tolerance_pct` on throughput (`output_tokens_per_s`) and
    interactivity (`output_tokens_per_s_per_user`, when present in reference).
    """
    ref_curves = reference.get("curves")
    if not isinstance(ref_curves, dict):
        if isinstance(reference.get("cells"), list):
            ref_curves = build_curves(reference["cells"])
        else:
            ref_curves = reference

    comparisons: list[dict[str, Any]] = []
    for shape, ref_curve in ref_curves.items():
        if not isinstance(ref_curve, dict):
            continue
        ref_pts = (ref_curve.get("pareto_frontier") or ref_curve.get("points")
                   or [])
        measured_curve = curves.get(shape, {})
        measured_by_conc = {
            int(p["concurrency"]): p
            for p in measured_curve.get("points", [])
        }
        for ref_pt in ref_pts:
            conc = int(ref_pt["concurrency"])
            m_pt = measured_by_conc.get(conc)
            if m_pt is None:
                comparisons.append({
                    "shape":
                    shape,
                    "concurrency":
                    conc,
                    "consistent":
                    False,
                    "reason":
                    f"missing concurrency {conc} in measured sweep",
                })
                continue
            ref_tput = ref_pt.get("output_tokens_per_s")
            m_tput = m_pt.get("output_tokens_per_s")
            if not ref_tput or m_tput is None:
                continue
            tput_delta_pct = 100.0 * (m_tput - ref_tput) / ref_tput
            ref_inter = ref_pt.get("output_tokens_per_s_per_user")
            m_inter = m_pt.get("output_tokens_per_s_per_user")
            inter_delta_pct = None
            if ref_inter and m_inter is not None:
                inter_delta_pct = 100.0 * (m_inter - ref_inter) / ref_inter
            point_ok = tput_delta_pct >= -tolerance_pct and (
                inter_delta_pct is None or inter_delta_pct >= -tolerance_pct)
            comparisons.append({
                "shape":
                shape,
                "concurrency":
                conc,
                "reference_output_tokens_per_s":
                ref_tput,
                "measured_output_tokens_per_s":
                m_tput,
                "throughput_delta_pct":
                round(tput_delta_pct, 2),
                "reference_output_tokens_per_s_per_user":
                ref_inter,
                "measured_output_tokens_per_s_per_user":
                m_inter,
                "interactivity_delta_pct":
                round(inter_delta_pct, 2)
                if inter_delta_pct is not None else None,
                "consistent":
                point_ok,
            })

    consistent = bool(comparisons) and all(c["consistent"]
                                           for c in comparisons)
    return {
        "tolerance_pct": tolerance_pct,
        "consistent": consistent,
        "promote_to_bkc": consistent,
        "comparisons": comparisons,
    }


def run_cell(args: argparse.Namespace, model: dict[str, Any], isl: int,
             osl: int, concurrency: int, raw_dir: Path, first_seed: int,
             reset: bool) -> dict[str, Any]:
    name = cell_name(isl, osl, concurrency)
    cell: dict[str, Any] = {
        "cell": name,
        "isl": isl,
        "osl": osl,
        "concurrency": concurrency,
        "status": "ok",
        "replicates": [],
    }
    allowed = ((0,
                1) if getattr(args, "engine", "vllm") == "custom_fastapi" else
               (0, ))
    for rep in range(1, args.replicates + 1):
        filename = f"{name}_r{rep}.json"
        seed = args.seed if reset else first_seed + rep - 1
        cell.setdefault("seeds", []).append(seed)
        cmd = build_command(args, model, isl, osl, concurrency, raw_dir,
                            filename, seed)
        cell.setdefault("command", shlex.join(cmd))
        if args.dry_run:
            print(shlex.join(cmd))
            continue
        log_path = raw_dir / f"{name}_r{rep}.log"
        err = _run_client(args,
                          cmd,
                          log_path,
                          reset,
                          f"{name} replicate {rep}/{args.replicates}",
                          allowed=allowed)
        result_path = raw_dir / filename
        if err or not result_path.exists():
            cell["status"] = "failed"
            cell[
                "error"] = err or f"{client_name(args)} wrote no {result_path}"
            break
        raw = load_raw_result(result_path)
        reason = failure_reason(raw, args.max_failed_pct)
        if reason:
            cell["status"] = "failed"
            cell["error"] = reason
            break
        cell["replicates"].append(normalize(raw))

    if cell["replicates"]:
        cell["metrics"] = aggregate(cell["replicates"])
    return cell


# ------------------------------------------------------------------ AgentX


def agentx_corpus(max_model_len: int | None, override: str | None) -> str:
    """Picks the AgentX trace corpus to replay.

    The unfiltered corpus has sessions of up to about 1M tokens, so it is
    picked only when the server's context holds them. Otherwise the corpus
    capped at 256k tokens is picked.

    Args:
        max_model_len: The server's context length, or None if unknown.
        override: The corpus named by --agentx-corpus, or None.

    Returns:
        The aiperf public-dataset loader name of the corpus.
    """
    if override:
        return override
    if max_model_len and max_model_len >= AGENTX_FULL_CONTEXT:
        return AGENTX_CORPUS_FULL
    return AGENTX_CORPUS_256K


def agentx_cell_name(concurrency: int) -> str:
    """Returns the name of the AgentX cell at `concurrency`."""
    return f"agentx_c{concurrency}"


def _setup_step(cmd: list[str], log: Any, **kwargs: Any) -> None:
    """Runs one command of the AgentX client setup.

    Args:
        cmd: The command.
        log: The setup log, open in binary mode. The command and its output
            are appended to it.
        **kwargs: Passed to subprocess.run, e.g. `cwd` and `env`.

    Raises:
        RuntimeError: The command exited non-zero.
    """
    log.write(f"$ {shlex.join(cmd)}\n".encode())
    log.flush()
    proc = subprocess.run(cmd,
                          stdout=log,
                          stderr=subprocess.STDOUT,
                          check=False,
                          **kwargs)
    if proc.returncode:
        raise RuntimeError(f"{shlex.join(cmd)} exited {proc.returncode}; "
                           f"see {log.name}")


def agentx_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Returns the AgentX client's paths under --agentx-cache-dir.

    Args:
        args: The parsed command line.

    Returns:
        A dict with the keys "cache" (the cache directory), "clone" (the
        InferenceX clone) and "venv" (aiperf's venv).
    """
    cache = args.agentx_cache_dir.resolve()
    return {
        "cache": cache,
        "clone": cache / f"InferenceX-{INFERENCEX_SHA[:9]}",
        "venv": cache / "venv",
    }


def prepare_agentx(args: argparse.Namespace, raw_dir: Path) -> Path:
    """Sets up the AgentX client. Called once per sweep.

    Clones InferenceX at INFERENCEX_SHA with its aiperf submodule, and builds
    aiperf's venv with the installer in benchmark_lib.sh. A clone and a venv
    that are already in --agentx-cache-dir are reused. Sets the coverage
    ratio in the clone to --agentx-coverage-ratio, or back to the scenario's
    own value when the flag is not given.

    Args:
        args: The parsed command line.
        raw_dir: Where the setup log, agentx_setup.log, is written.

    Returns:
        The path of the InferenceX clone.

    Raises:
        RuntimeError: A setup step failed, or the clone in the cache is not at
            INFERENCEX_SHA.
    """
    paths = agentx_paths(args)
    clone = paths["clone"]
    paths["cache"].mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "agentx_setup.log", "ab") as log:
        if not (clone / ".git").exists():
            tmp = clone.with_name(clone.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            git = ["git", "-C", str(tmp)]
            for step in (["init", "-q"], [
                    "remote", "add", "origin", INFERENCEX_REPO
            ], ["fetch", "-q", "--depth", "1", "origin", INFERENCEX_SHA], [
                    "checkout", "-q", "FETCH_HEAD"
            ], ["submodule", "update", "-q", "--init", AIPERF_SUBMODULE]):
                _setup_step(git + step, log)
            os.replace(tmp, clone)
        head = subprocess.run(
            ["git", "-C", str(clone), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False).stdout.strip()
        if head != INFERENCEX_SHA:
            raise RuntimeError(f"{clone} is at {head or 'no commit'}, not "
                               f"{INFERENCEX_SHA}; delete it to re-clone")
        scenario = clone / AGENTX_SCENARIO_FILE
        _setup_step([
            "git", "-C",
            str(clone / AIPERF_SUBMODULE), "checkout", "-q", "--",
            str(scenario.relative_to(clone / AIPERF_SUBMODULE))
        ], log)
        if args.agentx_coverage_ratio is not None:
            text = scenario.read_text(encoding="utf-8")
            start = text.find(AGENTX_COVERAGE_KEY)
            if start < 0:
                raise RuntimeError(f"{scenario} sets no {AGENTX_COVERAGE_KEY}")
            start += len(AGENTX_COVERAGE_KEY)
            end = text.index(",", start)
            scenario.write_text(text[:start] +
                                str(args.agentx_coverage_ratio) + text[end:],
                                encoding="utf-8")
        venv = paths["venv"]
        if not ((venv / "bin" / "aiperf").exists() and
                (venv / "bin" / "hf").exists()):
            _setup_step(["bash", "-c", AGENTX_INSTALL_SCRIPT],
                        log,
                        cwd=clone,
                        env=agentx_base_env(args))
    return clone


def agentx_base_env(args: argparse.Namespace) -> dict[str, str]:
    """Returns the environment that benchmark_lib.sh reads its paths from.

    This is this process's environment without PYTHONPATH, with the paths of
    the clone, the venv and the caches under --agentx-cache-dir.

    Args:
        args: The parsed command line.

    Returns:
        The environment, as a dict.
    """
    paths = agentx_paths(args)
    env = dict(os.environ)
    # PYTHONPATH can point at a torch_tpu sitecustomize, which would then load
    # into aiperf's Python environment.
    env.pop("PYTHONPATH", None)
    if not env.get("HF_HOME") and not env.get("HF_HUB_CACHE"):
        env["HF_HUB_CACHE"] = str(paths["cache"] / "hf_hub")
    env.update({
        "INFMAX_CONTAINER_WORKSPACE": str(paths["clone"]),
        "AIPERF_RUNTIME_DIR": str(paths["cache"] / "runtime"),
        "AIPERF_VENV": str(paths["venv"]),
        "AIPERF_UV_CACHE_DIR": str(paths["cache"] / "uv-cache"),
        "AIPERF_UV_INSTALL_DIR": str(paths["cache"] / "uv" / "bin"),
    })
    return env


def agentx_run_env(args: argparse.Namespace, model: dict[str, Any],
                   concurrency: int, run_dir: Path,
                   track: str) -> dict[str, str]:
    """Returns the variables that configure one AgentX replay.

    These are the variables that an AgentX recipe sets before it calls
    benchmark_lib.sh. Run the replay with them added to agentx_base_env.

    Args:
        args: The parsed command line.
        model: The served model, from discover_model.
        concurrency: The number of client lanes.
        run_dir: The directory of this run. The aggregate result is written
            to it, and aiperf's artifacts to its replay/ subdirectory.
        track: The name of this run, e.g. "agentx_c4_r1". The aggregate
            result is written to `run_dir`/`track`.json.

    Returns:
        The variables, as a dict.
    """
    return {
        # benchmark_lib.sh passes MODEL to aiperf as --tokenizer and
        # SERVED_MODEL_NAME as --model.
        "MODEL":
        args.tokenizer or model["root"],
        "SERVED_MODEL_NAME":
        model["id"],
        "AIPERF_SERVER_URL":
        args.base_url,
        "CONC":
        str(concurrency),
        "DURATION":
        str(args.agentx_duration),
        "MAX_MODEL_LEN":
        str(model.get("max_model_len") or 0),
        "WEKA_LOADER_OVERRIDE":
        agentx_corpus(model.get("max_model_len"), args.agentx_corpus),
        "AIPERF_EXPERIMENTAL_FAST":
        "1" if args.agentx_fast else "0",
        "AIPERF_FAILED_REQUEST_THRESHOLD":
        str(args.max_failed_pct / 100.0),
        # Tells InferenceX to read the server's Prometheus metrics as vLLM
        # metrics.
        "FRAMEWORK":
        "vllm",
        # The power monitor reads GPU metrics, which a TPU host does not
        # have.
        "ENABLE_AGENTX_POWER":
        "0",
        "IS_MULTINODE":
        "false",
        # InferenceX requires this variable to compute the result. The server
        # keeps its KV cache in TPU memory only, so the value is none.
        "KV_OFFLOADING":
        "none",
        "AGENTIC_OUTPUT_DIR":
        str(run_dir),
        "RESULT_FILENAME":
        track,
        "RESULT_DIR":
        str(run_dir / "replay"),
    }


def normalize_agentx(agg: dict[str, Any]) -> dict[str, float]:
    """Maps one AgentX aggregate result to the summary's metric names.

    AgentX reports latencies in seconds, at the mean, p50, p75, p90 and p95.
    It reports no p99, so the result has no *_p99_ms metric. AgentX's TPOT is
    the same as its ITL. Throughput counts only the successful requests of
    the measured window.

    Args:
        agg: The aggregate result that process_agentic_result.py wrote.

    Returns:
        The metrics, by summary metric name. Metrics that the result does not
        have are left out.
    """
    req = agg.get("request_metrics") or {}
    latency = req.get("latency") or {}
    throughput = req.get("throughput") or {}
    metrics: dict[str, float] = {}
    for m in LATENCY_METRICS:
        stats = latency.get(m) or {}
        for stat in ("mean", "p50", "p90", "p95"):
            if stats.get(stat) is not None:
                metrics[f"{m}_{stat}_ms"] = stats[stat] * 1000.0
    duration = throughput.get("duration_seconds")
    for key in ("input", "output", "total"):
        tps = (throughput.get(key) or {}).get("tokens_per_second")
        if tps is None:
            continue
        metrics[f"{key}_tokens_per_s"] = tps
        if duration and key != "total":
            metrics[f"total_{key}_tokens"] = round(tps * duration)
    if duration is not None:
        metrics["duration_s"] = duration
    qps = (req.get("qps") or {}).get("mean")
    if qps is not None:
        metrics["requests_per_s"] = qps
    if agg.get("num_requests_successful") is not None:
        metrics["completed"] = agg["num_requests_successful"]
    errors = (agg.get("request_accounting") or {}).get("records_error_dropped")
    if errors is not None:
        metrics["failed"] = errors
    server_cache = (agg.get("server_metrics") or {}).get("cache") or {}
    if server_cache.get("overall_cache_hit_rate") is not None:
        metrics["prefix_cache_hit_rate"] = server_cache[
            "overall_cache_hit_rate"]
    theoretical = (req.get("cache") or {}).get("theoretical_cache_hit_rate")
    if theoretical is not None:
        metrics["theoretical_prefix_cache_hit_rate"] = theoretical
    tpot = metrics.get("tpot_mean_ms")
    if tpot:
        metrics["output_tokens_per_s_per_user"] = 1000.0 / tpot
    return metrics


def run_agentx_cell(args: argparse.Namespace, model: dict[str, Any],
                    concurrency: int, raw_dir: Path,
                    reset: bool) -> dict[str, Any]:
    """Runs the AgentX replay at one concurrency, --replicates times.

    Stops at the first replicate that fails.

    Args:
        args: The parsed command line.
        model: The served model, from discover_model.
        concurrency: The number of client lanes.
        raw_dir: Where each replicate's directory is created.
        reset: Whether to reset the server's prefix cache before each
            replicate.

    Returns:
        The cell, in the same format as run_cell's.
    """
    name = agentx_cell_name(concurrency)
    cell: dict[str, Any] = {
        "cell": name,
        "workload": "agentx",
        "concurrency": concurrency,
        "status": "ok",
        "replicates": [],
    }
    cmd = ["bash", "-c", AGENTX_REPLAY_SCRIPT]
    for rep in range(1, args.replicates + 1):
        track = f"{name}_r{rep}"
        run_dir = raw_dir / track
        run_env = agentx_run_env(args, model, concurrency, run_dir, track)
        cell.setdefault("replay_env", run_env)
        if args.dry_run:
            print(" ".join(f"{k}={shlex.quote(v)}"
                           for k, v in run_env.items()) + " " +
                  shlex.join(cmd))
            continue
        (run_dir / "replay").mkdir(parents=True, exist_ok=True)
        err = _run_client(args,
                          cmd,
                          run_dir / "client.log",
                          reset,
                          f"{name} replicate {rep}/{args.replicates}",
                          env={
                              **agentx_base_env(args),
                              **run_env
                          },
                          cwd=agentx_paths(args)["clone"])
        replay_cmd = run_dir / "replay" / "benchmark_command.txt"
        if replay_cmd.exists():
            cell.setdefault("command",
                            replay_cmd.read_text(encoding="utf-8").strip())
        result_path = run_dir / f"{track}.json"
        if err or not result_path.exists():
            cell["status"] = "failed"
            cell[
                "error"] = err or f"{client_name(args)} wrote no {result_path}"
            break
        agg = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = normalize_agentx(agg)
        reason = failure_reason(
            {
                "completed": metrics.get("completed"),
                "failed": metrics.get("failed")
            }, args.max_failed_pct)
        if reason:
            cell["status"] = "failed"
            cell["error"] = reason
            break
        cell["replicates"].append(metrics)
    if cell["replicates"]:
        cell["metrics"] = aggregate(cell["replicates"])
    return cell


def agentx_settings(args: argparse.Namespace,
                    model: dict[str, Any]) -> dict[str, Any]:
    """Returns the AgentX settings that summary.json records.

    Args:
        args: The parsed command line.
        model: The served model, from discover_model.

    Returns:
        The settings, as a dict.
    """
    canonical = (args.agentx_duration == AGENTX_DURATION_S
                 and not args.agentx_fast
                 and args.agentx_coverage_ratio is None)
    return {
        "inferencex_repo": INFERENCEX_REPO,
        "inferencex_sha": INFERENCEX_SHA,
        "scenario": "inferencex-agentx-mvp",
        "corpus": agentx_corpus(model.get("max_model_len"),
                                args.agentx_corpus),
        "duration_s": args.agentx_duration,
        "fast": args.agentx_fast,
        "coverage_ratio": args.agentx_coverage_ratio,
        # Only a standard run can be compared with published AgentX results.
        "canonical": canonical,
        "cache_dir": str(agentx_paths(args)["cache"]),
    }


def run_agentx(args: argparse.Namespace, model: dict[str, Any], raw_dir: Path,
               reset: bool, run: dict[str, Any]) -> int:
    """Runs the AgentX sweep and writes its summary.

    There is one cell per concurrency, run in ascending order. The summary is
    rewritten after each cell.

    Args:
        args: The parsed command line.
        model: The served model, from discover_model.
        raw_dir: Where the setup log and each run's directory go.
        reset: Whether the server can reset its prefix cache.
        run: The summary's fields other than "status", "error" and "cells".
            The AgentX settings are added to its "settings".

    Returns:
        The exit code.
    """
    run["settings"]["agentx"] = agentx_settings(args, model)
    max_len = model.get("max_model_len")
    if max_len and max_len < AGENTX_MIN_CONTEXT:
        print(
            f"[serving_benchmark] warning: the server's max_model_len, "
            f"{max_len}, is less than {AGENTX_MIN_CONTEXT}. aiperf skips "
            "every recorded session longer than max_model_len, so the load "
            "is lighter than in a standard AgentX run.",
            file=sys.stderr,
            flush=True)
    if not args.dry_run:
        try:
            prepare_agentx(args, raw_dir)
        except (RuntimeError, OSError) as e:
            print(
                json.dumps({
                    "status": "error",
                    "error": f"could not set up the AgentX client: {e}"
                }))
            return EXIT_SETUP
    cells: list[dict[str, Any]] = []
    server_down = None
    for conc in sorted(set(args.concurrencies)):
        if server_down:
            cells.append({
                "cell": agentx_cell_name(conc),
                "workload": "agentx",
                "concurrency": conc,
                "status": "skipped",
                "error": f"not run: {server_down}",
                "replicates": [],
            })
            continue
        cell = run_agentx_cell(args, model, conc, raw_dir, reset)
        cells.append(cell)
        if (cell["status"] == "failed" and not args.dry_run
                and not server_alive(args.base_url)):
            server_down = (f"the server stopped answering {args.base_url}"
                           f"/health during {cell['cell']}")
        if not args.dry_run:
            write_summary(args.output_dir, {
                "status": "partial",
                "error": server_down,
                **run, "cells": cells
            })
    if args.dry_run:
        return EXIT_OK
    return finish(args, run, cells, server_down,
                  all(c["status"] == "ok" for c in cells))


def finish(args: argparse.Namespace, run: dict[str, Any],
           cells: list[dict[str, Any]], server_down: str | None,
           cells_ok: bool) -> int:
    """Writes the final summary and prints its table and path.

    Args:
        args: The parsed command line.
        run: The summary's fields other than "status", "error" and "cells".
        cells: All the cells of the sweep.
        server_down: Why the server stopped answering, or None if it did not.
        cells_ok: Whether the cells pass.

    Returns:
        The exit code.
    """
    ok = cells_ok and server_down is None
    curves = build_curves(cells)
    summary: dict[str, Any] = {
        "status": "ok" if ok else "failed",
        "error": server_down,
        **run,
        "cells": cells,
        "curves": curves,
    }
    reference_curve = getattr(args, "reference_curve", None)
    export_bkc = getattr(args, "export_bkc", None)
    if reference_curve is not None:
        try:
            ref_data = json.loads(reference_curve.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            print(
                json.dumps({
                    "status": "error",
                    "error": f"failed to read --reference-curve: {e}"
                }))
            return EXIT_SETUP
        repro = evaluate_pareto_reproduction(
            curves,
            ref_data,
            tolerance_pct=getattr(args, "tolerance_pct", 5.0),
        )
        summary["reproduction"] = repro
        if not repro["consistent"]:
            ok = False
            summary["status"] = "failed"

    if export_bkc is not None and reference_curve is not None and ok:
        bkc = {
            "status": "best_known_config",
            "created_at": summary["created_at"],
            "model": run.get("model"),
            "client": summary["client"],
            "settings": summary["settings"],
            "curves": curves,
            "reproduction": summary.get("reproduction"),
        }
        export_bkc.parent.mkdir(parents=True, exist_ok=True)
        export_bkc.write_text(json.dumps(bkc, indent=2) + "\n",
                              encoding="utf-8")
        summary["bkc_path"] = str(export_bkc)

    table = write_summary(args.output_dir, summary)
    summary_path = args.output_dir / "summary.json"
    print(table, file=sys.stderr)
    for cell in cells:
        if cell["status"] != "ok":
            print(f"{cell['cell']}: {cell['status']}: {cell.get('error')}",
                  file=sys.stderr)
    print(
        json.dumps({
            "status": summary["status"],
            "summary": str(summary_path)
        }))
    if server_down:
        print(server_down, file=sys.stderr)
        return EXIT_SETUP
    return EXIT_OK if ok else EXIT_CELL_FAILED


def write_summary(output_dir: Path, summary: dict[str, Any]) -> str:
    """summary.json and summary.md, each replaced whole; the table.

    Rewritten after every cell, so a sweep cut short keeps the cells it
    finished. Replaced, never rewritten in place, so whoever copies the files
    mid-sweep (compute sends declared outputs back as they change) never gets
    half of one.
    """
    table = format_table(summary["cells"])
    for name, text in (("summary.json", json.dumps(summary, indent=1)),
                       ("summary.md", table)):
        tmp = output_dir / f".{name}.tmp"
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, output_dir / name)
    return table


def format_table(cells: list[dict[str, Any]]) -> str:
    columns = [
        ("ttft_p50_ms", "TTFT p50 ms"),
        ("ttft_p99_ms", "TTFT p99 ms"),
        ("tpot_p50_ms", "TPOT p50 ms"),
        ("itl_p50_ms", "ITL p50 ms"),
        ("itl_p99_ms", "ITL p99 ms"),
        ("e2el_p50_ms", "E2E p50 ms"),
        ("output_tokens_per_s", "out tok/s"),
        ("total_tokens_per_s", "total tok/s"),
        ("output_tokens_per_s_per_user", "tok/s/user"),
        ("requests_per_s", "req/s"),
    ]
    # Only an AgentX replay reads the server's prefix cache hit rate.
    if any("prefix_cache_hit_rate" in c.get("metrics", {}) for c in cells):
        columns.append(("prefix_cache_hit_rate", "prefix hit"))
    header = ["cell", "status"] + [label for _, label in columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for cell in cells:
        metrics = cell.get("metrics", {})
        row = [cell["cell"], cell["status"]]
        for key, _ in columns:
            m = metrics.get(key)
            if m is None:
                row.append("-")
            elif len(m["values"]) > 1:
                row.append(f"{m['mean']:.2f} ±{m['cv_pct']:.1f}%")
            else:
                row.append(f"{m['mean']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def run_as_vllm_cli(argv: list[str]) -> int:
    """Runs the vllm CLI with `argv`, keeping the TPU stack out of the process.

    On a pod of the torchtpu image without chips the plain CLI does not start:
    `import torch` autoloads torch_tpu, which brings up libtpu; with torch_tpu
    not loaded, nothing names the PrivateUse1 backend and vllm fails with
    "Expected one of cpu, cuda, ... device type: tpu"; and the torchtpu vllm
    platform plugin imports torch_tpu. A client needs none of them.

    Nothing this module imports at the top loads torch, so this runs before
    torch does. It is for `vllm bench serve`: with no platform, vllm commands
    that build an engine config, even `--version`, fail to infer a device.
    """
    # Forced, not defaulted: a caller's values, say copied from the server's
    # recipe, would bring torch_tpu back. torch reads the first one once, at
    # import; vllm bench serve needs no plugin.
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["VLLM_PLUGINS"] = ""
    for name in ("libtpu", "torch_tpu"):
        # `None` makes any later import of it raise ImportError.
        sys.modules.setdefault(name, None)
    import torch  # pylint: disable=import-outside-toplevel
    try:
        # Names the backend as torch_tpu would, for vllm code that still asks
        # for the "tpu" device.
        torch.utils.rename_privateuse1_backend("tpu")
    except RuntimeError:
        pass  # Already named.
    from vllm.entrypoints.cli.main import \
        main as vllm_main  # pylint: disable=import-outside-toplevel
    sys.argv = ["vllm", *argv]
    return vllm_main()


def resolve_vllm_cmd(value: str | None) -> list[str]:
    if value:
        return shlex.split(value)
    if importlib.util.find_spec("vllm") is not None:
        return [sys.executable, str(Path(__file__).resolve()), AS_VLLM_CLI]
    # Right where there are chips, or vllm is not torchtpu's; on a pod of the
    # torchtpu image without them, the plain CLI does not start.
    print(
        f"[serving_benchmark] warning: {sys.executable} has no vllm, so "
        f"vllm runs without {AS_VLLM_CLI} and will not start on a pod "
        "without TPU chips",
        file=sys.stderr,
        flush=True)
    found = shutil.which("vllm")
    if found:
        return [found]
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def resolve_sglang_cmd(value: str | None) -> list[str]:
    if value:
        return shlex.split(value)
    return [sys.executable, "-m", "sglang.bench_serving"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url",
                   default="http://localhost:8000",
                   help="Server root, without /v1 (default: %(default)s)")
    p.add_argument("--output-dir",
                   required=True,
                   type=Path,
                   help="Where raw results, logs and summary.json go")
    p.add_argument("--engine",
                   choices=ENGINES,
                   default="vllm",
                   help="Serving engine, which picks the client "
                   "(default: %(default)s)")
    p.add_argument("--task",
                   choices=FASTAPI_TASKS,
                   default="generate",
                   help="custom_fastapi only: the recipe the server was "
                   "built from (default: %(default)s)")
    p.add_argument("--image-url",
                   default=None,
                   help="custom_fastapi --task vqa only: the image every "
                   "request sends")
    p.add_argument("--workload",
                   choices=WORKLOADS,
                   default="random",
                   help="The requests to send. random: random prompts of "
                   "fixed length, one cell per ISL:OSL pair and concurrency. "
                   "agentx: the AgentX benchmark, which replays recorded "
                   "agentic-coding sessions (default: %(default)s)")
    p.add_argument("--agentx-duration",
                   type=int,
                   default=AGENTX_DURATION_S,
                   help="agentx only. Length of the measured phase of each "
                   "run, in seconds. AgentX marks a run shorter than "
                   f"{AGENTX_MIN_DURATION_S} seconds as not valid for "
                   "submission (default: %(default)s)")
    p.add_argument("--agentx-fast",
                   action="store_true",
                   help="agentx only. Use the AgentX quick mode for smoke "
                   "tests: one warmup request per replayed session and a "
                   "1200-second measured phase")
    p.add_argument("--agentx-corpus",
                   choices=(AGENTX_CORPUS_FULL, AGENTX_CORPUS_256K),
                   default=None,
                   help="agentx only. The set of recorded sessions to "
                   "replay (default: the full set when the server's "
                   f"max_model_len is at least {AGENTX_FULL_CONTEXT}, "
                   "otherwise the set capped at 256k tokens)")
    p.add_argument("--agentx-cache-dir",
                   type=Path,
                   default=None,
                   help="agentx only. Directory for the InferenceX "
                   "repository, aiperf's Python environment and, unless "
                   "HF_HOME or HF_HUB_CACHE is set, the recorded sessions. "
                   "Later runs reuse its contents (default: "
                   "<output-dir>/agentx_cache)")
    p.add_argument("--agentx-coverage-ratio",
                   type=float,
                   default=None,
                   help="agentx only. AgentX fails a run when latency "
                   "measurements cover less than 0.95 of the measured "
                   "phase. This flag sets a lower share, for a server too "
                   "slow to meet it. The run is then not a standard "
                   "AgentX run")
    p.add_argument("--isl-osl",
                   default=None,
                   help="Comma-separated ISL:OSL cells (default: 1024:128)")
    p.add_argument("--concurrency",
                   default="1",
                   help="Comma-separated max concurrency levels "
                   "(default: %(default)s)")
    p.add_argument("--replicates",
                   type=int,
                   default=1,
                   help="Runs per cell; report mean, stdev and CV across them")
    p.add_argument("--num-prompts",
                   type=int,
                   default=None,
                   help="Prompts per run (default: max(--min-prompts, "
                   "--prompts-per-concurrency x concurrency))")
    p.add_argument("--prompts-per-concurrency", type=int, default=10)
    p.add_argument("--min-prompts", type=int, default=16)
    p.add_argument("--num-warmups",
                   type=int,
                   default=None,
                   help="Discarded warmup requests (default: concurrency)")
    p.add_argument("--range-ratio",
                   type=float,
                   default=0.0,
                   help="Length jitter, sampled in [(1-r)L, (1+r)L]; 0 keeps "
                   "every request at the nominal length")
    p.add_argument("--request-rate",
                   default="inf",
                   help="Requests/s; inf sends as fast as concurrency allows")
    p.add_argument("--endpoint",
                   choices=sorted(ENDPOINTS),
                   default="completions")
    p.add_argument("--temperature",
                   type=float,
                   default=0.0,
                   help="Sampling temperature (default greedy; pass a "
                   "negative value to omit and use the server default)")
    p.add_argument("--seed",
                   type=int,
                   default=42,
                   help="Prompt seed. When the server cannot reset its "
                   "prefix cache, each later run of the sweep adds one, so "
                   "no run resends an earlier run's prompts")
    p.add_argument("--no-ignore-eos",
                   action="store_true",
                   help="Let requests stop at EOS instead of running to OSL")
    p.add_argument("--model",
                   default=None,
                   help="Served model name (default: from /v1/models)")
    p.add_argument("--tokenizer",
                   default=None,
                   help="Tokenizer path, id or gs:// checkpoint folder "
                   "(default: the path the server loaded, from /v1/models); a "
                   "gs:// folder's tokenizer files are copied here")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--max-failed-pct",
                   type=float,
                   default=None,
                   help="Fail a cell when more than this percentage of its "
                   "requests fail (default: 0 for --workload random, 10 for "
                   "--workload agentx, which is the AgentX limit)")
    p.add_argument("--ready-timeout",
                   type=float,
                   default=60.0,
                   help="Seconds to wait for /health before giving up")
    p.add_argument("--run-timeout",
                   type=float,
                   default=None,
                   help="Seconds one client run may take before it is killed "
                   "and its cell failed; 0 waits forever (default: 1800 for "
                   "--workload random; for --workload agentx, the measured "
                   f"phase plus {AGENTX_TIMEOUT_SLACK_S})")
    p.add_argument("--bench-client",
                   choices=BENCH_CLIENTS,
                   default="vllm",
                   help="Benchmark client CLI to invoke: vllm (`vllm bench "
                   "serve`) or sglang (`python3 -m sglang.bench_serving`)")
    p.add_argument("--vllm-cmd",
                   default=None,
                   help="Command that runs the vllm CLI (default: this "
                   f"script with {AS_VLLM_CLI} when this interpreter has "
                   "vllm, else vllm on PATH, else python -m "
                   "vllm.entrypoints.cli.main)")
    p.add_argument("--sglang-cmd",
                   default=None,
                   help="Command that runs sglang.bench_serving (default: "
                   "python -m sglang.bench_serving)")
    p.add_argument("--extra-bench-args",
                   default="",
                   help="Extra flags appended to every benchmark call")
    p.add_argument("--max-concurrency-limit",
                   type=int,
                   default=None,
                   help="Static filter: skip concurrency points above this "
                   "limit (e.g. max_num_seqs * dp)")
    p.add_argument("--avoid",
                   default="",
                   help="Static filter: comma-separated isl:osl:c rules to "
                   "skip (* matches any, e.g. 8192:*:64)")
    p.add_argument("--stop-below-gain-pct",
                   type=float,
                   default=None,
                   help="Dynamic early-stop: stop sweeping higher "
                   "concurrencies for an ISL:OSL shape after two consecutive "
                   "steps where output_tokens_per_s improves by less than "
                   "this percentage over the best throughput so far "
                   "(e.g. 10.0)")
    p.add_argument("--stop-on-failure",
                   action="store_true",
                   help="Dynamic early-stop: stop sweeping higher "
                   "concurrencies for an ISL:OSL shape after a failed cell")
    p.add_argument("--reference-curve",
                   type=Path,
                   default=None,
                   help="Optional reference summary.json or curve JSON to "
                   "verify Pareto curve reproduction against")
    p.add_argument("--tolerance-pct",
                   type=float,
                   default=5.0,
                   help="Allowed percentage drop below a reference Pareto "
                   "point when verifying reproduction (default: %(default)s)")
    p.add_argument("--export-bkc",
                   type=Path,
                   default=None,
                   help="Where to write the Best Known Config (bkc.json) "
                   "manifest when Pareto reproduction is consistent")
    p.add_argument("--dry-run",
                   action="store_true",
                   help="Print the commands without running them")
    args = p.parse_args(argv)
    args.base_url = args.base_url.rstrip("/").removesuffix("/v1")
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None
    if args.replicates < 1:
        p.error("--replicates must be at least 1")
    if (args.max_concurrency_limit is not None
            and args.max_concurrency_limit < 1):
        p.error("--max-concurrency-limit must be at least 1")
    agentx = args.workload == "agentx"
    if agentx:
        if args.engine != "vllm":
            p.error("--workload agentx requires --engine vllm. AgentX "
                    "sends its requests to an OpenAI-compatible chat "
                    "endpoint, which only the vllm engine has.")
        shaped = [
            flag for flag, value in (
                ("--isl-osl", args.isl_osl),
                ("--avoid", args.avoid),
                ("--stop-below-gain-pct", args.stop_below_gain_pct),
                ("--stop-on-failure", args.stop_on_failure),
                ("--max-concurrency-limit", args.max_concurrency_limit),
            ) if value
        ]
        if shaped:
            p.error(f"{', '.join(shaped)}: these flags apply only to "
                    "--workload random and cannot be used with --workload "
                    "agentx.")
        if args.agentx_duration < 1:
            p.error("--agentx-duration must be at least 1")
        if (args.agentx_coverage_ratio is not None
                and not 0 < args.agentx_coverage_ratio <= 1):
            p.error("--agentx-coverage-ratio must be in (0, 1]")
    if args.agentx_cache_dir is None:
        args.agentx_cache_dir = args.output_dir / "agentx_cache"
    if args.max_failed_pct is None:
        args.max_failed_pct = 10.0 if agentx else 0.0
    if args.run_timeout is None:
        duration = 1200 if args.agentx_fast else args.agentx_duration
        args.run_timeout = (duration +
                            AGENTX_TIMEOUT_SLACK_S if agentx else 1800.0)
    try:
        args.cells = parse_cells(args.isl_osl or "1024:128")
        args.concurrencies = parse_ints(args.concurrency)
        args.avoid_rules = parse_avoid_rules(args.avoid)
    except ValueError as e:
        p.error(str(e))
    if args.engine == "custom_fastapi":
        if not args.model:
            p.error("--engine custom_fastapi needs --model: the server has "
                    "no /v1/models to read it from")
        if args.task == "vqa" and not args.image_url:
            p.error("--task vqa needs --image-url")
        if args.stop_below_gain_pct is not None and (args.task != "generate"
                                                     or not args.tokenizer):
            p.error("--stop-below-gain-pct needs output_tokens_per_s, which "
                    "--engine custom_fastapi measures only for --task "
                    "generate with --tokenizer")
    # The AgentX replay runs aiperf, never the vllm CLI.
    args.vllm_cmd = [] if agentx else resolve_vllm_cmd(args.vllm_cmd)
    args.sglang_cmd = [] if agentx else resolve_sglang_cmd(args.sglang_cmd)
    return args


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv[:1] == [AS_VLLM_CLI]:
        return run_as_vllm_cli(argv[1:])
    args = parse_args(argv)
    if args.export_bkc is not None and args.reference_curve is None:
        print(
            json.dumps({
                "status": "error",
                "error": "--export-bkc requires --reference-curve"
            }))
        return EXIT_SETUP
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fastapi = args.engine == "custom_fastapi"
    try:
        if (args.dry_run and args.model) or fastapi:
            model = {
                "id": args.model,
                "root": args.tokenizer or args.model,
                "max_model_len": None
            }
            if not args.dry_run:
                wait_ready(args.base_url, args.ready_timeout)
        else:
            wait_ready(args.base_url, args.ready_timeout)
            model = discover_model(args.base_url)
        # The FastAPI recipe keeps no prefix cache, so there is none to reset.
        reset = (not args.dry_run and not fastapi
                 and reset_prefix_cache(args.base_url) is not None)
    except (TimeoutError, RuntimeError, urllib.error.URLError, OSError,
            KeyError, ValueError) as e:
        print(json.dumps({"status": "error", "error": str(e)}))
        return EXIT_SETUP
    if args.model:
        model["id"] = args.model
    # A gs:// checkpoint folder -- named by --tokenizer, or the path the
    # server loaded -- is copied here, tokenizer and config files only, so
    # vllm bench serve and the FastAPI client read a local directory.
    source = args.tokenizer if fastapi else (args.tokenizer or model["root"])
    if not args.dry_run and str(source or "").startswith(GCS_SCHEME):
        try:
            args.tokenizer = local_tokenizer(source)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            print(
                json.dumps({
                    "status":
                    "error",
                    "error":
                    f"could not copy the tokenizer from {source}: {e}"
                }))
            return EXIT_SETUP
    # The client reaches a remote server through a port forward, so the path
    # the server loaded is on the server's filesystem, not necessarily here.
    if (not args.tokenizer and not args.dry_run and not fastapi
            and model["root"].startswith("/")
            and not Path(model["root"]).exists()):
        print(
            json.dumps({
                "status":
                "error",
                "error":
                f"the server loaded {model['root']}, which is not on this "
                "host; pass --tokenizer with the model's Hugging Face id, a "
                "local copy, or the gs:// folder it was loaded from"
            }))
        return EXIT_SETUP

    # No run may hit prompts an earlier run of the sweep left in the server's
    # prefix cache. The cache is reset before every run where the server
    # allows it; otherwise every run gets its own seed. Either way the
    # baseline and every arm send the identical sequence of prompts.
    # An AgentX run has no seed that the client could change. Without a reset,
    # a run can use the cache entries that the previous run left behind.
    agentx = args.workload == "agentx"
    if not reset and not args.dry_run and agentx:
        print(
            "[serving_benchmark] The server has no /reset_prefix_cache "
            "endpoint, so the client cannot empty its prefix cache between "
            "AgentX runs. Each run can reuse cache entries from the run "
            "before it. To enable the endpoint, start the server with "
            "VLLM_SERVER_DEV_MODE=1.",
            file=sys.stderr,
            flush=True)
    elif not reset and not args.dry_run and not fastapi:
        print(
            "[serving_benchmark] the server has no /reset_prefix_cache or "
            "/flush_cache (start vLLM with VLLM_SERVER_DEV_MODE=1); giving "
            "each run its own seed instead",
            file=sys.stderr,
            flush=True)
    run = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_url": args.base_url,
        "model": model,
        "client": client_name(args),
        "settings": {
            "engine":
            args.engine,
            "bench_client":
            args.bench_client,
            "task":
            args.task if fastapi else None,
            "num_prompts":
            args.num_prompts,
            "endpoint":
            None if fastapi else args.endpoint,
            "replicates":
            args.replicates,
            "range_ratio":
            args.range_ratio,
            "request_rate":
            args.request_rate,
            "temperature":
            args.temperature,
            "ignore_eos":
            not args.no_ignore_eos,
            "seed":
            args.seed,
            "max_concurrency_limit":
            args.max_concurrency_limit,
            "avoid":
            args.avoid or None,
            "stop_below_gain_pct":
            args.stop_below_gain_pct,
            "stop_on_failure":
            args.stop_on_failure,
            "sweep_mode":
            sweep_mode(args),
            "prefix_cache": ("none: the engine keeps no prefix cache"
                             if fastapi else "reset per run" if reset else
                             "not reset" if agentx else "seed per run"),
            "workload":
            args.workload,
        },
    }
    if agentx:
        return run_agentx(args, model, raw_dir, reset, run)
    sweep_plan = [(isl, osl, sorted(set(args.concurrencies)))
                  for isl, osl in args.cells]
    cells: list[dict[str, Any]] = []
    server_down = None
    shapes_ok = True
    for shape_idx, (isl, osl, c_range) in enumerate(sweep_plan):
        kept_c_range, shape_cells = ((
            [],
            [],
        ) if server_down else filter_c_range(
            isl,
            osl,
            c_range,
            max_model_len=model.get("max_model_len"),
            range_ratio=args.range_ratio,
            max_concurrency_limit=args.max_concurrency_limit,
            avoid_rules=args.avoid_rules,
        ))
        ok_cells: list[dict[str, Any]] = []
        stop_error: str | None = None
        for conc in (c_range if server_down else kept_c_range):
            if server_down or stop_error:
                shape_cells.append(
                    skipped_cell(isl, osl, conc,
                                 f"not run: {server_down or stop_error}"))
                continue
            grid_idx = shape_idx * len(c_range) + c_range.index(conc)
            cell = run_cell(
                args,
                model,
                isl,
                osl,
                conc,
                raw_dir,
                args.seed + grid_idx * args.replicates,
                reset,
            )
            shape_cells.append(cell)
            # After a failure, tell a dead server from a bad cell, so the rest
            # of the sweep does not wait out a timeout per cell against nothing.
            if (cell["status"] == "failed" and not args.dry_run
                    and not server_alive(args.base_url)):
                server_down = (f"the server stopped answering {args.base_url}"
                               f"/health during {cell['cell']}")
            else:
                if not args.dry_run:
                    stop_error = check_early_stop(
                        ok_cells,
                        cell,
                        stop_below_gain_pct=args.stop_below_gain_pct,
                        stop_on_failure=args.stop_on_failure,
                    )
                if cell["status"] == "ok":
                    ok_cells.append(cell)
            if not args.dry_run:
                # Kept as each cell ends, so a sweep cut short loses no cell
                # it finished. The final summary, below, replaces it.
                write_summary(
                    args.output_dir, {
                        "status":
                        "partial",
                        "error":
                        server_down,
                        **run, "cells":
                        cells +
                        sorted(shape_cells, key=lambda c: c["concurrency"])
                    })
        if not shape_ok(shape_cells, args.stop_on_failure):
            shapes_ok = False
        shape_cells.sort(key=lambda c: c["concurrency"])
        cells.extend(shape_cells)
    if args.dry_run:
        return EXIT_OK
    return finish(args, run, cells, server_down, shapes_ok)


if __name__ == "__main__":
    sys.exit(main())

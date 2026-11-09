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
"""Runs the serving detectors on XProf Events DB parquets of a vLLM server.

    python3 analyze.py --deployments <inference_work>/deployments.json \\
        --launch-recipe <inference_work>/launch_vllm.sh \\
        --points 1024:1024:128,8192:1024:128 \\
        --hf-config <model>/config.json \\
        --events <events>/prefill.parquet,role=prefill,point=8192:1024:128 \\
        --events <events>/decode.parquet,role=decode,point=1024:1024:128 \\
        --out-dir <inference_work>/optimization-heuristics

Each `--events` is one Events DB parquet, as tpu-diagnostics exports it from
an XProf trace, with the profile's role (prefill, decode or mixed) and the
benchmark point it was captured at. The server's launch (flags, env, TP and
DP) comes from the deployments.json and launch_vllm.sh tpu-model-onboarding
served it from, or from `--serve-command`.

Writes `findings.json` (one entry per profile, each the analyzer's summary:
findings with their evidence, knob changes and status tier, phase step
stats and notes) and `report.md`. Nothing here is a model's judgement: every
finding is a fixed detector rule over the trace.
"""

import argparse
import json
import os
import shlex
import sys
from typing import Any

# serving_analysis sits beside this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# pylint: disable=g-import-not-at-top,wrong-import-position
import deployment  # noqa: E402
from serving_analysis import analyze as analyze_lib
from serving_analysis import context as context_lib
from serving_analysis import trace_table

_ROLES = ("prefill", "decode", "mixed", "auto")


def parse_point(text: str) -> tuple[int, int, int]:
    """`isl:osl:c` as three positive ints."""
    isl, osl, conc = (int(v) for v in text.split(":"))
    if min(isl, osl, conc) < 1:
        raise ValueError(f"benchmark point {text!r} must be positive")
    return isl, osl, conc


def parse_events(text: str) -> dict[str, Any]:
    """`PATH[,role=R][,point=ISL:OSL:C][,label=L]` as a dict."""
    path, *opts = text.split(",")
    out: dict[str, Any] = {"parquet": path, "role": "auto", "point": None}
    for opt in opts:
        key, _, value = opt.partition("=")
        if key == "role":
            if value not in _ROLES:
                raise ValueError(f"role {value!r} is not one of {_ROLES}")
            out["role"] = value
        elif key == "point":
            out["point"] = parse_point(value)
        elif key == "label":
            out["label"] = value
        else:
            raise ValueError(f"unknown --events option {key!r}")
    out.setdefault(
        "label", out["role"] if out["role"] != "auto" else os.path.splitext(
            os.path.basename(path))[0])
    return out


def serve_command(bench: dict[str, Any]) -> str:
    """The `vllm serve` launch a `deployment.baseline()` describes."""
    env = " ".join(f"{k}={shlex.quote(str(v))}"
                   for k, v in sorted((bench.get("env") or {}).items()))
    argv = ["vllm", "serve", bench.get("model_path") or "/model"]
    for flag, key in (("--tensor-parallel-size",
                       "tp"), ("--data-parallel-size", "dp"),
                      ("--max-num-seqs", "max_num_seqs"), ("--max-model-len",
                                                           "max_model_len")):
        if bench.get(key) is not None:
            argv += [flag, str(bench[key])]
    argv += [str(a) for a in bench.get("vllm_args") or []]
    cmd = " ".join(shlex.quote(a) for a in argv)
    return f"{env} {cmd}" if env else cmd


def parse_points(text: str | None) -> list[tuple[int, int, int]]:
    """ISL:OSL:C,... as a list of points."""
    return [parse_point(p) for p in (text or "").split(",") if p.strip()]


def build_context(
    command: str,
    hf_config: str | None,
    point: tuple[int, int, int] | None,
    server_log: str | None,
    ttft_sensitive: bool,
    points: list[tuple[int, int, int]] | None = None
) -> context_lib.ServingContext:
    model = (context_lib.ModelConfig.from_json_file(hf_config)
             if hf_config else context_lib.ModelConfig())
    log = None
    if server_log:
        with open(server_log, errors="replace", encoding="utf-8") as f:
            log = f.read()
    isl, osl, conc = point or (None, None, None)
    return context_lib.ServingContext(
        framework="vllm_torchtpu",
        model=model,
        serving_config=context_lib.parse_serving_command(command),
        serving_command=command,
        input_len=isl,
        output_len=osl,
        concurrency=conc,
        benchmark_concurrencies=tuple(sorted({c
                                              for _, _, c in points or ()})),
        server_log=log,
        ttft_sensitive=ttft_sensitive,
    )


def run(events: list[dict[str, Any]],
        command: str,
        hf_config: str | None,
        default_point: tuple[int, int, int] | None,
        server_log: str | None,
        ttft_sensitive: bool,
        points: list[tuple[int, int, int]] | None = None) -> dict[str, Any]:
    """Analyzes each profile; returns the findings.json payload."""
    profiles = []
    for ev in events:
        point = ev["point"] or default_point
        ctx = build_context(command, hf_config, point, server_log,
                            ttft_sensitive, points)
        table = trace_table.TraceTable.from_parquet(ev["parquet"],
                                                    session_id=ev["label"])
        result = analyze_lib.analyze(table, ctx, session_id=ev["label"])
        profiles.append({
            "label": ev["label"],
            "role": ev["role"],
            "parquet": ev["parquet"],
            "point": list(point) if point else None,
            "result": result.summary_dict(),
            "markdown": result.to_markdown(),
        })
    return {"serve_command": command, "profiles": profiles}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events",
                        action="append",
                        required=True,
                        help="PATH[,role=prefill|decode|mixed][,point="
                        "ISL:OSL:C][,label=NAME]; repeat per profile")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--deployments",
                     help="The deployments.json the server was launched from")
    src.add_argument("--serve-command", help="The server's launch command")
    parser.add_argument("--launch-recipe",
                        help="The launch_vllm.sh the server ran (with"
                        " --deployments)")
    parser.add_argument("--points",
                        help="ISL:OSL:C,... the benchmark measured")
    parser.add_argument("--hf-config", help="The model's HF config.json")
    parser.add_argument("--point",
                        help="ISL:OSL:C for profiles that name no point")
    parser.add_argument("--server-log", help="The vLLM server's startup log")
    parser.add_argument("--ttft-insensitive",
                        action="store_true",
                        help="Time to first token does not matter here")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    try:
        events = [parse_events(e) for e in args.events]
        default_point = parse_point(args.point) if args.point else None
    except ValueError as e:
        parser.error(str(e))
    points = []
    if args.deployments:
        recipe = None
        if args.launch_recipe:
            with open(args.launch_recipe, encoding="utf-8") as f:
                recipe = f.read()
        try:
            points = parse_points(args.points)
        except ValueError as e:
            parser.error(str(e))
        bench = deployment.baseline(deployment.load(args.deployments), recipe)
        command = serve_command(bench)
        if default_point is None and points:
            default_point = max(points, key=lambda p: (p[2], p[0]))
    else:
        command = args.serve_command
    payload = run(events, command, args.hf_config, default_point,
                  args.server_log, not args.ttft_insensitive, points)
    os.makedirs(args.out_dir, exist_ok=True)
    report = [p.pop("markdown") for p in payload["profiles"]]
    with open(os.path.join(args.out_dir, "findings.json"),
              "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    with open(os.path.join(args.out_dir, "report.md"), "w",
              encoding="utf-8") as f:
        for p, md in zip(payload["profiles"], report):
            f.write(f"<!-- profile {p['label']} ({p['role']}) -->\n{md}\n\n")
    for p in payload["profiles"]:
        ids = [x.get("detector_id") for x in p["result"].get("findings", [])]
        print(f"{p['label']}: {', '.join(ids) or 'no findings'}")
    print(f"Wrote {args.out_dir}/findings.json and report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

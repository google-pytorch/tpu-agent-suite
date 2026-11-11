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
"""Argparse CLI entrypoint for the serving hill-climbing analytics engine.

Subcommands are grouped into two lifecycle phases backed by shared stateless
readers (`ledger.py`, `run_history.py`, `lessons.py`, `metrics.py`):

  arm_preflight (gap digging & pre-launch gates before spending TPU slice hours):
    preflight         Pre-launch configuration, P/D DCP, kernel, VMEM & leg gates
    legs              1P1D prefill-vs-decode bottleneck attribution from `/metrics`
    kv-replay         Prefix-cache HBM + CPU tier replay & Little's Law lanes
    roofline          Trace-to-roofline HBM/ICI/MXU attainment & target ranker
    code-limits       Static scanner for crossed kernel row/bucket limits
    code-gap          Upstream vs served env-registry switch & gate-chain analyzer
    ports             Rank local/upstream git PR refs by active switches & merge
    microbench        Emit single-host TPU pytest harness (`--repo`)
    predict           Closed queueing-network throughput/interactivity predictor

  arm_verdict (in-flight & post-run advisory readings):
    early-verdict     Equal-time `aiperf` realtime telemetry early stopping
    verify            Mechanism + noise-aware KEEP/DISCARD read (advisory)
    noise             Per-concurrency replicate noise vs pooled-sweep variance
    diagnose-failure  Classify failed container logs (`vmem_oom` vs `hbm_oom`, etc.)
    microbench        Grade 1-host kernel `RESULT` log (`--bench-log`)
"""

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
from collections.abc import Mapping, Sequence

# Ensure `hillclimb` is importable whether invoked as `python3 hillclimb_cli.py`
# or `python3 -m hillclimb.hillclimb_cli`.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# pylint: disable=g-import-not-at-top,wrong-import-position
from hillclimb import code_gap  # noqa: E402
from hillclimb import code_limits  # noqa: E402
from hillclimb import early_verdict  # noqa: E402
from hillclimb import kv_replay  # noqa: E402
from hillclimb import ledger  # noqa: E402
from hillclimb import legs  # noqa: E402
from hillclimb import lessons  # noqa: E402
from hillclimb import microbench  # noqa: E402
from hillclimb import model_plugins  # noqa: E402
from hillclimb import ports  # noqa: E402
from hillclimb import predictor  # noqa: E402
from hillclimb import preflight  # noqa: E402
from hillclimb import run_history  # noqa: E402
from hillclimb import trace_roofline  # noqa: E402
from hillclimb import verify  # noqa: E402


class CliError(Exception):
    """Bad or missing input; `main` prints it and exits 2 without a traceback."""


def _read_ledgers(
        paths: Sequence[str]) -> tuple[list[ledger.Entry], list[str]]:
    """Reads close-gaps outcome lines from `paths`; returns entries and misses."""
    entries: list[ledger.Entry] = []
    missing: list[str] = []
    for path in paths:
        if not ledger.state_files(path):
            missing.append(path)
            continue
        entries.extend(ledger.read(path))
    return entries, missing


def _parse_kv_pairs(items: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _load_knobs_input(
    knob_items: Sequence[str],
    config_file: str = "",
) -> tuple[str, dict[str, str]]:
    recipe = "1p1d_long_context"
    merged: dict[str, str] = {}
    if config_file:
        raw = json.loads(pathlib.Path(config_file).read_text(encoding="utf-8"))
        if isinstance(raw, Mapping):
            if "recipe" in raw:
                recipe = str(raw["recipe"])
            inner = raw.get("knobs", raw)
            if isinstance(inner, Mapping):
                for k, v in inner.items():
                    if k != "recipe" and not isinstance(v, (Mapping, list)):
                        merged[str(k)] = str(v)
    merged.update(_parse_kv_pairs(knob_items))
    return recipe, merged


def _cmd_preflight(args: argparse.Namespace) -> int:
    recipe, arm_knobs = _load_knobs_input(args.knob, args.config)
    if args.recipe:
        recipe = args.recipe
    _, defaults = _load_knobs_input(args.default, args.defaults_file)
    history: list[run_history.RunRecord] = []
    if args.history:
        history = run_history.load_jsonl(
            pathlib.Path(
                args.history).read_text(encoding="utf-8").splitlines())
    code_flags = None
    if args.envs_file:
        code_flags = code_gap.parse_env_registry(
            pathlib.Path(args.envs_file).read_text(encoding="utf-8"))
    elif args.repo:
        envs_candidate = pathlib.Path(args.repo) / args.envs_path
        if envs_candidate.is_file():
            code_flags = code_gap.parse_env_registry(
                envs_candidate.read_text(encoding="utf-8"))

    leg_verdict = None
    if args.server_metrics:
        p_loads, d_loads = legs.parse_server_metrics(
            pathlib.Path(args.server_metrics).read_text(encoding="utf-8"))
        if p_loads:
            leg_verdict = legs.assess(p_loads, d_loads)

    knob_map, plugin_warning = _knob_map(args.plugin)
    entries, missing = _read_ledgers(args.ledger)
    findings = preflight.check_launch(
        recipe,
        arm_knobs,
        history,
        code_flags,
        defaults,
        repo_dir=args.repo or None,
        ledger_entries=entries,
        knob_map=knob_map,
        leg_verdict=leg_verdict,
    )
    findings.extend(
        preflight.Finding(
            preflight.INFO,
            "prior_outcome",
            f"No close-gaps ledger at {path}; prior outcomes not checked.",
        ) for path in missing)
    if plugin_warning:
        findings.append(
            preflight.Finding(preflight.WARNING, "plugin", plugin_warning))
    has_error = any(f.severity == preflight.ERROR for f in findings)
    if args.json:
        payload = {
            "passed": not has_error,
            "has_errors": has_error,
            "recipe": recipe,
            "findings": [f.to_dict() for f in findings],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(preflight.render_findings(findings))
    return 1 if has_error and not args.no_fail_on_error else 0


def _cmd_early_verdict(args: argparse.Namespace) -> int:
    arm_text = pathlib.Path(args.arm_log).read_text(encoding="utf-8",
                                                    errors="replace")
    base_text = pathlib.Path(args.baseline_log).read_text(encoding="utf-8",
                                                          errors="replace")
    verdict = early_verdict.assess(
        early_verdict.parse_blocks(arm_text),
        args.arm_chips,
        early_verdict.parse_blocks(base_text),
        args.baseline_chips,
        same_shape=not args.cross_shape,
        target_ratio=args.target_ratio,
    )
    if args.json:
        sys.stdout.write(
            json.dumps(dataclasses.asdict(verdict), indent=2) + "\n")
    else:
        sys.stdout.write(
            early_verdict.render(verdict, args.arm_name, args.baseline_name))
    return 0


def _cmd_legs(args: argparse.Namespace) -> int:
    text = pathlib.Path(args.server_metrics).read_text(encoding="utf-8")
    p_loads, d_loads = legs.parse_server_metrics(text)
    verdict = legs.assess(p_loads, d_loads)
    _, knobs = _load_knobs_input(args.knob, args.config)
    _, defaults = _load_knobs_input(args.default, args.defaults_file)
    off_leg = legs.warn_off_leg_arm(verdict,
                                    preflight.changed_knobs(knobs, defaults))
    if args.json:
        payload = verdict.to_dict()
        payload["off_leg_warning"] = off_leg
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        out = legs.render(verdict)
        if off_leg:
            out += f"\nWARNING off_leg: {off_leg}\n"
        sys.stdout.write(out)
    return 0


def _cmd_kv_replay(args: argparse.Namespace) -> int:
    policy_totals: dict[str, dict[str, float | int]] = {}
    if args.requests:
        reqs = kv_replay.load_requests_jsonl(args.requests)
        scale = args.dcp_size / 16.0
        gpu_blocks = (args.gpu_blocks
                      if args.gpu_blocks is not None else int(5806 * scale))
        cpu_blocks = (args.cpu_blocks
                      if args.cpu_blocks is not None else int(3816 * scale))
        base = kv_replay.CacheConfig(
            gpu_blocks=gpu_blocks,
            cpu_blocks=cpu_blocks,
            tier_pop=not args.keep_eagle_tail,
        )
        policies = {
            "as_served": base,
            "keep_eagle_tail": dataclasses.replace(base, tier_pop=False),
            "no_cpu_tier": dataclasses.replace(base, cpu_blocks=0),
            "infinite_cache": dataclasses.replace(base, infinite=True),
        }
        for name, cfg in policies.items():
            outcomes = kv_replay.replay(reqs, cfg)
            policy_totals[name] = kv_replay.totals(reqs, outcomes)

    lanes = None
    if args.throughput_req_s > 0:
        lanes = kv_replay.lane_saturation(
            client_concurrency=args.concurrency,
            decode_batch_cap=args.decode_cap,
            throughput_req_s=args.throughput_req_s,
            mean_prefill_s=args.mean_prefill_s,
            mean_decode_s=args.mean_decode_s,
            mean_admit_wait_s=args.mean_admit_wait_s,
            mean_kv_pull_s=args.mean_kv_pull_s,
        )

    if args.json:
        sys.stdout.write(
            json.dumps(
                {
                    "policies": policy_totals,
                    "lanes": lanes.to_dict() if lanes else None,
                },
                indent=2,
            ) + "\n")
    else:
        sys.stdout.write(kv_replay.render_replay_markdown(
            policy_totals, lanes))
    return 0


def _knob_map(plugin_name: str) -> tuple[preflight.KnobMap, str]:
    """Resolves `--plugin` (name, alias or JSON path) to its knob map.

    Returns the map and, when none is registered for the plugin, a warning: the
    checks then fall back to the GLM-5 P/D knob names and limits, which another
    model's recipe may not use.
    """
    kmap = preflight.KNOB_MAPS.get(plugin_name)
    if kmap is None:
        try:
            kmap = preflight.KNOB_MAPS.get(
                model_plugins.get_plugin(plugin_name).name)
        except (KeyError, ValueError, OSError):
            kmap = None
    if kmap is not None:
        return kmap, ""
    fallback = preflight.GLM5_PD_KNOB_MAP
    known = ", ".join(sorted(preflight.KNOB_MAPS))
    return fallback, (
        f"No knob map is registered for --plugin {plugin_name!r}; "
        f"checks use {fallback.name}'s knob names and limits. "
        f"Registered: {known}.")


def _resolve_steps(
    step_args: Sequence[int],
    knobs: Mapping[str, str],
    plugin_name: str,
) -> list[int]:
    if step_args:
        return [int(s) for s in step_args]
    kmap, plugin_warning = _knob_map(plugin_name)
    if plugin_warning:
        sys.stderr.write(f"warning: {plugin_warning}\n")
    shape = preflight.decode_shape(knobs, kmap)
    if shape is not None and shape.cap > 0:
        return [shape.step_tokens]
    return []


def _cmd_code_limits(args: argparse.Namespace) -> int:
    _, knobs = _load_knobs_input(args.knob, args.config)
    steps = _resolve_steps(args.steps, knobs, args.plugin)
    if not steps:
        raise ValueError(
            "Provide --steps or DECODE_BATCH_CAP via --knob/--config.")
    env = code_gap.recipe_env("", knobs)
    limits = code_limits.scan(args.repo)
    if args.base_steps:
        found = code_limits.new_crossings(limits, steps, args.base_steps, env)
    else:
        found = code_limits.crossings(limits, steps, env)
    if args.json:
        sys.stdout.write(
            json.dumps([c.to_dict() for c in found], indent=2) + "\n")
    else:
        sys.stdout.write(code_limits.render(found, args.repo))
    return 0


def _read_envs_file(path: str) -> dict[str, code_gap.EnvFlag]:
    target = pathlib.Path(path)
    if not target.is_file():
        raise CliError(f"envs file {target} not found.")
    return code_gap.parse_env_registry(target.read_text(encoding="utf-8"))


def _read_env_registry(repo: str, ref: str,
                       envs_path: str) -> dict[str, code_gap.EnvFlag]:
    if not ref:
        path = pathlib.Path(repo)
        target = path if path.is_file() else (path / envs_path)
        if not target.is_file():
            raise CliError(
                f"{target} not found: pass --repo <vllm-torchtpu checkout> "
                "(or --served-envs-file and --upstream-envs-file).")
        return code_gap.parse_env_registry(target.read_text(encoding="utf-8"))
    proc = subprocess.run(
        ["git", "-C", repo, "show", f"{ref}:{envs_path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise CliError(f"git show {ref}:{envs_path} failed in {repo}: "
                       f"{proc.stderr.strip()}")
    return code_gap.parse_env_registry(proc.stdout)


def _cmd_code_gap(args: argparse.Namespace) -> int:
    _, knobs = _load_knobs_input(args.knob, args.config)
    steps = _resolve_steps(args.steps, knobs, args.plugin)
    arm_env = code_gap.recipe_env("", knobs)
    if args.served_envs_file:
        served = _read_envs_file(args.served_envs_file)
    else:
        served = _read_env_registry(args.repo, args.served_ref, args.envs_path)
    if args.upstream_envs_file:
        upstream = _read_envs_file(args.upstream_envs_file)
    else:
        upstream = _read_env_registry(args.repo, args.upstream_ref
                                      or args.served_ref, args.envs_path)
    entries, missing = _read_ledgers(args.ledger)
    for path in missing:
        sys.stderr.write(
            f"No close-gaps ledger at {path}; prior outcomes not checked.\n")
    priors = lessons.flag_verdicts(entries)
    ep_flag = True if args.ep else (False if args.tp else None)
    gaps = code_gap.find_gaps(
        served,
        upstream,
        arm_env,
        priors=priors,
        steps=steps,
        ep_enabled=ep_flag,
    )
    if args.json:
        sys.stdout.write(
            json.dumps([dataclasses.asdict(g) for g in gaps], indent=2) + "\n")
    else:
        sys.stdout.write(
            code_gap.render_markdown(
                gaps,
                args.served_ref or "served",
                args.upstream_ref or "upstream",
            ))
    return 0


def _cmd_ports(args: argparse.Namespace) -> int:
    _, knobs = _load_knobs_input(args.knob, args.config)
    steps = _resolve_steps(args.steps, knobs, args.plugin)
    arm_env = code_gap.recipe_env("", knobs)
    ranked = ports.scan_local_refs(
        args.repo,
        args.served_ref,
        args.refs,
        envs_path=args.envs_path,
        arm_env=arm_env,
        steps=steps,
    )
    if args.json:
        sys.stdout.write(
            json.dumps([c.to_dict() for c in ranked], indent=2) + "\n")
    else:
        sys.stdout.write(ports.render(ranked, args.served_ref, steps))
    return 0


def _cmd_microbench(args: argparse.Namespace) -> int:
    if args.bench_log:
        stdout = pathlib.Path(args.bench_log).read_text(encoding="utf-8")
        verdicts = [
            microbench.evaluate_result(p, steps=args.steps)
            for p in microbench.parse_results(stdout)
        ]
        failed = any(not v.passed for v in verdicts)
        if args.json:
            body = (verdicts[0].to_dict()
                    if len(verdicts) == 1 else [v.to_dict() for v in verdicts])
            sys.stdout.write(json.dumps(body, indent=2) + "\n")
        else:
            sys.stdout.write("\n".join(
                microbench.render_verdict(v) for v in verdicts))
        return 1 if failed else 0

    if not args.repo:
        raise ValueError(
            "Provide --bench-log to evaluate or --repo to generate.")
    _, knobs = _load_knobs_input(args.knob, args.config)
    steps = _resolve_steps(args.steps, knobs, args.plugin)
    if not steps:
        raise ValueError(
            "Provide --steps or DECODE_BATCH_CAP via --knob/--config.")
    env = code_gap.recipe_env("", knobs)
    limits = code_limits.scan(args.repo)
    found = (code_limits.new_crossings(limits, steps, args.base_steps, env)
             if args.base_steps else code_limits.crossings(limits, steps, env))
    if not found:
        raise ValueError(f"No crossed size limits found at steps {steps}.")
    if args.limit_name:
        matches = [c for c in found if c.limit.name == args.limit_name]
        if not matches:
            raise ValueError(
                f"Limit {args.limit_name!r} not among crossed limits.")
        chosen = matches[0]
    else:
        non_env = [c for c in found if not c.limit.overridable]
        chosen = non_env[0] if non_env else found[0]
    spec = microbench.spec_from_crossing(chosen,
                                         base_steps=args.base_steps,
                                         env=env)
    script = microbench.render_pytest(spec)
    if args.out:
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(script, encoding="utf-8")
    sys.stdout.write(script)
    return 0


def _cmd_roofline(args: argparse.Namespace) -> int:
    plugin = model_plugins.get_plugin(args.plugin)
    if args.vpu_unpack_gweights_per_ms is not None:
        if args.vpu_unpack_gweights_per_ms <= 0:
            raise CliError("--vpu-unpack-gweights-per-ms must be positive.")
        plugin = dataclasses.replace(plugin,
                                     hardware=dataclasses.replace(
                                         plugin.hardware,
                                         vpu_unpack_gweights_per_ms_per_tc=(
                                             args.vpu_unpack_gweights_per_ms)))
    report = trace_roofline.analyze_trace_file(
        args.trace,
        plugin,
        section=args.section,
        batch_reqs=args.batch_reqs,
        padded_rows=args.padded_rows,
        context_tokens=args.context_tokens,
        weight_dtype=args.weight_dtype,
        row_compacted=args.row_compacted,
        moe_mode=args.moe_mode,
    )
    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(report.render_markdown())
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if args.chip_hour_usd is not None and args.chip_hour_usd <= 0:
        raise CliError("--chip-hour-usd must be positive.")
    pd_params = None
    sh_params = None
    if args.history:
        calibrated = predictor.calibrate_from_jsonl(args.history, args.plugin)
        if isinstance(calibrated, predictor.PDParams):
            pd_params = calibrated
        else:
            sh_params = calibrated
    buckets = None
    if args.token_buckets:
        moe = "ep" if args.decode_family == predictor.EP else "tp"
        buckets = predictor.decode_buckets(args.token_buckets, moe,
                                           args.decode_cap * 4)
    row_adj: dict[int, float] = {}
    if args.microbench_log:
        row_adj = predictor.parse_microbench_row_deltas_ms(
            pathlib.Path(args.microbench_log).read_text(encoding="utf-8"))
    res = predictor.predict_serving(
        predictor.PredictConfig(
            plugin_name=args.plugin,
            topology=args.topology,
            concurrency=args.concurrency,
            cap=args.decode_cap,
            family=args.decode_family,
            buckets=buckets,
            step_delta_ms=args.delta_step_ms,
            row_adjust_ms=row_adj,
            chunk_tokens=args.chunk_tokens,
            chip_hour_usd=args.chip_hour_usd,
        ),
        pd_params=pd_params,
        single_host_params=sh_params,
    )
    if args.json:
        sys.stdout.write(json.dumps(res.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(res.render_markdown())
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    before = json.loads(
        pathlib.Path(args.baseline).read_text(encoding="utf-8"))
    after = json.loads(
        pathlib.Path(args.candidate).read_text(encoding="utf-8"))
    comp = verify.compare(before, after)
    mech = None
    if args.target_detector and args.before_findings and args.after_findings:
        b_raw = json.loads(
            pathlib.Path(args.before_findings).read_text(encoding="utf-8"))
        a_raw = json.loads(
            pathlib.Path(args.after_findings).read_text(encoding="utf-8"))
        b_list = b_raw.get("findings", b_raw) if isinstance(b_raw,
                                                            Mapping) else b_raw
        a_list = a_raw.get("findings", a_raw) if isinstance(a_raw,
                                                            Mapping) else a_raw
        mech = verify.check_mechanism(args.target_detector, b_list, a_list)

    verdict = verify.decide_verdict(
        comp,
        noise_rel=args.noise_rel,
        mechanism=mech,
        eval_drop=args.eval_drop,
    )
    if args.json:
        sys.stdout.write(json.dumps(verdict.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(verify.render_verdict(verdict))
    return 0


def _cmd_noise(args: argparse.Namespace) -> int:
    lines = pathlib.Path(args.history).read_text(encoding="utf-8").splitlines()
    runs = run_history.load_jsonl(lines)
    recipe = args.recipe or (runs[0].recipe if runs else "default")
    est = run_history.noise_estimate(runs, recipe)
    if args.json:
        body = dataclasses.asdict(est) if est is not None else None
        sys.stdout.write(
            json.dumps({
                "recipe": recipe,
                "noise": body
            }, indent=2) + "\n")
    else:
        sys.stdout.write(run_history.render_noise_markdown(est, recipe))
    return 0


def _cmd_diagnose_failure(args: argparse.Namespace) -> int:
    log_text = pathlib.Path(args.log).read_text(encoding="utf-8",
                                                errors="replace")
    diag = lessons.diagnose_log(
        log_text,
        run_id=args.run_id,
        container=args.container,
    )
    if args.json:
        sys.stdout.write(json.dumps(diag.to_dict(), indent=2) + "\n")
    else:
        sys.stdout.write(lessons.render_diagnosis(diag))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the `hillclimb_cli.py` argument parser."""
    parser = argparse.ArgumentParser(
        prog="hillclimb_cli.py",
        description=
        "Deterministic gates, roofline analysis, and predictors for TPU serving hill-climbing.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. preflight
    p_pre = sub.add_parser(
        "preflight",
        help="Run pre-launch P/D, kernel, VMEM, and ledger gates.")
    p_pre.add_argument("--recipe", default="", help="Recipe or workload name.")
    p_pre.add_argument("--config",
                       default="",
                       help="JSON file of candidate knobs.")
    p_pre.add_argument("--knob",
                       action="append",
                       default=[],
                       help="KEY=VALUE knob.")
    p_pre.add_argument("--default",
                       action="append",
                       default=[],
                       help="Default KEY=VALUE.")
    p_pre.add_argument(
        "--defaults-file",
        "--defaults",
        dest="defaults_file",
        default="",
        help="JSON file of baseline/default knobs.",
    )
    p_pre.add_argument("--repo", default="", help="Served checkout directory.")
    p_pre.add_argument("--envs-file",
                       default="",
                       help="Direct path to envs.py.")
    p_pre.add_argument(
        "--envs-path",
        default="src/vllm_torchtpu/envs.py",
        help="Repo-relative path to envs.py.",
    )
    p_pre.add_argument(
        "--server-metrics",
        default="",
        help="Baseline /metrics or server_metrics_export for off-leg gating.",
    )
    p_pre.add_argument("--history", default="", help="Run history JSONL.")
    p_pre.add_argument(
        "--ledger",
        action="append",
        default=[],
        help=("close-gaps ledger.md (or its directory); reads `outcome:` "
              "lines there and in gaps/*.md. Repeatable."),
    )
    p_pre.add_argument("--plugin",
                       default=model_plugins.DEFAULT_PLUGIN_NAME,
                       help="Model plugin.")
    p_pre.add_argument(
        "--no-fail-on-error",
        action="store_true",
        help="Return exit code 0 even when ERROR findings fire.",
    )
    p_pre.add_argument("--json", action="store_true", help="Emit JSON output.")
    p_pre.set_defaults(func=_cmd_preflight)

    # 2. early-verdict
    p_ev = sub.add_parser(
        "early-verdict",
        help="Equal-time early stopping from aiperf realtime logs.",
    )
    p_ev.add_argument("--arm-log", required=True, help="Candidate aiperf log.")
    p_ev.add_argument("--baseline-log",
                      required=True,
                      help="Baseline aiperf log.")
    p_ev.add_argument("--arm-chips",
                      type=int,
                      default=16,
                      help="Candidate chip count.")
    p_ev.add_argument("--baseline-chips",
                      type=int,
                      default=16,
                      help="Baseline chip count.")
    p_ev.add_argument("--arm-name", default="candidate")
    p_ev.add_argument("--baseline-name", default="baseline")
    p_ev.add_argument("--target-ratio", type=float, default=1.0)
    p_ev.add_argument("--cross-shape", action="store_true")
    p_ev.add_argument("--json", action="store_true")
    p_ev.set_defaults(func=_cmd_early_verdict)

    # 3. legs
    p_legs = sub.add_parser(
        "legs", help="Attribute 1P1D bottleneck leg from /metrics export.")
    p_legs.add_argument("--server-metrics", required=True)
    p_legs.add_argument("--config", default="")
    p_legs.add_argument("--knob", action="append", default=[])
    p_legs.add_argument("--default",
                        action="append",
                        default=[],
                        help="Default KEY=VALUE.")
    p_legs.add_argument(
        "--defaults-file",
        "--defaults",
        dest="defaults_file",
        default="",
        help=("JSON file of baseline knobs; only knobs that differ from it "
              "count as changed."),
    )
    p_legs.add_argument("--json", action="store_true")
    p_legs.set_defaults(func=_cmd_legs)

    # 4. kv-replay
    p_kv = sub.add_parser(
        "kv-replay",
        help="Replay prefix-cache policies and Little's Law lane saturation.",
    )
    p_kv.add_argument("--requests", default="", help="Requests JSONL path.")
    p_kv.add_argument("--dcp-size", type=int, default=16)
    p_kv.add_argument("--gpu-blocks", type=int, default=None)
    p_kv.add_argument("--cpu-blocks", type=int, default=None)
    p_kv.add_argument("--keep-eagle-tail", action="store_true")
    p_kv.add_argument("--concurrency", type=int, default=48)
    p_kv.add_argument("--decode-cap", type=int, default=16)
    p_kv.add_argument("--throughput-req-s", type=float, default=0.0)
    p_kv.add_argument("--mean-prefill-s", type=float, default=0.0)
    p_kv.add_argument("--mean-decode-s", type=float, default=0.0)
    p_kv.add_argument("--mean-admit-wait-s", type=float, default=0.0)
    p_kv.add_argument("--mean-kv-pull-s", type=float, default=0.0)
    p_kv.add_argument("--json", action="store_true")
    p_kv.set_defaults(func=_cmd_kv_replay)

    # 5. code-limits
    p_lim = sub.add_parser(
        "code-limits",
        help="Scan served code for hardcoded kernel row/bucket limits.",
    )
    p_lim.add_argument("--repo",
                       required=True,
                       help="Source directory or file.")
    p_lim.add_argument("--steps", type=int, nargs="*", default=[])
    p_lim.add_argument("--base-steps", type=int, nargs="*", default=[])
    p_lim.add_argument("--config", default="")
    p_lim.add_argument("--knob", action="append", default=[])
    p_lim.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_lim.add_argument("--json", action="store_true")
    p_lim.set_defaults(func=_cmd_code_limits)

    # 6. code-gap
    p_gap = sub.add_parser(
        "code-gap",
        help=
        "Analyze unset switches, prerequisite gate chains, and MoE conflicts.",
    )
    p_gap.add_argument(
        "--repo",
        default=".",
        help=("vllm-torchtpu checkout; envs.py is read from it (or from "
              "--served-ref/--upstream-ref) unless --*-envs-file is given."),
    )
    p_gap.add_argument("--served-ref", default="")
    p_gap.add_argument("--upstream-ref", default="")
    p_gap.add_argument("--served-envs-file", default="")
    p_gap.add_argument("--upstream-envs-file", default="")
    p_gap.add_argument("--envs-path", default="src/vllm_torchtpu/envs.py")
    p_gap.add_argument("--steps", type=int, nargs="*", default=[])
    p_gap.add_argument("--config", default="")
    p_gap.add_argument("--knob", action="append", default=[])
    p_gap.add_argument(
        "--ledger",
        action="append",
        default=[],
        help="close-gaps ledger.md (or its directory). Repeatable.",
    )
    p_gap.add_argument("--ep", action="store_true")
    p_gap.add_argument("--tp", action="store_true")
    p_gap.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_gap.add_argument("--json", action="store_true")
    p_gap.set_defaults(func=_cmd_code_gap)

    # 7. ports
    p_ports = sub.add_parser(
        "ports",
        help="Rank local git PR/branch refs by active switches & merge.")
    p_ports.add_argument("--repo", required=True)
    p_ports.add_argument("--served-ref", required=True)
    p_ports.add_argument("--refs", nargs="+", required=True)
    p_ports.add_argument("--envs-path", default="src/vllm_torchtpu/envs.py")
    p_ports.add_argument("--steps", type=int, nargs="*", default=[])
    p_ports.add_argument("--config", default="")
    p_ports.add_argument("--knob", action="append", default=[])
    p_ports.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_ports.add_argument("--json", action="store_true")
    p_ports.set_defaults(func=_cmd_ports)

    # 8. microbench
    p_mb = sub.add_parser(
        "microbench",
        help="Generate single-host TPU pytest microbench or gate RESULT log.",
    )
    p_mb.add_argument("--repo", default="")
    p_mb.add_argument("--bench-log", default="")
    p_mb.add_argument("--out", default="")
    p_mb.add_argument("--limit-name", default="")
    p_mb.add_argument("--steps", type=int, nargs="*", default=[])
    p_mb.add_argument("--base-steps", type=int, nargs="*", default=[])
    p_mb.add_argument("--config", default="")
    p_mb.add_argument("--knob", action="append", default=[])
    p_mb.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_mb.add_argument("--json", action="store_true")
    p_mb.set_defaults(func=_cmd_microbench)

    # 9. roofline
    p_rf = sub.add_parser(
        "roofline",
        help="Analyze trace op profile against HBM/ICI/MXU rooflines.")
    p_rf.add_argument("--trace", required=True)
    p_rf.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_rf.add_argument("--section", default="new")
    p_rf.add_argument("--batch-reqs", type=int, default=None)
    p_rf.add_argument("--padded-rows", type=int, default=None)
    p_rf.add_argument("--context-tokens", type=int, default=None)
    p_rf.add_argument("--weight-dtype", default="fp8")
    p_rf.add_argument("--row-compacted", action="store_true")
    p_rf.add_argument("--moe-mode", default="")
    p_rf.add_argument(
        "--vpu-unpack-gweights-per-ms",
        type=float,
        default=None,
        help=("Measured VPU 4-bit unpack rate (billions of weights per ms per "
              "TensorCore). Without it, 4-bit compute floors leave out the "
              "unpack time."),
    )
    p_rf.add_argument("--json", action="store_true")
    p_rf.set_defaults(func=_cmd_roofline)

    # 10. predict
    p_pred = sub.add_parser(
        "predict",
        help=("Predict tok/s/chip (Mtok/$ with --chip-hour-usd), p90 "
              "interactivity, and binding leg."),
    )
    p_pred.add_argument("--plugin", default=model_plugins.DEFAULT_PLUGIN_NAME)
    p_pred.add_argument("--topology", default="")
    p_pred.add_argument("--concurrency", type=int, default=48)
    p_pred.add_argument("--decode-cap", type=int, default=16)
    p_pred.add_argument("--decode-family", default="")
    p_pred.add_argument("--token-buckets", type=int, nargs="*", default=[])
    p_pred.add_argument(
        "--delta-step-ms",
        type=float,
        default=0.0,
        help="Decode step latency reduction in ms (positive = faster step).",
    )
    p_pred.add_argument("--microbench-log", default="")
    p_pred.add_argument("--chunk-tokens", type=int, default=4096)
    p_pred.add_argument("--history", default="")
    p_pred.add_argument(
        "--chip-hour-usd",
        type=float,
        default=None,
        help=("Your chip-hour rate in USD. Only the Mtok/$ figure uses it; "
              "without it Mtok/$ is omitted. No price is built in."),
    )
    p_pred.add_argument("--json", action="store_true")
    p_pred.set_defaults(func=_cmd_predict)

    # 11. verify
    p_ver = sub.add_parser(
        "verify", help="Verify mechanism & compute noise-aware A/B verdict.")
    p_ver.add_argument("--baseline",
                       required=True,
                       help="Baseline metrics JSON.")
    p_ver.add_argument("--candidate",
                       required=True,
                       help="Candidate metrics JSON.")
    p_ver.add_argument("--noise-rel", type=float, default=0.01)
    p_ver.add_argument("--eval-drop", type=float, default=None)
    p_ver.add_argument(
        "--target-detector",
        action="append",
        default=[],
        help=(
            "Detector ID (`D8_kv_head_padding`), unambiguous `D<N>` (`D8`), "
            "or semantic slug (`batched_rpa_decode_untunable`). Repeatable."),
    )
    p_ver.add_argument("--before-findings", default="")
    p_ver.add_argument("--after-findings", default="")
    p_ver.add_argument("--json", action="store_true")
    p_ver.set_defaults(func=_cmd_verify)

    # 12. noise
    p_noise = sub.add_parser(
        "noise",
        help="Estimate per-concurrency replicate noise from history JSONL.",
    )
    p_noise.add_argument("--history", required=True)
    p_noise.add_argument("--recipe", default="")
    p_noise.add_argument("--json", action="store_true")
    p_noise.set_defaults(func=_cmd_noise)

    # 13. diagnose-failure
    p_diag = sub.add_parser(
        "diagnose-failure",
        help="Classify failed container log (vmem_oom vs hbm_oom, etc.).",
    )
    p_diag.add_argument("--log", required=True)
    p_diag.add_argument("--run-id", default="failed-run")
    p_diag.add_argument("--container", default="")
    p_diag.add_argument("--json", action="store_true")
    p_diag.set_defaults(func=_cmd_diagnose_failure)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as err:
        sys.stderr.write(f"hillclimb_cli.py {args.subcommand}: error: {err}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())

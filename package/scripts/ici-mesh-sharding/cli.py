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
"""Unified pure-stdlib CLI for ICI mesh sharding analysis, search, and verification.

Subcommands:
- `enumerate`: Enumerate legal pairwise power-of-two ICI mesh moves around a baseline.
- `breakdown`: Attribute XProf HLO op-profile collective time to mesh axes and roles.
- `rank-portfolio`: Build the ranked 7-proposal analytical mesh portfolio.
- `check-topology`: Check physical torus fit (`allow_split_physical_axes`) and partition layout.
- `plan-topology`: Compute target TPU generation slice shape for compute parity and HBM fit.
- `validate-model`: Validate OSS / trainer model sharding constraints, HBM fit, and CLI flags.
- `verify-compile`: Decide `LANDED` / `NO_OP` / `DROPPED` from AOT compile logs and XLA HLO dumps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import collective_breakdown  # noqa: E402
import compile_verdict  # noqa: E402
import mesh_search  # noqa: E402
import mesh_topology  # noqa: E402
import oss_sharding_rules  # noqa: E402
import rank_mesh_proposals  # noqa: E402
import sharding_adapter  # noqa: E402
import xprof_parsers  # noqa: E402


def _load_json_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    stripped = raw.strip()
    if not stripped:
        return {}
    if os.path.isfile(stripped):
        with open(stripped, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.loads(stripped)
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object.")
    return data


def _load_text_or_file(raw: str | None) -> str:
    if not raw:
        return ""
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8", errors="replace") as f:
            return f.read()
    return raw


def _split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _add_output_format(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-format",
        "--output_format",
        dest="output_format",
        choices=("table", "json"),
        default="table",
        help="Output format: 'table' (human/Markdown) or 'json'.",
    )


def _cmd_enumerate(args: argparse.Namespace) -> int:
    mesh = mesh_search.parse_mesh(args.mesh)
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "topology": args.topology,
        "extra_axes": _split_csv(args.extra_axes),
        "colocate": _split_csv(args.colocate),
        "max_moves": args.max_moves,
    }
    if args.fixed_axes:
        kwargs["fixed_axes"] = _split_csv(args.fixed_axes)
    result = mesh_search.enumerate_mesh_candidates(mesh, **kwargs)
    if args.output_format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(mesh_search.format_candidates_table(result))
    return 0


def _cmd_breakdown(args: argparse.Namespace) -> int:
    ops = xprof_parsers.parse_hlo_op_profile(args.ops_json)
    mesh = mesh_search.parse_mesh(args.mesh)
    summary = collective_breakdown.summarize(
        ops,
        mesh,
        top_k=args.top_k,
        sequence_length=args.sequence_length,
    )
    if args.output_format == "json":
        print(json.dumps(summary, indent=2))
    else:
        print(collective_breakdown.format_markdown(summary))
    return 0


def _cmd_rank_portfolio(args: argparse.Namespace) -> int:
    result = rank_mesh_proposals.rank_mesh_proposals(
        args.ops_json,
        args.mesh,
        memory_profile=args.memory_profile,
        memory_viewer_module=args.memory_viewer,
        module_names=_split_csv(args.module_names),
        megascale_stats=args.megascale_stats,
        overview_page=args.overview_page,
        kpi_metrics=args.kpi_metrics,
        num_hosts=args.num_hosts,
        dcn_pct=args.dcn_pct,
        architecture=args.arch_json,
        topology=args.topology,
        num_slices=args.num_slices,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        extra_axes=args.extra_axes,
        fixed_axes=args.fixed_axes,
        token_axes=args.token_axes,
        fsdp_axes=args.fsdp_axes,
        allow_reshape=args.allow_reshape,
        allow_slice_change=args.allow_slice_change,
        num_proposals=args.num_proposals,
    )
    print(rank_mesh_proposals.format_result(result, args.output_format))
    return 0


def _cmd_check_topology(args: argparse.Namespace) -> int:
    mesh = mesh_search.parse_mesh(args.mesh)
    cores_per_chip = args.cores_per_chip
    if cores_per_chip is None:
        cores_per_chip = (mesh_topology.get_devices_per_chip(args.platform)
                          if args.platform else None) or 1
    independent = _split_csv(args.independent_axes) or None
    fit = mesh_topology.check_mesh_topology_fit(
        mesh,
        args.topology,
        cores_per_chip=cores_per_chip,
        independent_axes=independent,
    )
    if args.partition_config:
        part_cfg = _load_json_dict(args.partition_config)
        dims = mesh_topology.parse_topology(args.topology)
        fit["partition_layout"] = mesh_topology.describe_partition_layout(
            part_cfg, mesh, dims, cores_per_chip=cores_per_chip)
    if args.output_format == "json":
        print(json.dumps(fit, indent=2))
    else:
        lines = [
            f"Mesh: {mesh}",
            f"Topology: {args.topology} (cores_per_chip={cores_per_chip})",
            f"Device count matches: {fit['device_count_matches']}",
            f"Buildable: {fit['buildable']}",
            f"Fits without splitting physical axes: {fit['fits_without_split']}",
            f"allow_split_physical_axes: {fit['allow_split_physical_axes']}",
            f"Reason: {fit['reason']}",
        ]
        if fit.get("partition_layout"):
            lines.append(f"Partition layout: {fit['partition_layout']}")
        print("\n".join(lines))
    return 0


def _cmd_plan_topology(args: argparse.Namespace) -> int:
    mem_gib = (args.memory_per_chip_gib if args.memory_per_chip_gib
               and args.memory_per_chip_gib > 0 else None)
    plan = mesh_topology.plan_target_topology(
        source_chips=args.source_chips,
        source_platform=args.source_platform,
        target_platform=args.target_platform,
        total_model_memory_per_chip_gib=mem_gib,
    )
    if args.output_format == "json":
        print(json.dumps(plan.to_dict(), indent=2))
    else:
        print(str(plan))
    return 0


def _cmd_validate_model(args: argparse.Namespace) -> int:
    proposal = mesh_search.parse_mesh(args.proposal)
    model_cfg = _load_json_dict(args.model_config)
    if args.tpu_type and "accelerator_type" not in model_cfg:
        model_cfg["accelerator_type"] = args.tpu_type

    is_valid, errors, warnings = oss_sharding_rules.validate_oss_model_sharding(
        proposal, model_cfg, args.total_tpu_chips)
    adapter = sharding_adapter.ShardingNomenclatureAdapter
    payload = {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "standardized": adapter.standardize_proposal(proposal),
        "maxtext_config": adapter.to_maxtext_config(proposal),
        "vllm_flags": adapter.to_vllm_flags(proposal),
    }
    if args.output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        status = "VALID" if is_valid else "INVALID"
        lines = [
            f"Verdict: {status}",
            f"MaxText ICI config: {json.dumps(payload['maxtext_config'])}",
            f"vLLM CLI flags: {payload['vllm_flags']['cli_string']}",
        ]
        if errors:
            lines.append("Errors:")
            lines.extend(f"  - {err}" for err in errors)
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {warn}" for warn in warnings)
        print("\n".join(lines))
    return 0


def _serialize_verdict(res: compile_verdict.VerdictResult) -> dict[str, Any]:
    mod_ev = res.module_evidence
    mod_dict: dict[str, Any] | None = None
    if mod_ev is not None:
        mod_dict = {
            "module":
            mod_ev.module,
            "optimized_hlo":
            mod_ev.files.optimized_hlo,
            "memory_report":
            mod_ev.files.memory_report,
            "peak_hbm_bytes":
            mod_ev.peak_hbm_bytes,
            "peak_hbm_formatted":
            compile_verdict.format_bytes(mod_ev.peak_hbm_bytes),
            "peak_host_bytes":
            mod_ev.peak_host_bytes,
            "instruction_count":
            (mod_ev.stats.instruction_count if mod_ev.stats else None),
            "collectives":
            (dict(mod_ev.stats.collectives) if mod_ev.stats else None),
        }
    return {
        "verdict":
        res.verdict.value,
        "error_class":
        res.error_class,
        "evidence":
        res.evidence,
        "module_evidence":
        mod_dict,
        "sharding_check":
        (dict(res.sharding_check) if res.sharding_check else None),
    }


def _cmd_verify_compile(args: argparse.Namespace) -> int:
    if args.plan_path and args.trial_json:
        plan_paths = compile_verdict.load_plan_paths(args.plan_path)
        if plan_paths is not None:
            trial = _load_json_dict(args.trial_json)
            unknown = compile_verdict.find_unknown_paths(trial, plan_paths)
            if unknown:
                dropped = compile_verdict.VerdictResult(
                    verdict=compile_verdict.Verdict.DROPPED,
                    error_class=compile_verdict.ERROR_UNKNOWN_CONFIG_PATH,
                    evidence=(
                        f"{len(unknown)} config trial path(s) not in baseline"
                        f" plan: {', '.join(unknown)}"),
                )
                payload = _serialize_verdict(dropped)
                if args.output_format == "json":
                    print(json.dumps(payload, indent=2))
                else:
                    print(
                        f"Verdict: {dropped.verdict.value}"
                        f" ({dropped.error_class})\nEvidence: {dropped.evidence}"
                    )
                return 0

    compile_passed = (args.exit_code == 0) and not args.compile_failed
    compile_log = _load_text_or_file(args.compile_log)
    checker_cmd = _split_csv(args.checker_cmd) if args.checker_cmd else None
    res = compile_verdict.decide_verdict(
        compile_passed=compile_passed,
        compile_log=compile_log,
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        module=args.module,
        checker_cmd=checker_cmd,
    )
    payload = _serialize_verdict(res)
    if args.output_format == "json":
        print(json.dumps(payload, indent=2))
    else:
        header = f"Verdict: {res.verdict.value}"
        if res.error_class:
            header += f" (error_class={res.error_class})"
        lines = [header, f"Evidence: {res.evidence}"]
        if res.module_evidence and res.module_evidence.stats:
            lines.append(
                "HLO stats: " +
                compile_verdict.format_collectives(res.module_evidence.stats))
            lines.append("Peak HBM: " + compile_verdict.format_bytes(
                res.module_evidence.peak_hbm_bytes))
        print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the top-level `argparse.ArgumentParser` for `cli.py`."""
    parser = argparse.ArgumentParser(
        prog="ici-mesh-sharding",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # 1. enumerate
    p_enum = sub.add_parser(
        "enumerate",
        help="Enumerate legal pairwise power-of-two ICI mesh candidates.",
    )
    p_enum.add_argument("--mesh",
                        required=True,
                        help="Baseline mesh JSON or k=v string.")
    p_enum.add_argument("--batch-size", "--batch_size", type=int, default=None)
    p_enum.add_argument("--sequence-length",
                        "--sequence_length",
                        type=int,
                        default=None)
    p_enum.add_argument("--num-heads", "--num_heads", type=int, default=None)
    p_enum.add_argument("--num-kv-heads",
                        "--num_kv_heads",
                        type=int,
                        default=None)
    p_enum.add_argument("--topology", default=None)
    p_enum.add_argument("--extra-axes", "--extra_axes", default=None)
    p_enum.add_argument("--fixed-axes", "--fixed_axes", default=None)
    p_enum.add_argument("--colocate", default=None)
    p_enum.add_argument("--max-moves", "--max_moves", type=int, default=1)
    _add_output_format(p_enum)
    p_enum.set_defaults(func=_cmd_enumerate)

    # 2. breakdown
    p_bd = sub.add_parser(
        "breakdown",
        help="Attribute XProf HLO op-profile collective time to mesh axes.",
    )
    p_bd.add_argument(
        "--ops-json",
        "--ops_json",
        required=True,
        help="Path or JSON string for XProf hlo_op_profile / op_profile.",
    )
    p_bd.add_argument("--mesh",
                      required=True,
                      help="Baseline mesh JSON or k=v string.")
    p_bd.add_argument("--sequence-length",
                      "--sequence_length",
                      type=int,
                      default=None)
    p_bd.add_argument("--top-k", "--top_k", type=int, default=15)
    _add_output_format(p_bd)
    p_bd.set_defaults(func=_cmd_breakdown)

    # 3. rank-portfolio
    p_rank = sub.add_parser(
        "rank-portfolio",
        help="Build the ranked 7-proposal analytical ICI mesh portfolio.",
    )
    p_rank.add_argument("--ops-json", "--ops_json", required=True)
    p_rank.add_argument("--mesh", required=True)
    p_rank.add_argument("--topology", default=None)
    p_rank.add_argument("--num-slices", "--num_slices", type=int, default=1)
    p_rank.add_argument("--batch-size", "--batch_size", type=int, default=None)
    p_rank.add_argument("--sequence-length",
                        "--sequence_length",
                        type=int,
                        default=None)
    p_rank.add_argument("--num-heads", "--num_heads", type=int, default=None)
    p_rank.add_argument("--num-kv-heads",
                        "--num_kv_heads",
                        type=int,
                        default=None)
    p_rank.add_argument("--extra-axes", "--extra_axes", default=None)
    p_rank.add_argument("--fixed-axes", "--fixed_axes", default=None)
    p_rank.add_argument("--token-axes", "--token_axes", default=None)
    p_rank.add_argument("--fsdp-axes", "--fsdp_axes", default=None)
    p_rank.add_argument("--arch-json", "--arch_json", default=None)
    p_rank.add_argument("--memory-profile", "--memory_profile", default=None)
    p_rank.add_argument("--memory-viewer", "--memory_viewer", default=None)
    p_rank.add_argument("--module-names", "--module_names", default=None)
    p_rank.add_argument("--megascale-stats", "--megascale_stats", default=None)
    p_rank.add_argument("--overview-page", "--overview_page", default=None)
    p_rank.add_argument("--kpi-metrics", "--kpi_metrics", default=None)
    p_rank.add_argument("--num-hosts", "--num_hosts", type=int, default=1)
    p_rank.add_argument("--dcn-pct", "--dcn_pct", type=float, default=None)
    p_rank.add_argument("--allow-reshape",
                        "--allow_reshape",
                        action="store_true",
                        default=False)
    p_rank.add_argument(
        "--allow-slice-change",
        "--allow_slice_change",
        action="store_true",
        default=False,
    )
    p_rank.add_argument(
        "--num-proposals",
        "--num_proposals",
        type=int,
        default=rank_mesh_proposals.DEFAULT_NUM_PROPOSALS,
    )
    _add_output_format(p_rank)
    p_rank.set_defaults(func=_cmd_rank_portfolio)

    # 4. check-topology
    p_topo = sub.add_parser(
        "check-topology",
        help=
        "Check physical torus fit and allow_split_physical_axes requirement.",
    )
    p_topo.add_argument("--mesh", required=True)
    p_topo.add_argument("--topology", required=True)
    p_topo.add_argument("--platform", default=None)
    p_topo.add_argument("--cores-per-chip",
                        "--cores_per_chip",
                        type=int,
                        default=None)
    p_topo.add_argument("--independent-axes",
                        "--independent_axes",
                        default=None)
    p_topo.add_argument("--partition-config",
                        "--partition_config",
                        default=None)
    _add_output_format(p_topo)
    p_topo.set_defaults(func=_cmd_check_topology)

    # 5. plan-topology
    p_plan = sub.add_parser(
        "plan-topology",
        help=
        "Plan target TPU generation slice shape for compute parity and HBM fit.",
    )
    p_plan.add_argument("--source-chips",
                        "--source_chips",
                        type=int,
                        required=True)
    p_plan.add_argument("--source-platform",
                        "--source_platform",
                        required=True)
    p_plan.add_argument("--target-platform",
                        "--target_platform",
                        required=True)
    p_plan.add_argument(
        "--memory-per-chip-gib",
        "--memory_per_chip_gib",
        type=float,
        default=0.0,
    )
    _add_output_format(p_plan)
    p_plan.set_defaults(func=_cmd_plan_topology)

    # 6. validate-model
    p_val = sub.add_parser(
        "validate-model",
        help=
        "Validate OSS / trainer model sharding constraints and emit engine flags.",
    )
    p_val.add_argument("--proposal", required=True)
    p_val.add_argument("--model-config", "--model_config", default="{}")
    p_val.add_argument("--total-tpu-chips",
                       "--total_tpu_chips",
                       type=int,
                       default=0)
    p_val.add_argument("--tpu-type", "--tpu_type", default="")
    _add_output_format(p_val)
    p_val.set_defaults(func=_cmd_validate_model)

    # 7. verify-compile
    p_ver = sub.add_parser(
        "verify-compile",
        help=
        "Decide LANDED / NO_OP / DROPPED from AOT compile logs and XLA dumps.",
    )
    p_ver.add_argument("--candidate-dir", "--candidate_dir", required=True)
    p_ver.add_argument("--baseline-dir", "--baseline_dir", default=None)
    p_ver.add_argument("--compile-log", "--compile_log", default="")
    p_ver.add_argument("--exit-code", "--exit_code", type=int, default=0)
    p_ver.add_argument(
        "--compile-failed",
        "--compile_failed",
        action="store_true",
        default=False,
    )
    p_ver.add_argument("--module", default=None)
    p_ver.add_argument("--plan-path", "--plan_path", default=None)
    p_ver.add_argument("--trial-json", "--trial_json", default=None)
    p_ver.add_argument("--checker-cmd", "--checker_cmd", default=None)
    _add_output_format(p_ver)
    p_ver.set_defaults(func=_cmd_verify_compile)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

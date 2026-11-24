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
"""Command-line argument parser and top-level dispatch for `runtime hc`."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import io
import json
import os
import pathlib
import sys
from typing import Any, NoReturn, Sequence

from framework import paths as base_paths
from framework import utils
from framework.hc import (campaign_ops, candidate_ops, constants, db,
                          hypothesis_ops, paths, review_ops, round_ops,
                          telemetry)
from framework.hooks import events

Path = pathlib.Path


def parse_positive_int(raw: str) -> int:
    """Parse a positive integer CLI identifier, raising `ValidationError` on bad input."""
    stripped = str(raw).strip()
    if not (stripped.isascii() and stripped.isdecimal()) or int(stripped) <= 0:
        raise constants.ValidationError(
            f"expected positive integer ID, got {raw!r}")
    return int(stripped)


class HcArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that converts CLI usage errors into structured `HcError`s."""

    def error(self, message: str) -> NoReturn:
        """Raise an `INVALID_ARGS` `HcError` instead of exiting directly."""
        raise constants.HcError(message,
                                error_code="INVALID_ARGS",
                                exit_code=2)


def make_common_parent() -> argparse.ArgumentParser:
    """Shared context flags accepted at the root or leaf subcommand level."""
    parent = HcArgumentParser(add_help=False)
    parent.add_argument("--workspace",
                        type=Path,
                        default=argparse.SUPPRESS,
                        help="Optional workspace root directory override")
    parent.add_argument("--run-id",
                        default=argparse.SUPPRESS,
                        help="Optional run ID override")
    return parent


def build_parser(prog: str = "runtime hc") -> HcArgumentParser:
    """Build the `runtime hc` CLI parser."""
    common = make_common_parent()
    parser = HcArgumentParser(
        prog=prog,
        parents=[common],
        description="Hillclimber state management agent interface.",
    )
    sub = parser.add_subparsers(dest="domain", required=True)

    def _add_domain(name: str, help_text: str) -> Any:
        return sub.add_parser(name, parents=[common],
                              help=help_text).add_subparsers(dest="action",
                                                             required=True)

    def _add_cmd(
        group: Any,
        name: str,
        fn: Any,
        help_text: str,
    ) -> argparse.ArgumentParser:
        leaf = group.add_parser(name, parents=[common], help=help_text)
        if fn is not None:
            params = tuple(inspect.signature(fn).parameters)[1:]
            leaf.set_defaults(_handler=lambda conn, a, ws, run_id, telem: fn(
                conn,
                **{
                    k: ({
                        "run_id": run_id,
                        "workspace": ws,
                        "telem": telem,
                    }[k] if k in
                        ("run_id", "workspace", "telem") else getattr(a, k))
                    for k in params
                },
            ))
        return leaf

    # campaign
    camp_sub = _add_domain("campaign",
                           "Manage hillclimbing campaign lifecycle")
    p_open = _add_cmd(camp_sub, "open", campaign_ops.open_campaign,
                      "Open a new campaign in Unanchored state")
    p_open.add_argument("campaign_config_path",
                        metavar="campaign_config",
                        type=Path,
                        help="Campaign Config JSON file")
    p_open.add_argument("protocol_config_path",
                        metavar="protocol_config",
                        type=Path,
                        help="Protocol Config JSON file")
    p_open.add_argument("--code-dir",
                        type=Path,
                        required=True,
                        help="Source directory imported as initial_version")
    p_open.add_argument("--microbenchmark-protocol",
                        dest="microbenchmark_protocol_path",
                        type=Path,
                        help="Optional microbenchmark Protocol Config JSON")
    p_open.add_argument("--package", help="Optional package tag override")

    p_micro = _add_cmd(camp_sub, "set-microbenchmark-protocol",
                       campaign_ops.set_microbenchmark_protocol,
                       "Set or replace the campaign microbenchmark protocol")
    p_micro.add_argument("protocol_config_path",
                         metavar="protocol_config",
                         type=Path,
                         help="Microbenchmark Protocol Config JSON")
    p_micro.add_argument("--package", help="Optional package tag override")

    p_block = _add_cmd(camp_sub, "block", campaign_ops.block_campaign,
                       "Halt the campaign pending human intervention")
    p_block.add_argument("reason",
                         help="Explanation for blocking the campaign")

    _add_cmd(camp_sub, "unblock", campaign_ops.unblock_campaign,
             "Resume a Blocked campaign")

    p_close = _add_cmd(camp_sub, "close", campaign_ops.close_campaign,
                       "Close the active campaign")
    p_close.add_argument("--abort",
                         action="store_true",
                         help="Record close_reason=Aborted instead of Manual")

    # round
    round_sub = _add_domain("round", "Manage hillclimbing rounds")
    _add_cmd(round_sub, "open", round_ops.open_round,
             "Open the next hillclimbing round")

    # hypothesis
    hyp_sub = _add_domain("hypothesis", "Manage optimization hypotheses")
    p_hyp_add = _add_cmd(hyp_sub, "add", hypothesis_ops.add_hypothesis,
                         "Record a new Open hypothesis in the current round")
    p_hyp_add.add_argument("hypothesis_json_path",
                           metavar="hypothesis_config",
                           type=Path,
                           help="Hypothesis JSON file")

    p_hyp_refute = _add_cmd(hyp_sub, "refute",
                            hypothesis_ops.refute_hypothesis,
                            "Mark an Open hypothesis as Refuted")
    p_hyp_refute.add_argument("hypothesis_id",
                              type=parse_positive_int,
                              help="Hypothesis primary key")
    p_hyp_refute.add_argument("reason",
                              help="Immutable refutation explanation")

    # candidate
    cand_sub = _add_domain("candidate", "Manage optimization candidates")
    p_prep = _add_cmd(
        cand_sub, "prepare", None,
        "Populate <workspace>/src/ from 1 or 2 parent candidates")
    p_prep.add_argument("first_parent",
                        type=parse_positive_int,
                        help="First parent candidate ID")
    p_prep.add_argument("second_parent",
                        nargs="?",
                        type=parse_positive_int,
                        help="Optional second parent candidate ID")
    p_prep.add_argument("--discard",
                        action="store_true",
                        help="Overwrite uncommitted edits in <workspace>/src/")
    p_prep.set_defaults(
        _handler=lambda conn, a, ws, run_id, telem: candidate_ops.
        prepare_candidate(
            conn,
            run_id=run_id,
            workspace=ws,
            parent_ids=([a.first_parent] + (
                [a.second_parent] if a.second_parent is not None else [])),
            discard=bool(a.discard),
            telem=telem,
        ))

    p_prop = _add_cmd(cand_sub, "propose", candidate_ops.propose_candidate,
                      "Commit <workspace>/src/ as an immutable candidate")
    p_prop.add_argument("--hypotheses",
                        dest="hypothesis_ids",
                        nargs="+",
                        type=parse_positive_int,
                        default=[],
                        help="Open hypothesis ID(s) implemented")
    p_prop.add_argument("--supersedes",
                        dest="supersedes_id",
                        type=parse_positive_int,
                        help="Prior candidate ID superseded by this candidate")
    p_prop.add_argument("--innovation",
                        required=True,
                        help="Prose description of expected advantage")
    p_prop.add_argument("--short-innovation",
                        required=True,
                        help="Summary prose description (<= 500 chars)")

    p_aban = _add_cmd(cand_sub, "abandon", candidate_ops.abandon_candidate,
                      "Mark a non-Promoted candidate as Abandoned")
    p_aban.add_argument("candidate_id",
                        type=parse_positive_int,
                        help="Candidate primary key")
    p_aban.add_argument("reason", help="Explanation for abandoning candidate")

    # review
    rev_sub = _add_domain("review", "Manage qualitative reviews")
    p_rev_add = _add_cmd(
        rev_sub, "add", review_ops.add_review,
        "Record a qualitative review for a candidate or hypothesis")
    rev_target = p_rev_add.add_mutually_exclusive_group(required=True)
    rev_target.add_argument("--candidate",
                            dest="candidate_id",
                            type=parse_positive_int,
                            help="Target candidate ID")
    rev_target.add_argument("--hypothesis",
                            dest="hypothesis_id",
                            type=parse_positive_int,
                            help="Target hypothesis ID")
    p_rev_add.add_argument("--approval",
                           required=True,
                           help="Judgement (Pass | ConditionalPass | Fail)")
    p_rev_add.add_argument("--comments",
                           required=True,
                           help="Freeform review comments")
    p_rev_add.add_argument("--short-comments",
                           help="Optional summary comments (<= 500 chars)")
    p_rev_add.add_argument("--conditions",
                           help="Required caveats for ConditionalPass")

    return parser


def resolve_context(args: argparse.Namespace | None,
                    raw_argv: Sequence[str]) -> tuple[Path, str]:
    """Resolve `(workspace, run_id)` from parsed args, workspace markers, or env."""
    ws_override = getattr(args, "workspace",
                          None) if args is not None else None
    run_override = getattr(args, "run_id", None) if args is not None else None

    if args is None and raw_argv:
        lenient = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        lenient.add_argument("--workspace", type=Path, default=None)
        lenient.add_argument("--run-id", default=None)
        with contextlib.suppress(SystemExit), contextlib.redirect_stderr(
                io.StringIO()):
            known, _ = lenient.parse_known_args(list(raw_argv))
            ws_override = known.workspace
            run_override = known.run_id

    workspace = events.resolve_workspace_root(ws_override)
    run_id = (run_override or base_paths.get_active_run_id(workspace)
              or os.environ.get("TPU_AGENT_SUITE_RUN_ID") or "").strip()
    if not run_id:
        raise constants.InvalidStateError(
            "no active run ID found; run `runtime.py start-run` or pass --run-id"
        )
    try:
        utils.validate_run_id(run_id)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    try:
        base_paths.get_started_run_root(run_id)
    except utils.TpuAgentSuiteError as exc:
        raise constants.InvalidStateError(str(exc)) from exc
    return workspace, run_id


def _best_effort_rejection_context(
        raw_argv: Sequence[str]) -> tuple[str, int | None, int | None]:
    """Best-effort `(run_id, campaign_id, round_id)` lookup when startup/parse fails."""
    try:
        _, run_id = resolve_context(None, raw_argv)
    except (constants.HcError, utils.TpuAgentSuiteError):
        return "unknown", None, None

    cid = paths.get_active_campaign_id(run_id)
    if cid is None:
        return run_id, None, None
    db_path = paths.get_campaign_db_path()
    if not db_path.is_file():
        return run_id, cid, None
    try:
        with db.open_db(db_path, init_schema=False) as conn:
            open_rnd = campaign_ops.get_open_round(conn, cid)
            rid = int(open_rnd["id"]) if open_rnd is not None else None
            return run_id, cid, rid
    except Exception:  # noqa: BLE001
        return run_id, cid, None


def _emit_rejection(
    telem: telemetry.TelemetryBuffer | None,
    raw_argv: Sequence[str],
    *,
    error_code: str,
    message: str,
) -> None:
    if telem is None:
        fb_run, fb_cid, fb_rid = _best_effort_rejection_context(raw_argv)
        telem = telemetry.TelemetryBuffer(
            run_id=fb_run,
            campaign_id=fb_cid,
            round_id=fb_rid,
        )
    telem.record_rejection(args=raw_argv,
                           error_code=error_code,
                           message=message)


def main(argv: Sequence[str] | None = None,
         *,
         prog: str = "runtime hc") -> int:
    """Main CLI entry point for `runtime.py hc`."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    telem: telemetry.TelemetryBuffer | None = None

    try:
        parser = build_parser(prog=prog)
        args = parser.parse_args(raw_argv)
        workspace, run_id = resolve_context(args, raw_argv)

        with db.open_db() as conn:
            active_cid = paths.get_active_campaign_id(run_id)
            open_rid: int | None = None
            if active_cid is not None:
                open_rnd = campaign_ops.get_open_round(conn, active_cid)
                if open_rnd is not None:
                    open_rid = int(open_rnd["id"])

            telem = telemetry.TelemetryBuffer(
                run_id=run_id,
                campaign_id=active_cid,
                round_id=open_rid,
            )

            handler = getattr(args, "_handler", None)
            if handler is None:
                raise constants.ValidationError(
                    f"unsupported command: {args.domain!r}")
            res = handler(conn, args, workspace, run_id, telem)
            print(json.dumps(res, ensure_ascii=False))

        # Emit telemetry strictly after all database transactions have committed.
        telem.flush()
        subcommands = [
            getattr(args, "domain", None),
            getattr(args, "action", None),
        ]
        telem.maybe_record_call_fallback(
            subcommands=[s for s in subcommands if s],
            args=raw_argv,
        )
        return 0
    except constants.HcError as exc:
        _emit_rejection(telem,
                        raw_argv,
                        error_code=exc.error_code,
                        message=str(exc))
        print(f"{prog} error [{exc.error_code}]: {exc}", file=sys.stderr)
        return exc.exit_code
    except utils.TpuAgentSuiteError as exc:
        _emit_rejection(telem,
                        raw_argv,
                        error_code="FRAMEWORK_ERROR",
                        message=str(exc))
        print(f"{prog} error [FRAMEWORK_ERROR]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        _emit_rejection(telem,
                        raw_argv,
                        error_code="INTERNAL_ERROR",
                        message=msg)
        print(f"{prog} internal error: {msg}", file=sys.stderr)
        return 1

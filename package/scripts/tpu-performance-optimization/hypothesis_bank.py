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
"""Hypothesis bank: records the researcher's hypotheses as study arms.

The researcher model generates and prioritizes optimization hypotheses. This
script is their bookkeeper. It never invents, scores, rescales or reorders a
hypothesis. For each proposal, in the order given, it:

1. Rejects on objective checks only: a malformed or unknown field, a missing
   id, kind, claim or evidence, a share or headroom outside [0, 1], a code_ref
   that escapes or is missing from --repo-root, or an arm id or recipe
   (fingerprint) the study already holds.
2. Expands the hypothesis into one study arm per variant (one arm when it has
   no variants) and admits all of them or none of them, within --max-new-arms
   arms per call.
3. Returns every judgment as a warning for the model to act on: a missing or
   low expected gain, hlo-op evidence or a share the XProf summary does not
   support, TPU hardware gotchas (`arm_microbench.check_hardware_gotchas`),
   and code_refs that overlap an open or in_flight arm.

Arms in study.json carry only study-round.json fields. Everything else -- the
hypothesis with its evidence and code_refs, and each arm's fingerprint and
merged target_parameters -- lives in the `hypothesis_bank.json` sidecar beside
study.json, written under the same `study.study_lock`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import posixpath
import re
import sys
from typing import Any, Mapping, Sequence

_HERE = pathlib.Path(__file__).resolve()
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
# The directory holding the framework package: the package's own in a build,
# the repository root in the repository.
for _root in (_HERE.parents[2], _HERE.parents[3]):
    if (_root / "framework").is_dir():
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        break

import arm_microbench  # noqa: E402
import evaluate_arm  # noqa: E402

from framework import study  # noqa: E402

DEFAULT_MAX_NEW_ARMS = 5
DEFAULT_MIN_EXPECTED_GAIN = 0.005
DEFAULT_JOURNAL_DEDUP_LIMIT = 64
DEFAULT_RETIRED_LIMIT = 64
PROFILE_SHARE_TOLERANCE = 0.05

# One slug grammar for hypothesis ids and variant suffixes, so the arm id
# `<id>-<suffix>` of a variant is itself a valid hypothesis id.
ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"
ACTIVE_ARM_STATUSES = ("open", "in_flight")
SUPPORTED_TPU_VERSIONS = tuple(k for k in arm_microbench.TPU_HARDWARE_SPECS
                               if k not in ("trillium", "ironwood"))

# The hypothesis item. The same rules live in the item schema under
# `properties.hypotheses.items` of schemas/hypothesis-bank-result.json; the
# code checks them itself because the framework validator has no minItems,
# $ref or anyOf.
HYPOTHESIS_FIELDS = (
    "id",
    "kind",
    "claim",
    "details",
    "targeted_roofline",
    "evidence",
    "code_refs",
    "share",
    "headroom",
    "target_parameters",
    "kb_references",
    "variants",
)
REQUIRED_HYPOTHESIS_FIELDS = ("id", "kind", "claim", "evidence")
_SUPERSEDED_FIELDS = {
    "attributed_ops": "evidence[type=hlo-op]",
    "target_files": "code_refs",
}
# Reasons that leave a proposal too malformed to echo back or expand into arms.
_SHAPE_CODES = frozenset(
    {"invalid_shape", "missing_field", "invalid_id", "out_of_range"})

# The sidecar beside study.json. study-round.json arm items are
# `additionalProperties: false`, so any extra field on an arm makes the next
# `study.record_round` echo of that arm fail validation. Hypothesis metadata
# therefore lives here, written under the same `study.study_lock`:
#   {"hypotheses": {"a1-fused-recurrence": {<the hypothesis, known keys only>,
#                   "request_id": "20260925T120000Z-0123456789ab"}},
#    "arms": {"a1-fused-recurrence-b256": {
#                 "hypothesis_id": "a1-fused-recurrence",
#                 "fingerprint": "<16 hex>",
#                 "target_parameters": {"fuse_recurrence": true,
#                                       "block_size": 256}}},
#    "retired": [{"arm": "...", "hypothesis_id": "...",
#                 "fingerprint": "<16 hex>"}]}
# `arms` holds only arms still in study.json. An arm that leaves study.json
# moves to `retired` (newest last, at most DEFAULT_RETIRED_LIMIT) so its
# recipe keeps blocking duplicates, and a hypothesis with no arms left is
# dropped.
HYPOTHESIS_BANK_FILE = "hypothesis_bank.json"

_EXIT_CODES = {"completed": 0, "failed": 1, "invalid_input": 2}


def hypothesis_bank_path(study_id: str) -> pathlib.Path:
    """Path of the hypothesis bank sidecar for `study_id`."""
    return study.paths.get_study_root(study_id) / HYPOTHESIS_BANK_FILE


def load_hypothesis_bank(study_id: str) -> dict[str, Any]:
    """Read the bank as {hypotheses, arms, retired}.

    Each part defaults to empty when the file is absent or malformed.
    """
    raw = study.jsonio.load_json_or_empty(hypothesis_bank_path(study_id))
    hypotheses = raw.get("hypotheses")
    arms = raw.get("arms")
    retired = raw.get("retired")
    return {
        "hypotheses": ({
            str(k): dict(v)
            for k, v in hypotheses.items() if isinstance(v, Mapping)
        } if isinstance(hypotheses, Mapping) else {}),
        "arms": ({
            str(k): dict(v)
            for k, v in arms.items() if isinstance(v, Mapping)
        } if isinstance(arms, Mapping) else {}),
        "retired":
        ([dict(r) for r in retired
          if isinstance(r, Mapping)] if isinstance(retired, list) else []),
    }


def with_hypothesis_metadata(
    arms: Sequence[Any],
    bank: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Overlay bank metadata under each arm; study.json fields win.

    Adds `hypothesis_id`, `fingerprint` and merged `target_parameters` from
    the arm's bank entry, and `code_refs` from its hypothesis. Arms the bank
    does not know are returned unchanged.
    """
    bank_arms = bank.get("arms") or {}
    bank_hypotheses = bank.get("hypotheses") or {}
    out: list[dict[str, Any]] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            continue
        meta = bank_arms.get(str(arm.get("arm") or ""))
        overlay: dict[str, Any] = {}
        if isinstance(meta, Mapping):
            hypothesis_id = str(meta.get("hypothesis_id") or "")
            if hypothesis_id:
                overlay["hypothesis_id"] = hypothesis_id
            if meta.get("fingerprint"):
                overlay["fingerprint"] = str(meta["fingerprint"])
            if isinstance(meta.get("target_parameters"), Mapping):
                overlay["target_parameters"] = dict(meta["target_parameters"])
            hypothesis = bank_hypotheses.get(hypothesis_id)
            if isinstance(hypothesis, Mapping) and isinstance(
                    hypothesis.get("code_refs"), list):
                overlay["code_refs"] = copy.deepcopy(hypothesis["code_refs"])
        out.append({**overlay, **dict(arm)})
    return out


def _verify_against_profile_summary(
    attributed_ops: Sequence[str],
    claimed_share: float,
    profile_summary: Mapping[str, Any] | None,
) -> list[str]:
    """Cross-check candidate op attribution and share against a measured XProf summary."""
    if not isinstance(profile_summary, Mapping):
        return []
    raw_ops = profile_summary.get("ops") or profile_summary.get("top_ops")
    if not isinstance(raw_ops, Sequence):
        return []

    known_shares: dict[str, float] = {}
    for entry in raw_ops:
        if isinstance(entry, Mapping):
            name = str(entry.get("name") or entry.get("op_name") or "").strip()
            share_val = entry.get("share")
            if share_val is None and isinstance(entry.get("self_time_pct"),
                                                (int, float)):
                share_val = float(entry["self_time_pct"]) / 100.0
            if name and isinstance(share_val, (int, float)):
                known_shares[name] = float(share_val)

    if not known_shares:
        return []

    issues: list[str] = []
    matched_share = 0.0
    unmatched: list[str] = []
    for op in attributed_ops:
        op_str = str(op).strip()
        if op_str in known_shares:
            matched_share += known_shares[op_str]
        else:
            substr_matches = [
                s for k, s in known_shares.items()
                if op_str in k or k in op_str
            ]
            if substr_matches:
                matched_share += max(substr_matches)
            else:
                unmatched.append(op_str)

    if unmatched:
        issues.append(
            f"unverified_profile_ops: ops {unmatched} not found in XProf profile summary"
        )
    elif matched_share > 0 and claimed_share > matched_share + PROFILE_SHARE_TOLERANCE:
        issues.append(
            f"inflated_profile_share: claimed share {claimed_share:.3f} exceeds "
            f"profiled op share {matched_share:.3f} (+{PROFILE_SHARE_TOLERANCE:.2f} tol)"
        )
    return issues


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and (not isinstance(value, float)
                                  or math.isfinite(value))


def _reason_code(reason: str) -> str:
    """The machine-readable code of `code`, `code:arg` or `code: detail`."""
    return reason.split(":", 1)[0].strip()


def _generate_run_id() -> str:
    """Generate a fresh run ID (imported lazily to avoid loading `compute` on `--help`)."""
    from framework import run_state
    return run_state.generate_run_id()


def _request_id(run_id: Any) -> str:
    if isinstance(run_id, str) and re.fullmatch(study.utils.RUN_ID_PATTERN,
                                                run_id):
        return run_id
    return _generate_run_id()


def _empty_result(status: str, study_id: Any, request_id: str,
                  error: str) -> dict[str, Any]:
    return {
        "status": status,
        "study_id": str(study_id),
        "request_id": request_id,
        "admitted": [],
        "rejected": [],
        "error": error,
    }


def _input_problem(
    study_id: Any,
    hypotheses: Any,
    *,
    run_id: Any,
    repo_root: Any,
    tpu_version: Any,
    max_new_arms: Any,
    min_expected_gain: Any,
    journal_dedup_limit: Any,
    profile_summary: Any,
) -> str:
    """Why this call cannot be screened at all, or "" when it can."""
    if not isinstance(study_id, str):
        return f"study_id must be a string, got {type(study_id).__name__}"
    try:
        study.utils.validate_study_id(study_id)
    except study.TpuAgentSuiteError as exc:
        return str(exc)
    if run_id and not (isinstance(run_id, str)
                       and re.fullmatch(study.utils.RUN_ID_PATTERN, run_id)):
        return (f"invalid run_id {run_id!r}: expected"
                f" {study.utils.RUN_ID_PATTERN}")
    if (not isinstance(tpu_version, str)
            or tpu_version not in SUPPORTED_TPU_VERSIONS):
        return (f"unknown tpu_version {tpu_version!r}: expected one of"
                f" {list(SUPPORTED_TPU_VERSIONS)}")
    if not isinstance(hypotheses, (list, tuple)):
        return ("hypotheses must be an array of hypothesis objects, got"
                f" {type(hypotheses).__name__}")
    if not _is_int(max_new_arms) or max_new_arms < 1:
        return f"max_new_arms must be an integer >= 1, got {max_new_arms!r}"
    if not _is_finite_number(min_expected_gain):
        return ("min_expected_gain must be a finite number, got"
                f" {min_expected_gain!r}")
    if not _is_int(journal_dedup_limit) or journal_dedup_limit < 1:
        return ("journal_dedup_limit must be an integer >= 1, got"
                f" {journal_dedup_limit!r}")
    if repo_root is not None and (not str(repo_root).strip()
                                  or not pathlib.Path(repo_root).is_dir()):
        return f"repo_root {str(repo_root)!r} is not a directory"
    if profile_summary is not None and not isinstance(profile_summary,
                                                      Mapping):
        return ("profile_summary must be a JSON object, got"
                f" {type(profile_summary).__name__}")
    return ""


def _object_item_errors(
        where: str,
        item: Any,
        *,
        required: tuple[str, ...],
        optional: tuple[str, ...] = (),
        non_empty: tuple[str, ...] = (),
) -> list[str]:
    """Shape errors of an evidence or code_refs item: string fields only."""
    if not isinstance(item, Mapping):
        return [f"invalid_shape: {where} must be an object"]
    allowed = (*required, *optional)
    errors: list[str] = []
    unknown = sorted(str(key) for key in item if key not in allowed)
    if unknown:
        errors.append(f"invalid_shape: {where} has unknown key(s) {unknown};"
                      f" allowed: {list(allowed)}")
    for key in required:
        if key not in item:
            errors.append(f"invalid_shape: {where}.{key} is required")
    for key in allowed:
        if key in item:
            value = item[key]
            if not isinstance(value, str) or (key in non_empty and not value):
                wanted = ("a non-empty string"
                          if key in non_empty else "a string")
                errors.append(f"invalid_shape: {where}.{key} must be {wanted}")
    return errors


def _variant_errors(where: str, variant: Any) -> list[str]:
    if not isinstance(variant, Mapping):
        return [f"invalid_shape: {where} must be an object"]
    allowed = ("suffix", "target_parameters", "claim")
    errors: list[str] = []
    unknown = sorted(str(key) for key in variant if key not in allowed)
    if unknown:
        errors.append(f"invalid_shape: {where} has unknown key(s) {unknown};"
                      f" allowed: {list(allowed)}")
    if "suffix" not in variant:
        errors.append(f"invalid_shape: {where}.suffix is required")
    elif not isinstance(variant["suffix"], str) or not re.fullmatch(
            ID_PATTERN, variant["suffix"]):
        errors.append(f"invalid_shape: {where}.suffix must match {ID_PATTERN}")
    if "target_parameters" in variant and not isinstance(
            variant["target_parameters"], Mapping):
        errors.append(
            f"invalid_shape: {where}.target_parameters must be an object")
    if "claim" in variant and (not isinstance(variant["claim"], str)
                               or not variant["claim"]):
        errors.append(
            f"invalid_shape: {where}.claim must be a non-empty string")
    return errors


def _hypothesis_errors(raw: Any) -> list[str]:
    """Every objective reason one proposal is malformed; [] when it is not.

    Mirrors the item schema in hypothesis-bank-result.json, plus the rules
    that schema cannot express: kind membership, at least one evidence item,
    finite numbers, and unique variant suffixes.
    """
    if not isinstance(raw, Mapping):
        return [
            "invalid_shape: a hypothesis must be an object, got"
            f" {type(raw).__name__}"
        ]
    reasons: list[str] = []
    for key in raw:
        if key not in HYPOTHESIS_FIELDS:
            hint = _SUPERSEDED_FIELDS.get(str(key))
            reasons.append(f"unknown_field:{key}: " +
                           (f"superseded by {hint}" if hint else
                            f"allowed fields are {list(HYPOTHESIS_FIELDS)}"))
    for key in REQUIRED_HYPOTHESIS_FIELDS:
        if key not in raw:
            reasons.append(f"missing_field:{key}")

    if "id" in raw:
        if not isinstance(raw["id"], str):
            reasons.append("invalid_shape: id must be a string")
        elif not re.fullmatch(ID_PATTERN, raw["id"]):
            reasons.append(
                f"invalid_id: {raw['id']!r} must match {ID_PATTERN}")
    if "kind" in raw:
        if not isinstance(raw["kind"], str):
            reasons.append("invalid_shape: kind must be a string")
        elif raw["kind"] not in study.BOTTLENECK_KINDS:
            reasons.append(f"invalid_kind: {raw['kind']!r} is not one of"
                           f" {list(study.BOTTLENECK_KINDS)}")
    if "claim" in raw and (not isinstance(raw["claim"], str)
                           or not raw["claim"]):
        reasons.append("invalid_shape: claim must be a non-empty string")
    if "details" in raw and not isinstance(raw["details"], str):
        reasons.append("invalid_shape: details must be a string")
    if "targeted_roofline" in raw and not isinstance(raw["targeted_roofline"],
                                                     Mapping):
        reasons.append("invalid_shape: targeted_roofline must be an object")

    evidence = raw.get("evidence")
    if "evidence" in raw and not isinstance(evidence, list):
        reasons.append(
            "invalid_shape: evidence must be an array of {type, ref, note?}")
    elif isinstance(evidence, list):
        for index, item in enumerate(evidence):
            reasons.extend(
                _object_item_errors(f"evidence[{index}]",
                                    item,
                                    required=("type", "ref"),
                                    optional=("note", ),
                                    non_empty=("type", "ref")))
    if "evidence" not in raw or evidence == []:
        reasons.append(
            "no_evidence: cite at least one evidence item, e.g. an xprof"
            " trace, an hlo-op, code, sharding, config, a prior arm or a kb"
            " page")

    if "code_refs" in raw:
        if not isinstance(raw["code_refs"], list):
            reasons.append("invalid_shape: code_refs must be an array of"
                           " {path, lines?, symbol?}")
        else:
            for index, item in enumerate(raw["code_refs"]):
                reasons.extend(
                    _object_item_errors(f"code_refs[{index}]",
                                        item,
                                        required=("path", ),
                                        optional=("lines", "symbol"),
                                        non_empty=("path", )))

    for key in ("share", "headroom"):
        if key in raw:
            value = raw[key]
            if not _is_finite_number(value) or not 0 <= value <= 1:
                reasons.append(f"out_of_range:{key}: must be a finite number"
                               f" in [0, 1], got {value!r}")

    if "target_parameters" in raw and not isinstance(raw["target_parameters"],
                                                     Mapping):
        reasons.append("invalid_shape: target_parameters must be an object")
    if "kb_references" in raw and not (isinstance(raw["kb_references"], list)
                                       and all(
                                           isinstance(ref, str)
                                           for ref in raw["kb_references"])):
        reasons.append("invalid_shape: kb_references must be an array of"
                       " strings")

    if "variants" in raw:
        variants = raw["variants"]
        if not isinstance(variants, list):
            reasons.append("invalid_shape: variants must be an array of"
                           " {suffix, target_parameters?, claim?}")
        else:
            seen: set[str] = set()
            repeated: list[str] = []
            for index, variant in enumerate(variants):
                reasons.extend(_variant_errors(f"variants[{index}]", variant))
                suffix = (variant.get("suffix") if isinstance(
                    variant, Mapping) else None)
                if isinstance(suffix, str):
                    if suffix in seen and suffix not in repeated:
                        repeated.append(suffix)
                    seen.add(suffix)
            for suffix in repeated:
                reasons.append(
                    f"duplicate_variant_suffix:{suffix}: each variant needs"
                    " its own suffix, which becomes the arm id <id>-<suffix>")
    return reasons


def _normalized(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    """The hypothesis with known keys only, in field order."""
    return {
        key: copy.deepcopy(hypothesis[key])
        for key in HYPOTHESIS_FIELDS if key in hypothesis
    }


def _expand_arms(hypothesis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The arms one hypothesis becomes: one per variant, or one without."""
    hypothesis_id = hypothesis["id"]
    claim = hypothesis["claim"]
    base = dict(hypothesis.get("target_parameters") or {})
    specs = [(hypothesis_id, claim, base)]
    if hypothesis.get("variants"):
        specs = [(
            f"{hypothesis_id}-{variant['suffix']}",
            variant.get("claim") or f"{claim} [{variant['suffix']}]",
            {
                **base,
                **dict(variant.get("target_parameters") or {})
            },
        ) for variant in hypothesis["variants"]]
    return [{
        "arm":
        arm_id,
        "claim":
        arm_claim,
        "target_parameters":
        params,
        "fingerprint":
        evaluate_arm.compute_arm_fingerprint(
            kind=hypothesis["kind"],
            target_parameters=params or None,
            claim=arm_claim,
        ),
    } for arm_id, arm_claim, params in specs]


def _claim_fingerprint(arm: Mapping[str, Any]) -> str:
    """Fingerprint of an arm the bank holds no recipe for: kind and claim."""
    return evaluate_arm.compute_arm_fingerprint(
        kind=str(arm.get("kind") or ""),
        target_parameters=None,
        claim=str(arm.get("claim") or ""),
    )


def _arm_ids(arms: Any) -> set[str]:
    ids = {
        str(arm.get("arm") or "")
        for arm in arms or [] if isinstance(arm, Mapping)
    }
    ids.discard("")
    return ids


def _dedup_ledger(
    doc: Mapping[str, Any],
    bank: Mapping[str, Any],
    journal: Sequence[Any],
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """What a new arm must not collide with.

    Returns (arm id -> where it lives, fingerprint -> arm holding it, ids of
    hypotheses that still have arms in study.json). Bank arms missing from
    both study.json and the journal tail -- a write that died before
    `save_study` -- are deliberately absent, so a retry is not a duplicate.
    """
    taken_ids: dict[str, str] = {}
    known_fps: dict[str, str] = {}
    study_arms = [
        arm for arm in doc.get("arms") or [] if isinstance(arm, Mapping)
    ]
    for arm in with_hypothesis_metadata(study_arms, bank):
        arm_id = str(arm.get("arm") or "")
        if arm_id:
            taken_ids.setdefault(arm_id,
                                 f"study.json (status {arm.get('status')})")
        known_fps.setdefault(
            str(arm.get("fingerprint") or "") or _claim_fingerprint(arm),
            arm_id or "a study.json arm")

    retired_fps: dict[str, str] = {}
    for entry in bank.get("retired") or []:
        fingerprint = str(entry.get("fingerprint") or "")
        if fingerprint:
            arm_id = str(entry.get("arm") or "")
            known_fps.setdefault(fingerprint, arm_id or "a retired arm")
            retired_fps[arm_id] = fingerprint

    # `_enforce_study_state_caps` archives overflow as
    # {"section": "arms", "evicted": <arm>}. The bank still knows the recipe
    # of an arm evicted since its last prune, so use that fingerprint rather
    # than the kind/claim one, which could never match a parameter recipe.
    evicted = [
        entry["evicted"] for entry in journal
        if isinstance(entry, Mapping) and entry.get("section") == "arms"
        and isinstance(entry.get("evicted"), Mapping)
    ]
    for arm in with_hypothesis_metadata(evicted, bank):
        arm_id = str(arm.get("arm") or "")
        if arm_id:
            taken_ids.setdefault(arm_id, "journal.jsonl (evicted)")
        known_fps.setdefault(
            str(arm.get("fingerprint") or "") or retired_fps.get(arm_id)
            or _claim_fingerprint(arm), arm_id or "an evicted arm")

    live_arm_ids = _arm_ids(study_arms)
    live_hypotheses = {
        str(meta.get("hypothesis_id") or "")
        for arm_id, meta in (bank.get("arms") or {}).items()
        if arm_id in live_arm_ids
    }
    live_hypotheses.discard("")
    return taken_ids, known_fps, live_hypotheses


def _code_ref_errors(code_refs: Sequence[Mapping[str, Any]],
                     root: pathlib.Path) -> list[str]:
    reasons: list[str] = []
    for ref in code_refs:
        rel = ref["path"]
        try:
            resolved = (root / rel).resolve()
        except (OSError, RuntimeError) as exc:
            reasons.append(f"missing_code_ref:{rel}: cannot resolve ({exc})")
            continue
        try:
            resolved.relative_to(root)
        except ValueError:
            reasons.append(
                f"code_ref_outside_repo:{rel}: resolves outside repo_root")
            continue
        if not resolved.is_file():
            reasons.append(
                f"missing_code_ref:{rel}: no such file under repo_root")
    return reasons


def _duplicate_errors(
    arms: Sequence[Mapping[str, Any]],
    taken_ids: Mapping[str, str],
    known_fps: Mapping[str, str],
    batch_ids: Mapping[str, str],
    batch_fps: Mapping[str, str],
) -> list[str]:
    reasons: list[str] = []
    sibling_fps: dict[str, str] = {}
    for arm in arms:
        arm_id = arm["arm"]
        fingerprint = arm["fingerprint"]
        if arm_id in taken_ids:
            reasons.append(
                f"duplicate_arm_id:{arm_id}: already in {taken_ids[arm_id]}")
        elif arm_id in batch_ids:
            reasons.append(f"duplicate_arm_id:{arm_id}: admitted earlier in"
                           f" this call by {batch_ids[arm_id]}")
        for source in (known_fps, batch_fps, sibling_fps):
            if fingerprint in source:
                reasons.append(
                    f"duplicate_recipe:{source[fingerprint]}: arm {arm_id} has"
                    f" the same recipe (fingerprint {fingerprint})")
                break
        sibling_fps.setdefault(fingerprint, arm_id)
    return reasons


def _screen(
    hypotheses: Sequence[Any],
    *,
    taken_ids: Mapping[str, str],
    known_fps: Mapping[str, str],
    live_hypotheses: set[str],
    root: pathlib.Path | None,
    max_new_arms: int,
) -> dict[str, Any]:
    """Walk proposals in the model's order: admit, reject or defer each."""
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deferred: list[str] = []
    deferred_notes: list[str] = []
    echoes: list[dict[str, Any]] = []
    batch_ids: dict[str, str] = {}
    batch_fps: dict[str, str] = {}
    seen_hypothesis_ids: set[str] = set()
    remaining = max_new_arms
    for index, raw in enumerate(hypotheses):
        reasons = _hypothesis_errors(raw)
        codes = {_reason_code(reason) for reason in reasons}
        raw_id = raw.get("id") if isinstance(raw, Mapping) else None
        display_id = raw_id if isinstance(raw_id,
                                          str) and raw_id else f"#{index}"
        if isinstance(raw_id, str) and "invalid_id" not in codes:
            if raw_id in seen_hypothesis_ids:
                reasons.append(f"duplicate_hypothesis_id:{raw_id}: an earlier"
                               " hypothesis in this call has the same id")
            elif raw_id in live_hypotheses:
                reasons.append(
                    f"duplicate_hypothesis_id:{raw_id}: the bank holds a"
                    " hypothesis with this id whose arms are still in"
                    " study.json; use a new id")
            seen_hypothesis_ids.add(raw_id)

        arms: list[dict[str, Any]] = []
        if not codes & (_SHAPE_CODES | {"duplicate_variant_suffix"}):
            if root is not None:
                reasons.extend(
                    _code_ref_errors(raw.get("code_refs") or [], root))
            arms = _expand_arms(raw)
            reasons.extend(
                _duplicate_errors(arms, taken_ids, known_fps, batch_ids,
                                  batch_fps))
        if not codes & _SHAPE_CODES:
            echoes.append(_normalized(raw))

        if reasons:
            rejected.append({"hypothesis_id": display_id, "reasons": reasons})
        elif len(arms) > remaining:
            deferred.append(display_id)
            deferred_notes.append(f"{display_id} (needs {len(arms)} arm(s),"
                                  f" {remaining} left)")
        else:
            remaining -= len(arms)
            for arm in arms:
                batch_ids[arm["arm"]] = raw_id
                batch_fps.setdefault(arm["fingerprint"], arm["arm"])
            admitted.append({
                "hypothesis": _normalized(raw),
                "arms": arms,
                "warnings": [],
            })
    return {
        "admitted": admitted,
        "rejected": rejected,
        "deferred": deferred,
        "deferred_notes": deferred_notes,
        "hypotheses": echoes,
    }


def _study_arm(hypothesis: Mapping[str, Any],
               arm: Mapping[str, Any]) -> dict[str, Any]:
    """The study.json entry for one arm: study-round.json fields only."""
    entry: dict[str, Any] = {
        "arm": arm["arm"],
        "kind": hypothesis["kind"],
        "claim": arm["claim"],
        "status": "open",
    }
    for key in ("share", "headroom"):
        if key in hypothesis:
            entry[key] = float(hypothesis[key])
    return entry


def _gain_warning(hypothesis: Mapping[str, Any], entry: Mapping[str, Any],
                  min_expected_gain: float) -> str:
    gain = study.arm_expected_gain(entry)
    missing = [key for key in ("share", "headroom") if key not in hypothesis]
    if missing:
        if gain == 0.0:
            consequence = (
                "it ranks last for claim_arm and is among the first open arms"
                f" the {study.MAX_STUDY_ARMS}-arm cap evicts")
        else:
            consequence = "claim_arm and the arm cap rank it by that gain"
        return (f"no_gain_estimate: {' and '.join(missing)} missing, so"
                f" arm_expected_gain is {gain:.4f}; {consequence}. Add share"
                " and headroom from the profile to rank it on evidence.")
    if gain < min_expected_gain:
        return (f"low_expected_gain: arm_expected_gain {gain:.4f} is below"
                f" {min_expected_gain:.4f}; claim_arm takes it after, and the"
                " arm cap evicts it before, open arms with a higher gain")
    return ""


def _profile_warnings(hypothesis: Mapping[str, Any],
                      profile_summary: Mapping[str, Any] | None) -> list[str]:
    ops = [
        item["ref"] for item in hypothesis.get("evidence") or []
        if str(item.get("type") or "").strip().lower() == "hlo-op"
    ]
    if not ops or profile_summary is None:
        return []
    return _verify_against_profile_summary(ops,
                                           float(hypothesis.get("share", 0.0)),
                                           profile_summary)


def _gotchas(path: pathlib.Path, config: Mapping[str, Any] | None,
             tpu_version: str) -> list[str]:
    try:
        return list(
            arm_microbench.check_hardware_gotchas(
                path,
                config=dict(config) if config else None,
                tpu_version=tpu_version))
    except (ArithmeticError, TypeError, ValueError) as exc:
        # A malformed override in target_parameters (e.g. a zero
        # systolic_lane_dim) must not abort the call: report, don't veto.
        return [f"check_failed: {type(exc).__name__}: {exc}"]


def _hardware_warnings(hypothesis: Mapping[str, Any],
                       arms: Sequence[Mapping[str,
                                              Any]], root: pathlib.Path | None,
                       tpu_version: str) -> list[str]:
    """`arm_microbench.check_hardware_gotchas` findings, prefixed `hardware:`.

    Code checks run once per existing .py code_ref and name the file; config
    checks run once per arm on its merged target_parameters against
    os.devnull, which parses as an empty module.
    """
    found: list[str] = []
    if root is not None:
        for ref in hypothesis.get("code_refs") or []:
            path = (root / ref["path"]).resolve()
            if path.suffix == ".py" and path.is_file():
                for message in _gotchas(path, None, tpu_version):
                    found.append(f"hardware:{ref['path']}: {message}")
    for arm in arms:
        for message in _gotchas(pathlib.Path(os.devnull),
                                arm["target_parameters"], tpu_version):
            found.append(f"hardware: {message}")
    return list(dict.fromkeys(found))


def _ref_paths(code_refs: Any) -> set[str]:
    if not isinstance(code_refs, list):
        return set()
    return {
        posixpath.normpath(ref["path"])
        for ref in code_refs if isinstance(ref, Mapping)
        and isinstance(ref.get("path"), str) and ref["path"]
    }


def _add_overlap_warnings(admitted: Sequence[dict[str, Any]],
                          final_arms: Sequence[Any],
                          bank: Mapping[str, Any]) -> None:
    active = [
        arm for arm in final_arms if isinstance(arm, Mapping)
        and arm.get("status") in ACTIVE_ARM_STATUSES
    ]
    paths_by_arm = {
        str(arm.get("arm") or ""): _ref_paths(arm.get("code_refs"))
        for arm in with_hypothesis_metadata(active, bank)
    }
    for item in admitted:
        mine = _ref_paths(item["hypothesis"].get("code_refs"))
        own = {arm["arm"] for arm in item["arms"]}
        hits = [
            arm_id for arm_id, paths in paths_by_arm.items()
            if arm_id not in own and paths & mine
        ]
        if hits:
            shared = sorted(set().union(*(paths_by_arm[arm_id] & mine
                                          for arm_id in hits)))
            item["warnings"].append(
                f"overlaps_active_arm:{','.join(hits)}: code_refs {shared} are"
                " also targeted by these open or in_flight arms; parallel"
                " workers editing the same code may collide")


def _with_batch(bank: Mapping[str, Any], admitted: Sequence[dict[str, Any]],
                request_id: str) -> dict[str, Any]:
    """The bank plus this call's entries. Entries are replaced, never merged."""
    hypotheses = dict(bank["hypotheses"])
    arms = dict(bank["arms"])
    for item in admitted:
        hypothesis = item["hypothesis"]
        hypotheses[hypothesis["id"]] = {**hypothesis, "request_id": request_id}
        for arm in item["arms"]:
            meta: dict[str, Any] = {
                "hypothesis_id": hypothesis["id"],
                "fingerprint": arm["fingerprint"],
            }
            if arm["target_parameters"]:
                meta["target_parameters"] = copy.deepcopy(
                    arm["target_parameters"])
            arms[arm["arm"]] = meta
    return {
        "hypotheses": hypotheses,
        "arms": arms,
        "retired": list(bank["retired"])
    }


def _pruned_bank(bank: Mapping[str, Any],
                 live_arm_ids: set[str]) -> dict[str, Any]:
    """Retire arms that left study.json; drop hypotheses with none left."""
    arms: dict[str, Any] = {}
    retired = list(bank["retired"])
    for arm_id, meta in bank["arms"].items():
        if arm_id in live_arm_ids:
            arms[arm_id] = meta
            continue
        fingerprint = str(meta.get("fingerprint") or "")
        if not fingerprint:
            continue
        retired = [
            entry for entry in retired
            if (entry.get("arm"), entry.get("fingerprint")) != (arm_id,
                                                                fingerprint)
        ]
        retired.append({
            "arm": arm_id,
            "hypothesis_id": str(meta.get("hypothesis_id") or ""),
            "fingerprint": fingerprint,
        })
    live_hypotheses = {
        str(meta.get("hypothesis_id") or "")
        for meta in arms.values()
    }
    return {
        "hypotheses": {
            hypothesis_id: entry
            for hypothesis_id, entry in bank["hypotheses"].items()
            if hypothesis_id in live_hypotheses
        },
        "arms": arms,
        "retired": retired[-DEFAULT_RETIRED_LIMIT:],
    }


def _summary(admitted: Sequence[dict[str, Any]], rejected: Sequence[Any],
             deferred: Sequence[str], deferred_notes: Sequence[str],
             active_count: int, dry_run: bool) -> str:
    arm_count = sum(len(item["arms"]) for item in admitted)
    parts = [
        f"Admitted {len(admitted)} hypothesis(es) as {arm_count} arm(s);"
        f" rejected {len(rejected)}; deferred {len(deferred)}."
    ]
    if deferred_notes:
        parts.append("Deferred over the --max-new-arms budget, resubmit on the"
                     " next call: " + "; ".join(deferred_notes) + ".")
    parts.append(f"Open or in-flight arms: {active_count}.")
    if dry_run:
        parts.append("Dry run: nothing was written, and eviction by the"
                     f" {study.MAX_STUDY_ARMS}-arm cap was not simulated.")
    return " ".join(parts)


def admit_hypotheses(
    study_id: str,
    hypotheses: Sequence[Any],
    *,
    run_id: str = "",
    repo_root: str | pathlib.Path | None = None,
    tpu_version: str = arm_microbench.DEFAULT_TPU_VERSION,
    max_new_arms: int = DEFAULT_MAX_NEW_ARMS,
    min_expected_gain: float = DEFAULT_MIN_EXPECTED_GAIN,
    journal_dedup_limit: int = DEFAULT_JOURNAL_DEDUP_LIMIT,
    profile_summary: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Record the model's hypotheses as open study arms; see the module doc.

    Returns a hypothesis-bank-result.json document: `completed`, or
    `invalid_input` when the call itself is unusable (bad study_id, run_id,
    tpu_version, limits, repo_root or payload types). Infrastructure failures
    -- a study that was never started, an unwritable store -- raise
    `study.TpuAgentSuiteError` or `OSError`; `main` reports those as `failed`.
    """
    request_id = _request_id(run_id)
    problem = _input_problem(
        study_id,
        hypotheses,
        run_id=run_id,
        repo_root=repo_root,
        tpu_version=tpu_version,
        max_new_arms=max_new_arms,
        min_expected_gain=min_expected_gain,
        journal_dedup_limit=journal_dedup_limit,
        profile_summary=profile_summary,
    )
    if problem:
        return _empty_result("invalid_input", study_id, request_id, problem)
    root = pathlib.Path(repo_root).resolve() if repo_root is not None else None

    with study.study_lock(study_id):
        doc = study.load_study(study_id)
        bank = load_hypothesis_bank(study_id)
        journal = study.read_study_journal(study_id, limit=journal_dedup_limit)
        taken_ids, known_fps, live_hypotheses = _dedup_ledger(
            doc, bank, journal)
        screened = _screen(
            hypotheses,
            taken_ids=taken_ids,
            known_fps=known_fps,
            live_hypotheses=live_hypotheses,
            root=root,
            max_new_arms=max_new_arms,
        )
        admitted: list[dict[str, Any]] = screened["admitted"]
        rejected: list[dict[str, Any]] = screened["rejected"]

        arm_schema = study.load_study_round_schema(
        )["properties"]["arms"]["items"]
        for item in admitted:
            hypothesis = item["hypothesis"]
            item["entries"] = [
                _study_arm(hypothesis, arm) for arm in item["arms"]
            ]
            for entry in item["entries"]:
                errors = study.validate_instance(entry, arm_schema)
                if errors:
                    raise study.TpuAgentSuiteError(
                        f"arm {entry['arm']!r} does not match the study-round"
                        " arm schema:\n  " + "\n  ".join(errors))
            gain_warning = _gain_warning(hypothesis, item["entries"][0],
                                         min_expected_gain)
            item["warnings"] = (
                ([gain_warning] if gain_warning else []) +
                _profile_warnings(hypothesis, profile_summary) +
                _hardware_warnings(hypothesis, item["arms"], root,
                                   tpu_version))

        bank_view = _with_batch(bank, admitted, request_id)
        if dry_run:
            final_arms = list(doc.get("arms") or []) + [
                entry for item in admitted for entry in item["entries"]
            ]
        else:
            written = bank
            if admitted:
                # Bank first: if the process dies before save_study, the
                # orphan entries are inert (no study arm references them),
                # whereas the reverse order would leave open arms without
                # their recipe.
                study.jsonio.write_json(hypothesis_bank_path(study_id),
                                        bank_view)
                written = bank_view
                doc.setdefault("arms", []).extend(entry for item in admitted
                                                  for entry in item["entries"])
                doc = study.save_study(doc)
                kept = _arm_ids(doc.get("arms"))
                evicted = [
                    item for item in admitted
                    if any(arm["arm"] not in kept for arm in item["arms"])
                ]
                if evicted:
                    # The arm cap evicted part of this call on admission.
                    # A hypothesis is all-or-nothing, so withdraw any arm of
                    # an evicted hypothesis that survived.
                    withdraw = {
                        arm["arm"]
                        for item in evicted
                        for arm in item["arms"]
                    } & kept
                    if withdraw:
                        doc["arms"] = [
                            arm for arm in doc["arms"]
                            if arm.get("arm") not in withdraw
                        ]
                        doc = study.save_study(doc)
                    for item in evicted:
                        admitted.remove(item)
                        lost = [
                            arm["arm"] for arm in item["arms"]
                            if arm["arm"] not in kept
                        ]
                        rejected.append({
                            "hypothesis_id":
                            item["hypothesis"]["id"],
                            "reasons": [
                                f"evicted_by_arm_cap: study.json holds at most"
                                f" {study.MAX_STUDY_ARMS} arms and the cap"
                                f" evicted {', '.join(lost)} on admission"
                                " (open arms with the lowest"
                                " arm_expected_gain go first); none of this"
                                " hypothesis's arms were kept and its recipes"
                                " are retired"
                            ],
                        })
            pruned = _pruned_bank(written, _arm_ids(doc.get("arms")))
            if pruned != written:
                study.jsonio.write_json(hypothesis_bank_path(study_id), pruned)
            if admitted:
                study.append_study_event(
                    study_id,
                    "hypotheses_admitted",
                    run_id=request_id,
                    hypothesis_ids=[
                        item["hypothesis"]["id"] for item in admitted
                    ],
                    arm_ids=[
                        arm["arm"] for item in admitted for arm in item["arms"]
                    ],
                    admitted_count=len(admitted),
                    rejected_count=len(rejected),
                    deferred_count=len(screened["deferred"]),
                )
            final_arms = list(doc.get("arms") or [])

        _add_overlap_warnings(admitted, final_arms, bank_view)
        active_count = sum(1 for arm in final_arms if isinstance(arm, Mapping)
                           and arm.get("status") in ACTIVE_ARM_STATUSES)
        return {
            "status":
            "completed",
            "study_id":
            study_id,
            "request_id":
            request_id,
            "admitted": [{
                "hypothesis_id":
                item["hypothesis"]["id"],
                "arms": [{
                    "arm": arm["arm"],
                    "fingerprint": arm["fingerprint"]
                } for arm in item["arms"]],
                "warnings":
                item["warnings"],
            } for item in admitted],
            "rejected":
            rejected,
            "deferred":
            screened["deferred"],
            "open_arms":
            active_count,
            "hypotheses":
            screened["hypotheses"],
            "summary":
            _summary(admitted, rejected, screened["deferred"],
                     screened["deferred_notes"], active_count, dry_run),
        }


class _UsageError(Exception):
    """An argparse usage error, reported as an invalid_input result."""


class _ArgumentParser(argparse.ArgumentParser):

    def error(self, message: str) -> None:  # type: ignore[override]
        raise _UsageError(message)


def _read_json_arg(value: str) -> Any:
    """Inline JSON when it looks like JSON, else the JSON file it names.

    The inline check comes first: `Path.is_file()` on a long JSON string can
    raise ENAMETOOLONG instead of returning False.
    """
    text = value.strip()
    if text.startswith(("{", "[")):
        return json.loads(text)
    path = pathlib.Path(text)
    if not path.is_file():
        raise ValueError(f"{value!r} is neither inline JSON nor a JSON file")
    return json.loads(path.read_text(encoding="utf-8"))


def _emit(result: Mapping[str, Any]) -> int:
    print(json.dumps(result, indent=2))
    return _EXIT_CODES[result["status"]]


def main(argv: list[str] | None = None) -> int:
    parser = _ArgumentParser(description=(
        "Record the researcher's hypotheses as open arms of a study."
        " Rejects only on objective checks; every judgment comes back as"
        " a warning. Prints a hypothesis-bank-result.json document."))
    parser.add_argument("--study-id",
                        required=True,
                        help="Study slug (lower-case hyphen-case).")
    parser.add_argument(
        "--run-id",
        default="",
        help="The researcher's session run_id, echoed as request_id and"
        " recorded on each bank entry. Generated when omitted.")
    parser.add_argument(
        "--hypotheses-json",
        required=True,
        help="A JSON file, or inline JSON: {\"hypotheses\": [...]} or a bare"
        " array of hypotheses.")
    parser.add_argument(
        "--profile-summary-json",
        default=None,
        help="Optional XProf summary (file or inline JSON). hlo-op evidence"
        " and share are checked against it; mismatches are warnings.")
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Root that code_refs paths are relative to. When given, a"
        " code_ref that escapes it or does not exist is rejected, and .py"
        " code_refs get hardware gotcha warnings.")
    parser.add_argument("--tpu-version",
                        default=arm_microbench.DEFAULT_TPU_VERSION,
                        help="TPU generation for hardware warnings, one of"
                        f" {', '.join(SUPPORTED_TPU_VERSIONS)}. Default:"
                        f" {arm_microbench.DEFAULT_TPU_VERSION}.")
    parser.add_argument(
        "--max-new-arms",
        type=int,
        default=DEFAULT_MAX_NEW_ARMS,
        help="Maximum arms this call admits, walking hypotheses in the given"
        f" order; a hypothesis that does not fit is deferred. Default:"
        f" {DEFAULT_MAX_NEW_ARMS}.")
    parser.add_argument(
        "--min-expected-gain",
        type=float,
        default=DEFAULT_MIN_EXPECTED_GAIN,
        help="An arm_expected_gain below this adds a low_expected_gain"
        f" warning; it never rejects. Default: {DEFAULT_MIN_EXPECTED_GAIN}.")
    parser.add_argument(
        "--journal-dedup-limit",
        type=int,
        default=DEFAULT_JOURNAL_DEDUP_LIMIT,
        help="Journal tail records read for arm id and recipe dedup."
        f" Default: {DEFAULT_JOURNAL_DEDUP_LIMIT}.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run every check and report the outcome without writing"
        " study.json or the bank.")
    try:
        args = parser.parse_args(argv)
    except _UsageError as exc:
        parser.print_usage(sys.stderr)
        return _emit(
            _empty_result("invalid_input", "", _request_id(""),
                          f"usage: {exc}"))

    request_id = _request_id(args.run_id)
    try:
        payload = _read_json_arg(args.hypotheses_json)
        profile_summary = (_read_json_arg(args.profile_summary_json)
                           if args.profile_summary_json else None)
    except (OSError, ValueError) as exc:
        return _emit(
            _empty_result("invalid_input", args.study_id, request_id,
                          f"cannot read input: {exc}"))
    if isinstance(payload, Mapping) and "hypotheses" not in payload:
        return _emit(
            _empty_result(
                "invalid_input", args.study_id, request_id,
                "expected {\"hypotheses\": [...]} or a bare array; got an"
                f" object with keys {sorted(payload)}"))
    hypotheses = payload.get("hypotheses") if isinstance(payload,
                                                         Mapping) else payload

    try:
        result = admit_hypotheses(
            args.study_id,
            hypotheses,
            run_id=args.run_id,
            repo_root=args.repo_root,
            tpu_version=args.tpu_version,
            max_new_arms=args.max_new_arms,
            min_expected_gain=args.min_expected_gain,
            journal_dedup_limit=args.journal_dedup_limit,
            profile_summary=profile_summary,
            dry_run=args.dry_run,
        )
    except (study.TpuAgentSuiteError, OSError) as exc:
        result = _empty_result("failed", args.study_id, request_id,
                               f"{type(exc).__name__}: {exc}")
    return _emit(result)


if __name__ == "__main__":
    sys.exit(main())

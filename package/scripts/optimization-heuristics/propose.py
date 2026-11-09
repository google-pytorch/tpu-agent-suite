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
"""Turns analyzer findings into at most N ranked candidates to measure.

    python3 propose.py --findings <work>/findings.json \\
        --deployments <inference_work>/deployments.json \\
        --launch-recipe <inference_work>/launch_vllm.sh \\
        --points 1024:1024:128,8192:1024:128 --image <digest> \\
        --hf-config <model>/config.json --hbm-gib-per-device 94.75 \\
        [--knobs-present <work>/knobs_present.json] --max 3 \\
        --out <work>/candidates.json --changes-dir <work>/changes

Two kinds of candidate, each tested alone against the baseline:

- knob candidates: one per finding with a knob change the server has not
  set (env vars, vLLM flags), on the baseline mesh;
- mesh candidates: TP / DP layouts from `oss_meshes.enumerate_oss_meshes`,
  which argues for a mesh change only when the trace shows it (a
  collective-bound prefill, D7) and scores each layout at every benchmark
  point.

Each candidate gets a status: `workload_dependent` for a mesh whose
benchmark points disagree, else `unmeasured`. Every candidate is measured
on this deployment; no earlier result on another model keeps one from being
tried. Knobs the serving image does not have (`--knobs-present`, from
`knob_probe.py`) are reported, never launched.
Ranking and selection are fixed rules; nothing is chosen by a model.

Each candidate carries `serving_change`: the `env`, `vllm_args`, `tp`, `dp`,
`max_num_seqs` and `patches` of the server that runs it, already merged with
the baseline. With `--changes-dir`, each also gets
`<changes-dir>/<cand_id>/deployments.json`: the baseline manifest with the
change applied, which tpu-model-onboarding launches the way it launched the
baseline (its `env` and `vllm_patches` go into the launch recipe).
"""

import argparse
import json
import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
# serving_analysis sits beside this script; oss_meshes in scripts/common.
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "common"))

# pylint: disable=g-import-not-at-top,wrong-import-position
import deployment  # noqa: E402
import oss_meshes  # noqa: E402
from serving_analysis import composer  # noqa: E402
from serving_analysis import knobs  # noqa: E402

# Candidate status: a mesh whose benchmark points disagree, or neither.
TIERS = ("workload_dependent", "unmeasured")
# Baseline fields that are vLLM flags.
_FIELD_FLAGS = deployment.FIELD_FLAGS


def image_digest(image: str | None) -> str | None:
    if not image:
        return None
    return image.split("@", 1)[1] if "@" in image else image


def bench_points(bench: dict[str, Any]) -> list[dict[str, int]]:
    return [{
        "isl": i,
        "osl": o,
        "c": c
    } for i, o, c in bench.get("points") or []]


def _label(point: tuple[int, int, int]) -> str:
    isl, osl, c = point
    return oss_meshes.Point.from_dict({"isl": isl, "osl": osl, "c": c}).label


_PREFILL_PHASES = ("prefill", "prefill_or_mixed", "prefill_heavy", "mixed")


def _phase_kind(phase: str | None) -> str | None:
    """`prefill`, `decode`, `all`, or None for a finding's phase label."""
    if not phase:
        return None
    if phase == "decode" or phase.startswith("decode"):
        return "decode"
    if phase in _PREFILL_PHASES or phase.startswith("prefill"):
        return "prefill"
    return "all" if phase == "all" else None


def phase_mix(findings: dict[str, Any],
              point: tuple[int, int, int],
              dp: int = 1) -> dict[str, Any]:
    """Where the time goes at one benchmark point: prefill vs decode.

    The phased captures time each phase in its own window, so their step
    counts say nothing about the mix. The mix comes from the request shape
    instead: per engine, prefill runs ISL x C / DP tokens in steps of the
    captured prefill bucket, and decode runs OSL steps.
    """
    isl, osl, c = point
    pre = dec = None
    for prof in findings.get("profiles") or []:
        if tuple(prof.get("point") or ()) != tuple(point):
            continue
        for st in (prof.get("result") or {}).get("phase_stats") or []:
            kind = _phase_kind(st.get("phase"))
            dur, bucket = st.get("mean_dur_ms"), st.get("mean_bucket")
            if not dur:
                continue
            if kind == "decode" and prof.get("role") == "decode":
                dec = dur
            elif kind == "prefill" and prof.get(
                    "role") == "prefill" and bucket:
                pre = (dur, bucket)
    if pre is None or dec is None:
        phase = "prefill" if isl >= 2 * osl else "decode"
        return {
            "phase": phase,
            "prefill_share": None,
            "basis": "input/output lengths (no captured step times)"
        }
    prefill_ms = isl * c / max(dp, 1) / pre[1] * pre[0]
    decode_ms = osl * dec
    share = prefill_ms / (prefill_ms + decode_ms)
    return {
        "phase": "prefill" if share >= 0.5 else "decode",
        "prefill_share": round(share, 3),
        "basis": "captured step times at this point"
    }


def _finding_phases(findings: dict[str, Any],
                    points: list[tuple[int, int, int]]) -> dict[str, set]:
    """Phase kinds each detector fired in, at the given points."""
    out: dict[str, set] = {}
    for prof in findings.get("profiles") or []:
        if points and tuple(prof.get("point") or ()) not in points:
            continue
        for f in (prof.get("result") or {}).get("findings") or []:
            kind = _phase_kind(f.get("phase"))
            if kind:
                out.setdefault(f.get("detector_id"), set()).add(kind)
    return out


def _set_flag(args: list[str], flag: str, value: str | None) -> list[str]:
    """`args` with `flag` set to `value` (None: a bare flag), once."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == flag or a.startswith(flag + "="):
            skip = (a == flag and i + 1 < len(args)
                    and not args[i + 1].startswith("--"))
            i += 2 if skip else 1
            continue
        out.append(a)
        i += 1
    return out + ([flag] if value is None else [flag, value])


def _drop_flag(args: list[str], flag: str) -> list[str]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a == flag or a.startswith(flag + "="):
            skip = (a == flag and i + 1 < len(args)
                    and not args[i + 1].startswith("--"))
            i += 2 if skip else 1
            continue
        out.append(a)
        i += 1
    return out


def serving_change(c: composer.Candidate, bench: dict[str,
                                                      Any]) -> dict[str, Any]:
    """The server fields that run candidate `c`, merged with the baseline."""
    env = dict(bench.get("env") or {})
    env.update(c.env_set)
    args = [str(a) for a in bench.get("vllm_args") or []]
    fields: dict[str, Any] = {}
    for flag, value in c.argv_set.items():
        if flag in _FIELD_FLAGS:
            fields[_FIELD_FLAGS[flag]] = int(value)
        else:
            args = _set_flag(args, flag, str(value))
    for flag in c.argv_add_flags:
        args = _set_flag(args, flag, None)
    for flag in c.argv_remove_flags:
        args = _drop_flag(args, flag)
    if c.additional_config_set:
        args = _set_flag(args, "--additional-config",
                         json.dumps(c.additional_config_set, sort_keys=True))
    if c.mesh:
        fields.setdefault("tp", int(c.mesh.get("tp", bench.get("tp", 1))))
        fields.setdefault("dp", int(c.mesh.get("dp", bench.get("dp", 1))))
    out = {
        "env": env,
        "vllm_args": args,
        "patches": list(bench.get("patches") or [])
    }
    for key in ("tp", "dp", "max_num_seqs", "max_model_len"):
        if key in fields or bench.get(key) is not None:
            out[key] = fields.get(key, bench.get(key))
    return out


def _patch_for(name: str) -> str:
    """The package patch that adds knob `name`, or ''."""
    for k in knobs.REGISTRY.values():
        if k.name == name and k.requires_patch.endswith(".patch"):
            return k.requires_patch
    return ""


def select_kinds(
        ranked: list[tuple[int, float, dict[str, Any]]],
        max_candidates: int) -> list[tuple[int, float, dict[str, Any]]]:
    """`ranked` reordered so the first `max_candidates` hold the best of each
    kind (mesh, knob) when both exist: a run of mesh candidates cannot crowd
    out every serving setting, or the reverse."""
    ranked = list(ranked)
    for kind in ("mesh", "knob"):
        top = [r for r in ranked[:max_candidates] if r[2]["kind"] == kind]
        rest = [r for r in ranked[max_candidates:] if r[2]["kind"] == kind]
        if max_candidates > 1 and not top and rest:
            others = ranked[:max_candidates]
            last = max(i for i, r in enumerate(others) if sum(
                o[2]["kind"] == r[2]["kind"] for o in others) > 1)
            ranked.remove(rest[0])
            ranked.insert(last, rest[0])
    return ranked


def _knobs_absent(c: composer.Candidate,
                  probe: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    """(knobs the image lacks and no patch adds, patches that add the rest).

    A knob the image lacks is still measurable when a package patch adds it
    and the probe did not find that the patch fails to apply.
    """
    if probe is None:
        return [], []
    present = probe.get("present") or {}
    status = probe.get("patches") or {}
    names = list(c.env_set) + [
        f for f in list(c.argv_set) + c.argv_add_flags if f not in _FIELD_FLAGS
    ]
    absent, patches = [], []
    for n in names:
        if present.get(n) is not False:
            continue
        patch = _patch_for(n)
        if patch and status.get(os.path.basename(patch)) != "fails":
            if patch not in patches:
                patches.append(patch)
        else:
            absent.append(n)
    return absent, patches


def propose(findings: dict[str, Any],
            bench: dict[str, Any],
            hf_config: dict[str, Any],
            hbm_gib_per_device: float,
            *,
            devices: int | None = None,
            knobs_present: dict[str, Any] | None = None,
            max_candidates: int = 3,
            target_points: list[tuple[int, int, int]] | None = None,
            target_phase: str | None = None) -> dict[str, Any]:
    """The candidates.json payload.

    `target_points` default to the points the profiles were captured at (each
    regime's peak unless the user named traffic), else every point. Candidates whose findings fired in the target
    phase rank first; with no `target_phase`, it is the phase that dominates
    the target points' time (`phase_mix`).
    """
    tp, dp = int(bench.get("tp") or 1), int(bench.get("dp") or 1)
    devices = devices or tp * dp
    image = image_digest(bench.get("image"))
    points = bench_points(bench)
    meshes = oss_meshes.enumerate_oss_meshes(hf_config,
                                             devices=devices,
                                             hbm_gib=hbm_gib_per_device,
                                             framework="vllm_torchtpu",
                                             image=image,
                                             baseline={
                                                 "tp": tp,
                                                 "dp": dp
                                             },
                                             serving=findings,
                                             points=points or None)
    labels = [oss_meshes.Point.from_dict(p).label for p in points]
    plan = composer.compose_plan(findings,
                                 target_points=labels,
                                 oss_meshes=meshes,
                                 baseline_mesh={
                                     "tp": tp,
                                     "dp": dp
                                 })
    # Default targets: the points the profiles were captured at, which the
    # round takes at each regime's peak unless the user named traffic.
    captured = []
    for prof in findings.get("profiles") or []:
        pt = tuple(prof.get("point") or ())
        if len(pt) == 3 and pt not in captured:
            captured.append(pt)
    targets = [tuple(p) for p in target_points or []
               ] or captured or [tuple(p) for p in bench.get("points") or []]
    mixes = {_label(t): phase_mix(findings, t, dp) for t in targets}
    if target_phase:
        phases, source = {target_phase}, "requested"
    else:
        # Each target point's own dominant phase: a decode-heavy and a
        # prefill-heavy target both count, rather than one outvoting the other.
        phases = {m["phase"] for m in mixes.values() if m.get("phase")}
        source = "dominant at each target point"
    phase = (next(iter(phases))
             if len(phases) == 1 else "mixed" if phases else None)
    fired = _finding_phases(findings, targets)
    ranked, reported, seen = [], [], []
    for c in plan.candidates:
        tier = ("workload_dependent" if c.mesh is not None and not c.env_set
                and c.workload_dependent else "unmeasured")
        entry = c.to_json(0)
        entry["status"] = tier
        entry["kind"] = "mesh" if c.mesh is not None else "knob"
        entry["serving_change"] = serving_change(c, bench)
        # One detector can fire at several target points with different
        # values (D45 / D46 block sizes per context length): give each its
        # own id so its manifest is not overwritten, and drop exact repeats.
        twin = next((e for e in seen if e["cand_id"] == entry["cand_id"]),
                    None)
        if twin is not None:
            if twin["serving_change"] == entry["serving_change"]:
                entry["reason"] = f"same change as {twin['cand_id']}"
                reported.append(entry)
                continue
            n = sum(e["cand_id"].startswith(entry["cand_id"]) for e in seen)
            entry["cand_id"] = f"{entry['cand_id']}-{n + 1}"
        seen.append(entry)
        cand_phases = set().union(*[
            fired.get(fid, set())
            for fid in (entry.get("provenance") or {}).get("findings") or []
        ] or [set()])
        entry["target_phase_match"] = bool(
            phases and (phases & cand_phases or "all" in cand_phases))
        absent, patches = _knobs_absent(c, knobs_present)
        if patches and not absent:
            entry["serving_change"]["patches"] += [
                p for p in patches
                if p not in entry["serving_change"]["patches"]
            ]
            entry["needs_patches"] = patches
        if absent:
            entry["reason"] = f"not in the serving image: {', '.join(absent)}"
            reported.append(entry)
        else:
            ranked.append((TIERS.index(tier), -c.score, entry))
    # Candidates for the target phase first; then evidence tier and score.
    ranked.sort(key=lambda x: (not x[2]["target_phase_match"], x[0], x[1]))
    ranked = select_kinds(ranked, max_candidates)
    chosen = [e for _, _, e in ranked[:max_candidates]]
    for rank, e in enumerate(chosen, 1):
        e["rank"] = rank
    for _, _, e in ranked[max_candidates:]:
        e["reason"] = f"beyond the first {max_candidates} candidates"
        reported.append(e)
    return {
        "baseline": {
            "tp": tp,
            "dp": dp,
            "devices": devices,
            "image": image,
            "points": labels,
        },
        "target": {
            "points": [_label(t) for t in targets],
            "phase": phase,
            "phase_source": source,
            "phase_mix": mixes,
        },
        "candidates":
        chosen,
        "reported_only":
        reported,
        "rejected_meshes": [{
            "label": m["label"],
            "rejected_by": m["rejected_by"]
        } for m in meshes["meshes"] if not m["feasible"]],
        "rejected":
        plan.rejected,
        "notes":
        plan.notes,
        "prefill_collective_bound":
        meshes["prefill_collective_bound"],
        "kv_head_padding":
        meshes.get("kv_head_padding"),
        "not_argued_for":
        meshes.get("not_argued_for", []),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--findings", required=True)
    parser.add_argument("--deployments",
                        required=True,
                        help="The baseline's deployments.json")
    parser.add_argument("--launch-recipe",
                        help="The baseline's launch_vllm.sh")
    parser.add_argument("--points",
                        required=True,
                        help="ISL:OSL:C,... the baseline was measured at")
    parser.add_argument("--image", help="The serving image's digest")
    parser.add_argument("--hf-config", required=True)
    parser.add_argument("--hbm-gib-per-device", type=float, required=True)
    parser.add_argument("--devices",
                        type=int,
                        help="Devices one deployment gets (default TP x DP)")
    parser.add_argument("--knobs-present",
                        help="knob_probe.py output for the serving image")
    parser.add_argument("--target-points",
                        help="ISL:OSL:C,... to optimize (default: each"
                        " regime's peak point, else every point)")
    parser.add_argument("--target-phase",
                        choices=("prefill", "decode"),
                        help="Phase to optimize (default: the one that"
                        " dominates the target points)")
    parser.add_argument("--max", type=int, default=3)
    parser.add_argument("--out", required=True)
    parser.add_argument("--changes-dir",
                        help="Also write each candidate's deployments.json"
                        " under <dir>/<cand_id>/")
    args = parser.parse_args(argv)
    loaded = []
    for path in (args.findings, args.hf_config):
        with open(path, encoding="utf-8") as f:
            loaded.append(json.load(f))
    findings, hf_config = loaded
    recipe = None
    if args.launch_recipe:
        with open(args.launch_recipe, encoding="utf-8") as f:
            recipe = f.read()
    dep = deployment.load(args.deployments)
    points = [
        tuple(int(x) for x in p.split(":")) for p in args.points.split(",")
        if p.strip()
    ]
    bench = deployment.baseline(dep, recipe, args.image, points)
    present = None
    if args.knobs_present:
        with open(args.knobs_present, encoding="utf-8") as f:
            present = json.load(f)
    payload = propose(
        findings,
        bench,
        hf_config,
        args.hbm_gib_per_device,
        devices=args.devices,
        knobs_present=present,
        max_candidates=args.max,
        target_points=[
            tuple(int(x) for x in p.split(":"))
            for p in (args.target_points or "").split(",") if p.strip()
        ] or None,
        target_phase=args.target_phase)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, default=str)
    if args.changes_dir:
        for e in payload["candidates"]:
            d = os.path.join(args.changes_dir, str(e["cand_id"]))
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, "deployments.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "deployments": [
                            deployment.apply_change(dep, e["serving_change"],
                                                    bench)
                        ]
                    },
                    f,
                    indent=2)
            e["deployments_json"] = path
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1, default=str)
    for e in payload["candidates"]:
        print(f"#{e['rank']} {e['cand_id']} [{e['kind']}, {e['status']}]")
    for e in payload["reported_only"]:
        print(f"  reported: {e['cand_id']} ({e['reason']})")
    if not payload["candidates"]:
        print("No candidate to launch: keep the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
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
"""workload-microbenchmark CLI.

  cli.py [--work DIR] [--jobs DIR] <command> ...
                                                          --work: configs, plans, new models
                                                          (default ./.work); --jobs: fetched job
                                                          directories (default <work>/jobs)
  cli.py onboard  <MODEL_DIR> --job <id> [--name N]      reference config (reviewed, merged)
  cli.py derive   <MODEL_DIR> --job <id> [--override k=v]... [--name N]
                                                          working config under <work>/
  cli.py project  <config.json> [--twin-world 8]          real workload -> twin config for a small slice
  cli.py show     <config.json>                           human-readable summary
  cli.py selftest                                         re-derive every reference, check bars
  cli.py points   <twin.json> [--override dcp=8]... [--min-share F] [--point NAME=c/i/o@C]... [--combined]
                  [--top K]                               the job's step mix -> one self-contained .steps.json
                                                          per cell (parallel jobs), or one with --combined;
                                                          --top K: the K heaviest cells, one job, all profiled
  cli.py mix-rollup --plans <dir|top.steps.json> <job-id>... [--baseline <ids>]
                                                          the cost model: T_mix, spread; with --baseline the
                                                          speedup, class deltas, guardrails -> cost_model.json
  cli.py forward-cmd --steps <f.steps.json> [--no-profile]   recipe + `compute submit` command for a
                                                          plan (profile on by default)
  cli.py collect  <job-id> --steps <f.steps.json>        after `compute job artifacts`: cache, times,
                                                          profile split (from tpu-diagnostics's export),
                                                          evidence.json (the facts REPORT.md is written from)
  cli.py cache-pull  <job-id>                            file its compile cache under ~/.xla_cache
  cli.py run      ...                                     L1 A/B on 2x2x1 (M1+, not built yet)

Run as `python3 <this package's workload-microbenchmark scripts>/core/cli.py ...`, from the
workspace: relative --work / --jobs paths are taken from the current directory.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
sys.path.insert(0, SCRIPTS)

from core import cachelocal  # noqa: E402
from core import jobdata  # noqa: E402
from core import jobdata_agentx  # noqa: E402
from core import report  # noqa: E402
from core import derive as derive_mod  # noqa: E402
from core import mix as mix_mod  # noqa: E402
from core import project as project_mod  # noqa: E402

#: the work directory (`--work`): working configs, snapshots, plans, models being onboarded
WORK = os.path.join(os.getcwd(), ".work")


def _rel(path: str) -> str:
    """A path as printed: relative to the current directory when it is under it."""
    ap = os.path.abspath(path)
    return os.path.relpath(ap) if ap.startswith(os.getcwd() + os.sep) else ap


def _commit() -> str | None:
    """The harness's own commit, when the scripts run from a git checkout (None when installed)."""
    try:
        r = subprocess.run(
            ["git", "-C", SCRIPTS, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True)
    except OSError:
        return None
    return r.stdout.strip() or None


def _slice(world: int) -> str:
    return {
        4: "2x2x1-4c",
        8: "2x2x1",
        16: "2x2x2",
        32: "2x2x4"
    }.get(world, f"{world}cores")


def _name(cfg: dict, overrides: dict) -> str:
    d = cfg["deploy"]
    world = d["tp"] * d["dp"] * d["pp"] * d["pcp"]
    lab = f"tp{d['tp']}" + (f"-dcp{d['dcp']}" if d["dcp"] > 1 else "") + \
          (f"-dp{d['dp']}" if d["dp"] > 1 else "") + ("-ep" if d["ep"] else "")
    base = f"{cfg.get('workload', 'bench')}-{_slice(world)}-{lab}"
    if overrides:
        base += "+" + ",".join(f"{k}={v}"
                               for k, v in sorted(overrides.items()))
    return base


def _overrides(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--override expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _header(cmd: str, snap_path: str) -> dict:
    return {
        "generated_by":
        cmd,
        "generated_at":
        datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "harness_commit":
        _commit(),
        "snapshot":
        snap_path
    }


def _snapshot(job: str, model_dir: str, cache_dir: str) -> tuple[dict, str]:
    """Reuse a checked-in snapshot for this job if one exists, else read the job."""
    mdir = derive_mod.model_path(model_dir)
    for p in glob.glob(os.path.join(mdir, "config", "*.source.json")):
        s = json.load(open(p))
        if s.get("job_id") == job:
            return s, os.path.relpath(p, os.path.dirname(mdir))
    snap = jobdata_agentx.snapshot(job)
    os.makedirs(cache_dir, exist_ok=True)
    p = os.path.join(cache_dir, f"{job}.source.json")
    with open(p, "w") as fh:
        fh.write(jobdata.dumps_snapshot(snap))
    return snap, p


def cmd_onboard(a) -> int:
    mdir = derive_mod.model_path(a.model_dir)
    if not os.path.exists(os.path.join(mdir, "model.py")):
        print(
            f"! {mdir}/model.py missing. Write it first (copy an existing model's "
            f"model.py as the template), then re-run onboard.",
            file=sys.stderr)
        return 2
    snap = jobdata_agentx.snapshot(a.job)
    cfg = derive_mod.derive(snap, a.model_dir)
    name = a.name or _name(cfg, {})
    cdir = os.path.join(mdir, "config")
    os.makedirs(cdir, exist_ok=True)
    src = os.path.join(cdir, f"{name}.source.json")
    with open(src, "w") as fh:
        fh.write(jobdata.dumps_snapshot(snap))
    cfg = {
        "_header":
        _header(f"cli.py onboard {a.model_dir} --job {a.job}",
                os.path.relpath(src, os.path.dirname(mdir))),
        **cfg
    }
    json.dump(cfg, open(os.path.join(cdir, f"{name}.json"), "w"), indent=1)
    tw = _project(cfg, snap, a.model_dir, a.twin_world)
    twp = os.path.join(cdir, f"{name}.twin{tw['twin_world']}.json")
    json.dump(tw, open(twp, "w"), indent=1)
    bars = report.bars(cfg, snap, a.model_dir)
    bars["twin"] = {
        "twin_world": tw["twin_world"],
        "expect": report.twin_bars(tw)
    }
    json.dump(bars,
              open(os.path.join(cdir, f"{name}.bars.json"), "w"),
              indent=1)
    ex = os.path.join(mdir, "examples")
    os.makedirs(ex, exist_ok=True)
    open(os.path.join(ex, f"{name}.md"),
         "w").write(report.worked_example(cfg, snap, tw))
    print(f"onboarded {a.model_dir} from {a.job} as {name}:")
    for f in (src, os.path.join(cdir, f"{name}.json"), twp,
              os.path.join(cdir, f"{name}.bars.json"),
              os.path.join(ex, f"{name}.md")):
        print("  " + _rel(f))
    print(
        "Review %s/ before anyone relies on it; then run `cli.py selftest`." %
        _rel(mdir))
    return 0


def _project(cfg: dict, snap: dict, model_dir: str,
             twin_world: int) -> dict | None:
    """`twin_world` 0 = mirror mode: the served world, so nothing scales."""
    if not twin_world and cfg.get("deploy"):
        from core.deploy import Deploy
        twin_world = Deploy(**cfg["deploy"]).world
    if not cfg["feasibility"]["feasible"]:
        return None
    return project_mod.project(cfg, snap, derive_mod.load_model(model_dir),
                               twin_world)


def _resolve(p: str) -> str:
    """A snapshot path as a header records it: absolute, or relative to a model root."""
    if os.path.isabs(p):
        return p
    for root in derive_mod.MODEL_ROOTS:
        if os.path.exists(os.path.join(root, p)):
            return os.path.join(root, p)
    return os.path.join(SCRIPTS, p)


def _snap_for(cfg: dict) -> dict:
    return json.load(open(_resolve(cfg["_header"]["snapshot"])))


def cmd_project(a) -> int:
    """Real-workload L1 config -> twin config for a smaller slice (default 8 cores)."""
    cfg = json.load(open(a.config))
    tw = _project(cfg, _snap_for(cfg), cfg["model_dir"], a.twin_world)
    if tw is None:
        print("infeasible config: nothing to project", file=sys.stderr)
        return 3
    out = a.out or a.config.replace(".json", f".twin{tw['twin_world']}.json")
    json.dump(tw, open(out, "w"), indent=1)
    print(report.twin_summary(tw))
    print(f"\nwrote {_rel(out)}")
    return 0


def cmd_derive(a) -> int:
    ov = _overrides(a.override)
    wdir = os.path.join(WORK, a.model_dir, "workload-microbenchmark")
    snap, snap_path = _snapshot(a.job, a.model_dir,
                                os.path.join(wdir, "snapshots"))
    cfg = derive_mod.derive(snap, a.model_dir, ov)
    name = a.name or _name(cfg, ov)
    cmd = f"cli.py derive {a.model_dir} --job {a.job}" + "".join(
        f" --override {k}={v}" for k, v in sorted(ov.items())) \
        + (" --mode mirror" if a.mode == "mirror" else "")
    cfg = {"_header": _header(cmd, snap_path), **cfg}
    os.makedirs(os.path.join(wdir, "configs"), exist_ok=True)
    out = os.path.join(wdir, "configs", f"{name}.json")
    json.dump(cfg, open(out, "w"), indent=1)
    print(report.summary(cfg))
    print(f"\nwrote {_rel(out)}")
    tw = _project(cfg, snap, a.model_dir, a.twin_world)
    if tw:
        twp = out.replace(".json", f".twin{tw['twin_world']}.json")
        json.dump(tw, open(twp, "w"), indent=1)
        print("\n" + report.twin_summary(tw))
        print(f"wrote {_rel(twp)}")
    return 0 if cfg["feasibility"]["feasible"] else 3


def cmd_show(a) -> int:
    print(report.summary(json.load(open(a.config))))
    return 0


def cmd_selftest(a) -> int:
    fails = 0
    bars_files = sorted(
        f for root in dict.fromkeys(derive_mod.MODEL_ROOTS)
        for f in glob.glob(os.path.join(root, "*", "config", "*.bars.json")))
    if not bars_files:
        print("no reference configs found")
        return 1
    for bf in bars_files:
        model_dir = os.path.basename(os.path.dirname(os.path.dirname(bf)))
        bars = json.load(open(bf))
        snap = json.load(open(_resolve(bars["snapshot"])))
        for case in bars["cases"]:
            cfg = derive_mod.derive(snap, model_dir, case.get("overrides", {}))
            errs = report.check(cfg, case["expect"])
            tag = f"{os.path.basename(bf)} {case.get('overrides') or '{}'}"
            if errs:
                fails += 1
                print(f"FAIL {tag}")
                for e in errs:
                    print(f"     {e}")
            else:
                print(f"ok   {tag}  ({len(case['expect'])} bars)")
        if bars.get("twin"):
            cfg = derive_mod.derive(snap, model_dir)
            tw = project_mod.project(cfg, snap,
                                     derive_mod.load_model(model_dir),
                                     bars["twin"]["twin_world"])
            errs = report.check(tw, bars["twin"]["expect"])
            tag = f"{os.path.basename(bf)} projection->{bars['twin']['twin_world']} cores"
            if errs:
                fails += 1
                print(f"FAIL {tag}")
                for e in errs:
                    print(f"     {e}")
            else:
                print(f"ok   {tag}  ({len(bars['twin']['expect'])} bars)")
    print(
        f"\n{'FAILED' if fails else 'passed'}: {len(bars_files)} reference config(s)"
    )
    return 1 if fails else 0


def _job_image(job_dir: str) -> str | None:
    """The image a fetched job ran: the `image.txt` a forward-cmd recipe writes into its outputs,
    else the first container in the job's JobSet manifest (`recipe.yaml`)."""
    txt = os.path.join(job_dir, "outputs", "image.txt")
    if os.path.isfile(txt):
        return open(txt).read().strip() or None
    if not os.path.isfile(os.path.join(job_dir, "recipe.yaml")):
        return None
    import yaml
    r = yaml.safe_load(open(os.path.join(job_dir, "recipe.yaml")))
    for rj in r.get("spec", {}).get("replicatedJobs", []):
        for c in rj["template"]["spec"]["template"]["spec"].get(
                "containers", []):
            if c.get("image"):
                return c["image"]
    return None


#: a warm `--top` job (all its shapes, timed and profiled) should finish inside this
TOP_JOB_BUDGET_S = 300

#: the one file a forward-cmd recipe returns: its output directory, packed
OUT_TGZ = "wmb_out.tgz"

#: the recipe forward-cmd writes next to the plan; it runs on the TPU host, beside its inputs
_RECIPE = '''#!/usr/bin/env python3
"""workload-microbenchmark recipe, written by `cli.py forward-cmd` for {plan}.

Runs forward_run.py on the TPU host and packs its output directory into {out}, the
job's one output. The inputs of the job are in the same directory as this file. The
recipe exits with the exit code of forward_run.py.

The recipe runs forward_run.py through forward_run.supervise, which stops a run whose
engine start-up fails or lasts longer than START_TIMEOUT_S seconds. supervise writes
failure.json if the run ends without result.json."""
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
from forward_run import supervise  # noqa: E402

IMAGE = {image!r}
ARGS = {args!r}
START_TIMEOUT_S = {start_timeout_s!r}
os.makedirs("wmb_out", exist_ok=True)
with open(os.path.join("wmb_out", "image.txt"), "w") as f:
    f.write(IMAGE + "\\n")
rc = supervise(ARGS, os.path.join("wmb_out", "wmb", "forward"), START_TIMEOUT_S)
with tarfile.open({out!r}, "w:gz", compresslevel=1) as t:
    t.add("wmb_out", arcname=".")
sys.exit(rc)
'''


def _pin_digest(image: str) -> str:
    """A tag moves (`:latest` is rebuilt): run, record and cache-key the exact image. Resolved
    with Artifact Registry; falls back to the tag, with a warning, if it can't be read."""
    if "@sha256:" in image:
        return image
    r = subprocess.run([
        "gcloud", "artifacts", "docker", "images", "describe", image,
        "--format=value(image_summary.digest)"
    ],
                       capture_output=True,
                       text=True)
    dig = r.stdout.strip().splitlines(
    )[-1] if r.returncode == 0 and r.stdout.strip() else ""
    if not dig.startswith("sha256:"):
        print(
            f"# WARN could not resolve {image} to a digest; running the tag as is",
            file=sys.stderr)
        return image
    return image.rsplit(":", 1)[0] + "@" + dig if ":" in image.rsplit(
        "/", 1)[-1] else image + "@" + dig


def cmd_forward_cmd(a) -> int:
    """Write the recipe for a plan and print the `compute submit` command that runs it, wiring
    in a matching local compile cache.
    The image is the code under test. Default `job`: the image the source job ran, so the run
    reproduces the workload's own code. For an A/B, pass each arm's build explicitly (SKILL.md
    "Change -> build -> microbenchmark"). No moving tag is a default. Tags are pinned to their
    digest. On a backend that runs jobs on the host, the arm is an environment there
    (`--interpreter`), not an image."""
    from core import steps as steps_mod
    from core.forward_run import engine_argv
    pl = json.load(open(a.steps))
    top = pl.get("top")
    # a --top plan: enough repeats for a spread, enough profiles to pool, <= 50 steps a shape
    # iters 10 for every plan: the median drops the first, and with 3 one slow host iteration
    # moved a repeat by 20-45%
    for k, plain, topd in (("iters", 10, 10), ("repeats", 1, 2),
                           ("profile_iters", 1, 5)):
        if getattr(a, k) is None:
            setattr(a, k, topd if top else plain)
    if top:
        cap = top["max_steps_per_shape"]
        n = {
            x["point"]["name"]:
            steps_mod.steps_per_shape(x, a.iters, a.repeats, a.profile_iters,
                                      not a.no_profile)
            for x in pl["points"]
        }
        print(f"# top {top['k']}: forward steps per shape (cap {cap}): " +
              ", ".join(f"{k} {v}" for k, v in n.items()))
        over = [k for k, v in n.items() if v > cap]
        if over:
            print(
                f"! {over} exceed {cap} steps a shape: lower --iters / --repeats / --profile-iters",
                file=sys.stderr)
            return 2
    src = pl.get("source_image")
    if a.interpreter and a.image != "job":
        raise SystemExit(
            "Pass only one of --image or --interpreter, not both. Use --image when the "
            "benchmark runs as a container job on GKE (gke_job); use --interpreter "
            "when it runs directly on a machine (local, a TPU VM, or an existing GKE pod)."
        )
    if a.interpreter:
        # the arm is an environment on the host; its label keys the compile cache and is
        # what `collect` files the job's cache under (outputs/image.txt)
        tag = image = f"interpreter:{a.interpreter}"
        print(
            f"# code under test: the vllm-torchtpu of {a.interpreter} on the host "
            "(result.json image_source records its version and commit)")
    else:
        image = src if a.image == "job" else a.image
        if not image:
            raise SystemExit(
                "this plan does not record the source job's image (a server read from its "
                "vLLM log has no image record): pass --image <repo:tag@sha256:...> on a "
                "container backend (gke_job), or --interpreter <python> on a host "
                "backend (local, gce_tpu_vm, gke_pod)")
        tag = image
        image = _pin_digest(image)
        print(f"# image: {image}" + (
            "  (the source job's image)" if tag == src else
            f"\n#   source job ran {src}: the plan's engine args and env come from that job. Flags"
            " the new image lacks are translated or reported by forward_run (result.json 'compat')."
        ))
    argv_k = engine_argv(pl["twin"], "/tmp/wmb_model", pl["compile_buckets"],
                         "/tmp/wmb_xprof")
    hit = cachelocal.find(image, argv_k, pl["twin"]["env"])
    is_immutable_tag = tag != image and not tag.endswith(":latest")
    if not hit and not a.interpreter and is_immutable_tag:
        # entries filed under an immutable tag before digests were pinned; never for a moving tag
        hit = cachelocal.find(tag, argv_k, pl["twin"]["env"])
    if pl["twin"].get("multihost"):
        # mirror mode: the twin spans two hosts on one Ray cluster (forward_run --node-prep on
        # every host before `ray start`, then forward_run with the Ray executor on host 0). The
        # recipe below runs on one host only, so it cannot run this plan.
        print(
            f"! mirror plan: {pl['twin'].get('hosts', 2)} hosts ({pl['sharding']['twin']} on "
            f"{pl['sharding']['twin_world']} cores) on one Ray cluster. forward-cmd writes a "
            "one-host recipe and cannot run it; use a scaled plan (derive --mode scaled)",
            file=sys.stderr)
        return 2
    args = (["--steps",
             os.path.basename(a.steps), "--iters",
             str(a.iters)] +
            (["--repeats", str(a.repeats)] if a.repeats > 1 else []) +
            (["--save-cache"] if a.resave or not hit else []) +
            ([] if a.no_profile else ["--profile"] +
             (["--profile-iters", str(a.profile_iters)]
              if a.profile_iters > 1 else [])) +
            (["--cache-in", "compile_cache_in.tgz"] if hit else []) +
            ["--out", os.path.join("wmb_out", "wmb", "forward")])
    steps_path = os.path.abspath(a.steps)
    recipe = a.recipe or steps_path.replace(".steps.json", "") + ".recipe.py"
    # The start-up limit: the flag if given, else the plan's own vLLM ready timeout,
    # else 45 minutes.
    plan_ready_s = pl["twin"]["env"].get("VLLM_ENGINE_READY_TIMEOUT_S")
    if a.engine_start_timeout is not None:
        start_timeout_s = a.engine_start_timeout * 60
    elif plan_ready_s is not None:
        start_timeout_s = float(plan_ready_s)
    else:
        start_timeout_s = 45 * 60
    with open(recipe, "w") as f:
        f.write(
            _RECIPE.format(plan=os.path.basename(a.steps),
                           out=OUT_TGZ,
                           image=image,
                           args=args,
                           start_timeout_s=start_timeout_s))
    inputs = [os.path.join(HERE, "forward_run.py"), steps_path]
    if hit:
        inputs.append(
            os.path.join(hit, "compile_cache.tgz") + ":compile_cache_in.tgz")
    print(
        f"# local cache: {'HIT ' + hit if hit else 'miss (cold compile; collect files it after)'}"
    )
    print(f"# recipe: {recipe}")
    arm = (f"--interpreter {a.interpreter}"
           if a.interpreter else f"--image {image}")
    print(f"{a.compute} submit --recipe {recipe} {arm} " +
          " ".join(f"--input {x}" for x in inputs) + f" --output {OUT_TGZ}")
    return 0


def cmd_cache_pull(a) -> int:
    """File a finished forward job's compile cache under ~/.xla_cache (fetch the job first)."""
    job_dir = os.path.join(jobdata.DEFAULT_ROOT, a.job)
    if not os.path.isdir(job_dir):
        print(
            f"{job_dir} missing: run `compute job artifacts --job-id {a.job} --out-dir "
            f"{job_dir}` first",
            file=sys.stderr)
        return 2
    _unpack(job_dir)
    image = a.image or _job_image(job_dir)
    if not image:
        raise SystemExit(
            f"cannot read the image from {job_dir}/outputs/image.txt or "
            f"{job_dir}/recipe.yaml; pass --image")
    env = json.load(open(a.steps))["twin"]["env"] if a.steps else None
    d = cachelocal.store(job_dir, image, env)
    print(f"image {image}")
    print(
        f"stored {os.path.join(d, 'compile_cache.tgz')} "
        f"({os.path.getsize(os.path.join(d, 'compile_cache.tgz')) / 2**20:.0f} MiB)"
    )
    return 0


def _unpack(job_dir: str) -> None:
    """A forward-cmd job returns one file, `wmb_out.tgz`: unpack it into `<job_dir>/outputs/`,
    the layout every reader here expects. Once only."""
    tgz = os.path.join(job_dir, OUT_TGZ)
    out = os.path.join(job_dir, "outputs")
    if os.path.isfile(tgz) and not os.path.isdir(out):
        import tarfile
        with tarfile.open(tgz) as t:
            t.extractall(out, filter="data") if hasattr(
                tarfile, "data_filter") else t.extractall(out)


#: what tpu-diagnostics writes next to result.json, read by xprof_steps / evidence
TRACE_EVENTS = "trace_events.json"
HLO_CATEGORIES = "hlo_categories.json"


def cmd_collect(a) -> int:
    """After `compute job artifacts --job-id <job> --out-dir <jobs>/<job>`: unpack the job's
    output, file its compile cache (if it saved one), print its phase and step times, and, if it
    was profiled and tpu-diagnostics has exported the rank-0 trace's events, split the profile into
    steps and layers with the served-model projection (core/xprof_steps.py)."""
    job_dir = os.path.join(jobdata.DEFAULT_ROOT, a.job)
    _unpack(job_dir)
    res = sorted(
        glob.glob(os.path.join(job_dir, "outputs", "**", "wmb", "forward",
                               "result.json"),
                  recursive=True))
    if not res:
        failed = sorted(
            glob.glob(os.path.join(job_dir, "outputs", "**", "wmb", "forward",
                                   "failure.json"),
                      recursive=True))
        if failed:
            # forward_run.supervise writes failure.json when a run ends without
            # result.json.
            f = json.load(open(failed[0]))
            print(
                f"! job {a.job} failed during {f['stage']}. Reason: "
                f"{f['reason']}. Cause: {f['detail']}",
                file=sys.stderr)
            ctx = f.get("cause_context")
            print(("  The cause, as logged:\n    "
                   if ctx else "  The last lines of its log:\n    ") +
                  "\n    ".join(ctx or f.get("log_tail", [])[-15:]),
                  file=sys.stderr)
            return 2
        print(
            f"no wmb/forward/result.json under {job_dir}: fetch the job first "
            f"(`compute job artifacts --job-id {a.job} --out-dir {job_dir}`)",
            file=sys.stderr)
        return 2
    r = json.load(open(res[0]))
    print(
        f"job {a.job}: {r.get('sharding', {}).get('real')} -> twin "
        f"{r.get('sharding', {}).get('twin')}; drafter {r.get('drafter', 'not run')}"
    )
    print("phases (s): " + ", ".join(f"{k.split(' (')[0]} {v}"
                                     for k, v in r.get("times_s", {}).items()
                                     if not k.startswith(("P", "WARM"))) +
          f"; total {r.get('total_s')}")
    if a.steps and json.load(open(
            a.steps)).get("top") and (r.get("total_s")
                                      or 0) > TOP_JOB_BUDGET_S:
        print(
            f"! a --top job should finish in {TOP_JOB_BUDGET_S // 60} min warm; this one took "
            f"{r['total_s']} s (cold compile? see the phases above)",
            file=sys.stderr)
    for p in r.get("points", []):
        m = [x["median_after_first"] for x in p["mixed_ms"]]
        d = p["decode_ms"] or {}
        dr = [
            x.get("draft_ms", {}).get("median_after_first")
            for x in p["mixed_ms"]
        ]
        flag = any(x.get("compiled_on_first_use") for x in p["mixed_ms"] + [d])
        tm = p.get("twin_model") or {}
        print(f"  {p['point']['name']:18} mixed {m} ms" +
              (f" (+draft {dr})" if any(dr) else "") +
              f", decode {d.get('median_after_first')} ms" +
              (f" (+draft {d['draft_ms']['median_after_first']})" if d.
               get("draft_ms") else "") +
              (f"; x {tm['steps']:,.0f} steps = {tm['weighted_s']} s"
               if "weighted_s" in tm else "") +
              ("  ! a step compiled on first use: its bucket is missing"
               if flag else ""))
    if r.get("mix"):
        mx = r["mix"]
        print(
            f"  step mix (twin ms x the job's steps): {mx['busy_twin_s']} s "
            f"{mx['by_class_twin_s']}; covers {mx['share_of_busy_covered']} of the job's "
            f"measured busy {mx['job_busy_s_measured']} s (served model)")
    tgz = os.path.join(os.path.dirname(res[0]), "compile_cache.tgz")
    if os.path.isfile(tgz):
        if os.path.getsize(tgz) != r.get("compile_cache_bytes"):
            print(
                f"! {tgz} is {os.path.getsize(tgz)} bytes, result.json says "
                f"{r.get('compile_cache_bytes')}: fetch again once the job has finished",
                file=sys.stderr)
        else:
            a.image = a.image or _job_image(job_dir)
            cmd_cache_pull(a)
    if r.get("profile_dir"):
        from core.evidence import fetch_rank0
        pb = fetch_rank0(job_dir)
        ev_json = os.path.join(os.path.dirname(res[0]), TRACE_EVENTS)
        if pb and a.steps and os.path.isfile(ev_json):
            sys.stdout.flush()
            subprocess.run([
                sys.executable,
                os.path.join(HERE, "xprof_steps.py"), ev_json, "--plan",
                a.steps, "--result", res[0], "--json",
                os.path.join(os.path.dirname(res[0]), "xprof_steps.json")
            ])
        elif pb:
            print(
                f"profiled: rank-0 trace {pb}\n"
                f"  tpu-diagnostics has not exported it yet: ask it for {ev_json} and "
                f"{os.path.join(os.path.dirname(res[0]), HLO_CATEGORIES)} (SKILL.md, "
                "\"Reading the profile\"), then run collect again" +
                ("" if a.steps else
                 "; pass --steps <plan> to split it into steps and layers"))
        else:
            print(
                f"! profiled, but no rank-0 .xplane.pb under {job_dir}/outputs",
                file=sys.stderr)
    if a.steps:
        # facts only (numbers, shapes, ids, cross-checks); the agent writes REPORT.md from them
        from core import evidence
        ev = evidence.build(job_dir, os.path.abspath(a.steps))
        ep = os.path.join(os.path.dirname(res[0]), "evidence.json")
        json.dump(ev, open(ep, "w"), indent=1)
        cen = [
            layer for c in ev.get("layer_census_check") or []
            for layer in (c["layers"] or [])
        ]
        print(
            f"evidence: {ep}" +
            "\n  layer census (tests the boundary rule): " +
            (f"{sum(layer['ok'] for layer in cen)}/{len(cen)} layers as expected"
             if cen else "not run") +
            f"\n  offload flips: {len(ev['offload']['flips'])}; image: " +
            str((ev.get("image_source") or {}).get("version")))
    return 0


def cmd_requests(a) -> int:
    """Per-request cached / new / output for a fetched job, checked against the server."""
    from dataclasses import asdict

    from core import cells
    from core.deploy import apply_overrides, deploy_from_args
    job_dir = os.path.join(jobdata.DEFAULT_ROOT, a.job)
    if not os.path.isdir(job_dir):
        print(
            f"! {job_dir} missing. Fetch it first: `compute job artifacts --job-id {a.job} "
            f"--out-dir {job_dir}`",
            file=sys.stderr)
        return 2
    rec = jobdata_agentx.read_recipe(job_dir)
    d = apply_overrides(deploy_from_args(rec["server_args"], rec["env"]),
                        _overrides(a.override))
    reqs = jobdata.read_requests(job_dir)
    hook = getattr(derive_mod.load_model(a.model), "cache_hit",
                   None) if a.model else None
    rows = [cells.with_hit(r, d, hook) for r in jobdata.prefix_chain(reqs)]
    print(
        report.request_report(a.job,
                              rows, {
                                  **asdict(d), "eagle_drop": d.eagle_drop
                              },
                              jobdata.read_server_metrics(job_dir),
                              jobdata.concurrency(reqs),
                              conv=a.conv,
                              bars=a.bars,
                              gate=cells.REPLAY_GATE))
    return 0


def sampling_block(snap: dict,
                   mod,
                   want: bool | None,
                   job_root: str | None = None) -> dict:
    """The plan's `sampling` block: the source job's sampling params and whether the twin step
    runs the sample stage. Params, first found: the snapshot (`sampling_defaults`, read from the
    server's startup log), the job's fetched server log, the model's `SAMPLING` (model.py). Run:
    `want` if given (--sampling / --no-sampling), else on when the job sampled non-greedily
    (temperature > 0). Unknown params: not run unless asked, and then refused."""
    sd = snap.get("sampling_defaults")
    if sd is None and snap.get("job_id"):
        jd = os.path.join(job_root or jobdata.DEFAULT_ROOT, snap["job_id"])
        if os.path.isdir(jd):
            sd = jobdata.read_sampling_defaults(jd)
    if sd is None and getattr(mod, "SAMPLING", None):
        sd = dict(mod.SAMPLING)
    if sd is None:
        if want:
            raise SystemExit(
                "--sampling: the job's sampling params are unknown (no server-log "
                "line, no model.py SAMPLING)")
        return {
            "run":
            False,
            "why":
            "sampling params unknown (no server-log line, no model.py "
            "SAMPLING)"
        }
    greedy = float(sd.get("temperature", 1.0) or 0.0) <= 1e-5
    run = (not greedy) if want is None else bool(want)
    why = (
        "--sampling" if want else "--no-sampling"
        if want is False else "the source job sampled greedily" if greedy else
        "the source job sampled non-greedily (default on; --no-sampling to leave it out)"
    )
    return {
        "run": run,
        "temperature": float(sd.get("temperature", 1.0)),
        "top_k": int(sd.get("top_k") or 0),
        "top_p": float(sd.get("top_p", 1.0)),
        "greedy": greedy,
        "source": sd.get("source", "snapshot"),
        "why": why
    }


def cmd_points(a) -> int:
    """The job's step mix (and/or typed what-if points) -> the batches to run."""
    from core import steps
    from core.deploy import Deploy
    cfg = json.load(open(a.config))
    snap = _snap_for(cfg)
    ov = _overrides(a.override)
    if ov:
        # a different sharding (or any deploy override): re-derive, then re-project the twin
        real = derive_mod.derive(snap, cfg["model_dir"], ov)
        real["_header"] = cfg["_header"]
        if not real["feasibility"]["feasible"]:
            print(report.summary(real), file=sys.stderr)
            return 3
        # keep the config's mode unless --mode says otherwise: an env-only override of a
        # mirror config must not silently re-project it onto 8 cores
        tw = 0 if (a.mode or cfg.get("mode")) == "mirror" else a.twin_world
        cfg = _project(real, snap, cfg["model_dir"], tw)
    d = Deploy(**cfg["real_deploy"])
    mod = derive_mod.load_model(cfg["model_dir"])
    cache_hit = getattr(mod, "cache_hit", None)
    # a serving_benchmark.py job records no step mix: its cells are already fixed shapes
    if not a.point and not snap.get("step_slices") and snap.get(
            "summary_cells"):
        a.point = [
            f"{c['cell']}=0/{c['isl']}/{c['osl']}@{c['concurrency']}"
            for c in snap["summary_cells"]
        ]
        print(
            "no step mix: the source job is a serving_benchmark.py run (fixed lengths and "
            "concurrency, every prompt new); one point per benchmark cell: " +
            ", ".join(a.point))
    # the workload: the job's step mix (core/mix.py), unless explicit --point shapes replace it
    mixd = None
    if not a.point or a.with_mix:
        try:
            mixd = mix_mod.build(snap, d, cache_hit)
        except ValueError as e:
            print(f"! {e}", file=sys.stderr)
            return 2
    pts = [steps.parse_point(x) for x in a.point or []]
    # the drafter follows the source job: on when it served with speculative decoding
    if a.mtp is None:
        a.mtp = bool(d.spec_method)
        if a.mtp:
            print(
                f"MTP: the source job serves {d.spec_method} x{d.spec_tokens}, so the drafter is "
                f"built and timed (pass --no-mtp to leave it out)")
    acc, src = steps.accept_per_step(snap["server"], d)
    if a.accept:
        acc, src = a.accept, "given (--accept)"
    cs = cfg["twin"]["server_args"].get("compilation-config")
    cs = (json.loads(cs)
          if isinstance(cs, str) else cs or {}).get("compile_sizes")
    pl = steps.plan(pts,
                    d,
                    acc,
                    src,
                    a.budget,
                    a.last_chunks,
                    steps.job_measured(snap),
                    env=cfg["twin"].get("env"),
                    compile_sizes=cs,
                    cache_hit=cache_hit,
                    mix=mixd,
                    min_share=a.min_share,
                    top=a.top)
    cmd = f"points {os.path.relpath(os.path.abspath(a.config), SCRIPTS)}" + "".join(
        f" --override {k}={v}" for k, v in sorted(ov.items())) \
        + (f" --top {a.top}" if a.top else "") + (" --mtp" if a.mtp else " --no-mtp" if d.spec_method else "") \
        + (" --combined" if a.combined else "") + "".join(f" --point {x}" for x in a.point or []) \
        + (" --sampling" if a.sampling else " --no-sampling" if a.sampling is False else "")
    pl["_header"] = _header(cmd, cfg["_header"]["snapshot"])
    # self-contained: forward_run.py needs nothing else
    pl["sharding"] = {
        "real": d.label(),
        "twin": Deploy(**cfg["twin_deploy"]).label(),
        "twin_world": cfg["twin_world"],
        "overrides": ov
    }
    pl["twin"], pl["context_scale"] = dict(cfg["twin"]), cfg["context_scale"]
    # per-rank real vs twin table and deviations, for the report's layers & sharding section
    pl["projection"], pl["deviations"] = cfg.get("projection"), cfg.get(
        "deviations")
    if a.mtp:
        if not d.spec_method:
            raise SystemExit(
                "--mtp: the deployment has no speculative decoding")
        pl["twin"]["run_drafter"] = True
        if not getattr(mod, "DRAFTER", None):
            raise SystemExit(
                f"--mtp: {cfg['model_dir']}/model.py declares no DRAFTER replay"
            )
        pl["twin"]["drafter"] = mod.DRAFTER
    pl["sampling"] = sampling_block(snap, mod, a.sampling)
    sb = pl["sampling"]
    print(f"SAMPLING: {'run' if sb['run'] else 'not run'} -- {sb['why']}")
    # the image the job ran; forward-cmd runs whatever image the caller picks
    pl["source_image"] = snap["recipe"].get("image")
    pl["env"] = derive_mod.env_provenance(snap["recipe"], cfg["twin"]["env"],
                                          ov, getattr(mod, "TWIN_ENV", {}))
    f = mod.facts(snap["checkpoint_config"], d.kv_cache_dtype)
    # twin layer i has kind layers.twin[i]; the served model has layers.real[kind] of each
    pl["model_dir"] = cfg["model_dir"]
    pl["layers"] = {"twin": cfg["layers"], "real": f.layer_kinds}
    # how to read this model's profile, and what to re-draw on dummy weights (model.py)
    pl["trace"] = getattr(mod, "TRACE", None)
    if getattr(mod, "ROUTER", None):
        pl["twin"]["router"] = mod.ROUTER
    pl["checkpoint_config"] = {
        k: v
        for k, v in snap["checkpoint_config"].items() if not k.startswith("_")
    }
    tag = pl["sharding"]["real"] + "".join(f"+{k}={v}" for k, v in sorted(ov.items()) if "." not in k) \
        + ("+mtp" if a.mtp else "") + ("+mirror" if cfg.get("mode") == "mirror" else "")
    sdir = os.path.join(WORK, cfg["model_dir"], "workload-microbenchmark",
                        "steps")
    print(
        f"sharding: real {pl['sharding']['real']} -> twin {pl['sharding']['twin']} "
        f"on {pl['sharding']['twin_world']} cores")
    print(steps.table(pl))
    print(report.env_block(pl["env"], pl["source_image"]))
    if a.top:
        # the K fixed shapes in ONE plan: one job runs them back to back and profiles each
        out = a.out or os.path.join(
            sdir, f"{tag}-top{a.top}-{snap['job_id'][:10]}.steps.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(pl, open(out, "w"), indent=1)
        rel = _rel(out)
        print(
            f"\nwrote {_rel(out)}: {len(pl['points'])} shapes, one job\n"
            f"run:   cli.py forward-cmd --steps {rel}\n"
            f"after: cli.py collect <job-id> --steps {rel}\n"
            f"       cli.py mix-rollup --plans {rel} <job-id> [--baseline <job-id>]"
        )
        return 0
    if not a.combined:
        # One plan (one job, one profile) per cell / point, to run as parallel jobs. Every plan
        # compiles the UNION of the buckets, so all of them are one engine config and share one
        # compile cache. A step-mix split goes in its own directory; `mix-rollup` sums it back.
        names = [x["point"]["name"] for x in pl["points"]]
        sub = os.path.join(
            a.out or sdir,
            f"{tag}-mix-{snap['job_id'][:10]}") if mixd else (a.out or sdir)
        for x in pl["points"]:
            one = {
                **pl, "points": [{
                    **x, "profile": True
                }],
                "split": {
                    "of": names,
                    "compile_buckets": "union of all plans' buckets"
                }
            }
            out = os.path.join(
                sub, f"{x['point']['name']}.steps.json"
                if mixd else f"{tag}-{x['point']['name']}.steps.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            json.dump(one, open(out, "w"), indent=1)
        print(
            f"\nwrote {len(names)} plans under {_rel(sub)}/ -- one parallel job "
            f"each; all compile buckets {pl['compile_buckets']} (one cache)")
        if mixd:
            print(
                f"after the jobs: cli.py mix-rollup --plans {_rel(sub)} <job-id> ..."
            )
        return 0
    out = a.out or os.path.join(
        sdir,
        f"{tag}-" + "-".join(([f"mix-{snap['job_id'][:10]}"] if mixd else []) +
                             [p.name for p in pts]) + ".steps.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(pl, open(out, "w"), indent=1)
    print(f"\nwrote {_rel(out)}")
    return 0


def cmd_mix_rollup(a) -> int:
    """After a split step mix's per-cell jobs have run and been fetched: the cost model
    (core/costmodel.py). T_mix = sum of steps x step time over the cells; with --baseline, the
    speedup against another set of jobs of the same cells, per-class deltas, guardrails and the
    modelled served gain. Writes cost_model.json, the report's source for these numbers."""
    from core import costmodel

    # a directory of per-cell plans, or one plan file (a --top plan: its shapes are the cells)
    plans = [a.plans] if os.path.isfile(a.plans) else \
        sorted(glob.glob(os.path.join(a.plans, "*.steps.json")))
    if not plans:
        print(f"! no .steps.json under {a.plans}", file=sys.stderr)
        return 2
    cells, mix = {}, None
    for f in plans:
        pl = json.load(open(f))
        mix = mix or pl.get("mix")
        for x in pl["points"]:
            if x.get("cell"):
                cells[x["cell"]["name"]] = x["cell"]

    def times(jobs: list[str], profile_jobs: list[str]) -> tuple[dict, dict]:
        """Per cell: twin wall times (result.json of `jobs`) and profiles (xprof_steps.json of
        `jobs` + `profile_jobs`, pooled: more profiles = less per-layer sampling noise)."""
        out, prof = {}, {}
        for job in profile_jobs:
            xs = sorted(
                glob.glob(os.path.join(jobdata.DEFAULT_ROOT, job, "outputs",
                                       "**", "wmb", "forward",
                                       "xprof_steps.json"),
                          recursive=True))
            if xs:
                costmodel.cell_profiles(
                    json.load(open(xs[0])).get("steps", []), cells, prof, job)
            else:
                print(
                    f"! {job}: no xprof_steps.json -- fetch and `collect` it",
                    file=sys.stderr)
        for job in jobs:
            res = sorted(
                glob.glob(os.path.join(jobdata.DEFAULT_ROOT, job, "outputs",
                                       "**", "wmb", "forward", "result.json"),
                          recursive=True))
            if not res:
                print(
                    f"! {job}: no result.json -- fetch and `collect` it first",
                    file=sys.stderr)
                continue
            for n, t in costmodel.cell_times(
                    json.load(open(res[0])).get("points", [])).items():
                out[n] = {**t, "job": job}
            xs = os.path.join(os.path.dirname(res[0]), "xprof_steps.json")
            if os.path.exists(xs):
                costmodel.cell_profiles(
                    json.load(open(xs)).get("steps", []), cells, prof, job)
            else:
                print(
                    f"! {job}: no xprof_steps.json -- run `collect`; served basis skips it",
                    file=sys.stderr)
        return out, prof

    busy, idle = (mix or {}).get("busy_s"), (mix or {}).get("idle_s")

    def ids(x):
        return [j for j in (x or "").split(",") if j]

    arms = {"candidate": times(a.jobs, ids(a.profiles))}
    if a.baseline:
        arms["baseline"] = times(ids(a.baseline), ids(a.baseline_profiles))
    out = {
        "plans": _rel(a.plans),
        "jobs": a.jobs,
        "baseline_jobs": a.baseline,
        "source_job_busy_s": busy,
        "source_job_idle_s": idle
    }
    for basis in ("twin", "served", "served_device"):
        sec = {}
        for arm, (tw, pf) in arms.items():
            t = tw if basis == "twin" else costmodel.served_times(cells, tw, pf) \
                if basis == "served" else costmodel.served_device_times(cells, tw, pf)
            sec[arm] = costmodel.evaluate(cells, t)
            if basis == "served":
                sec[arm]["profile_spread"] = costmodel.profile_spread(
                    cells, tw, pf)
            if basis == "served_device":
                sec[arm]["boot_spread"] = costmodel.boot_spread(cells, t)
        if a.baseline:
            sec["compare"] = costmodel.compare(cells, sec["baseline"],
                                               sec["candidate"], busy, idle)
            if basis == "served_device":
                # per-profile T_mix samples of each arm (evaluate's t_mix_by_repeat_s on this
                # basis), Welch's t; replaces the range-vs-spread rule for this basis
                floors = [
                    x for x in (sec["baseline"]["boot_spread"]["spread_pct"],
                                sec["candidate"]["boot_spread"]["spread_pct"],
                                a.noise_floor_pct) if x is not None
                ]
                sig = costmodel.significance(
                    sec["baseline"]["t_mix_by_repeat_s"],
                    sec["candidate"]["t_mix_by_repeat_s"],
                    floor_pct=max(floors) if floors else None)
                sec["compare"]["significance"] = sig
                if sig.get("significant") is not None:
                    sec["compare"]["significant"] = sig["significant"]
        out[basis] = sec
    # the served basis (modelled at the served layer counts) is primary when it covers every cell
    full = all(not out["served"][arm]["missing"] for arm in arms)
    out["basis"] = "served" if full else "twin"
    out["basis_rule"] = (
        "served: twin layer ms x served layer counts + the step's fixed (host, "
        "before/after) time, per cell from its profile or its class's profiled "
        "cell; twin: twin wall step. Primary = served when every cell projects. "
        "served_device: every term from device time (before/after busy + "
        "per-step ops + n_k x layer + drafter busy + sampling busy), no host "
        "wall; primary for kernel / device-time changes.")
    # which basis decides which kind of change: a device-time change (kernel, fusion, layout,
    # collective) is judged on served_device, whose terms carry no host jitter; a host-time change
    # (dispatch, glue) is invisible there and is judged on served / twin
    dev_full = all(not out["served_device"][arm]["missing"] for arm in arms)
    out["primary_for"] = {
        "kernel / device-time changes":
        "served_device" if dev_full else out["basis"],
        "host / per-step changes":
        out["basis"]
    }
    out["sampling_rule"] = (
        "the sample stage (LM head + bonus-token sampler + rejection sampler, "
        "plans with sampling.run): its device busy from the profile, once per "
        "step, on the served and served_device bases; its wall on twin.")
    out["drafter_rule"] = (
        "MTP drafter, served basis: its DEVICE BUSY per step from the profile "
        "(xprof_steps draft_busy_ms, union of its ops), added once, never x "
        "n_layers. Its replay wall (draft_wall_ms) is host-driven module by "
        "module and mostly host gaps a served drafter does not have, so it is "
        "kept beside it and used only on the twin basis, or (labelled in "
        "drafter_basis) when a profile has no drafter busy.")
    out.update(out[out["basis"]])

    def show(tag: str, e: dict, unit: str) -> None:
        print(
            f"  {tag}: T_mix {e['t_mix_s']:.3f} s ({unit}), repeats {e['t_mix_by_repeat_s']}, "
            f"spread {e['spread_pct']}%, covers {100 * e['covered_share']:.1f}% of busy; "
            + "  ".join(f"{k} {v:.3f}" for k, v in e["by_class_s"].items()))
        for n, c in e["cells"].items():
            if c.get("parts_ms"):
                pp = c["parts_ms"]
                print(
                    f"    {n}: {c['step_ms']:.3f} ms/step = before/after {pp['before_after']:.3f} + "
                    f"per-step ops {pp['per_step_ops']:.3f} + layers {pp['layers']:.3f} + drafter "
                    f"{pp['drafter']:.3f} + sampling {pp['sampling']:.3f} ({c.get('profiles')} profiles)"
                )
        ps = e.get("profile_spread")
        if ps:
            print(
                f"    profiled jobs (boots) per cell <= {ps['profiled_jobs_max']}; T_mix by job "
                f"{ps['t_mix_by_profile_s']}, spread {ps['spread_pct']}%" +
                ("  (one boot: profile sampling noise unmeasured)"
                 if ps["profiled_jobs_max"] < 2 else ""))
        if e["missing"]:
            print(
                f"    ! {len(e['missing'])} cell(s) without a job or profile: {e['missing']}",
                file=sys.stderr)

    for basis in ("served_device", "served", "twin"):
        sec = out[basis]
        unit = {
            "served": "served layers, modelled",
            "twin": "twin, as run",
            "served_device": "served layers, device time only, modelled"
        }[basis]
        print(f"[{basis}]" +
              "".join(f"  <- primary for {k}"
                      for k, v in out["primary_for"].items() if v == basis))
        if a.baseline:
            show("baseline ", sec["baseline"], unit)
        show("candidate" if a.baseline else "T_mix", sec["candidate"], unit)
        if a.baseline:
            c = sec["compare"]
            print(
                f"  speedup {c['speedup']}x over {c['cells_compared']} cells "
                f"({'beats' if c['significant'] else 'within'} the {c['noise_pct']}% "
                f"{'repeat/profile' if basis == 'served' else 'per-profile' if basis == 'served_device' else 'repeat'} spread); modelled served output gain {c['served_output_gain_modelled']}x"
            )
            sg = c.get("significance")
            if sg and sg.get("p_two_sided") is not None:
                ck = next(k for k in sg if k.startswith("ci"))
                fl = sg.get("boot_noise_floor_pct")
                print(
                    f"    Welch t over per-profile T_mix ({sg['n_base']} vs {sg['n_cand']} profiles): "
                    f"t {sg['t']}, df {sg['df']}, p {sg['p_two_sided']:.3g}, {ck} {sg[ck]}; "
                    +
                    (f"boot-noise floor {fl}%" if fl is not None else
                     "boot-to-boot noise UNMEASURED (one boot per arm; pass --noise-floor-pct "
                     "from an A/A)") + " -> " +
                    ("SIGNIFICANT" if sg["significant"] else "not significant")
                )
            if c.get("sampling_mismatch"):
                print(
                    f"    ! the arms differ in whether the sample stage ran: {c['sampling_in_steps']}"
                )
            print("    by class (s saved): " +
                  "  ".join(f"{k} {v:+.3f}"
                            for k, v in c["by_class_delta_s"].items()))
            for g in c["guardrails"]:
                print(
                    f"    guardrail {g['cell']} ({g['role']}): {g['base_ms']} -> {g['cand_ms']} ms"
                    + ("  ! regressed" if g["regressed"] else ""))
    path = a.json or (a.plans.removesuffix(".steps.json") +
                      ".cost_model.json" if os.path.isfile(a.plans) else
                      os.path.join(a.plans, "cost_model.json"))
    json.dump(out, open(path, "w"), indent=1)
    print(f"wrote {_rel(path)}")
    missing = out["candidate"]["missing"] + (out.get("baseline") or {}).get(
        "missing", [])
    return 1 if missing else 0


def cmd_run(a) -> int:
    print(
        "`run` (L1 A/B on 2x2x1) is milestone M1+ and not built yet. See proposal §9.3.",
        file=sys.stderr)
    return 2


def main() -> int:
    global WORK
    p = argparse.ArgumentParser(prog="workload-microbenchmark")
    p.add_argument(
        "--work",
        default=WORK,
        help=
        "work directory: working configs, snapshots, plans, models being onboarded "
        "(models/<MODEL_DIR>/) (default ./.work)")
    p.add_argument("--jobs",
                   dest="jobs_root",
                   help="directory of fetched jobs, one <job-id>/ each "
                   "(default <work>/jobs)")
    sub = p.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("onboard")
    o.add_argument("model_dir")
    o.add_argument("--job", required=True)
    o.add_argument("--name")
    o.add_argument("--twin-world", type=int, default=8)
    o.add_argument(
        "--mode",
        choices=("scaled", "mirror"),
        default="scaled",
        help=
        "scaled (default): the twin on --twin-world cores (8 = one 2x2x1 host), "
        "split dims scaled to keep per-rank shapes. mirror: the served sharding "
        "on the served slice (e.g. 2x2x2), only layers shrink")
    d = sub.add_parser("derive")
    d.add_argument("model_dir")
    d.add_argument("--job", required=True)
    d.add_argument("--override", action="append")
    d.add_argument("--name")
    d.add_argument("--twin-world", type=int, default=8)
    d.add_argument(
        "--mode",
        choices=("scaled", "mirror"),
        default="scaled",
        help=
        "scaled (default): the twin on --twin-world cores (8 = one 2x2x1 host), "
        "split dims scaled to keep per-rank shapes. mirror: the served sharding "
        "on the served slice (e.g. 2x2x2), only layers shrink")
    pj = sub.add_parser("project")
    pj.add_argument("config")
    pj.add_argument("--out")
    pj.add_argument("--twin-world", type=int, default=8)
    pj.add_argument(
        "--mode",
        choices=("scaled", "mirror"),
        default="scaled",
        help=
        "scaled (default): the twin on --twin-world cores (8 = one 2x2x1 host), "
        "split dims scaled to keep per-rank shapes. mirror: the served sharding "
        "on the served slice (e.g. 2x2x2), only layers shrink")
    s = sub.add_parser("show")
    s.add_argument("config")
    sub.add_parser("selftest")
    fc = sub.add_parser("forward-cmd")
    fc.add_argument("--steps",
                    required=True,
                    help="a .steps.json from `points`")
    fc.add_argument("--iters",
                    type=int,
                    help="timed iterations per batch (default 10)")
    fc.add_argument(
        "--repeats",
        type=int,
        help=
        "time the whole plan N times in one job (default 1; 2 for a --top plan)"
    )
    fc.add_argument(
        "--resave",
        action="store_true",
        help=
        "save the compile cache even on a hit (e.g. to replace a cold-boot cache)"
    )
    fc.add_argument(
        "--no-profile",
        action="store_true",
        help="skip the xprof window (profiling is on by default; saves ~40 s)")
    fc.add_argument(
        "--engine-start-timeout",
        type=float,
        help=
        "The number of minutes that engine start-up may last before the job "
        "stops itself. The limit stops an engine that hangs without logging an "
        "error. Set it longer than the plan takes to compile without cached "
        "compile results. Default: the plan's VLLM_ENGINE_READY_TIMEOUT_S if it "
        "sets one, otherwise 45 minutes.")
    fc.add_argument(
        "--profile-iters",
        type=int,
        help="(default 1; 5 for a --top plan) "
        "profile each profiled step N times: the served basis multiplies a "
        "profile's per-layer wobble by the layer count, so pool several "
        "(about 90 MB of trace per mixed step on the mirror twin)")
    fc.add_argument(
        "--compute",
        default="<runtime> compute",
        help=
        "the command that invokes the runtime's compute commands, printed in "
        "front of `submit`")
    fc.add_argument(
        "--recipe",
        help="where to write the recipe (default: next to the plan, "
        "<plan>.recipe.py)")
    fc.add_argument(
        "--image",
        default="job",
        help=
        "image under test, on a backend that runs jobs in containers (gke_job): "
        "`job` (default) = the source job's image, recorded in the plan; "
        "or an image path, e.g. an arm's build. Pinned to its digest")
    fc.add_argument(
        "--interpreter",
        help="code under test on a backend that runs jobs on the host (local, "
        "gce_tpu_vm, gke_pod), which ignores --image: the Python of the environment "
        "holding the arm's vllm-torchtpu, passed to `submit --interpreter`")
    cp = sub.add_parser("cache-pull")
    cp.add_argument("job")
    cp.add_argument("--image",
                    help="default: the image the job ran (from its recipe)")
    cp.add_argument(
        "--steps",
        help=
        "the plan the job ran; needed only if its result.json predates plan_env"
    )
    co = sub.add_parser("collect")
    co.add_argument("job")
    co.add_argument(
        "--steps",
        help="the .steps.json the job ran (profile split + cache key)")
    co.add_argument("--image", help="default: the image the job ran")
    rq = sub.add_parser("requests")
    rq.add_argument("job")
    rq.add_argument("--conv",
                    help="one conversation: its #index or an id prefix")
    rq.add_argument("--bars",
                    action="store_true",
                    help="also one text bar per request")
    rq.add_argument("--override",
                    action="append",
                    help="deploy key=value, e.g. dcp=8")
    rq.add_argument("--model",
                    help="MODEL_DIR whose model.py supplies a cache-hit rule "
                    "(default: vLLM's full-attention rule)")
    pt = sub.add_parser("points")
    pt.add_argument("config", help="twin config (.twin8.json)")
    pt.add_argument(
        "--point",
        action="append",
        help="an explicit what-if request shape NAME=cached/input/output@C "
        "(e.g. X=299K/6K/1K@8); replaces the step mix unless --with-mix")
    pt.add_argument("--with-mix",
                    action="store_true",
                    help="keep the job's step mix alongside --point shapes")
    pt.add_argument(
        "--min-share",
        type=float,
        default=0.002,
        help="drop step-mix cells under this share of busy time (default 0.002; "
        "the dropped share is printed)")
    pt.add_argument(
        "--override",
        action="append",
        help=
        "deploy key=value, repeatable (e.g. dcp=8): re-derives and re-projects the twin"
    )
    pt.add_argument("--twin-world", type=int, default=8)
    pt.add_argument(
        "--mode",
        choices=("scaled", "mirror"),
        default=None,
        help=
        "how an --override re-projects the twin. Default: the config's own mode "
        "(a .twin16 mirror config stays mirror). scaled: the twin on --twin-world "
        "cores (8 = one 2x2x1 host), split dims scaled to keep per-rank shapes. "
        "mirror: the served sharding on the served slice (e.g. 2x2x2), only "
        "layers shrink")
    pt.add_argument("--budget",
                    type=int,
                    help="token budget per step (default: the deploy's mnbt)")
    pt.add_argument("--accept",
                    type=float,
                    help="tokens per decode step (default: from the job)")
    pt.add_argument(
        "--top",
        type=int,
        metavar="K",
        help=
        "the K (2-5) step-mix cells carrying the most busy time, as fixed shapes "
        "in ONE plan: one job runs them back to back and profiles every one; "
        "the cost model counts only them")
    pt.add_argument(
        "--combined",
        action="store_true",
        help=
        "everything in ONE plan (one job, one profile of the heaviest cell per "
        "class). Default: one plan per cell / point, run as parallel jobs")
    pt.add_argument(
        "--mtp",
        action="store_true",
        default=None,
        help=
        "build and time the MTP drafter after every step (compiles it too). "
        "Default: on when the source job serves with speculative decoding")
    pt.add_argument(
        "--no-mtp",
        dest="mtp",
        action="store_false",
        help="leave the drafter out even though the source job runs it")
    pt.add_argument(
        "--sampling",
        action="store_true",
        default=None,
        help=
        "run the serving step's sample stage (LM head + sampler + rejection "
        "sampler) after every forward, with the job's sampling params. Default: "
        "on when the source job sampled non-greedily")
    pt.add_argument(
        "--no-sampling",
        dest="sampling",
        action="store_false",
        help="leave the sample stage out (plan `sampling.run` false)")
    pt.add_argument(
        "--last-chunks",
        action="store_true",
        help=
        "also run each point's partial last chunk (compiles its smaller bucket)"
    )
    pt.add_argument("--out")
    mr = sub.add_parser(
        "mix-rollup",
        help="sum steps x step time over a split step mix's per-cell jobs")
    mr.add_argument(
        "--plans",
        required=True,
        help=
        "the directory `points` wrote the cell plans to, or one --top plan file"
    )
    mr.add_argument("jobs",
                    nargs="+",
                    help="the per-cell job ids (fetched and collected)")
    mr.add_argument(
        "--baseline",
        help="comma-separated per-cell job ids of the arm to compare "
        "against (same plans, e.g. the incumbent image)")
    mr.add_argument(
        "--profiles",
        help="comma-separated extra jobs of the candidate arm whose "
        "profiles (xprof_steps.json) pool into the served basis; "
        "their step times are not used")
    mr.add_argument("--baseline-profiles",
                    help="the same, for the baseline arm")
    mr.add_argument(
        "--json", help="where to write cost_model.json (default: in --plans)")
    mr.add_argument(
        "--noise-floor-pct",
        type=float,
        help=
        "boot-to-boot noise on the served_device basis from an A/A of two baseline "
        "boots (Qwen3.5 AgentX top5: 0.37%%); a speedup must beat it to be "
        "significant. An arm with >= 2 boots measures its own")
    sub.add_parser("run")
    a = p.parse_args()
    WORK = os.path.abspath(a.work)
    jobdata.DEFAULT_ROOT = os.path.abspath(
        a.jobs_root) if a.jobs_root else os.path.join(WORK, "jobs")
    derive_mod.MODEL_ROOTS[:] = [os.path.join(WORK, "models"), SCRIPTS]
    if getattr(a, "mode", None) == "mirror":
        a.twin_world = 0  # _project resolves it to the served world
    return {
        "onboard": cmd_onboard,
        "derive": cmd_derive,
        "show": cmd_show,
        "project": cmd_project,
        "selftest": cmd_selftest,
        "run": cmd_run,
        "requests": cmd_requests,
        "forward-cmd": cmd_forward_cmd,
        "points": cmd_points,
        "collect": cmd_collect,
        "cache-pull": cmd_cache_pull,
        "mix-rollup": cmd_mix_rollup
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

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
"""Onboarding's checks that need the task image, in one compute job.

Every compute job takes a whole TPU node, and each queues for it on its own.
The image probe, the full preflight check, the vLLM engine compatibility
check and the vLLM flag discovery used to be up to four jobs; here they are
steps of one. Each still runs as its own process, so one that crashes or
leaves state behind (the vLLM patch, the TPU runtime) does not change the
next. The vLLM steps run when --devices is given, the check after the
compatibility check, so a patch it applied (--vllm-patch) is in place for it.

On a CPU node (a gke_job --cpu-requests job, which sets
TPU_AGENT_SUITE_CHIPLESS_TPU to the profile's TPU generation), it first has
chipless_tpu.py tell torch_tpu it has --devices chips of that generation, and
the probe skips the device; the server's start is then the first use of a
chip. If that self-test fails, nothing else runs and it exits 75, which
compute answers by running the job again on a TPU node.

Runs in the working directory, where compute puts the inputs and collects
the outputs; the script itself can sit elsewhere (on gke_job, a read-only
mount). Every step runs even when one fails, and every file below is written
whatever happens, so none is ever missing:

  preflight.json    each step's exit code (null: not run) and, for one that
                    failed, the end of its output; `chipless` is the
                    self-test's (null on a TPU node)
  image_probe.json  image_probe.py --touch_device; {"error": ...} if it wrote
                    nothing
  full_check.log    preflight_check.py's output, which the Rules are read from
  tp_plan.json      engine_compatibility_check.py (vLLM or SGLang); {"ok": false,
                    "reason": ...} if it stopped before planning
  vllm_flags.txt    `vllm serve --help` (vLLM); a `# ` line saying why if not
  sglang_flags.txt  `python3 -m sglang.launch_server --help` (SGLang); a `# `
                    line saying why if not

It exits 0 once every step has run; judge each step by preflight.json.

Usage:
  compute run --code .../onboarding_preflight.py \
    --input .../image_probe.py:image_probe.py \
    --input .../preflight_check.py:preflight_check.py \
    [--cpu-requests cpu=2,memory=8Gi --input .../chipless_tpu.py:chipless_tpu.py] \
    [--input .../engine_compatibility_check.py:engine_compatibility_check.py \
     --input .../oss_meshes.py:oss_meshes.py \
     --input .../discover_vllm_flags.sh:discover_vllm_flags.sh] \
    --output preflight.json:<local> ... \
    -- --model <id> --mode inference \
       [--engine vllm|sglang] [--trust-remote-code] \
       [--devices <n> --hbm-gb-per-device <gb> [--tp <n>] [--vllm-patch <py>]
        [--points <isl:osl:c,...>]
        [--objective throughput|latency]]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

TAIL_LINES = 20
# Set by compute's gke_job in a --cpu-requests job; chipless_tpu.py reads it.
CHIPLESS_ENV = "TPU_AGENT_SUITE_CHIPLESS_TPU"
# What gke_job's `run` answers by running the job again on a TPU node.
NEEDS_TPU = 75


def run_step(name: str, cmd: list[str], env: dict[str, str]) -> dict:
    """Run one step, echoing its output; its exit code and output lines."""
    print(f"=== onboarding_preflight: {name}", flush=True)
    lines = []
    try:
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True,
                              errors="replace",
                              env=env) as proc:
            for line in proc.stdout:
                print(line, end="", flush=True)
                lines.append(line.rstrip("\n"))
        code = proc.returncode
    except OSError as e:
        lines.append(f"could not start {cmd[0]}: {e}")
        print(lines[-1], flush=True)
        code = 127
    return {"exit_code": code, "lines": lines}


def failure_reason(lines: list[str]) -> str:
    """What a failed step said about why: its error lines, else its last."""
    said = [line.strip() for line in lines]
    for i, line in enumerate(said):
        if "CRITICAL ERROR" in line:
            return " ".join(filter(None, said[i:i + 3]))
        if "PRE-FLIGHT FAILED" in line:
            reason = next((later for later in said[i + 1:i + 3]
                           if later.startswith("Reason:")), "")
            return f"{line} {reason}".strip()
    last = [line for line in said if line]
    return last[-1] if last else "it printed nothing"


def write_if_missing(path: str, content: str) -> None:
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode",
                        required=True,
                        choices=("benchmark", "inference"))
    parser.add_argument("--engine", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--hbm-gb-per-device", type=float)
    parser.add_argument("--tp", type=int)
    parser.add_argument("--vllm-patch")
    # Passed to the engine check: the target points the KV check and the
    # traffic rule use.
    parser.add_argument("--points")
    # What the target traffic needs; the check picks the layout by it.
    parser.add_argument("--objective", choices=("throughput", "latency"))
    args = parser.parse_args(argv)
    if args.devices is not None and args.hbm_gb_per_device is None:
        parser.error("--devices needs --hbm-gb-per-device")

    py = sys.executable
    # The scripts the steps run (discover_vllm_flags.sh, apply_vllm_patch.sh)
    # use this one rather than resolving their own.
    env = dict(os.environ, TPU_AGENT_SUITE_PYTHON=py, PYTHONUNBUFFERED="1")
    steps = {}

    probe_args = ["--touch_device"]
    if env.get(CHIPLESS_ENV):
        os.makedirs(".chipless", exist_ok=True)
        if os.path.isfile("chipless_tpu.py"):
            shutil.copy("chipless_tpu.py", ".chipless/")
        with open(".chipless/sitecustomize.py", "w", encoding="utf-8") as f:
            f.write("import chipless_tpu\n")
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, [os.path.abspath(".chipless"),
                          env.get("PYTHONPATH")]))
        # The device count the plan is for; torch_tpu's layout lookups use it.
        env["CHIPLESS_TPU_DEVICES"] = str(args.devices or 1)
        chipless = run_step(
            f"chipless: as {env['CHIPLESS_TPU_DEVICES']} device(s) of "
            f"{env[CHIPLESS_ENV]}, no chip",
            [py, ".chipless/chipless_tpu.py", "--check"], env)
        steps["chipless"] = chipless
        if chipless["exit_code"]:
            # A check without the device would fail as if the model did.
            print(
                "=== onboarding_preflight: this job needs a TPU node; "
                f"exiting {NEEDS_TPU}",
                flush=True)
            return NEEDS_TPU
        probe_args = []

    probe = run_step("image probe", [py, "image_probe.py"] + probe_args +
                     ["--output", "image_probe.json"], env)
    steps["image_probe"] = probe
    write_if_missing(
        "image_probe.json",
        json.dumps({
            "error":
            f"image_probe.py exited {probe['exit_code']} without "
            f"writing this file: {failure_reason(probe['lines'])}"
        }) + "\n")

    # No --cuda_report_json: shipped alone, it cannot find the playbooks a
    # finding maps to, and the host's Step 2 report is the report.
    full = run_step("full preflight check", [
        py, "preflight_check.py", f"--model={args.model}",
        "--allow_gpu_imports", f"--mode={args.mode}"
    ], env)
    steps["full_check"] = full
    with open("full_check.log", "w", encoding="utf-8") as f:
        f.write("\n".join(full["lines"]) + "\n")

    if args.devices is not None:
        check_args = [
            "--model", args.model, "--devices",
            str(args.devices), "--hbm-gb-per-device",
            f"{args.hbm_gb_per_device:g}"
        ]
        if args.engine != "vllm":
            check_args += ["--engine", args.engine]
        if args.trust_remote_code:
            check_args += ["--trust-remote-code"]
        if args.tp is not None:
            check_args += ["--tp", str(args.tp)]
        if args.vllm_patch:
            check_args += ["--vllm-patch", args.vllm_patch]
        if args.points:
            check_args += ["--points", args.points]
        if args.objective:
            check_args += ["--objective", args.objective]
        engine_label = "SGLang" if args.engine == "sglang" else "vLLM"
        check = run_step(f"{engine_label} engine compatibility check", [
            py, "engine_compatibility_check.py", "--plan-json", "tp_plan.json"
        ] + check_args, env)
        steps["engine_compatibility_check"] = check
        write_if_missing(
            "tp_plan.json",
            json.dumps(
                {
                    "ok":
                    False,
                    "tensor_parallel_size":
                    None,
                    "data_parallel_size":
                    None,
                    "source":
                    "not_planned",
                    "reason":
                    "the check stopped before planning: " +
                    failure_reason(check["lines"]),
                },
                indent=2) + "\n")
        if args.engine == "sglang":
            flag_step = "discover_sglang_flags"
            flag_script = "discover_sglang_flags.sh"
            flag_file = "sglang_flags.txt"
        else:
            flag_step = "discover_vllm_flags"
            flag_script = "discover_vllm_flags.sh"
            flag_file = "vllm_flags.txt"
        flags = run_step(f"{engine_label} flag discovery",
                         ["bash", flag_script, flag_file], env)
        steps[flag_step] = flags
        write_if_missing(
            flag_file,
            f"# flag discovery exited {flags['exit_code']} without writing "
            f"this file: {failure_reason(flags['lines'])}\n")

    flag_step_name = ("discover_sglang_flags"
                      if args.engine == "sglang" else "discover_vllm_flags")
    report = {
        name: {
            "exit_code": None
        }
        for name in ("chipless", "image_probe", "full_check",
                     "engine_compatibility_check", flag_step_name)
    }
    for name, step in steps.items():
        report[name] = {"exit_code": step["exit_code"]}
        if step["exit_code"]:
            report[name]["tail"] = step["lines"][-TAIL_LINES:]
    with open("preflight.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("=== onboarding_preflight: exit codes " +
          json.dumps({
              name: step["exit_code"]
              for name, step in report.items()
          }),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

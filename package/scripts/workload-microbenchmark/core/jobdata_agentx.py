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
"""Read an AgentX serving job's `recipe.yaml`: the Kubernetes JobSet manifest that AgentX's job
platform saves with every job.

Builds on `jobdata`, which reads a server this package started (`compute submit --serve`) from
vLLM's own log. The manifest holds the served command and env, so a job that has one is read from
it: the server container's script is run in restricted bash up to its `vllm` call
(`_eval_script`), and the checkpoint's config.json comes from the GCS mirror the recipe names
(`_manifest_checkpoint_config`). A job with no manifest falls through to `jobdata`'s readers.
Everything after the recipe (the client's records, the server's counters and log) is the same
for both, and `jobdata` reads it.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any

import yaml

from . import jobdata

# ---------------------------------------------------------------- recipe


def _containers(recipe: dict) -> list[dict]:
    out = []
    for rj in recipe.get("spec", {}).get("replicatedJobs", []):
        pod = rj["template"]["spec"]["template"]["spec"]
        for c in pod.get("containers", []):
            out.append({"replicated_job": rj["name"], **c})
    return out


#: Runs before the recipe script. fd 3 is opened on the capture file *before* restricted mode,
#: because rbash refuses redirection to files but allows `>&3`. Restricted mode also refuses
#: commands with a `/`, `cd`, `exec` and changing PATH; PATH is an empty directory, so every
#: external command (gcloud, ray, python, sleep, tee) is a no-op that succeeds.
_PRELUDE = r'''exec 3>"$WMB_CAPTURE"
unset WMB_CAPTURE
builtin set -r
command_not_found_handle() { return 0; }
cd() { return 0; }
exit() { return 0; }
set() {
  local a=() x skip=0
  for x in "$@"; do
    if [ $skip = 1 ]; then skip=0; continue; fi
    case $x in -o|+o|-?*o|+?*o) skip=1;; -?*|+?*) ;; *) a+=("$x");; esac
  done
  [ ${#a[@]} -gt 0 ] && builtin set -- "${a[@]}"
  return 0
}
exec() { [ $# -gt 0 ] && "$@"; return 0; }
vllm() {
  local n
  for n in "$@"; do printf 'A\037%s\0' "$n" >&3; done
  for n in $(compgen -e); do printf 'E\037%s=%s\0' "$n" "${!n}" >&3; done
  builtin kill -KILL 0
}
'''

#: `> file`, `>> file`, `&> file`, `2>/dev/null`; not fd duplication (`2>&1`, `>&2`).
#: rbash turns any file redirect into a failed command, which is harmless everywhere except
#: on the `vllm` call itself, so only lines that call it are stripped.
_FILE_REDIRECT = re.compile(
    r"""(?<=\s)(?:\d|&)?>>?\|?[ \t]*(?!&)(?:"[^"\n]*"|'[^'\n]*'|[^\s;|&)"'<>]+)"""
)
_CALLS_VLLM = re.compile(r"(^|[\s;(])vllm\s|VLLM_ARGS\[@\]")

#: Set by bash or by the prelude, not by the recipe.
_SHELL_ENV = {"PATH", "HOME", "PWD", "OLDPWD", "SHLVL", "_"}


def _logical_lines(script: str) -> list[str]:
    """Join `\\`-continued lines as bash does: a `#` comment consumes to newline (including
    any trailing `\\`), so a `#` line inside a `\\`-continuation ends that command."""
    out: list[str] = []
    cur = ""
    for raw in script.splitlines():
        if raw.lstrip().startswith("#"):
            if cur:
                out.append(cur)
                cur = ""
            continue
        r = raw.rstrip()
        if r.endswith("\\"):
            cur += r[:-1] + " "
        else:
            out.append(cur + raw)
            cur = ""
    if cur:
        out.append(cur)
    return out


def _eval_script(
        script: str,
        env: dict[str, str],
        timeout_s: float = 30) -> tuple[list[str], dict[str, str]] | None:
    """Run the recipe script in restricted bash until it calls `vllm`, and return the argv
    and the exported environment at that call, exactly as bash expanded them: `export`s,
    `${X:-default}`, `if` branches, line continuations, arrays, `VAR=x vllm ...` prefixes.
    Returns None if the script never reaches `vllm` (e.g. `python -m vllm...`).

    Filesystem tests (`[ -d /vllm-cache/... ]`) see this machine, not the pod."""
    import signal
    import subprocess
    import tempfile
    body_lines: list[str] = []
    in_vllm = False
    for line in script.splitlines():
        if line.lstrip().startswith("#"):
            in_vllm = False
            body_lines.append(line)
            continue
        is_vllm = in_vllm or bool(_CALLS_VLLM.search(line))
        body_lines.append(_FILE_REDIRECT.sub("", line) if is_vllm else line)
        in_vllm = is_vllm and line.rstrip().endswith("\\")
    body = "\n".join(body_lines)
    with tempfile.TemporaryDirectory() as tmp:
        cap, empty = os.path.join(tmp, "capture"), os.path.join(tmp, "bin")
        os.mkdir(empty)
        path = os.path.join(tmp, "recipe.sh")
        with open(path, "w") as f:
            f.write(_PRELUDE + "\n" + body + "\nwait\n")
        run_env = {**env, "PATH": empty, "HOME": tmp, "WMB_CAPTURE": cap}
        with open(os.path.join(tmp, "stderr"), "w") as err:
            p = subprocess.Popen(["/bin/bash", path],
                                 cwd=tmp,
                                 env=run_env,
                                 stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL,
                                 stderr=err,
                                 start_new_session=True)
            try:
                p.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        recs = []
        if os.path.exists(cap):
            with open(cap, "rb") as f:
                recs = f.read().decode(errors="replace").split("\0")
    argv = [r[2:] for r in recs if r.startswith("A\x1f")]
    if not argv:
        return None
    out_env = {}
    for r in recs:
        if r.startswith("E\x1f") and "=" in r:
            k, v = r[2:].split("=", 1)
            if k not in _SHELL_ENV:
                out_env[k] = v
    return argv, out_env


def _args_from_tokens(tokens: list[str]) -> tuple[dict[str, Any], list[str]]:
    if "serve" in tokens:
        tokens = tokens[tokens.index("serve") + 1:]
    model = tokens[0] if tokens and not tokens[0].startswith("-") else None
    args: dict[str, Any] = {}
    i = 1 if model else 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("--"):
            i += 1
            continue
        if "=" in t:
            k, v = t[2:].split("=", 1)
            args[k] = v
            i += 1
        elif i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[t[2:]] = tokens[i + 1]
            i += 2
        else:
            args[t[2:]] = True
            i += 1
    return args, ([model] if model else [])


def _parse_serve_args(script: str) -> tuple[dict[str, Any], list[str]]:
    """Fallback when `_eval_script` cannot reach `vllm`: pull `vllm serve` arguments out of
    the script text. Handles a `VLLM_ARGS=( ... )` array whose expansion may carry extra flags
    (`"${VLLM_ARGS[@]}" --no-dcp-q-replicate`), and a `vllm serve ...` command (including
    `\\`-continued lines). Skips `#` comment lines.
    """
    tokens: list[str] = []
    m = re.search(r"VLLM_ARGS=\(\s*\n(.*?)\n\s*\)", script, re.S)
    if m:
        body = "\n".join(line for line in m.group(1).splitlines()
                         if not line.strip().startswith(('"${', "#"))
                         and "OFFLOAD_ARGS" not in line)
        tokens = shlex.split(body, comments=True)
        tail = re.search(r'"\$\{VLLM_ARGS\[@\]\}"([^|\n]*)', script)
        if tail:
            # stop at the first shell redirection: `--flag 2>&1 | tee log`
            tokens += shlex.split(re.split(r"\s\d?>", tail.group(1))[0],
                                  comments=True)
    else:
        line = next(
            (s for logical in _logical_lines(script)
             for s in logical.split(";") if not s.lstrip().startswith("#")
             and re.search(r"(?:^|[\s(])vllm\s+serve\b", s)),
            "",
        )
        tokens = shlex.split(re.split(r"\s\d?>|\|", line)[0], comments=True)
    return _args_from_tokens(tokens)


def read_recipe(job_dir: str) -> dict[str, Any]:
    """The served command and env: from an AgentX job's `recipe.yaml`, else from the vLLM log
    of a server this package started (`jobdata.read_recipe`)."""
    if os.path.isfile(os.path.join(job_dir, "recipe.yaml")):
        return _read_manifest(job_dir)
    try:
        return jobdata.read_recipe(job_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{job_dir}: no recipe.yaml (AgentX) and no vLLM server log with a "
            "`non-default args` line") from e


def _is_server_script(script: str, env: dict[str, Any] | None = None) -> bool:
    active = "\n".join(_logical_lines(script))
    if re.search(r"(?:^|[\s;(])vllm\s+serve\b|VLLM_ARGS=", active):
        return True
    if env and re.search(r'\beval\s+"?\$\{?VLLM_SERVER_CMD\}?"?', active):
        return _is_server_script(str(env.get("VLLM_SERVER_CMD") or ""))
    return False


def _container_script(c: dict[str, Any]) -> str:
    args = c.get("args") or []
    if args:
        return args[0]
    cmd = c.get("command") or []
    return cmd[2] if len(cmd) > 2 else ""


def _read_manifest(job_dir: str) -> dict[str, Any]:
    recipe = yaml.safe_load(open(os.path.join(job_dir, "recipe.yaml")))
    server = bench = None
    for c in _containers(recipe):
        script = _container_script(c)
        ce = {
            e["name"]: e.get("value")
            for e in c.get("env", []) if "value" in e
        }
        if _is_server_script(script, ce):
            server = (c, script)
        elif any(k in script or k in str(ce.get("BENCH_CMD") or "")
                 for k in ("aiperf", "vllm bench", "REPLAY_CMD")):
            bench = (c, script)
    if server is None:
        raise ValueError(f"{job_dir}: no container runs `vllm serve`")
    c, script = server
    cenv = {
        e["name"]: e.get("value")
        for e in c.get("env", []) if "value" in e
    }
    benv = {
        e["name"]: e.get("value")
        for e in (bench[0].get("env", []) if bench else []) if "value" in e
    }
    # the server runs on node 0 (multi-host recipes branch on the jobset index)
    ev = _eval_script(script, {
        "JOBSET_JOB_INDEX": "0",
        **{
            k: str(v)
            for k, v in cenv.items()
        }
    })
    if ev:
        argv, env = ev
        args, model = _args_from_tokens(argv)
        if "JOBSET_JOB_INDEX" not in cenv:
            env.pop("JOBSET_JOB_INDEX", None)
    else:
        args, model = _parse_serve_args(
            str(cenv.get("VLLM_SERVER_CMD") or "") or script)
        env = dict(cenv)
    meta = recipe.get("metadata", {})
    return {
        "job_id":
        meta.get("name"),
        "owner":
        _annotated_user(meta.get("annotations") or {}),
        "image":
        c.get("image"),
        "model_path":
        model[0] if model else None,
        "server_args":
        args,
        "env":
        env,
        # what the script itself set or changed at the `vllm` call, on top of container env
        "env_container":
        cenv,
        "env_from_script": {
            k: v
            for k, v in env.items() if cenv.get(k) != v
        },
        "serve_parse":
        "bash-eval" if ev else "regex-fallback (script exports not applied)",
        "bench_env":
        benv,
        "slice_workers":
        next((rj.get("replicas") for rj in recipe["spec"]["replicatedJobs"]
              if rj["name"] == c["replicated_job"]), None),
    }


# ---------------------------------------------------------------- checkpoint config


def _annotated_user(annotations: dict[str, Any]) -> str | None:
    """The submitting user, from the job platform's `<domain>/username` annotation."""
    return next(
        (v for k, v in sorted(annotations.items()) if k.endswith("/username")),
        None)


def read_checkpoint_config(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The checkpoint's config.json: for a server read from its vLLM log, from this host
    (`jobdata.read_checkpoint_config`) when it has it; else from the GCS mirror the recipe
    names."""
    return ((rec.get("serve_parse") == "vllm-log"
             and jobdata.read_checkpoint_config(rec))
            or _manifest_checkpoint_config(rec))


def _manifest_checkpoint_config(rec: dict[str, Any]) -> dict[str, Any] | None:
    """The checkpoint's config.json, from the GCS mirror the bench container names
    (`GCS_MODEL_SRC`). Read once at onboarding and stored in the snapshot, so later
    derivations are offline."""
    import subprocess
    src = rec.get("bench_env", {}).get("GCS_MODEL_SRC") or rec.get(
        "env", {}).get("GCS_MODEL_SRC")
    if not src:
        # recipes that copy `<prefix>/$MODEL_NAME` (infx-qwen35-agentx) or take MODEL_ID
        # (the generic bench recipes) instead of naming the mirror. The prefix is the job's
        # MODEL_GCS_PREFIX, else WMB_MODEL_GCS_PREFIX (e.g. gs://<bucket>/ckpt/hf); with
        # neither, there is no mirror to read.
        env = {**rec.get("env_container", {}), **rec.get("env", {})}
        mid = env.get("MODEL_NAME") or env.get("MODEL_ID")
        prefix = env.get("MODEL_GCS_PREFIX") or os.environ.get(
            "WMB_MODEL_GCS_PREFIX")
        if prefix and mid and "/" in mid and not mid.startswith(
            ("/", "gs://")):
            src = f"{prefix.rstrip('/')}/{mid}"
    if not src:
        return None
    r = subprocess.run(
        ["gcloud", "storage", "cat", f"{src.rstrip('/')}/config.json"],
        capture_output=True,
        text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"cannot read {src}/config.json: {r.stderr.strip()[:200]}")
    cfg = json.loads(r.stdout)
    cfg["_source"] = f"{src.rstrip('/')}/config.json"
    return cfg


# ---------------------------------------------------------------- snapshot


def snapshot(job_id: str,
             root: str | None = None,
             scan_log: bool = True) -> dict[str, Any]:
    """`jobdata.snapshot`, reading the server from an AgentX job's manifest when it has one."""
    return jobdata.snapshot(job_id,
                            root,
                            scan_log,
                            read_recipe=read_recipe,
                            read_checkpoint_config=read_checkpoint_config)

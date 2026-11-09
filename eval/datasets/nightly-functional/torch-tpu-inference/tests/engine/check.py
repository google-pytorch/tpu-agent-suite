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
"""Verify the model was deployed with the requested inference engine.

Evidence is gathered from the artifacts the inference workflow writes
(deployments.json, verification.json, launch_vllm.sh / fastapi_server.py) and
from the commands the agent ran (`compute submit --recipe ...`,
`verify_inference.py --engine ...`). The check passes only if the expected
engine was deployed and the other engine was not.
"""

import json
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion
from rewardkit.criteria._trajectory import collect_tool_calls, load_trajectory

EXPECTED_ENGINE = "vllm"
OTHER_ENGINE = "custom_fastapi"

_RECIPE_ENGINES = {
    "launch_vllm.sh": "vllm",
    "fastapi_server.py": "custom_fastapi",
}
_CMD_TOOLS = {
    "run_shell_command",
    "run_command",
    "bash",
    "execute_command",
    "sh",
    "terminal",
    "exec",
}
_SUBMIT_RE = re.compile(r"compute\s+submit\b.*?--recipe[\s=]+(\S+)", re.DOTALL)
# Only a run of the script names an engine: `cat verify_inference.py` or a
# grep for it does not, and `--help` deploys nothing.
_VERIFY_RE = re.compile(
    r"\bpython[\d.]*(?:\s+-\S+)*\s+"
    r"(?:\"[^\"\n]*/|'[^'\n]*/|['\"]?(?:\S*/)?)verify_inference\.py['\"]?(.*)",
    re.DOTALL)
_HELP_RE = re.compile(r"(?:^|\s)(?:-h|--help)\b")
_ENGINE_FLAG_RE = re.compile(r"--engine[\s=]+['\"]?([a-z_]+)")


def _search_dirs(workspace: Path) -> list[Path]:
    return [
        workspace,
        Path("/workspace"),
        Path("/logs/artifacts"),
        Path("/root/.tpu-agent-suite/runs"),
        Path.home() / ".tpu-agent-suite/runs",
    ]


def _find_files(workspace: Path, name: str) -> list[Path]:
    found: dict[str, Path] = {}
    for d in _search_dirs(workspace):
        try:
            if not d.is_dir():
                continue
            for p in d.glob(f"**/{name}"):
                if p.is_file():
                    found[str(p.resolve())] = p
        except Exception:
            pass
    return list(found.values())


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def _artifact_engines(workspace: Path) -> set[str]:
    """Engines named by deployments.json, verification.json, and recipes."""
    engines: set[str] = set()
    for path in _find_files(workspace, "deployments.json"):
        data = _load_json(path)
        if isinstance(data, dict):
            for dep in data.get("deployments") or []:
                if isinstance(dep, dict) and dep.get("engine"):
                    engines.add(str(dep["engine"]).lower())
    for path in _find_files(workspace, "verification.json"):
        data = _load_json(path)
        if isinstance(data, dict) and data.get("engine"):
            engines.add(str(data["engine"]).lower())
    for recipe, engine in _RECIPE_ENGINES.items():
        if _find_files(workspace, recipe):
            engines.add(engine)
    return engines


def _collect_commands() -> list[str]:
    calls: list[dict[str, Any]] = []
    for traj in ["/logs/agent/trajectory.json", "/logs/trajectory.json"]:
        data = load_trajectory(traj)
        if data:
            calls.extend(collect_tool_calls(data))

    log_dir = Path("/logs/agent")
    if log_dir.is_dir():
        for p in log_dir.glob("**/*"):
            if not (p.is_file() and p.suffix in (".log", ".json", ".jsonl")):
                continue
            try:
                lines = p.read_text(encoding="utf-8",
                                    errors="ignore").splitlines()
            except Exception:
                continue
            for line in lines:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                su = obj.get("step_update") or {}
                for key in ("tool_info", "tool_call"):
                    tc = su.get(key) if isinstance(su, dict) else None
                    if isinstance(tc, dict):
                        calls.append({
                            "name":
                            tc.get("name") or tc.get("function_name"),
                            "args":
                            tc.get("args") or tc.get("arguments")
                            or tc.get("parameters") or {},
                        })
                for tc in obj.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        calls.append(tc)

    commands: list[str] = []
    for tc in calls:
        name = str(tc.get("function_name") or tc.get("name") or "").lower()
        args = tc.get("arguments") or tc.get("args") or {}
        if name not in _CMD_TOOLS or not isinstance(args, dict):
            continue
        cmd = (args.get("command") or args.get("CommandLine")
               or args.get("cmd") or args.get("command_line") or "")
        if cmd:
            commands.append(str(cmd))
    return commands


def _command_engines(commands: list[str]) -> set[str]:
    """Engines named by `compute submit --recipe` and verify_inference.py."""
    engines: set[str] = set()
    for cmd in commands:
        for match in _SUBMIT_RE.finditer(cmd):
            recipe = match.group(1).strip("'\"")
            for name, engine in _RECIPE_ENGINES.items():
                if recipe.endswith(name):
                    engines.add(engine)
        for match in _VERIFY_RE.finditer(cmd):
            # The documented invocation continues onto the next line with
            # `\`, and --engine is on that line.
            args = re.sub(r"\\\n", " ", match.group(1)).split("\n")[0]
            if _HELP_RE.search(args):
                continue
            flag = _ENGINE_FLAG_RE.search(args)
            # verify_inference.py defaults to custom_fastapi.
            engines.add(flag.group(1) if flag else "custom_fastapi")
    return engines


@criterion(description=(
    f"Verify the model was deployed with {EXPECTED_ENGINE}, not {OTHER_ENGINE}"
))
def check_engine_used(workspace: Path) -> bool:
    engines = _artifact_engines(workspace) | _command_engines(
        _collect_commands())
    print(f"Engines observed: {sorted(engines) or 'none'}")
    return EXPECTED_ENGINE in engines and OTHER_ENGINE not in engines

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
"""RewardKit verifier to calculate the framework-directed turn penalty from trajectory.

Applies an exponential decay penalty for every framework-directed turn (F):
    framework_directed_penalty = gamma ** F
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from rewardkit import criterion

DEFAULT_GAMMA = 0.85

# Framework / hook intervention messages and continuation prompts injected into
# the conversation when the framework intercepts an error or forces a handoff.
_FRAMEWORK_INTERVENTION_RE = re.compile(
    r"(?:"
    r"Delegation payload to\s+['\"][^'\"]+['\"]\s+violated accepts schema"
    r"|Subagent\s+['\"][^'\"]+['\"]\s+output (?:must be a valid JSON object conforming to its produces schema|violated produces contract)"
    r"|The\s+\S+\s+skill doesn't allow\s+`[^`]+`"
    r"|Delegate this step to the specialist agent that handles it"
    r"|Denied by hook\s+\S+"
    r"|Cancelled by hook\s+\S+"
    r"|Resumed by hook\s+\S+"
    r"|Session Start Denied:"
    r"|Backend refused the request\s+\(attempt\s+\d+\s+of\s+\d+\)"
    r"|\[Early CLI Crash"
    r"|<user-prompt-submit-hook>"
    r"|hook_blocking_error"
    r"|(?:PreToolUse|SubagentStop|Stop):\S+\s+hook\s+(?:returned\s+blocking\s+error|blocked|denied)"
    r")",
    re.IGNORECASE,
)

# Depth tag prefix injected by session_supervisor.sh, e.g. "[Depth 0 | tpu-agent] ".
_DEPTH_TAG_PREFIX_RE = re.compile(r"^\s*\[Depth\s+\d+(?:\s*\|\s*[^\]]+)?\]\s*")

_TEXT_KEYS = (
    "text",
    "content",
    "output",
    "reason",
    "followup_message",
    "additionalContext",
    "ephemeralMessage",
)
_DIRECT_INTERVENTION_KEYS = ("followup_message", "permissionDecisionReason")
_CONTEXT_INTERVENTION_KEYS = (
    "additionalContext",
    "additional_context",
    "ephemeralMessage",
)
_BLOCKING_DECISIONS = {"deny", "block", "cancel", "continue", "retry"}


def _find_and_load_trajectory(workspace: Path | None) -> dict[str, Any] | None:
    """Finds and loads the ATIF trajectory JSON from standard candidate paths."""
    candidates: list[Path] = []
    for env_var in ("ATIF_TRAJECTORY", "TRAJECTORY_PATH"):
        if os.environ.get(env_var):
            candidates.append(Path(os.environ[env_var]))
    # Harness-owned paths only: a trajectory.json in the workspace is the
    # agent's to write, so it cannot be what scores the agent.
    del workspace
    candidates.extend([
        Path("/logs/agent/trajectory.json"),
        Path("/logs/trajectory.json"),
    ])

    for path in candidates:
        if path and path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                continue
    return None


def _load_verifier_params() -> dict[str, Any]:
    p = Path("/tests/verifier_params.json")
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        val = data.get(Path(__file__).parent.name, data)
        return val if isinstance(val, dict) else data
    except (json.JSONDecodeError, OSError):
        return {}


def _extract_text(content: Any) -> str:
    """Normalizes a message or ObservationResult content payload into a string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                val = next(
                    (item[k] for k in _TEXT_KEYS if item.get(k) is not None),
                    None,
                )
                if val is not None:
                    parts.append(str(val))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return json.dumps(content) if isinstance(content, dict) else str(content)


def _clean_snippet(text: str, limit: int = 80) -> str:
    return _DEPTH_TAG_PREFIX_RE.sub("", text).strip()[:limit]


def _check_dict_intervention(d: dict[str, Any] | None) -> str | None:
    """Returns a short reason if metadata dict `d` records a hook intervention."""
    if not isinstance(d, dict):
        return None

    for key in _DIRECT_INTERVENTION_KEYS:
        val = d.get(key)
        if isinstance(val, str) and val.strip():
            return f"{key}:{val.strip()[:80]}"

    for key in _CONTEXT_INTERVENTION_KEYS:
        val = d.get(key)
        if isinstance(val, str) and _FRAMEWORK_INTERVENTION_RE.search(val):
            return f"{key}:{val.strip()[:80]}"

    decision = str(d.get("decision") or d.get("permissionDecision")
                   or "").strip().lower()
    if decision in _BLOCKING_DECISIONS:
        reason = str(
            d.get("reason") or d.get("permissionDecisionReason")
            or "").strip()
        return f"decision={decision}" + (f":{reason[:80]}" if reason else "")

    if nested := _check_dict_intervention(d.get("hookSpecificOutput")):
        return nested

    for item in d.get("injectSteps") or []:
        if nested := _check_dict_intervention(item):
            return f"injectSteps:{nested}"

    return None


def _get_step_intervention_reason(
    step: dict[str, Any],
    seen_initial_user_prompt: bool,
) -> str | None:
    """Returns the intervention reason if `step` is framework-directed, else None."""
    source = str(step.get("source") or "").strip().lower()
    msg = _extract_text(step.get("message"))
    snippet = _clean_snippet(msg)
    extra = step.get("extra") if isinstance(step.get("extra"), dict) else {}

    # 1. Step-level hook metadata.
    if meta_reason := _check_dict_intervention(extra):
        return f"step_meta({meta_reason})"

    # 2. User-sourced turns: hook-injected prompts or non-sidechain follow-ups
    # after the initial benchmark prompt.
    if source == "user":
        if _FRAMEWORK_INTERVENTION_RE.search(msg):
            return f"injected_user_turn({snippet})"
        if seen_initial_user_prompt and not extra.get("is_sidechain",
                                                      False) and snippet:
            return f"followup_user_turn({snippet})"

    # 3. Agent message text containing synthesized crash/hook intervention markers.
    if _FRAMEWORK_INTERVENTION_RE.search(msg):
        return f"message_intervention({snippet})"

    # 4. Observation results carrying hook denials or forced handoffs.
    obs = step.get("observation")
    for res in (obs.get("results") or []) if isinstance(obs, dict) else []:
        if not isinstance(res, dict):
            continue
        res_extra = res.get("extra") if isinstance(res.get("extra"),
                                                   dict) else {}
        if reason := _check_dict_intervention(res_extra):
            return f"observation_meta({reason})"
        if reason := _check_dict_intervention(
                res_extra.get("tool_result_metadata")):
            return f"tool_result_meta({reason})"
        res_text = _extract_text(res.get("content"))
        if res_text and _FRAMEWORK_INTERVENTION_RE.search(res_text):
            return f"tool_intervention({_clean_snippet(res_text)})"

    return None


def _collect_framework_directed_turns(
        data: dict[str, Any]) -> list[tuple[Any, str]]:
    """Collects (step_id, reason) for every framework-directed turn in the trajectory."""
    interventions: list[tuple[Any, str]] = []
    seen_initial_user_prompt = False

    for idx, step in enumerate(data.get("steps") or []):
        if not isinstance(step, dict):
            continue
        if reason := _get_step_intervention_reason(step,
                                                   seen_initial_user_prompt):
            interventions.append((step.get("step_id", idx + 1), reason))

        extra = step.get("extra") if isinstance(step.get("extra"),
                                                dict) else {}
        if str(step.get("source")
               or "").strip().lower() == "user" and not extra.get(
                   "is_sidechain", False):
            seen_initial_user_prompt = True

    return interventions


@criterion(
    description=
    "Calculate exponential decay penalty (gamma^F) for framework-directed turns"
)
def framework_directed_penalty(workspace: Path) -> float:
    """Calculates framework-directed turn penalty from Harbor's ATIF trajectory."""
    traj = _find_and_load_trajectory(workspace)
    if not traj:
        print(
            "framework_directed_penalty: [FAIL] No trajectory found in candidates"
        )
        return 0.0

    gamma = float(_load_verifier_params().get("gamma", DEFAULT_GAMMA))
    interventions = _collect_framework_directed_turns(traj)
    f_count = len(interventions)
    score = float(gamma**f_count)

    details = f" interventions={interventions}" if interventions else ""
    print(f"framework_directed_penalty: F={f_count}, gamma={gamma:.4f}, "
          f"score={score:.4f}{details}")
    return score

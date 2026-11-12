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
"""String template generators for Harbor task bundles."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from inspect import cleandoc
from pathlib import Path
from typing import Any

from eval.utils import find_repo_root

from . import schema

_BARE_KEY_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(key: str) -> str:
    """Formats a TOML bare key or quoted string key."""
    return key if _BARE_KEY_RE.match(key) else json.dumps(key)


def toml_escape(val: Any) -> str:
    """Formats Python primitives and inline collections into valid TOML values."""
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (list, tuple)):
        return f"[{', '.join(toml_escape(x) for x in val)}]"
    if isinstance(val, Mapping):
        inner = ", ".join(f"{_toml_key(str(k))} = {toml_escape(v)}"
                          for k, v in val.items() if v is not None)
        return f"{{{inner}}}"
    return json.dumps(str(val))


def _is_table_array(val: Any) -> bool:
    return (isinstance(val, Sequence) and not isinstance(val, (str, bytes))
            and bool(val) and all(isinstance(x, Mapping) for x in val))


def _emit_toml_table(
    table: Mapping[str, Any],
    path: tuple[str, ...],
    header_kind: str | None,
    blocks: list[str],
) -> None:
    scalars: list[tuple[str, Any]] = []
    children: list[tuple[str, Any]] = []
    for raw_k, val in table.items():
        if val is None:
            continue
        key = _toml_key(str(raw_k))
        if isinstance(val, Mapping) or _is_table_array(val):
            children.append((key, val))
        else:
            scalars.append((key, val))

    lines: list[str] = []
    dotted = ".".join(path)
    if header_kind == "array":
        lines.append(f"[[{dotted}]]")
    elif header_kind == "table" and scalars:
        lines.append(f"[{dotted}]")

    lines.extend(f"{k} = {toml_escape(v)}" for k, v in scalars)
    if lines:
        blocks.append("\n".join(lines))

    for key, val in children:
        child_path = (*path, key)
        if isinstance(val, Mapping):
            if val:
                _emit_toml_table(val, child_path, "table", blocks)
        else:
            for item in val:
                _emit_toml_table(item, child_path, "array", blocks)


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serializes a nested Python mapping into clean, multi-section TOML."""
    blocks: list[str] = []
    _emit_toml_table(data, (), None, blocks)
    return "\n\n".join(blocks) + "\n" if blocks else ""


def _discover_prebuilt_images(repo_root: Path | None = None) -> dict[str, str]:
    """Maps prebuilt eval/images/* names to their /etc/<name>/constraints.txt path."""
    root = repo_root if repo_root is not None else find_repo_root()
    images_dir = root / "eval" / "images"
    if not images_dir.is_dir():
        return {
            name: f"/etc/{name}/constraints.txt"
            for name in ("nexus-host", "vllm-on-host", "sglang-on-host")
        }
    discovered: dict[str, str] = {}
    for entry in sorted(images_dir.iterdir()):
        dockerfile = entry / "Dockerfile"
        if not dockerfile.is_file():
            continue
        constraints = f"/etc/{entry.name}/constraints.txt"
        if constraints in dockerfile.read_text(encoding="utf-8"):
            discovered[entry.name] = constraints
    return discovered


def _prebuilt_image_constraints(
    base_image: str,
    repo_root: Path | None = None,
) -> str | None:
    """Returns /etc/<name>/constraints.txt when base_image is a prebuilt image."""
    image_ref = base_image.split("@", 1)[0].rsplit("/", 1)[-1].split(":", 1)[0]
    return _discover_prebuilt_images(repo_root=repo_root).get(image_ref)


def _script_invocation(target_expr: str, script_name: str) -> str:
    """Wraps target_expr with the appropriate interpreter for .py/.sh scripts."""
    if script_name.endswith(".py"):
        return f"python3 {target_expr}"
    if script_name.endswith(".sh"):
        return f"bash {target_expr}"
    return target_expr


def _setup_script_invocation(script_name: str) -> str:
    """Builds a shell command executing a staged setup script under /tmp/_harbor_setup."""
    return _script_invocation(f"/tmp/_harbor_setup/{shlex.quote(script_name)}",
                              script_name)


def generate_dockerfile(
    env: schema.EnvironmentConfig,
    has_data: bool = False,
    setup_scripts: Sequence[str] | None = None,
) -> str:
    """Generates a clean environment/Dockerfile tailored to the base image."""
    resolved_setup = (list(setup_scripts) if setup_scripts is not None else [
        Path(s).name for s in env.setup_scripts if s and s.strip()
    ])
    setup_lines: list[str] = []
    if resolved_setup:
        setup_cmds = " && ".join([
            *(_setup_script_invocation(s) for s in resolved_setup),
            "rm -rf /tmp/_harbor_setup",
        ])
        setup_lines = [
            "COPY setup/ /tmp/_harbor_setup/",
            f"RUN {setup_cmds}",
        ]

    copy_dest = f"{env.workdir.rstrip('/')}/"
    quoted_workdir = shlex.quote(env.workdir)

    constraints_path = _prebuilt_image_constraints(env.base_image)
    if constraints_path is not None:
        blocks = [f"FROM --platform=linux/amd64 {env.base_image}"]

        if env.apt_packages:
            dedup_apt = list(dict.fromkeys(env.apt_packages))
            quoted_apt = " ".join(shlex.quote(pkg) for pkg in dedup_apt)
            blocks.append(
                cleandoc(f"""
                    RUN apt-get update && apt-get install -y --no-install-recommends \\
                        {quoted_apt} \\
                        && rm -rf /var/lib/apt/lists/*
                """))

        if env.pip_packages:
            dedup_pip = list(dict.fromkeys(env.pip_packages))
            quoted_pip = " ".join(shlex.quote(pkg) for pkg in dedup_pip)
            blocks.append(
                f"RUN pip install --no-cache-dir -c {constraints_path} {quoted_pip}"
            )

        blocks.append(f"WORKDIR {env.workdir}")
        if has_data:
            blocks.append(f"COPY data/ {copy_dest}")
        blocks.extend(setup_lines)
        blocks.append(f"RUN chmod -R 777 {quoted_workdir}")
        return "\n\n".join(blocks) + "\n"

    pip_pkgs = list(dict.fromkeys(["harbor-rewardkit", *env.pip_packages]))
    quoted_pip = " ".join(shlex.quote(p) for p in pip_pkgs)

    combined_apt = list(
        dict.fromkeys([
            "curl", "ca-certificates", "gnupg", "jq", "sudo", *env.apt_packages
        ]))
    apt_lines = "\n".join(f"    {shlex.quote(pkg)} \\" for pkg in combined_apt)

    tail_lines: list[str] = []
    if has_data:
        tail_lines.append(f"COPY data/ {copy_dest}")
    tail_lines.extend(setup_lines)
    if has_data or tail_lines:
        tail_lines.append(f"RUN chmod -R 777 {quoted_workdir}")
    data_copy_section = "\n".join(tail_lines) + "\n" if tail_lines else ""

    dockerfile = cleandoc(f"""
        FROM --platform=linux/amd64 {env.base_image}

        # System packages
        RUN apt-get update && apt-get install -y --no-install-recommends \\
        {apt_lines}
            && rm -rf /var/lib/apt/lists/*

        # Create non-root agent user (uid 1000) and kube-gate system user (uid 1001)
        RUN groupadd -g 1000 agent && \\
            useradd -u 1000 -g agent -M -d /root -s /bin/bash agent && \\
            groupadd -g 1001 kube-gate && \\
            useradd -u 1001 -g kube-gate --system --no-create-home -s /usr/sbin/nologin kube-gate && \\
            mkdir -p /etc/kube-gate /usr/libexec /root {quoted_workdir} && \\
            chown -R kube-gate:kube-gate /etc/kube-gate && \\
            chmod 0700 /etc/kube-gate && \\
            chown -R agent:agent /root {quoted_workdir}

        # Install kubectl for Unified Compute cluster access
        RUN ARCH=$(dpkg --print-architecture) && \\
            curl -fsSL "https://dl.k8s.io/release/v1.31.0/bin/linux/${{ARCH}}/kubectl" -o /usr/local/bin/kubectl && \\
            chmod +x /usr/local/bin/kubectl

        # Install gke-gcloud-auth-plugin for GKE cluster authentication
        RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \\
            > /etc/apt/sources.list.d/google-cloud-sdk.list && \\
            curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \\
            | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg && \\
            apt-get update && apt-get install -y --no-install-recommends google-cloud-cli-gke-gcloud-auth-plugin \\
            && rm -rf /var/lib/apt/lists/*

        # Python dependencies
        RUN pip install --no-cache-dir {quoted_pip}

        WORKDIR {env.workdir}
    """)

    return f"{dockerfile}\n\n{data_copy_section}".rstrip() + "\n"


def generate_task_toml(
    case: schema.TestCase,
    suite_name: str,
    env: schema.EnvironmentConfig,
    eval_cfg: schema.EvaluationConfig | None = None,
    prompt_text: str | None = None,
) -> str:
    """Generates task.toml configuration for Harbor."""
    raw_prompt = prompt_text if prompt_text is not None else case.prompt
    if case.description and case.description.strip():
        desc = case.description.strip()
    else:
        desc = (raw_prompt.strip().splitlines()[0]
                if raw_prompt and raw_prompt.strip() else case.name)

    task_name = case.task_name or f"{suite_name}/{case.name}"
    agent_timeout = (case.timeout_sec if case.timeout_sec is not None else
                     (eval_cfg.timeout_sec if eval_cfg is not None else 300.0))
    verifier_timeout = (
        case.verifier_timeout_sec if case.verifier_timeout_sec is not None else
        (eval_cfg.verifier_timeout_sec if eval_cfg is not None else 300.0))

    eff_env = env.merged_with(case.environment)
    eff_verifier_env = {
        **(eval_cfg.verifier_env if eval_cfg is not None else {}),
        **case.verifier_env,
    }
    doc: dict[str, Any] = {
        "version":
        "1.0",
        "artifacts": [
            {
                "source": eff_env.workdir,
                "destination": "workspace",
            },
            {
                "source": "/root/.tpu-agent-suite/runs",
                "destination": "tpu_agent_suite_runs",
            },
        ],
        "task": {
            "name": task_name,
            "description": desc,
        },
        "metadata": {
            "category": suite_name,
            **(case.metadata or {}),
        },
        "agent": {
            "timeout_sec": float(agent_timeout),
        },
        "verifier": {
            "timeout_sec": float(verifier_timeout),
            "env": dict(sorted(eff_verifier_env.items())),
        },
        "environment": {
            "build_timeout_sec":
            (float(eff_env.build_timeout_sec)
             if eff_env.build_timeout_sec is not None else None),
            "workdir":
            eff_env.workdir,
            "cpus":
            eff_env.cpus,
            "memory_mb":
            eff_env.memory_mb,
            "storage_mb":
            eff_env.storage_mb,
            "env":
            dict(sorted(eff_env.env_vars.items())),
        },
    }
    return dumps_toml(doc)


def generate_reward_toml(
    aggregation: str = "weighted_mean",
    weights: dict[str, float] | None = None,
) -> str:
    """Generates tests/reward.toml configuration for RewardKit."""
    normalized_aggregation = aggregation.replace("_", "-")
    entry: dict[str, Any] = {
        "name": "reward",
        "aggregation": normalized_aggregation,
    }
    if weights:
        if normalized_aggregation != "weighted-sum" and any(
                w < 0 for w in weights.values()):
            raise ValueError(
                "Negative verifier weights require reward_aggregation='weighted-sum' "
                f"(got {aggregation!r}).")
        entry["weights"] = {k: float(v) for k, v in weights.items()}
    return dumps_toml({"reward": [entry]})


def generate_judge_toml(expectations: Sequence[Any],
                        judge_model: str) -> str | None:
    """Generates tests/expectations/judge.toml for LLM rubric evaluation."""
    norm = [
        schema.ExpectationCriterion.coerce(e) for e in expectations
        if str(e).strip()
    ]
    if not norm:
        return None
    if any(exp.weight < 0 for exp in norm):
        raise ValueError("Negative weights are not supported in judge.toml "
                         "(uses weighted_mean scoring).")

    doc: dict[str, Any] = {
        "judge": {
            "judge": judge_model,
            "atif-trajectory": "/logs/agent/trajectory.json",
        },
        "criterion": [{
            "description": exp.criterion,
            "type": exp.type,
            "weight": float(exp.weight),
        } for exp in norm],
        "scoring": {
            "aggregation": "weighted_mean",
        },
    }
    return dumps_toml(doc)


def _hook_invocation(script_name: str) -> str:
    """Builds a shell command executing a staged script under ${TESTS_DIR}."""
    return _script_invocation(f'"${{TESTS_DIR}}"/{shlex.quote(script_name)}',
                              script_name)


def generate_test_sh(
        pre_verify: Sequence[str] = (),
        post_verify: Sequence[str] = (),
        pythonpath_dirs: Sequence[str] = (),
) -> str:
    """Generates authoritative 3-phase executable tests/test.sh."""
    sections: list[str] = [
        cleandoc("""
            #!/bin/bash

            export TESTS_DIR="${TESTS_DIR:-/tests}"
            export WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
            export LOGS_DIR="${LOGS_DIR:-/logs}"

            mkdir -p "${LOGS_DIR}/verifier" 2>/dev/null || true
            if [ -d "${WORKSPACE_DIR}" ]; then
              cd "${WORKSPACE_DIR}" || exit 1
            fi

            export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-global}"
            export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:-tpu-nexus-767207}"
            export CLOUD_ML_REGION="${CLOUD_ML_REGION:-global}"
            export LITELLM_NUM_RETRIES="${LITELLM_NUM_RETRIES:-8}"
            export LITELLM_RETRY_AFTER="${LITELLM_RETRY_AFTER:-3}"
            export DEFAULT_MAX_RETRIES="${DEFAULT_MAX_RETRIES:-10}"
            export INITIAL_RETRY_DELAY="${INITIAL_RETRY_DELAY:-2.0}"
            export MAX_RETRY_DELAY="${MAX_RETRY_DELAY:-60.0}"
        """)
    ]

    if pythonpath_dirs:
        joined = ":".join(f"${{TESTS_DIR}}/{d}" for d in pythonpath_dirs)
        sections.append(
            f'export PYTHONPATH="{joined}:${{TESTS_DIR}}${{PYTHONPATH:+:${{PYTHONPATH}}}}"'
        )

    if pre_verify:
        pre_lines = ["# Phase 1: Pre-verification hooks"]
        for hook in pre_verify:
            pre_lines.append(f"{_hook_invocation(hook)} || exit $?")
        sections.append("\n".join(pre_lines))

    sections.append(
        cleandoc("""
            # Phase 2: RewardKit execution with transient retry handling
            MAX_RETRIES=5
            RETRY_DELAY=3
            EXIT_CODE=0

            for ((i=1; i<=MAX_RETRIES; i++)); do
              if command -v rewardkit >/dev/null 2>&1; then
                rewardkit "${TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
              else
                uvx --with google-auth --from harbor-rewardkit rewardkit "${TESTS_DIR}" --workspace "${WORKSPACE_DIR}" --output "${LOGS_DIR}/verifier/reward.json" --max-concurrent-llm 2 && EXIT_CODE=0 && break || EXIT_CODE=$?
              fi
              if [ "$i" -lt "$MAX_RETRIES" ]; then
                echo "[test.sh] Verifier transient error ($EXIT_CODE). Retrying in ${RETRY_DELAY}s ($i/$MAX_RETRIES)..."
                sleep "$RETRY_DELAY"
                RETRY_DELAY=$((RETRY_DELAY * 2))
              fi
            done

            if [ "$EXIT_CODE" -ne 0 ]; then
              exit "$EXIT_CODE"
            fi
        """))

    if post_verify:
        post_lines = ["# Phase 3: Post-verification hooks"]
        for hook in post_verify:
            post_lines.append(f"{_hook_invocation(hook)} || exit $?")
        sections.append("\n".join(post_lines))

    sections.append("exit 0")
    return "\n\n".join(sections) + "\n"

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
"""Mount and environment advisor for evaluation packages.

Synthesizes harness-aware container bind mounts and environment variables
from a staged composite workspace, dynamically injecting them into Harbor configs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.utils import USER_SCOPE_ANSWERS, USER_SCOPE_DIST, find_repo_root


@dataclass(frozen=True)
class MountUpdates:
    """Container mounts and environment variables produced by SkillMountAdvisor."""

    mounts: list[dict[str, Any]]
    env: dict[str, str]


class SkillMountAdvisor:
    """Synthesizes container bind mounts and environment variables for staged skills.

    Architectural Rationale:
      Mounts are dynamically synthesized at runtime rather than statically hardcoded
      in Harbor YAML configs for three critical reasons:
      1. Late-Binding of Dynamic Paths: Staging generates content-addressed,
         cache-invalidated temporary directories (e.g. `/tmp/eval_staged_<user>_<hash>`).
         YAML configs cannot know these paths statically before runtime.
      2. Harbor Variable Expansion Limitation: Harbor expands `${VAR}` references
         under `agents[].env`, but does NOT expand variables under `environment.mounts`.
         Therefore, declarative `${STAGED_WORKSPACE}` references in YAML are unresolvable
         by Harbor at container launch time.
      3. Control arms: dynamic synthesis guarantees a control arm receives exactly
         0 package mounts without a second YAML file per arm.
    """

    def __init__(
        self,
        staged_workspace: Path | str | None,
        harness: str,
        central_pkg_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
        pkg_name: str = "tpu-agent-suite",
        pkg_version: str = "0.1.0",
        is_empty: bool = False,
        user_scope: bool = False,
        staged_dist: Path | str | None = None,
        answers_file: Path | str | None = None,
    ) -> None:
        self.staged_workspace = Path(
            staged_workspace) if staged_workspace else None
        self.harness = harness
        self.central_pkg_dir = Path(
            central_pkg_dir) if central_pkg_dir else None
        self.repo_root = Path(repo_root) if repo_root else find_repo_root()
        self.pkg_name = pkg_name
        self.pkg_version = pkg_version
        self.is_empty = is_empty
        self.user_scope = user_scope
        self.staged_dist = Path(staged_dist) if staged_dist else None
        self.answers_file = Path(answers_file) if answers_file else None
        # A user-scope arm runs the installer inside the trial, so it gets
        # what the installer reads rather than what it would have written.
        self.mounts_staged_install = not is_empty and not user_scope

    def get_mounts(self) -> list[dict[str, Any]]:
        """Generate list of bind mounts for the staged skills and session supervisor."""
        mounts: list[dict[str, Any]] = []
        if self.harness == "claude-code":
            # Mount package files if not an empty/vanilla arm
            if self.mounts_staged_install and self.staged_workspace:
                staged_claude = self.staged_workspace / ".claude"
                if staged_claude.is_dir():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_claude),
                        "target": "/workspace/.claude",
                        "read_only": True,
                    })

                staged_claude_md = self.staged_workspace / "CLAUDE.md"
                if staged_claude_md.is_file():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_claude_md),
                        "target": "/workspace/CLAUDE.md",
                        "read_only": True,
                    })

                staged_tpu_agent_suite = self.staged_workspace / ".tpu-agent-suite"
                if staged_tpu_agent_suite.is_dir():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_tpu_agent_suite),
                        "target": "/workspace/.tpu-agent-suite",
                    })

            # Claude session supervisor hooks (always mounted by default)
            claude_hooks = self.repo_root / "eval" / "hooks" / "claude_code"
            hook_sh = claude_hooks / "session_supervisor_hook.sh"
            supervisor_sh = claude_hooks / "session_supervisor.sh"
            if hook_sh.is_file():
                mounts.append({
                    "type": "bind",
                    "source": str(hook_sh),
                    "target":
                    "/usr/local/bin/claude_session_supervisor_hook.sh",
                    "read_only": True,
                })
            if supervisor_sh.is_file():
                mounts.append({
                    "type": "bind",
                    "source": str(supervisor_sh),
                    "target": "/usr/local/bin/claude_session_supervisor.sh",
                    "read_only": True,
                })

        elif self.harness == "antigravity":
            # Mount package files if not an empty/vanilla arm
            if self.mounts_staged_install and self.staged_workspace:
                staged_agents = self.staged_workspace / ".agents"
                if staged_agents.is_dir():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_agents),
                        "target": "/workspace/.agents",
                        "read_only": True,
                    })
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_agents),
                        "target": "/root/.agents",
                        "read_only": True,
                    })

                staged_agents_md = self.staged_workspace / "AGENTS.md"
                if staged_agents_md.is_file():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_agents_md),
                        "target": "/workspace/AGENTS.md",
                        "read_only": True,
                    })

                staged_tpu_agent_suite = self.staged_workspace / ".tpu-agent-suite"
                if staged_tpu_agent_suite.is_dir():
                    mounts.append({
                        "type": "bind",
                        "source": str(staged_tpu_agent_suite),
                        "target": "/workspace/.tpu-agent-suite",
                    })

            # Agy session supervisor hooks (always mounted by default)
            agy_hooks = self.repo_root / "eval" / "hooks" / "agy"
            hook_sh = agy_hooks / "session_supervisor_hook.sh"
            supervisor_sh = agy_hooks / "session_supervisor.sh"
            if hook_sh.is_file():
                mounts.append({
                    "type": "bind",
                    "source": str(hook_sh),
                    "target": "/usr/local/bin/session_supervisor_hook.sh",
                    "read_only": True,
                })
            if supervisor_sh.is_file():
                mounts.append({
                    "type": "bind",
                    "source": str(supervisor_sh),
                    "target": "/usr/local/bin/session_supervisor.sh",
                    "read_only": True,
                })

        # Central package cache mount (only for packaged arms)
        if self.mounts_staged_install and self.central_pkg_dir and self.central_pkg_dir.is_dir(
        ):
            pkg_dirname = self.central_pkg_dir.name or f"{self.pkg_name}@{self.pkg_version}"
            target_path = f"/root/.tpu-agent-suite/packages/{pkg_dirname}"
            mounts.append({
                "type": "bind",
                "source": str(self.central_pkg_dir),
                "target": target_path,
                "read_only": True,
            })
            # The central environment.json staging propagates sits at the
            # store's root (packages/<tag>/../..), not in the package.
            central_env = self.central_pkg_dir.parent.parent / "environment.json"
            if central_env.is_file():
                mounts.append({
                    "type": "bind",
                    "source": str(central_env),
                    "target": "/root/.tpu-agent-suite/environment.json",
                    "read_only": True,
                })
            central_drivers = self.central_pkg_dir.parent.parent / "drivers"
            if central_drivers.is_dir():
                mounts.append({
                    "type": "bind",
                    "source": str(central_drivers),
                    "target": "/root/.tpu-agent-suite/drivers",
                    "read_only": True,
                })

        if self.user_scope and not self.is_empty:
            for source, target in ((self.staged_dist, USER_SCOPE_DIST),
                                   (self.answers_file, USER_SCOPE_ANSWERS)):
                if source is None or not source.exists():
                    raise FileNotFoundError(
                        f"a user-scope arm installs from {target}, but"
                        f" nothing was staged to mount there ({source})")
                mounts.append({
                    "type": "bind",
                    "source": str(source),
                    "target": target,
                    "read_only": True,
                })

        return mounts

    def get_container_env(self) -> dict[str, str]:
        """Generate environment variables to be injected at container level (`environment.env`)."""
        env: dict[str, str] = {}
        if self.harness == "claude-code":
            env["BASH_ENV"] = "/usr/local/bin/claude_session_supervisor_hook.sh"
            if not self.is_empty:
                env["AGENT_PERSONA"] = "tpu-agent"
        elif self.harness == "antigravity":
            env["BASH_ENV"] = "/usr/local/bin/session_supervisor_hook.sh"
            # The CLI is pinned (.ci/pins.conf, installed by the eval adapter);
            # a self-update mid-trial would replace it with the latest one.
            env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "true"
            if not self.is_empty:
                env["AGENT_PERSONA"] = "tpu-agent"
        return env

    def get_updates(self) -> MountUpdates:
        """Generate structured mounts and environment variables to apply to config."""
        return MountUpdates(
            mounts=self.get_mounts(),
            env=self.get_container_env(),
        )

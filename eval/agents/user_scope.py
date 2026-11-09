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
"""Harbor agents that run the package's user-scope install inside the trial.

Every other config mounts a host-staged, project-scope install into
/workspace. A `--scope user` install writes into $HOME instead, and whether
each harness reads what landed there is only learnt by starting it outside
$HOME: these agents run the compiled installer in the container, as a user
would, before the harness starts in /workspace.

A config selects one by `import_path`; run.py then mounts the compiled
distribution and the installer answers in place of the project-scope tree.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from harbor.agents.installed.antigravity_cli import AntigravityCli
from harbor.agents.installed.claude_code import ClaudeCode
from harbor.agents.installed.codex import Codex

from eval.utils import USER_SCOPE_ANSWERS as ANSWERS
from eval.utils import USER_SCOPE_DIST as DIST


def install_command() -> str:
    """The install a user runs, minus the prompts and the host checks.

    First the host packages docs/install.md has a user install from
    requirements-client.txt: every session checks for them and stops to ask
    for any missing, before doing what it was asked. The install asks nothing
    about compute; the runtime then saves the `default` profile from the
    answers, as staging.py does for every other config, so a session starts
    set up rather than offering to set up.
    """
    runtime = (
        'rt=$(ls -d "$HOME"/.tpu-agent-suite/packages/*/framework/runtime.py'
        ' | head -1)')
    return (
        f"python3 -m pip install -q -r {DIST}/requirements-client.txt"
        f" && python3 {DIST}/framework/installer.py install {DIST} --scope user"
        " --non-interactive --skip-harness-check --skip-host-check"
        f" && {runtime}"
        ' && python3 "$rt" compute'
        ' --package "$(basename "$(dirname "$(dirname "$rt")")")"'
        f" setup apply --profile default --answers {ANSWERS}"
        ' --workspace "$HOME"')


def install_record(record: str, version: str, memory: str, *listed:
                   str) -> str:
    """Shell that writes, beside the trial's logs, what the install left for
    the session: the harness version, the directories it reads, and the
    memory file's size, digest and block markers.

    Not the block's content: the record sits where the session can read it,
    and the memory-loaded answers are in the block. It can fail without
    failing the setup; the install before it still can.
    """
    dirs = " ".join(f'"{d}"' for d in listed)
    return (
        '{ { echo "HOME=$HOME user=$(id -un)";'
        f" {version};"
        f" ls -la {dirs};"
        f' wc -c < "{memory}";'
        f' sha256sum "{memory}";'
        f" grep -n -e '<!-- .*:begin -->' -e '<!-- .*:end -->' \"{memory}\"; }}"
        f" > {record} 2>&1 || true; }}")


def _then(first: str, rest: str | None) -> str:
    return f"{first} && {rest}" if rest else first


class UserScopeClaudeCode(ClaudeCode):
    """Claude Code after a user-scope install.

    Harbor points CLAUDE_CONFIG_DIR into the trial's log directory and copies
    only ~/.claude/skills across. Copying the rest of ~/.claude stands in for
    the default config dir, so the session sees exactly what the installer
    wrote there and nothing it wrote anywhere else.
    """

    def _build_register_skills_command(self) -> str | None:
        bridge = 'cp -R "$HOME/.claude/." "$CLAUDE_CONFIG_DIR/"'
        return _then(f"{install_command()} && {bridge}",
                     super()._build_register_skills_command())


class UserScopeCodex(Codex):
    """Codex after a user-scope install.

    Harbor sets CODEX_HOME to /tmp/codex-home, outside $HOME, so the install
    runs without it and lands in ~/.codex; its files are then carried into
    Harbor's home, with config.toml appended to the one Harbor wrote.
    """

    def _build_register_skills_command(self) -> str | None:
        bridge = (
            f"env -u CODEX_HOME {install_command()}"
            ' && cp -R "$HOME/.codex/agents" "$CODEX_HOME/"'
            ' && cp "$HOME/.codex/AGENTS.md" "$CODEX_HOME/AGENTS.md"'
            ' && { [ ! -f "$HOME/.codex/hooks.json" ]'
            ' || cp "$HOME/.codex/hooks.json" "$CODEX_HOME/"; }'
            ' && cat "$HOME/.codex/config.toml" >> "$CODEX_HOME/config.toml"')
        return _then(bridge, super()._build_register_skills_command())


class UserScopeCodexHome(Codex):
    """Codex after a user-scope install into a CODEX_HOME under $HOME.

    The case UserScopeCodex steps around: Harbor's /tmp/codex-home is outside
    $HOME, which the installer refuses, so here Harbor's home moves under
    $HOME and the install runs with CODEX_HOME set, as it does for a user who
    moved it. Nothing is bridged: the session reads what the installer put in
    $CODEX_HOME, or does not.
    """

    # The trial runs as root, so this is under $HOME.
    _REMOTE_CODEX_HOME = PurePosixPath("/root/.config/codex")

    def _build_register_skills_command(self) -> str | None:
        # Where the install landed, kept beside the session's logs: the block
        # belongs in $CODEX_HOME, and nothing in ~/.codex.
        seen = install_record(
            (self.environment_logs_dir / "user-scope-install.txt").as_posix(),
            self._INSTALL_VERSION_COMMAND, "$CODEX_HOME/AGENTS.md",
            "$CODEX_HOME", "$HOME/.codex")
        return _then(f"{install_command()} && {seen}",
                     super()._build_register_skills_command())


class UserScopeAntigravityCli(AntigravityCli):
    """Antigravity after a user-scope install.

    Harbor leaves HOME alone for agy, so nothing is bridged: the session reads
    ~/.gemini where the installer wrote it, or does not.
    """

    def _build_register_skills_command(self) -> str | None:
        # What the session will find, kept beside its logs: when it answers
        # as if the memory block were absent, this says whether the file was
        # there and which agy read it.
        seen = install_record(
            (self.environment_logs_dir / "user-scope-install.txt").as_posix(),
            '"$HOME/.local/bin/agy" --version', "$HOME/.gemini/AGENTS.md",
            "$HOME/.gemini", "$HOME/.gemini/config")
        return _then(f"{install_command()} && {seen}",
                     super()._build_register_skills_command())

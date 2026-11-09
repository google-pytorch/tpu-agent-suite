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
"""Content checks for TPU agent suite packages.

`compiler.py verify` shows a package is structurally sound: every declared file
exists, the delegation graph is acyclic and reachable, tokens resolve. It says
nothing about the prose inside instruction files, which is where a migrated
package tends to rot -- links that pointed somewhere in the source tree, paths
left over from the origin build system, references to agents by their old names.

The checks here cover that gap. They are deliberately kept out of `verify`:
several are heuristics that can produce false positives on legitimate content.

Every check returns a list of `Finding`, so callers decide what is fatal.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import re
from pathlib import Path
from typing import Any, Iterable

from framework import frontmatter, jsonio, utils

TpuAgentSuiteError = utils.TpuAgentSuiteError
parse_frontmatter_document = frontmatter.parse_frontmatter_document

# Markdown inline link: [text](target)
_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
# An inline code span, and a package path inside one. package/README.md tells
# authors to reference bundled files by package-root-relative path rather than
# by markdown link, so this -- not _LINK -- is the reference style that
# dominates the prompts, and the code span is as often a whole command
# (`python3 scripts/tpu-kernel-development/run_autotune.py --payload ...`) as a bare path.
# Matching inside the span rather than requiring it to be the span is what
# catches the README's own worked example.
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_PATH_TOKEN = re.compile(
    r"(?<![\w./-])"
    r"([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+\.(?:py|md|json|txt|ya?ml|sh|toml))"
    r"\b")
# First path segments that are not package files: `workspace/` is the agent's
# run directory and `.tpu-agent-suite/` is the install destination. Both are resolved
# at run time, so neither can be checked against the package root.
_RUNTIME_PATH_ROOTS = frozenset({"workspace", ".tpu-agent-suite"})
# A first segment written as a template placeholder the prompt defines itself,
# e.g. tpu-sharding's `<artifacts> = .tpu-agent-suite/<pkg>/runs/<run_id>/artifacts`. A
# path rooted at one is unresolvable at lint time by construction, so reporting
# it as "does not resolve from the package root" is always a false positive.
_PLACEHOLDER_ROOT = re.compile(r"^<[^/>]+>$")
# `[label](target)` written inside a code span. `check_instruction_links` owns
# link targets, and the label is display text rather than a path reference, so
# `check_bare_paths` drops the whole construct instead of tokenizing either half
# -- otherwise `` `[0/report_0.md](<artifacts>/0/report_0.md)` `` is reported
# for its label even though its target is correctly written.
_INLINE_LINK_SPAN = re.compile(r"\[[^\]]*\]\([^)]*\)")
# A `{{...}}` token names a resource rather than spelling a path, and `verify`
# holds it to what its document ships. Left in, the text after the colon of
# `{{script:dir/tool.py}}` reads as the bare path `dir/tool.py`, which is
# declared as `scripts/dir/tool.py` and so is reported for being correct.
_TEMPLATE_TOKEN = re.compile(r"\{\{[^{}\n]*\}\}")
# The module a Python file imports, as opposed to the filename it is stored
# under. A declared script reaches a declared sibling this way.
_PY_IMPORT = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)",
                        re.MULTILINE)
# {{agent:name}} template token
_AGENT_TOKEN = re.compile(r"\{\{agent:([a-z0-9]+(?:-[a-z0-9]+)*)}}")
_SKILL_TOKEN = re.compile(r"\{\{skill:([a-z0-9]+(?:-[a-z0-9]+)*)}}")
# `env.get("key")` -- how scripts read the TPU agent suite environment
_ENV_GET = re.compile(r"""env\.get\(\s*["']([A-Za-z0-9_]+)["']""")
# Asking the user is the one capability whose tool differs per harness and
# which some harnesses deny a subagent outright, so an instruction naming one
# of these is wrong in at least one harness and often in the calling context
# too. Deliberately just this family: file and shell tools are covered by the
# portable-tool pattern below, and names like `list_dir` or `run_command` read
# as ordinary prose often enough to cost more in false positives than they
# catch.
_ASK_USER_TOOLS = (
    "ask_question",
    "AskUserQuestion",
    "request_user_input",
)
# Paths and invocations left over from the origin tree, each with
# the remediation to apply. The hint is the useful half of the finding: the
# match alone says something is wrong, the hint says what replaces it.
_LEGACY_PATTERNS = {
    "bazel": (re.compile(r"\bbazel\b", re.IGNORECASE),
              "TPU agent suite packages build via framework/compiler.py"),
    "origin-agent-path":
    (re.compile(r"\blearning/(?:agents|gemini|brain)/[A-Za-z0-9_/]+"),
     "origin path; bundled files live under scripts/ and references/"),
    "experimental-path":
    (re.compile(r"(?<![\w/])experimental/[A-Za-z0-9_/.-]+"),
     "origin experimental path; bundled files live under scripts/ and references/"
     ),
    "workstation-path":
    (re.compile(r"/(?:usr/local/google/home|google/src)/[A-Za-z0-9_/.-]+"),
     "origin workstation path; use relative or workspace-scoped path"),
    "internal-shortlink":
    (re.compile(r"(?<![\w/])go/[A-Za-z0-9_\-./]+"
                r"|\b(?:b|cl)/[0-9]+\b"
                r"|\b[a-zA-Z0-9_-]+\.(?:corp\.google\.com|googleplex\.com)\b"),
     "internal link/shortlink; not accessible outside internal network"),
    "agy-subagents-dir": (re.compile(r"\b[A-Za-z0-9_]+/subagents/"),
                          "prompts live at agents/<name>.md"),
    "agy-agent-name":
    (re.compile(r"(?<![/\w])[a-z][a-z0-9]*(?:_[a-z0-9]+)*_agent\b"),
     "name the agent with {{agent:<hyphen-case-name>}}"),
    "agy-invoke":
    (re.compile(r"\b(?:invoke_subagent|define_subagent)\b"),
     "delegate with {{agent:<name>}}; the harness resolves the call"),
    "agy-tool":
    (re.
     compile(
         r"\b(?:view_file|write_to_file|replace_file_content|multi_replace_file_content|ask_question|code_search|search_for_files_codesearch|skill_search|run_command|manage_task|manage_subagents|send_message|search_web|read_url_content|notebook_edit|generate_image|find_by_name|grep_search|list_dir)\b"
     ),
     "Antigravity internal tool; use portable tool primitives (read/write/edit/search/shell/web_search/web_fetch)"
     ),
    # Claude Code tool names. Portable grants are the lowercase primitives, so
    # a capitalized tool name in prose names something codex and antigravity
    # do not have.
    "harness-tool":
    (re.compile(r"\b(?:TodoWrite|WebFetch|WebSearch|NotebookEdit|SlashCommand)"
                r"\b|\b(?:Bash|Glob|Grep|Task|Read|Write|Edit)\s+tool\b"),
     "not a portable tool; grant one of read/search/shell/write/edit/web_search/web_fetch"
     ),
    # Telling the agent to use one harness's ask tool, as opposed to naming it.
    # Checked in references too: a reference that says "use `ask_question`" is
    # an instruction wherever it is read, and only some harnesses have it.
    "harness-tool-directive":
    (re.compile(r"\b(?i:use|call|invoke)\s+(?:the\s+)?`?(?:" +
                "|".join(_ASK_USER_TOOLS) + r")`?(?:\s+tool)?\b"),
     "only some harnesses have this tool, and a subagent may have no user to"
     " ask; say \"ask the user\" and let the harness pick the tool, or return"
     " status needs_input (see package/README.md, Step 4.6)"),
}
# The subset of _LEGACY_PATTERNS whose match is a *leak* rather than migration
# debt: an origin-tree path. Those are checked in every bundled file --
# references and scripts included -- because a reference doc shipped from a
# public repo leaks just as effectively as a prompt does. The remaining
# patterns stay instruction-only, per check_legacy_paths: a doc may
# legitimately discuss a Bazel fallback or name an Antigravity tool while
# explaining what replaced it.
_ORIGIN_LEAK_LABELS = frozenset({
    "origin-agent-path",
    "experimental-path",
    "workstation-path",
    "internal-shortlink",
})
# What a bundled reference is held to: the leaks above, and any instruction to
# use a harness-specific tool, which a reference carries as surely as a prompt.
_REFERENCE_LABELS = _ORIGIN_LEAK_LABELS | {"harness-tool-directive"}
_NON_FILE_SCHEMES = ("http://", "https://", "mailto:", "#")
# Root modules that only exist in the origin build tree. An import
# rooted at one of these can never resolve here, so unlike "this package is not
# installed" -- which is routine for tensorflow, torch_tpu, matplotlib, and
# friends -- it is unambiguously broken.
_ORIGIN_MODULE_ROOTS = frozenset({"skills", "experimental", "learning"})
# Imports that read `.xplane.pb` traces directly rather than through the xprof
# CLI: the XPlane protobuf bindings (under TensorFlow or the standalone tsl
# package), xprof's private conversion layer, and the TensorBoard plugin the
# CLI supersedes. Matched against the dotted import name.
_XPLANE_IMPORT = re.compile(
    r"(?:^|\.)(?:xplane_pb2|tsl\.profiler|xprof\.convert"
    r"|tensorboard_plugin_profile)(?:\.|$)")
# An xprof CLI invocation in prose or a code block: `xprof get_overview ...`.
# Every tool subcommand starts with one of these verbs.
_XPROF_INVOCATION = re.compile(
    r"\bxprof\s+(?:get|list|aggregate|verify|check|compute)_[a-z_]+\b")


@dataclasses.dataclass(frozen=True)
class Finding:
    """One content problem, located in a specific file.

    `check` is the name it is registered under in `CHECKS`, `agent` the agent it
    is reported against, `file` the package-relative path, `line` the 1-based
    line number (0 when the finding is about the file as a whole rather than a
    place in it), and `detail` the human-readable message.
    """

    check: str
    agent: str
    file: str
    line: int
    detail: str

    def key(self) -> str:
        """Identity for baselining, unchanged when surrounding lines move.

        Returns: `"<check>|<file>|<detail>"`. Deliberately excludes the line
        number, so an unrelated edit above a finding does not create a new one.
        """
        return f"{self.check}|{self.file}|{self.detail}"

    def __str__(self) -> str:
        """Returns the one-line `<file>:<line>: [<check>] <detail>` form."""
        return f"{self.file}:{self.line}: [{self.check}] {self.detail}"


def _iter_lines(path: Path) -> Iterable[tuple[int, str]]:
    """Reads `path` and returns its lines as (1-based line number, text)."""
    return enumerate(path.read_text(encoding="utf-8").splitlines(), 1)


def _skills_and_agents(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every skill and agent, each once.

    A skill that delegates is also present as its twin agent, sharing the
    instructions file; the pair is one unit here so nothing is reported
    twice.
    """
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for unit in (*manifest.get("skills", []), *manifest.get("agents", [])):
        key = unit.get("instructions") or unit.get("name", "")
        if key in seen:
            continue
        seen.add(key)
        units.append(unit)
    return units


def _asset_roots(manifest: dict[str, Any]) -> frozenset[str]:
    """First path components of the manifest's verbatim `copy` assets."""
    return frozenset(
        Path(item).parts[0] for item in manifest.get("copy", [])
        if Path(item).parts)


def _is_asset_path(target: str, package_dir: Path,
                   manifest: dict[str, Any]) -> bool:
    """Whether `target` is a file the package ships verbatim as an asset."""
    root = target.split("/", 1)[0]
    return root in _asset_roots(manifest) and (package_dir / target).is_file()


@dataclasses.dataclass(frozen=True)
class _Prompt:
    """One instruction file, with the contracts it is checked against.

    `agent` is the agent's name and `rel` its instructions path. `declared` is
    the files this agent bundles (scripts + references), `delegates` the agents
    it may invoke, and `delegate_declared` the files bundled by every agent it
    can reach transitively -- not bundled for *this* agent, but naming one is
    description rather than a missing declaration. See
    `_delegate_declarations`.
    """

    agent: str
    rel: str
    declared: frozenset[str]
    delegates: frozenset[str]
    delegate_declared: frozenset[str]
    own_delegates: frozenset[str] = frozenset()


def _delegate_declarations(
        manifest: dict[str, Any]) -> dict[str, frozenset[str]]:
    """Maps each agent to the files its delegates bundle.

    Walks the delegation graph transitively from each agent, since a prompt may
    describe any agent in its subtree rather than only a direct child. `verify`
    has already proven the graph acyclic, so `seen` is bookkeeping rather than a
    cycle guard.

    Returns: `{agent name: frozenset of scripts/references declared by every
    agent it can reach}`. Callers use it to exempt a path a prompt merely
    *describes* -- naming a delegate's file is description, not a missing
    declaration.
    """
    agents_by_name = {
        agent["name"]: agent
        for agent in _skills_and_agents(manifest)
    }
    delegates_of = {
        name: agent.get("delegates", [])
        for name, agent in agents_by_name.items()
    }

    def declared(name: str) -> set[str]:
        agent = agents_by_name.get(name)
        if agent is None:
            return set()
        return set(agent.get("scripts", [])) | set(agent.get("references", []))

    closure: dict[str, frozenset[str]] = {}
    for name in delegates_of:
        seen: set[str] = set()
        stack = list(delegates_of[name])
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(delegates_of.get(current, []))
        files: set[str] = set()
        for reachable in seen:
            files |= declared(reachable)
        closure[name] = frozenset(files)
    return closure


def _prompts(manifest: dict[str, Any]) -> list[_Prompt]:
    """Collects every prompt in the package, skills included.

    A skill is a prompt like any other, so no check skips it. One that
    declares no `scripts` or `references` is not exempt: any file it names is
    one the build will not bundle for it.

    Returns: a list of `_Prompt`, skills first, then the agents they reach, in
    the order they were found.
    """
    downstream = _delegate_declarations(manifest)
    return [
        _Prompt(
            unit["name"],
            unit["instructions"],
            frozenset(unit.get("scripts", []))
            | frozenset(unit.get("references", [])),
            frozenset(unit.get("delegates", [])),
            downstream.get(unit["name"], frozenset()),
            frozenset(unit.get("own_delegates", unit.get("delegates", []))),
        ) for unit in _skills_and_agents(manifest)
    ]


def _is_runtime_path(target: str) -> bool:
    """Whether a path resolves at run time rather than against the package.

    Four spellings qualify, none of which names a file the build can check: the
    install destination (`.tpu-agent-suite/...`), the agent's run directory
    (`workspace/...`), a path rooted at a placeholder the prompt defines itself
    (`<artifacts>/...`), and any of those written home-relative (`~/...`).

    Returns: True when `target` is one of those, so path checks skip it instead
    of reporting a file they cannot resolve.
    """
    # `~` is the reader's home directory, so nothing under it is a package path
    # whatever follows -- and what follows is usually `.tpu-agent-suite/`, since the
    # install destination is written both bare and home-relative.
    if target.startswith("~/"):
        return True
    # Resource tokens ({{runs}}, {{package}}, ...) resolve at compile time to
    # installed locations, never to files inside the package source tree.
    if target.startswith("{{"):
        return True
    root = target.split("/", 1)[0]
    return root in _RUNTIME_PATH_ROOTS or bool(_PLACEHOLDER_ROOT.match(root))


def _file_link_targets(line: str) -> Iterable[str]:
    """The package paths `line` links to, anchors dropped.

    Skips external URLs, bare anchors, and run-time paths, none of which name
    a file the build bundles.
    """
    for raw in _LINK.findall(line):
        if raw.startswith(_NON_FILE_SCHEMES):
            continue
        target = raw.split("#", 1)[0]
        if target and not _is_runtime_path(target):
            yield target


def _declared_owners(manifest: dict[str, Any],
                     keys: tuple[str, ...] = ("references", ),
                     suffix: str = "") -> list[tuple[str, str]]:
    """Each file declared under `keys` ending in `suffix`, with its owner.

    A file several units declare is one file in the build, so it is paired
    with the first unit that declares it.

    Returns: (package-relative path, owning unit name), sorted by path.
    """
    owners: dict[str, str] = {}
    for agent in _skills_and_agents(manifest):
        for key in keys:
            for rel in agent.get(key, []):
                if rel.endswith(suffix):
                    owners.setdefault(rel, agent["name"])
    return sorted(owners.items())


def check_instruction_links(package_dir: Path,
                            manifest: dict[str, Any]) -> list[Finding]:
    """Checks that every markdown link in a prompt names a declared file.

    `translate` flattens prompts to `agents/<name>.md` and copies only the files
    an agent declares in `scripts`/`references`, so a link relative to the
    source instructions file dangles in the built distribution, and a link to
    an undeclared file dangles even when it resolves in the source tree.
    External URLs, bare anchors, and run-time paths are skipped.

    Returns: a list of `Finding` (check `"instruction-links"`), one per bad
    link, whose detail distinguishes a file that exists but is undeclared from
    one that does not resolve at all. Enforced by `verify.verify_package`.
    """
    findings: list[Finding] = []
    for prompt in _prompts(manifest):
        for lineno, line in _iter_lines(package_dir / prompt.rel):
            for path in _file_link_targets(line):
                if path in prompt.declared:
                    continue
                if _is_asset_path(path, package_dir, manifest):
                    continue
                if (package_dir / path).is_file():
                    detail = (f"links to {path!r}, which exists but is not"
                              f" declared in this agent's scripts/references"
                              f" -- it will not be bundled")
                else:
                    detail = (
                        f"links to {path!r}, which is not a declared"
                        f" script/reference and does not resolve from the"
                        f" package root")
                findings.append(
                    Finding("instruction-links", prompt.agent, prompt.rel,
                            lineno, detail))
    return findings


def check_bare_paths(package_dir: Path, manifest: dict[str,
                                                       Any]) -> list[Finding]:
    """Checks that every backticked package path in a prompt is declared.

    package/README.md tells authors to reference bundled files by
    package-root-relative path rather than by markdown link, so this is the
    reference style `check_instruction_links` cannot see. Skipped: run-time
    paths, and paths declared by an agent this one delegates to -- naming a
    delegate's file describes its work rather than reaching for the file.
    `check_instruction_links` grants no such exemption on purpose, since a
    *link* tells the reader to open the target whoever else declares it.

    Heuristic and therefore advisory: a backticked path may be illustrative
    rather than a real reference.

    Returns: a list of `Finding` (check `"bare-paths"`), at most one per path
    per line, with the same exists-but-undeclared / does-not-resolve split as
    `check_instruction_links`.
    """
    findings: list[Finding] = []
    for prompt in _prompts(manifest):
        for lineno, line in _iter_lines(package_dir / prompt.rel):
            targets = [
                t for span in _CODE_SPAN.findall(line)
                for t in _PATH_TOKEN.findall(
                    _TEMPLATE_TOKEN.sub(" ", _INLINE_LINK_SPAN.sub(" ", span)))
            ]
            for target in dict.fromkeys(targets):
                if _is_runtime_path(target):
                    continue
                if target in prompt.declared:
                    continue
                if target in prompt.delegate_declared:
                    continue
                if _is_asset_path(target, package_dir, manifest):
                    continue
                if (package_dir / target).is_file():
                    detail = (f"names {target!r}, which exists but is not"
                              f" declared in this agent's scripts/references"
                              f" -- it will not be bundled")
                else:
                    detail = (f"names {target!r}, which is not a declared"
                              f" script/reference and does not resolve from"
                              f" the package root")
                findings.append(
                    Finding("bare-paths", prompt.agent, prompt.rel, lineno,
                            detail))
    return findings


def check_reference_links(package_dir: Path,
                          manifest: dict[str, Any]) -> list[Finding]:
    """Checks that links in a bundled reference doc land on bundled files.

    Reference docs are copied to their package-relative path, so a link survives
    the build only if its target is itself declared by some agent. Both
    spellings are accepted -- sibling-relative (the docs sit together) and
    package-root-relative (what the README recommends). External URLs, bare
    anchors, and run-time paths are skipped.

    Unlike the prose checks this one is decidable, not heuristic: a link either
    resolves to a bundled file or it dangles.

    Returns: a list of `Finding` (check `"reference-links"`), reported against
    the doc's first declaring agent. Enforced by `verify.verify_package`.
    """
    findings: list[Finding] = []
    declared_all = {
        rel
        for agent in _skills_and_agents(manifest)
        for rel in agent.get("scripts", []) + agent.get("references", [])
    }
    for rel, owner in _declared_owners(manifest, suffix=".md"):
        doc = package_dir / rel
        if doc.suffix != ".md" or not doc.is_file():
            continue
        for lineno, line in _iter_lines(doc):
            for target in _file_link_targets(line):
                # Both spellings survive a build, provided the target is itself
                # declared: sibling-relative (the docs sit together) and
                # package-root-relative (what the README recommends).
                candidates = {
                    os.path.normpath(Path(rel).parent / target).replace(
                        os.sep, "/"),
                    os.path.normpath(target).replace(os.sep, "/"),
                }
                if candidates & declared_all:
                    continue
                findings.append(
                    Finding(
                        "reference-links", owner, rel, lineno,
                        f"links to {target!r}, which is not a declared script"
                        f" or reference -- it will not be bundled"))
    return findings


def check_unused_declarations(package_dir: Path,
                              manifest: dict[str, Any]) -> list[Finding]:
    """Flags a declared script or reference that nothing in the bundle names.

    A declaration nothing references is dead weight in every install. A file
    counts as used when the agent's prompt names it, when any of the agent's
    *other* declared files names it by path or basename, or -- for Python --
    when a declared sibling imports it by module name. Indirect use is real and
    common (e.g. a driver script invoking a helper), so without those
    allowances the check is pure noise.

    Returns: a list of `Finding` (check `"unused-declarations"`) with line 0,
    since the finding is about the declaration rather than a place in the file.
    Each means nothing shipped alongside the file mentions it at all.
    """
    findings: list[Finding] = []
    for agent in _skills_and_agents(manifest):
        prompt = (package_dir /
                  agent["instructions"]).read_text(encoding="utf-8")
        declared = (
            agent.get("declared_scripts", agent.get("scripts", [])) +
            agent.get("declared_references", agent.get("references", [])))
        bodies = {
            rel: (package_dir / rel).read_text(encoding="utf-8",
                                               errors="ignore")
            for rel in declared if (package_dir / rel).is_file()
        }
        for rel in declared:
            base = Path(rel).name
            if rel in prompt or base in prompt:
                continue
            # A directory reference is a deliberate bundle: every file under
            # it ships, and the prompt names the directory or any of them.
            if (package_dir / rel).is_dir():
                continue
            if any(base in body or rel in body
                   for other, body in bodies.items() if other != rel):
                continue
            # A Python dependency is named by its module, not its filename:
            # `query_tpu_sharding.py` reaches `entry.py` as `from entry import ...`.
            # A script naming a helper as a literal string is caught above;
            # the import spelling needs its own match, or every correctly declared
            # helper module is reported.
            if rel.endswith(".py") and any(
                    Path(rel).stem in _PY_IMPORT.findall(body)
                    for other, body in bodies.items()
                    if other != rel and other.endswith(".py")):
                continue
            findings.append(
                Finding(
                    "unused-declarations", agent["name"], rel, 0,
                    f"declared by {agent['name']!r} but named by neither its"
                    f" prompt nor any other file it declares"))
    return findings


def check_delegation_consistency(package_dir: Path,
                                 manifest: dict[str, Any]) -> list[Finding]:
    """Compares each agent's `{{agent:...}}` tokens against its delegates.

    Reports both directions: a token naming an agent outside the declared set,
    and a declared delegate the prompt never references.

    An agent that declares *no* delegates is skipped in the first direction,
    because it cannot be invoking anything: `translate` appends the harness
    delegation tool only when `delegates` is non-empty. A `{{agent:...}}` token
    there addresses the *caller* ("report your status to the worker agent and
    request it to invoke <next phase>") and is what renders the prefixed,
    caller-dispatchable id, so it is load bearing rather than a defect.

    Heuristic and therefore advisory: an agent may declare a delegate it
    invokes only conditionally. Treat findings as prompts to look, not failures.

    Returns: a list of `Finding` (check `"delegation-consistency"`). The
    undeclared-token direction carries the token's line; the
    declared-but-unreferenced direction uses line 0.
    """
    findings: list[Finding] = []
    for prompt in _prompts(manifest):
        used: dict[str, int] = {}
        for lineno, line in _iter_lines(package_dir / prompt.rel):
            for name in (*_AGENT_TOKEN.findall(line),
                         *_SKILL_TOKEN.findall(line)):
                used.setdefault(name, lineno)
        for name, lineno in sorted(used.items()):
            if not prompt.delegates:
                break
            if name not in prompt.delegates:
                findings.append(
                    Finding(
                        "delegation-consistency", prompt.agent, prompt.rel,
                        lineno,
                        f"references {{{{agent:{name}}}}} but does not declare"
                        f" it as a delegate"))
        for name in sorted(prompt.own_delegates - set(used)):
            findings.append(
                Finding(
                    "delegation-consistency", prompt.agent, prompt.rel, 0,
                    f"declares delegate {name!r} but never references it in"
                    f" its instructions"))
    return findings


def check_legacy_paths(package_dir: Path,
                       manifest: dict[str, Any]) -> list[Finding]:
    """Flags origin-tree paths and tool names that did not survive migration.

    Instruction files are matched against every pattern in `_LEGACY_PATTERNS`:
    experimental/origin paths, Bazel, AGY spellings, and non-portable harness
    tool names.

    Bundled references and scripts are matched against `_ORIGIN_LEAK_LABELS`
    only -- a doc may legitimately document a Bazel fallback, so the full set
    is noise there, but an origin path is a leak wherever it ships, and a
    bundled reference reaches the reader exactly as a prompt does. Markdown
    references are also matched against `harness-tool-directive`, for the same
    reason.

    Returns: a list of `Finding` (check `"legacy-paths"`), at most one per
    (file, line, pattern), each detail pairing the matched text with the
    remediation hint. A file several agents declare is reported once, against
    its first owner.
    """
    findings: list[Finding] = []
    seen: set[tuple[str, int, str]] = set()

    def scan(agent: str, rel: str, labels: Iterable[str]) -> None:
        subset = [(label, *_LEGACY_PATTERNS[label]) for label in labels]
        for lineno, line in _iter_lines(package_dir / rel):
            for label, pattern, hint in subset:
                match = pattern.search(line)
                if match and (rel, lineno, label) not in seen:
                    seen.add((rel, lineno, label))
                    findings.append(
                        Finding("legacy-paths", agent, rel, lineno,
                                f"{label}: {match.group(0)!r} -- {hint}"))

    for prompt in _prompts(manifest):
        scan(prompt.agent, prompt.rel, _LEGACY_PATTERNS)

    # Declared assets, deduped by path: a file several agents declare is one
    # file in the build, so it should be reported once, against its first owner.
    for rel, agent_name in _declared_owners(manifest,
                                            ("references", "scripts")):
        if (package_dir / rel).is_file():
            labels = (_REFERENCE_LABELS
                      if rel.endswith(".md") else _ORIGIN_LEAK_LABELS)
            scan(agent_name, rel, labels)
    return findings


def _python_script_trees(
    package_dir: Path, manifest: dict[str, Any]
) -> Iterable[tuple[str, str, ast.Module | SyntaxError]]:
    """Parses every declared `.py` script and the `*_test.py` beside it.

    A file several agents reach is parsed once, for its first owner.

    Returns: (owning agent, package-relative path, syntax tree) in path
    order, with the `SyntaxError` in place of the tree for a file that does
    not parse.
    """
    owners: dict[Path, str] = {}
    for agent in _skills_and_agents(manifest):
        for script_rel in agent.get("scripts", []):
            script = package_dir / script_rel
            if script.suffix != ".py" or not script.is_file():
                continue
            owners.setdefault(script, agent["name"])
            for sibling in sorted(script.parent.glob("*_test.py")):
                owners.setdefault(sibling, agent["name"])

    for path, agent_name in sorted(owners.items()):
        tree: ast.Module | SyntaxError
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            tree = exc
        yield agent_name, path.relative_to(package_dir).as_posix(), tree


def check_legacy_imports(package_dir: Path,
                         manifest: dict[str, Any]) -> list[Finding]:
    """Flags Python imports rooted in the origin build tree.

    Parses every declared `.py` script plus the companion `*_test.py` files
    beside them -- those are left out of frontmatter on purpose, so nothing
    a skill declares reaches them, and they are exactly where this rot
    hides: a test nothing runs keeps its `from skills.foo.scripts import foo`
    long after `skills` stopped existing. Relative imports are ignored.

    Deliberately narrow: only roots in `_ORIGIN_MODULE_ROOTS` are reported.
    "Cannot be imported right now" is not the signal, since bundled scripts
    legitimately import tensorflow, torch_tpu, and matplotlib, none of which
    need to be installed to author the package.

    Returns: a list of `Finding` (check `"legacy-imports"`), one per offending
    import, plus one per file that cannot be parsed at all.
    """
    findings: list[Finding] = []
    for agent_name, rel, tree in _python_script_trees(package_dir, manifest):
        if isinstance(tree, SyntaxError):
            findings.append(
                Finding("legacy-imports", agent_name, rel, tree.lineno or 0,
                        f"cannot be parsed: {tree.msg}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name.split(".")[0] in _ORIGIN_MODULE_ROOTS:
                    findings.append(
                        Finding(
                            "legacy-imports", agent_name, rel, node.lineno,
                            f"imports {name!r}, which only exists in the origin"
                            f" build tree and cannot resolve here"))
    return findings


_INSTALLED_ASSET_PATTERNS = (
    (re.compile(r"\S*\.tpu-agent-suite/(?:[\w.@\-]+/)*environment\.json"),
     "query effective config with"
     " '{{runtime}} compute inspect --package {{package_tag}} --json'"
     " instead of naming the file"),
    (re.compile(r"\S*\.tpu-agent-suite/(?:[\w.@\-]+/)*knowledgebase"),
     "name a provenance page with"
     " {{data:tpu-knowledge-base/pages/<name>.md}}; the knowledge base is not"
     " installed"),
    (re.compile(r"\S*\.tpu-agent-suite/runs"),
     "write {{runs}} so the compiler resolves the installed location"),
    (re.compile(r"\S*\.tpu-agent-suite/packages"),
     "write {{package}} so the compiler resolves the installed location"),
    (re.compile(r"~/\.tpu-agent-suite"),
     "write a resource token so the compiler resolves the installed"
     " location"),
)


def check_workspace_config_paths(package_dir: Path,
                                 manifest: dict[str, Any]) -> list[Finding]:
    """Flags instruction files that spell an installed asset path literally.

    The installer decides where environment.json and the package assets land,
    so any literal spelling in authored prose -- namespaced or not -- goes
    stale silently the moment the layout or package name changes: the compiler
    still builds, and an agent told to read a file that is not there invents
    one (observed live: an Antigravity agent ran `cp` to materialize the path
    its instructions promised). Authors write the matching template token
    instead, which the compiler resolves to the installed location.

    Example -- an instruction file containing either of

        Store results under ~/.tpu-agent-suite/runs/<run_id>/artifacts/.
        Read .tpu-agent-suite/environment.json before running.

    is flagged. The first is written as `{{runs}}/<run_id>/artifacts/`,
    which the compiler expands at build time; the second names a file whose
    location depends on install scope, so prose points the agent at
    `{{runtime}} compute inspect --package {{package_tag}} --json` instead.

    Returns: a list of `Finding` (check `"workspace-config-paths"`), one per
    (file, line, match).
    """
    scan_targets: list[tuple[str, str]] = []
    for prompt in _prompts(manifest):
        scan_targets.append((prompt.agent, prompt.rel))
    for agent in _skills_and_agents(manifest):
        for rel in agent.get("references", []):
            if rel.endswith(".md"):
                scan_targets.append((agent["name"], rel))
    for entry in manifest.get("workspace_memory", []):
        scan_targets.append(("(workspace memory)", entry["content"]))

    findings: list[Finding] = []
    seen: set[tuple[str, int, int]] = set()
    for agent_name, rel in scan_targets:
        if not (package_dir / rel).is_file():
            continue
        for lineno, line in _iter_lines(package_dir / rel):
            for idx, (pattern,
                      suggestion) in enumerate(_INSTALLED_ASSET_PATTERNS):
                for match in pattern.finditer(line):
                    if (rel, lineno, idx) in seen:
                        continue
                    seen.add((rel, lineno, idx))
                    findings.append(
                        Finding(
                            "workspace-config-paths", agent_name, rel, lineno,
                            f"hardcoded installed-asset path {match.group(0)!r}"
                            f" -- {suggestion}"))
    return findings


def check_xplane_access(package_dir: Path,
                        manifest: dict[str, Any]) -> list[Finding]:
    """Flags scripts that read XPlane traces directly instead of via the CLI.

    Profile analysis is owned by the `tpu-diagnostics` agent, which drives the
    `xprof` CLI and the skill shipped inside the xprof wheel. A script that
    parses `.xplane.pb` itself -- through TensorFlow's protobuf bindings, the
    private `xprof.convert` module, or the TensorBoard plugin -- re-implements
    metrics the CLI already computes, drifts from them silently, and drags
    TensorFlow into the package's requirements for the protobuf alone.

    Scans the same files as `check_legacy_imports`: declared `.py` scripts and
    the `*_test.py` companions beside them.

    Returns: a list of `Finding` (check `"xplane-access"`), one per offending
    import.
    """
    findings: list[Finding] = []
    for agent_name, rel, tree in _python_script_trees(package_dir, manifest):
        if isinstance(tree, SyntaxError):
            continue  # reported by legacy-imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                names = [
                    f"{node.module}.{alias.name}"
                    if node.module else alias.name for alias in node.names
                ]
            else:
                continue
            for name in names:
                if _XPLANE_IMPORT.search(name):
                    findings.append(
                        Finding(
                            "xplane-access", agent_name, rel, node.lineno,
                            f"imports {name!r} to read XPlane traces directly"
                            " -- delegate to tpu-diagnostics, which runs the"
                            " xprof CLI, instead of parsing the proto"))
    return findings


def check_xprof_outside_analyst(package_dir: Path,
                                manifest: dict[str, Any]) -> list[Finding]:
    """Flags `xprof <subcommand>` invocations in prompts outside the xprof package.

    Only `tpu-diagnostics` runs the xprof CLI; every other agent that needs
    profile numbers lists it in `delegates` and asks. A prompt elsewhere that
    spells out `xprof get_overview ...` restates the CLI's interface in a place
    that cannot stay in step with the wheel it describes -- which is the drift
    the analyst and its co-versioned skill exist to remove.

    The `tpu-diagnostics` agent's own prompt is exempt: that is the analyst.
    It is known by its name, since a prompt's path says nothing about whose
    it is.

    Returns: a list of `Finding` (check `"xprof-outside-analyst"`), at most
    one per (file, line).
    """
    findings: list[Finding] = []
    seen: set[tuple[str, int]] = set()
    for prompt in _prompts(manifest):
        if prompt.agent == "tpu-diagnostics":
            continue
        for lineno, line in _iter_lines(package_dir / prompt.rel):
            match = _XPROF_INVOCATION.search(line)
            if match and (prompt.rel, lineno) not in seen:
                seen.add((prompt.rel, lineno))
                findings.append(
                    Finding(
                        "xprof-outside-analyst", prompt.agent, prompt.rel,
                        lineno, f"invokes {match.group(0)!r} -- only"
                        " tpu-diagnostics runs the xprof CLI; add it to"
                        " `delegates` and delegate the analysis"))
    return findings


# A compute invocation in a prompt, e.g. `{{runtime}} compute run --code x.py`.
# Only the `{{runtime}}` spelling is matched: a bare `compute run` in prose is
# not a command, and the compiled path is not what a package source contains.
# Group 1 is any global option placed before the verb (`compute --backend x
# run`), group 2 the verb. A pre-verb option only consumes a value that is not
# itself a flag, so `--backend local run` reads as option, value, verb.
_COMPUTE_CALL = re.compile(
    r"\{\{runtime}}\s+compute((?:\s+--[a-z][a-z-]*(?:[= ](?!-)\S+)?)*)"
    r"\s+([a-z][a-z-]*)")

# Transports the unified runtime exists to own. A prompt naming one directly has
# either not been migrated or has drifted back to hand-rolled connection code,
# which is what `{{runtime}} compute` was introduced to delete.
_RAW_TRANSPORTS = {
    # Anything non-blank may follow: `ssh [-i key] host` uses the optional-arg
    # bracket notation, and a narrower class let exactly that form through.
    "ssh": (re.compile(r"(?<![\w-])(?:ssh|scp)\s+(?!-V\b)\S"),
            "reach the target with `{{runtime}} compute run` instead"),
    "ssh-config":
    (re.compile(r"(?<![\w/])~?/\.ssh/config\b"),
     "set up compute with `{{runtime}} compute setup` instead of ~/.ssh/config"
     ),
    "gcloud-compute": (re.compile(r"gcloud\s+compute\s+[a-z]"),
                       "use `{{runtime}} compute` instead of gcloud directly"),
    # Only the verbs that move code or drive a workload. `kubectl get` and
    # `describe` read cluster state, which is not what the runtime owns.
    "kubectl":
    (re.compile(r"(?<![\w-])kubectl\s+(?:exec|cp|apply|create|delete|run|"
                r"logs|attach|port-forward)\b"),
     "use `{{runtime}} compute` instead of kubectl directly"),
    "xpk": (re.compile(r"(?<![\w-])xpk\s+workload\s+"),
            "submit through `{{runtime}} compute submit --recipe`"),
    "gcloud-storage":
    (re.compile(r"gcloud\s+storage\s+[a-z]"),
     "pull results with `{{runtime}} compute job artifacts`"),
    "tpu-client":
    (re.compile(r"\b(?:tpu_client|tpu_server)\b"),
     "use `{{runtime}} compute run` instead of legacy TPU client/server scripts"
     ),
}

_COMPUTE_FLAGS_CACHE: dict[str, set[str]] | None = None
_COMPUTE_BACKENDS_CACHE: frozenset[str] | None = None
# Set when `runtime.py` could not be imported to read its flags and backends.
# `check_compute_contract` turns this into a lint finding. Without it, a
# broken import would just disable flag/backend validation and CI would stay
# green while nothing was actually being checked.
_COMPUTE_INTROSPECTION_ERROR: str | None = None

# Shell operators that end one command and begin another. A joined line is
# split on these before compute calls are matched, so the flags of a second
# call chained with `&&` are validated against its own verb, not the first's.
_SHELL_CHAIN = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")

# `--backend <name>` in a prompt; the value is checked against the registry.
_BACKEND_ARG = re.compile(r"--backend[= ]+([A-Za-z0-9_-]+)")


def _compute_flags() -> dict[str, set[str]]:
    """Maps each `compute` subcommand to the long options its parser accepts.

    Read out of the runtime's own argparse tree rather than restated here, so a
    flag added or renamed in `runtime.py` cannot drift from what this check
    believes exists -- the drift is the bug it is looking for. Imported lazily:
    `lint` is otherwise stdlib-only, and nothing outside this check needs it.

    Returns: `{subcommand: {"--code", "--timeout", ...}}`, empty on any import
    or introspection failure so a runtime that cannot be loaded silences the
    flag check rather than reporting every flag in the package as unknown.
    """
    global _COMPUTE_FLAGS_CACHE, _COMPUTE_BACKENDS_CACHE
    global _COMPUTE_INTROSPECTION_ERROR
    if _COMPUTE_FLAGS_CACHE is not None:
        return _COMPUTE_FLAGS_CACHE

    flags: dict[str, set[str]] = {}
    try:
        import argparse

        from framework import runtime
        try:
            _COMPUTE_BACKENDS_CACHE = frozenset(
                runtime.compute.get_valid_backends())
        except Exception as exc:  # noqa: BLE001 - reported below
            _COMPUTE_BACKENDS_CACHE = frozenset()
            _COMPUTE_INTROSPECTION_ERROR = (
                f"backend registry: {type(exc).__name__}: {exc}")
        compute = None
        for action in runtime.build_runtime_parser()._actions:
            if isinstance(action, argparse._SubParsersAction):
                compute = action.choices.get("compute")
                break
        if compute is not None:
            shared = {
                opt
                for act in compute._actions
                for opt in act.option_strings if opt.startswith("--")
            }

            def long_options(parser):
                # A verb's own subcommands (`setup plan --fields`) take their
                # flags after the verb too, so they count as the verb's.
                found = set()
                for a in parser._actions:
                    if isinstance(a, argparse._SubParsersAction):
                        for nested in a.choices.values():
                            found |= long_options(nested)
                    found |= {
                        opt
                        for opt in a.option_strings if opt.startswith("--")
                    }
                return found

            for act in compute._actions:
                if isinstance(act, argparse._SubParsersAction):
                    for name, sub in act.choices.items():
                        flags[name] = shared | long_options(sub)
    except Exception as exc:  # noqa: BLE001 - reported as a finding, not hidden
        flags = {}
        _COMPUTE_INTROSPECTION_ERROR = f"{type(exc).__name__}: {exc}"

    _COMPUTE_FLAGS_CACHE = flags
    return flags


def _joined_commands(path: Path) -> Iterable[tuple[int, str]]:
    """Yields (line number of the first line, text) with continuations folded in.

    A migrated invocation is routinely wrapped across lines with a trailing
    backslash, and its `--` delimiter often lands on a continuation. Reading
    line by line would split the command and read the target script's own flags
    as if they were the runtime's.
    """
    buffer, start = "", 0
    for lineno, line in _iter_lines(path):
        stripped = line.rstrip()
        if not buffer:
            start = lineno
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        yield start, buffer + stripped
        buffer = ""
    if buffer:
        yield start, buffer


def check_compute_contract(package_dir: Path,
                           manifest: dict[str, Any]) -> list[Finding]:
    """Flags compute invocations the runtime would reject, and unmigrated transports.

    Two failures this catches, both of which ship silently today: a prompt that
    names a `compute` subcommand or flag the runtime does not define (the agent
    is taught a command that exits with `unrecognized arguments`), and a prompt
    that still drives `ssh`, `kubectl`, `gcloud` or `xpk` by hand instead of
    going through the runtime.

    Only flags left of the `--` delimiter are checked; everything right of it
    belongs to the script being run, not to `compute`.

    Declared references and scripts are scanned alongside prompts. References
    reach the agent the same way a prompt does. Scripts are scanned for a
    different reason: a bundled tool that opens its own connection defeats the
    runtime whether or not any prompt mentions it, and a tool whose docstring
    tells the agent to run `tpu_client.py` teaches the same dead command a
    prompt would. Both arrive together when a package is ported from a codebase
    that shipped its own transport, which is the case this check exists for.

    Returns: a list of `Finding` (check `"compute-contract"`), each detail
    pairing what was found with what to write instead.
    """
    findings: list[Finding] = []
    known = _compute_flags()
    if _COMPUTE_INTROSPECTION_ERROR:
        findings.append(
            Finding(
                "compute-contract", manifest.get("name", "package"),
                "framework/runtime.py", 0, f"runtime introspection failed "
                f"({_COMPUTE_INTROSPECTION_ERROR}) -- compute verb, flag and "
                f"backend validation skipped for this package"))
    seen: set[tuple[str, int, str]] = set()

    def scan(agent: str, rel: str) -> None:
        path = package_dir / rel
        if not path.is_file():
            return
        for lineno, joined in _joined_commands(path):
            for label, (pattern, hint) in _RAW_TRANSPORTS.items():
                match = pattern.search(joined)
                if match and (rel, lineno, label) not in seen:
                    seen.add((rel, lineno, label))
                    findings.append(
                        Finding(
                            "compute-contract", agent, rel, lineno,
                            f"raw {label}: {match.group(0).strip()!r} -- {hint}"
                        ))
            for text in _SHELL_CHAIN.split(joined):
                scan_calls(agent, rel, lineno, text)

    def scan_calls(agent: str, rel: str, lineno: int, text: str) -> None:
        for call in _COMPUTE_CALL.finditer(text):
            verb = call.group(2)
            if known and verb not in known:
                key = (rel, lineno, f"verb:{verb}")
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        Finding(
                            "compute-contract", agent, rel, lineno,
                            f"unknown compute subcommand {verb!r} -- "
                            f"expected one of {sorted(known)}"))
                continue
            # Everything after `--` is the target script's own argv;
            # everything before it, on either side of the verb, is ours.
            segment = (call.group(1) + " " +
                       text[call.end():].split(" -- ")[0])
            backends = _COMPUTE_BACKENDS_CACHE or frozenset()
            for value in _BACKEND_ARG.findall(segment):
                if backends and value not in backends:
                    key = (rel, lineno, f"backend:{value}")
                    if key not in seen:
                        seen.add(key)
                        findings.append(
                            Finding(
                                "compute-contract", agent, rel, lineno,
                                f"unknown backend {value!r} -- registered "
                                f"backends are {sorted(backends)}"))
            for flag in re.findall(r"(?<![\w-])(--[a-z][a-z0-9_-]*)", segment):
                if known and flag not in known[verb]:
                    key = (rel, lineno, f"{verb}{flag}")
                    if key in seen:
                        continue
                    seen.add(key)
                    findings.append(
                        Finding(
                            "compute-contract", agent, rel, lineno,
                            f"`compute {verb}` has no {flag} flag -- pass "
                            f"script arguments after `--`, or add the flag "
                            f"to the runtime"))

    for prompt in _prompts(manifest):
        scan(prompt.agent, prompt.rel)

    for rel, agent_name in _declared_owners(manifest,
                                            ("references", "scripts")):
        scan(agent_name, rel)
    return findings


# Generic compute prose patterns. The compiled compute contract
# (compute_contract_section in builder/prompt_contracts.py) injects compute
# instructions (commands, flags, backends, run/submit semantics, artifact
# handling, interpreter setup). Authored package prompts should not duplicate
# generic explanations of these mechanisms.
_GENERIC_COMPUTE_PATTERNS = {
    "generic-run-vs-submit": (
        re.compile(
            r"(?i)\b(?:use\s+`?\{\{runtime\}\}\s+compute\s+run`?\s+for\s+synchronous"
            r"|synchronously\s+via\s+`?run`?\s+or\s+asynchronously\s+via\s+`?submit`?"
            r"|\b(?:synchronous\s+execution\s*\(`?run`?\)|asynchronous\s+batch\s+workload\s*\(`?submit`?\)))"
        ),
        "generic run vs submit explanation; the compiled compute contract "
        "owns compute run/submit semantics",
    ),
    "generic-compute-header": (
        re.compile(
            r"(?i)\b(?:unified\s+compute\s+access|universal\s+compute\s+operations)\b"
        ),
        "generic compute section or guideline heading; the compiled compute "
        "contract owns compute access guidelines",
    ),
    "generic-backend-list": (
        re.compile(
            r"(?i)\bsupports\s+`?local`?,\s*`?gce_tpu_vm`?,\s*`?gke_pod`?"),
        "generic compute backend listing; backends are dynamically registered "
        "and rendered in the compiled contract",
    ),
    "generic-compute-intro": (
        re.compile(
            r"(?i)when\s+execution\s+on\s+accelerator\s+compute\s+is\s+required,\s*"
            r"execute\s+via\s+`?\{\{runtime\}\}\s+compute`?:"),
        "generic compute execution intro; remove duplicated compute entry prose "
        "since the compiled contract injects accelerator access guidelines",
    ),
    "generic-job-lifecycle": (
        re.compile(
            r"(?i)\b(?:job\s+lifecycle\s+management"
            r"|status:\s*`?\{\{runtime\}\}\s+compute\s+job\s+status\b)"),
        "generic compute job lifecycle commands; the compiled contract "
        "automatically documents job subcommands",
    ),
}


def check_generic_compute_prose(package_dir: Path,
                                manifest: dict[str, Any]) -> list[Finding]:
    """Flags prompts that re-explain compute instead of just using it.

    The compiled accelerator contract already tells every agent how compute
    works -- run versus submit, which backends exist, how the interpreter is
    chosen, where artifacts go, the job commands. A prompt restating any of
    that is a second copy that drifts. Same files as `check_compute_contract`:
    the prompts and the references they declare.

    Returns: a list of `Finding` (check `"generic-compute-prose"`).
    """
    findings: list[Finding] = []

    def scan(agent: str, rel: str) -> None:
        path = package_dir / rel
        if not path.is_file():
            return
        for lineno, line in _iter_lines(path):
            for label, (pattern, hint) in _GENERIC_COMPUTE_PATTERNS.items():
                match = pattern.search(line)
                if match:
                    findings.append(
                        Finding(
                            "generic-compute-prose", agent, rel, lineno,
                            f"{label}: {match.group(0).strip()!r} -- {hint}"))
                    break  # one finding per line is enough to point at it

    for prompt in _prompts(manifest):
        scan(prompt.agent, prompt.rel)
    for rel, agent_name in _declared_owners(manifest):
        scan(agent_name, rel)
    return findings


def check_environment_contract(package_dir: Path,
                               manifest: dict[str, Any]) -> list[Finding]:
    """Checks that every environment key a bundled script reads is answerable.

    The questions in questions.json produce `environment.json`; scripts read it
    with `env.get("<key>")`. Two problems are reported: a key no question
    produces, and a question that is not a property of the environment schema.
    The first is what matters at runtime -- the script silently falls back to
    its default, which is how a misconfiguration reaches run time instead of
    setup. Scripts that never mention `environment.json` are skipped.

    Returns: a list of `Finding` (check `"environment-contract"`), deduplicated
    per (script, key), with schema mismatches at line 0. Empty when the manifest
    declares no environment. Enforced by `verify.verify_package`.
    """
    findings: list[Finding] = []
    env = manifest.get("environment") or {}
    questions_rel, schema_rel = env.get("questions"), env.get("schema")
    if not questions_rel or not schema_rel:
        return findings

    questions = jsonio.load_json(package_dir / questions_rel)
    schema = jsonio.load_json(package_dir / schema_rel)
    answerable = {q["name"] for q in questions.get("questions", [])}
    permitted = set(schema.get("properties", {}) or {})
    closed = schema.get("additionalProperties") is False

    # A question's answer lands at its `path` (dotted, default: its name), so
    # that is what must exist in the schema -- `compute.params.zone` is a
    # property three levels down, not a top-level one. `compute` stands for
    # a profile in `profiles` (see utils.question_path).
    profile_schema = ((schema.get("properties") or {}).get("profiles")
                      or {}).get("additionalProperties")
    for question in questions.get("questions", []):
        node: Any = schema
        path = str(question.get("path") or question["name"])
        for segment in path.split("."):
            if node is schema and segment == "compute" and isinstance(
                    profile_schema, dict):
                node = profile_schema
                continue
            node = (node.get("properties") or {}).get(segment)
            if node is None:
                findings.append(
                    Finding(
                        "environment-contract",
                        manifest.get("name", "package"), questions_rel, 0,
                        f"question {question['name']!r} lands at {path!r}, "
                        f"which is not a property of {schema_rel}"))
                break
        for key, allowed in (question.get("when") or {}).items():
            if key not in answerable:
                findings.append(
                    Finding(
                        "environment-contract",
                        manifest.get("name", "package"), questions_rel, 0,
                        f"question {question['name']!r} is conditional on "
                        f"{key!r}, which no question answers"))
            elif not isinstance(allowed, list):
                findings.append(
                    Finding(
                        "environment-contract",
                        manifest.get("name", "package"), questions_rel, 0,
                        f"question {question['name']!r}: `when.{key}` must "
                        f"be a list of accepted answers"))

    seen: set[tuple[str, str]] = set()
    for agent in _skills_and_agents(manifest):
        for script_rel in agent.get("scripts", []):
            script = package_dir / script_rel
            if script.suffix != ".py" or not script.is_file():
                continue
            text = script.read_text(encoding="utf-8")
            if "environment.json" not in text:
                continue
            for lineno, line in _iter_lines(script):
                for key in _ENV_GET.findall(line):
                    if key in answerable:
                        continue
                    if (script_rel, key) in seen:
                        continue
                    seen.add((script_rel, key))
                    why = ("no question produces it"
                           if not closed or key in permitted else
                           f"no question produces it and {schema_rel} forbids"
                           f" it (additionalProperties: false)")
                    findings.append(
                        Finding(
                            "environment-contract", agent["name"], script_rel,
                            lineno,
                            f"reads environment key {key!r} but {why}; the"
                            f" script's default is always used"))
    return findings


def _read_contract(path: Path, rel: str, owner: str,
                   findings: list[Finding]) -> dict[str, Any] | None:
    """Reads the frontmatter contract an instruction file opens with.

    Returns: the frontmatter, or None after appending to `findings` the
    reason there is none -- no leading `---`, or frontmatter that does not
    parse. Either way there is nothing left to check in the file.
    """
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("---"):
        findings.append(
            Finding("frontmatter-contracts", owner, rel, 1,
                    "missing leading '---' YAML frontmatter contract"))
        return None
    try:
        fm, _ = parse_frontmatter_document(text, rel)
    except TpuAgentSuiteError as exc:
        findings.append(
            Finding("frontmatter-contracts", owner, rel, 1, str(exc)))
        return None
    return fm


def _list_field_findings(
        fm: dict[str, Any],
        key: str,
        owner: str,
        rel: str,
        allowed_tools: set[str] | None = None) -> list[Finding]:
    """Reports a frontmatter `key` that is not a list, or grants a tool
    outside `allowed_tools` when that is given."""
    if key not in fm:
        return []
    if not isinstance(fm[key], list):
        return [
            Finding("frontmatter-contracts", owner, rel, 1,
                    f"'{key}' in frontmatter must be a list")
        ]
    if allowed_tools is None:
        return []
    invalid = set(fm[key]) - allowed_tools
    if invalid:
        return [
            Finding(
                "frontmatter-contracts", owner, rel, 1,
                "frontmatter contains invalid tool grants: "
                f"{sorted(invalid)}")
        ]
    return []


def check_frontmatter_contracts(package_dir: Path,
                                manifest: dict[str, Any]) -> list[Finding]:
    """Check that instruction markdown files define valid YAML contracts.

    Every skill and agent instruction file must begin with YAML
    frontmatter defining its manifest metadata (name, description, etc.).

    Returns: a list of `Finding` (check `"frontmatter-contracts"`).
    """
    findings: list[Finding] = []
    # A skill may hold every portable tool. An agent may not hold the ones
    # that need a user, which only a skill has.
    skill_tools = set(utils.PORTABLE_TOOLS)
    valid_tools = skill_tools - utils.SKILL_ONLY_TOOLS
    checked_files: set[Path] = set()

    # 1. Skill frontmatter contracts. Skills go first: a skill that delegates
    #    shares its document with its twin agent, and is checked as a skill.
    for skill in manifest.get("skills", []):
        # Indexed rather than fetched with a default: the manifest schema
        # requires `instructions`, and load_package validates before any
        # check runs, so a missing key is a bug worth raising, not skipping.
        rel = skill["instructions"]
        doc_path = package_dir / rel
        owner = skill.get("name", doc_path.stem)
        if not doc_path.is_file():
            # Nothing else catches this before a build: load_package verifies
            # that agent documents exist but has no equivalent for skills.
            findings.append(
                Finding("frontmatter-contracts", owner, rel, 1,
                        "skill instructions file does not exist"))
            continue
        if doc_path.resolve() in checked_files:
            continue
        checked_files.add(doc_path.resolve())
        fm = _read_contract(doc_path, rel, owner, findings)
        if fm is None:
            continue
        for req in ("name", "description"):
            if not fm.get(req):
                findings.append(
                    Finding("frontmatter-contracts", owner, rel, 1,
                            f"frontmatter missing required field {req!r}"))
        # Manifest and document must agree, or the harness advertises a skill
        # under one name while its body describes another.
        if fm.get("name") and fm["name"] != skill.get("name"):
            findings.append(
                Finding(
                    "frontmatter-contracts", skill["name"], rel, 1,
                    f"frontmatter name {fm['name']!r} does not match skill "
                    f"name {skill['name']!r}"))
        findings.extend(_list_field_findings(fm, "delegates", owner, rel))
        findings.extend(
            _list_field_findings(fm, "tools", owner, rel, skill_tools))

    # 2. Agent frontmatter contracts
    agents = manifest.get("agents", [])

    for agent in agents:
        rel = agent.get("instructions")
        if not rel:
            continue
        doc_path = package_dir / rel
        if not doc_path.is_file():
            continue
        if doc_path.resolve() in checked_files:
            continue
        checked_files.add(doc_path.resolve())
        owner = agent.get("name", doc_path.stem)
        fm = _read_contract(doc_path, rel, owner, findings)
        if fm is None:
            continue

        for req in ("name", "description"):
            if req not in fm:
                findings.append(
                    Finding(
                        "frontmatter-contracts", owner, rel, 1,
                        f"frontmatter missing required contract field {req!r}")
                )
        if "name" in fm and "name" in agent and fm["name"] != agent["name"]:
            findings.append(
                Finding(
                    "frontmatter-contracts", agent["name"], rel, 1,
                    f"frontmatter name {fm['name']!r} does not match agent "
                    f"name {agent['name']!r}"))
        findings.extend(
            _list_field_findings(fm, "tools", owner, rel, valid_tools))
        findings.extend(_list_field_findings(fm, "delegates", owner, rel))

    # 3. Check any authored instruction files in package
    authored = sorted({
        *package_dir.glob(f"{utils.AGENTS_DIRNAME}/*.md"),
        *package_dir.glob(f"{utils.SKILLS_DIRNAME}/*/{utils.SKILL_DOCUMENT}"),
    })
    for inst_file in authored:
        if inst_file.resolve() in checked_files:
            continue
        rel = str(inst_file.relative_to(package_dir))
        fm = _read_contract(inst_file, rel, inst_file.stem, findings)
        if fm is None:
            continue
        for req in ("name", "description"):
            if not fm.get(req):
                findings.append(
                    Finding("frontmatter-contracts", inst_file.stem, rel, 1,
                            f"frontmatter missing required field {req!r}"))

    return findings


CHECKS = {
    "instruction-links": check_instruction_links,
    "reference-links": check_reference_links,
    "bare-paths": check_bare_paths,
    "delegation-consistency": check_delegation_consistency,
    "legacy-paths": check_legacy_paths,
    "legacy-imports": check_legacy_imports,
    "xplane-access": check_xplane_access,
    "xprof-outside-analyst": check_xprof_outside_analyst,
    "environment-contract": check_environment_contract,
    "workspace-config-paths": check_workspace_config_paths,
    "unused-declarations": check_unused_declarations,
    "frontmatter-contracts": check_frontmatter_contracts,
    "compute-contract": check_compute_contract,
    "generic-compute-prose": check_generic_compute_prose,
}


def unreachable_files(package_dir: Path, manifest: dict[str,
                                                        Any]) -> list[str]:
    """Files under agents/, references/ and scripts/ that no released skill
    reaches.

    These files do not ship: a build holds only what its skills reach. They
    may stay in the repository, so this is a warning and not a finding. It is
    reported so a typo in a delegate or a token does not silently drop a file
    from the build.
    """
    reached: set[str] = set()
    for unit in _skills_and_agents(manifest):
        reached.add(unit.get("instructions", ""))
        reached.update(unit.get("references", []))
        reached.update(unit.get("scripts", []))
    for hook in manifest.get("hooks", []):
        action = hook.get("action", "")
        if action.startswith("script:"):
            reached.add(action[len("script:"):])
    unreachable: list[str] = []
    for root in (utils.AGENTS_DIRNAME, utils.REFERENCES_DIRNAME,
                 utils.SCRIPTS_DIRNAME):
        base = package_dir / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(package_dir).as_posix()
            if rel in reached:
                continue
            # A script's companion test ships with nothing and needs no reader.
            if root == utils.SCRIPTS_DIRNAME and (
                    path.name.endswith("_test.py")
                    or path.name.startswith("test_")):
                continue
            unreachable.append(rel)
    return unreachable


def run_checks(package_dir: Path,
               manifest: dict[str, Any],
               only: Iterable[str] | None = None) -> list[Finding]:
    """Runs the named checks against a package and collects their findings.

    `only` selects check names from `CHECKS`; every check runs by default.

    Returns: a flat list of `Finding`, grouped in the order the checks ran.
    Nothing here raises or decides severity -- callers choose what is fatal,
    which is how `verify` gates enforced checks while `lint` reports all checks.
    """
    names = list(only) if only else list(CHECKS)
    findings: list[Finding] = []
    for name in names:
        findings.extend(CHECKS[name](package_dir, manifest))
    return findings

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
"""Fills in `{{...}}` placeholders in agent prompts, skills and references.

Authors write `{{agent:name}}`, `{{skill:name}}`, `{{reference:path}}` and so
on instead of real names and install paths. This swaps in the right values for
the harness being built, and rejects misspelled placeholders and install paths
typed out by hand.
"""

from __future__ import annotations

import difflib
import re
from typing import Any

from framework import paths, utils

AGENT_REFERENCE_PATTERN = r"\{\{agent:([a-z0-9]+(?:-[a-z0-9]+)*)}}"

SKILL_REFERENCE_PATTERN = r"\{\{skill:([a-z0-9]+(?:-[a-z0-9]+)*)}}"

# `{{reference:<path>}}`, `{{script:<path>}}`, `{{data:<path>}}` and
# `{{schema:<path>}}` each name a file by its path under the package's
# references/, scripts/, data/ or schemas/ root, and render to the installed
# location, so prose never spells an install path itself.
RESOURCE_REFERENCE_PATTERN = r"\{\{(reference|script|data|schema):([A-Za-z0-9_][A-Za-z0-9_./-]*)}}"

# Token kind -> the package directory it names.
RESOURCE_ROOTS = {
    "reference": utils.REFERENCES_DIRNAME,
    "script": utils.SCRIPTS_DIRNAME,
    "data": utils.DATA_DIRNAME,
    "schema": utils.SCHEMAS_DIRNAME,
}

# Tokens whose value depends on the harness a distribution is compiled for,
# not on the manifest. `render_agent_references` accepts them and leaves them
# in place; `render_harness_tokens` expands them once the harness is known.
HARNESS_SCOPED_TOKENS = ("{{harness}}", "{{built-in:ask_user_question}}")

# What `{{built-in:ask_user_question}}` becomes, per harness. Two of the four
# have a tool for it under different names; the other two have none, so the
# token renders as the plain instruction and the sentence reads the same
# either way. Compiling is never refused over it: an author should be able to
# say "ask the user" once and ship to all four.
ASK_USER_QUESTION_TOOLS: dict[str, str] = {
    "claude-code": "AskUserQuestion",
    "antigravity": "ask_question",
}

ASK_USER_QUESTION_PROSE = "ask the user"


def render_harness_tokens(text: str, harness: str) -> str:
    """Expand the tokens whose value depends on the harness being built.

    `{{harness}}` resolves here because a skill that calls the runtime must
    pass the harness it is running under, and the harness is known only at
    translate time.

    `{{built-in:ask_user_question}}` is the same shape: the tool that puts a
    question to the user is named differently on the two harnesses that have
    one, and on the two that do not the token becomes the plain instruction,
    so one authored sentence ships to all four.

    Only harness-dependent tokens live here; `{{runtime}}`, `{{package}}`
    and `{{package_tag}}` are manifest-dependent and resolve in
    `render_agent_references`, which has no harness in scope.
    """
    text = text.replace("{{harness}}", harness)
    return text.replace(
        "{{built-in:ask_user_question}}",
        ASK_USER_QUESTION_TOOLS.get(harness, ASK_USER_QUESTION_PROSE))


# Every kind of `{{kind:name}}` token there is. A kind outside this set is a
# misspelling, and left alone it would reach the agent as literal braces.
TOKEN_KINDS = ("agent", "skill", "built-in", *RESOURCE_ROOTS)

_KINDED_TOKEN = re.compile(r"\{\{([A-Za-z_-]+):([^{}\n]*)}}")


def reject_unknown_token_kinds(text: str) -> None:
    """Fail on a `{{kind:name}}` token whose kind does not exist."""
    unknown = [(number, match.group(0), match.group(1))
               for number, line in enumerate(text.splitlines(), 1)
               for match in _KINDED_TOKEN.finditer(line)
               if match.group(1) not in TOKEN_KINDS]
    if not unknown:
        return
    lines = []
    for number, token, kind in unknown:
        close = difflib.get_close_matches(kind.lower(), TOKEN_KINDS, n=1)
        hint = f" (did you mean {{{{{close[0]}:...}}}}?)" if close else ""
        lines.append(f"  line {number}: {token}{hint}")
    raise utils.TpuAgentSuiteError(
        "instructions use a token of a kind that does not"
        f" exist; the kinds are {', '.join(TOKEN_KINDS)}:\n" +
        "\n".join(lines))


def render_markdown(text: str, manifest: dict[str, Any], harness: str) -> str:
    """Render a markdown file an agent reads: every token a prompt may use."""
    return render_harness_tokens(render_agent_references(text, manifest),
                                 harness)


def render_agent_references(text: str,
                            manifest: dict[str, Any],
                            prefix: str = "") -> str:
    """Replace {{agent:NAME}} template tokens with prefixed agent identifiers.

    A reference to an agent or skill the build does not hold is an error. A
    build holds everything its skills reach, so a name that fails here is one
    the document never declared.
    """
    reject_unknown_token_kinds(text)
    known_agents = {agent["name"] for agent in manifest["agents"]}
    known_skills = {
        skill["name"]
        for skill in manifest.get("skills", []) if isinstance(skill, dict)
    }

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in known_agents:
            raise utils.TpuAgentSuiteError(
                f"instructions reference unknown agent: {name}")
        return prefix + name

    def replace_skill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in known_skills:
            raise utils.TpuAgentSuiteError(
                f"instructions reference unknown skill: {name}")
        return prefix + name

    rendered = re.sub(AGENT_REFERENCE_PATTERN, replace, text)
    if "{{agent:" in rendered:
        raise utils.TpuAgentSuiteError(
            "instructions contain a malformed agent reference")
    rendered = re.sub(SKILL_REFERENCE_PATTERN, replace_skill, rendered)
    if "{{skill:" in rendered:
        raise utils.TpuAgentSuiteError(
            "instructions contain a malformed skill reference")

    # A reference or script token names a file under the package's
    # references/ or scripts/ root; it renders to the installed path. Whether
    # the file exists is checked against the source tree by
    # `package_checks.validate_resource_tokens`, which has the package
    # directory in scope.
    _, pkg_path, _ = package_runtime_paths(manifest)

    def replace_resource(match: re.Match[str]) -> str:
        return f"{pkg_path}/{RESOURCE_ROOTS[match.group(1)]}/{match.group(2)}"

    rendered = re.sub(RESOURCE_REFERENCE_PATTERN, replace_resource, rendered)
    if any(f"{{{{{kind}:" in rendered for kind in RESOURCE_ROOTS):
        raise utils.TpuAgentSuiteError(
            "instructions contain a malformed reference, script, data or"
            " schema token")

    # Automatically inject `--package {{package_tag}}` into `{{runtime}} compute` calls
    rendered = _RUNTIME_COMPUTE_PATTERN.sub(
        "{{runtime}} compute --package {{package_tag}}", rendered)

    for token, value in resource_registry(manifest).items():
        rendered = rendered.replace(token, value)

    # Harness-scoped tokens resolve in `render_harness_tokens`, which runs on
    # this function's output at translate time; here they are known, not
    # expanded, since no harness is in scope when a package is only verified.
    unknown = sorted(
        set(re.findall(r"\{\{[a-z_]+(?:-[a-z_]+)*(?::[a-z_]+)?\}\}", rendered))
        - set(HARNESS_SCOPED_TOKENS))
    if unknown:
        known = sorted({*resource_registry(manifest), *HARNESS_SCOPED_TOKENS})
        raise utils.TpuAgentSuiteError(
            "instructions reference unknown resource tokens: "
            f"{', '.join(unknown)} (known: {', '.join(known)})")

    return rendered


# `{{runtime}} compute` not already followed by `--package`.
_RUNTIME_COMPUTE_PATTERN = re.compile(
    r"\{\{runtime}}\s+compute(?!\s+--package\b)")


def resource_token_targets(text: str) -> list[tuple[str, str]]:
    """Every `{{reference:...}}`, `{{script:...}}`, `{{data:...}}` and
    `{{schema:...}}` token in `text`, as (kind, package-relative path)
    pairs."""
    return [(kind, f"{RESOURCE_ROOTS[kind]}/{rel}")
            for kind, rel in re.findall(RESOURCE_REFERENCE_PATTERN, text)]


# What a plain path must be written as instead. A path typed out in prose is
# read from wherever the agent happens to be running, which is its workspace
# and not the installed package, so it names a file that is not there.
_PLAIN_PACKAGE_PATH = re.compile(
    r"(?<![\w./{}:~-])((?:%s)/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)" %
    "|".join(sorted(set(RESOURCE_ROOTS.values()))))

_VIA_PACKAGE_PATH = re.compile(
    r"\{\{package}}/([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)")

_ANY_TOKEN = re.compile(r"\{\{[^{}\n]*}}")


def token_for(path: str) -> str:
    """The token that names the package path `path`, or what to do when no
    token can: the path is under a directory no build ships by that name."""
    root, _, rest = path.partition("/")
    kind = next((k for k, v in RESOURCE_ROOTS.items() if v == root), None)
    if kind is None or not rest:
        return ("no token names it; package files live under " +
                ", ".join(f"{v}/" for v in RESOURCE_ROOTS.values()))
    return f"{{{{{kind}:{rest}}}}}"


# A token this compiler once resolved. It named a directory, so a path built
# on it was checked against nothing, and it went on resolving after the files
# it was written for stopped shipping.
_RETIRED_TOKEN = re.compile(r"\{\{knowledgebase}}(/[A-Za-z0-9_.*/-]*)?")


def plain_package_paths(text: str) -> list[tuple[int, str]]:
    """(line, path) for every package file `text` names other than by token.

    Two spellings count: the path typed out (`scripts/fam/tool.py`) and any
    path built on the package token (`{{package}}/scripts/fam/tool.py`). The
    second renders to a place under the package but is checked against
    nothing, so it may name a directory the package no longer has.
    """
    found: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), 1):
        for match in _VIA_PACKAGE_PATH.finditer(line):
            found.append((number, match.group(1).rstrip(".")))
        bare = _ANY_TOKEN.sub(lambda m: " " * len(m.group(0)), line)
        for match in _PLAIN_PACKAGE_PATH.finditer(bare):
            found.append((number, match.group(1).rstrip(".")))
    return found


def body_first_line(document: str, body: str) -> int:
    """The line of `document` that `body`, its text after frontmatter, starts
    on."""
    return document.count("\n", 0, len(document) - len(body)) + 1


def reject_plain_package_paths(text: str,
                               label: str,
                               first_line: int = 1) -> None:
    """Fail when `text` names a package file other than by token."""
    retired = [(number, match.group(0))
               for number, line in enumerate(text.splitlines(), 1)
               for match in _RETIRED_TOKEN.finditer(line)]
    if retired:
        raise utils.TpuAgentSuiteError(
            f"{label} uses the retired {{{{knowledgebase}}}} token; name a"
            " provenance page with {{data:tpu-knowledge-base/pages/<name>.md}}:\n"
            + "\n".join(f"  line {first_line + number - 1}: {found}"
                        for number, found in retired))
    found = plain_package_paths(text)
    if found:
        raise utils.TpuAgentSuiteError(
            f"{label} names package files by path; a token is the only way"
            " to name one:\n" +
            "\n".join(f"  line {first_line + number - 1}: {path}"
                      f" -> {token_for(path)}" for number, path in found) +
            "\nA path that names a file in the workspace, not in the"
            " package, is written as one, with something in front of it:"
            f" ./{found[0][1]} or <work>/{found[0][1]}.")


def resource_registry(manifest: dict[str, Any]) -> dict[str, str]:
    """The single map from authored resource tokens to installed locations.

    Authors name WHAT they want; where it lives is routed here and nowhere
    else -- prose never states a location class, so relocating a resource
    (e.g. the knowledge base moving from workspace to central) is a one-line
    change with zero authored-file churn. `{{package_tag}}` is the exception:
    it expands to the `name@version` string itself, which compiled
    instructions pass to `--package` flags at run time -- an identifier, not
    a filesystem path.

    Expansion happens at compile time, and a distribution is compiled once
    then installed at any scope, so no token can encode scope. Every path
    token expands to a ~-anchored central location, valid from any cwd on
    the machine that installed it.

    Environment config deliberately has NO token: its location depends on
    install scope (workspace slot vs central store), which is unknowable at
    compile time, so a path in prose would be wrong at some scope. Prose
    that needs effective config directs the agent through the CLI
    (`{{runtime}} compute inspect --package {{package_tag}} --json`), and
    paths.resolve_environment applies the ownership-keyed rule at run time.
    """
    pkg_tag, pkg_path, runtime = package_runtime_paths(manifest)
    return {
        "{{package}}": pkg_path,
        "{{package_tag}}": pkg_tag,
        "{{runs}}": paths.RUNS_HOME_SPELLING,
        "{{runtime}}": runtime,
    }


def package_runtime_paths(manifest: dict[str, Any]) -> tuple[str, str, str]:
    """Return (pkg_tag, pkg_path, runtime_command) for manifest."""
    name = manifest["name"]
    version = manifest.get("version", "")
    pkg_tag = utils.format_package_tag(name, version)
    pkg_path = paths.central_home_spelling_for(name, version)
    runtime = f"python3 {pkg_path}/framework/runtime.py"
    return pkg_tag, pkg_path, runtime

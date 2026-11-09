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
"""Frontmatter parsing for agent markdown documents."""

from __future__ import annotations

import pathlib
from typing import Any

import yaml

from framework.utils import TpuAgentSuiteError

Path = pathlib.Path


def parse_frontmatter_document(content: str,
                               filename: str = ""
                               ) -> tuple[dict[str, Any], str]:
    """Parse a markdown document with optional leading YAML frontmatter.

    Returns (frontmatter_dict, markdown_body). If no frontmatter is present,
    returns ({}, content).
    """
    stripped = content.lstrip()
    if not stripped.startswith("---"):
        return {}, content

    lines = stripped.splitlines(keepends=True)
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        raise TpuAgentSuiteError(f"unclosed YAML frontmatter in '{filename}'")

    frontmatter_raw = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1:]).lstrip("\r\n")
    try:
        fm = yaml.safe_load(frontmatter_raw) or {}
    except yaml.YAMLError as exc:
        raise TpuAgentSuiteError(
            f"invalid YAML frontmatter in '{filename}': {exc}") from exc
    if not isinstance(fm, dict):
        raise TpuAgentSuiteError(
            f"YAML frontmatter in '{filename}' must be a mapping, got "
            f"{type(fm).__name__}. If this document begins with a markdown "
            "horizontal rule ('---'), frontmatter is expected; add required "
            "frontmatter or leading content.")
    return fm, body


def read_frontmatter_document(
        path: Path,
        filename: str | None = None,
        *,
        errors: str = "strict") -> tuple[dict[str, Any], str, str]:
    """Read a markdown file and split off its optional YAML frontmatter.

    `filename` names the file in parse errors, `path.name` by default, and
    `errors` is how undecodable bytes are handled, as for `bytes.decode`.

    Returns (frontmatter_dict, markdown_body, full_text); as with
    parse_frontmatter_document, a file without frontmatter is ({}, text, text).
    """
    text = path.read_text(encoding="utf-8", errors=errors)
    fm, body = parse_frontmatter_document(
        text, path.name if filename is None else filename)
    return fm, body, text

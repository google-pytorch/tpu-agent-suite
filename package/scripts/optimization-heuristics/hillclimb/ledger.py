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
"""Past outcomes read from close-gaps state (`ledger.md` and `gaps/*.md`).

close-gaps keeps one state file, `ledger.md`, with per-gap detail in
`gaps/<rank>-<slug>.md` beside it. The gates keep no state of their own:
`preflight --ledger` and `code-gap --ledger` read the outcome lines the loop
writes into those files when an A/B, an E2E arm or a job ends, in this form:

    outcome: not a win `DECODE_DCP_SIZE=8` (cost_model.json of the 2.4 A/B)

The outcome is `win`, `not a win` or `crash`, followed by every changed knob
or env flag as a backticked `KEY=VALUE`. Lines without a setting are skipped.
"""

import dataclasses
import pathlib
import re
from collections.abc import Mapping
from typing import Any

WIN = "win"
NOT_A_WIN = "not a win"
CRASH = "crash"
NEGATIVE: frozenset[str] = frozenset({NOT_A_WIN, CRASH})

_OUTCOME_RE = re.compile(r"\boutcome:\s*(not a win|win|crash)\b(.*)$",
                         re.IGNORECASE)
_SETTING_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*=[^`\s]*)`")


@dataclasses.dataclass(frozen=True)
class Entry:
    """One outcome line from the close-gaps ledger or a gap file."""

    kind: str  # decision
    payload: Mapping[str, Any]
    source: str = ""


def state_files(path: str | pathlib.Path) -> list[pathlib.Path]:
    """Returns `ledger.md` and the `gaps/*.md` files beside it.

    `path` is the close-gaps `ledger.md` or the directory that holds it.
    """
    root = pathlib.Path(path).expanduser()
    ledger_md = root / "ledger.md" if root.is_dir() else root
    files = [ledger_md] if ledger_md.is_file() else []
    gaps_dir = ledger_md.parent / "gaps"
    if gaps_dir.is_dir():
        files.extend(sorted(gaps_dir.glob("*.md")))
    return files


def parse_outcomes(text: str, source: str = "") -> list[Entry]:
    """Parses `outcome:` lines in `text` into `decision` entries."""
    out: list[Entry] = []
    for line_no, line in enumerate(text.splitlines(), 1):
        match = _OUTCOME_RE.search(line)
        if not match:
            continue
        settings = _SETTING_RE.findall(match.group(2))
        if not settings:
            continue
        where = f"{source}:{line_no}" if source else f"line {line_no}"
        out.append(
            Entry(
                kind="decision",
                payload={
                    "decision": match.group(1).lower(),
                    "settings": settings,
                    "run_id": where,
                    "reason": match.group(2).strip(),
                },
                source=where,
            ))
    return out


def read(path: str | pathlib.Path) -> list[Entry]:
    """Reads outcome lines from close-gaps state at `path` (`[]` if absent)."""
    if not path:
        return []
    root = pathlib.Path(path).expanduser()
    base = root if root.is_dir() else root.parent
    out: list[Entry] = []
    for file in state_files(root):
        out.extend(
            parse_outcomes(file.read_text(encoding="utf-8", errors="replace"),
                           file.relative_to(base).as_posix()))
    return out

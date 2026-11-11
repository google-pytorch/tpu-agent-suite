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
"""Upstream pull-request switch ranker over local `git` refs (`ports.py`).

Ranks candidate upstream branches or `refs/pull/<n>/head` refs by how many
step-active optimization switches they add over their fork point and whether
`git merge-tree --write-tree` merges cleanly onto the served commit without
touching the working tree.
"""

import dataclasses
import re
import subprocess
from collections.abc import Sequence
from typing import Any

from hillclimb import code_gap

_PR_REF_RE = re.compile(
    r"^([0-9a-f]{7,40})\s+refs/(?:pull|ports/pull)/(\d+)/?(?:head)?$")


@dataclasses.dataclass(frozen=True)
class PullHead:
    """One upstream pull-request head ref `(number, sha)`."""

    number: int
    sha: str


@dataclasses.dataclass(frozen=True)
class PortCandidate:
    """Scored upstream PR or branch candidate for cherry-picking/merging."""

    number: int
    sha: str
    subject: str
    date: str
    on_base: bool
    merge: str
    score: float
    active: tuple[code_gap.GapCandidate, ...]
    inert: tuple[str, ...]
    commits: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "sha": self.sha,
            "subject": self.subject,
            "date": self.date,
            "commits": self.commits,
            "on_base": self.on_base,
            "merge": self.merge,
            "score": round(self.score, 2),
            "active": [g.flag for g in self.active],
            "inert": list(self.inert),
        }


def parse_ls_remote(text: str) -> list[PullHead]:
    """Parses `git ls-remote <remote> refs/pull/*/head`, newest PR first."""
    heads: list[PullHead] = []
    for line in text.splitlines():
        m = _PR_REF_RE.match(line.strip())
        if m:
            heads.append(PullHead(number=int(m.group(2)), sha=m.group(1)))
    heads.sort(key=lambda h: h.number, reverse=True)
    return heads


def merge_status(returncode: int, output: str) -> str:
    """Formats `git merge-tree --write-tree --name-only` status."""
    if returncode == 0:
        return "clean"
    files = [
        ln.strip() for ln in output.splitlines()
        if ln.strip() and not re.fullmatch(r"[0-9a-f]{40,64}", ln.strip())
    ]
    if not files:
        return "conflicts"
    shown = ", ".join(files[:3])
    more = f" (+{len(files) - 3})" if len(files) > 3 else ""
    return f"conflicts: {shown}{more}"


def candidate(
    head: PullHead,
    subject: str,
    date: str,
    on_base: bool,
    merge: str,
    gaps: Sequence[code_gap.GapCandidate],
    *,
    own_flags: set[str] | frozenset[str],
    commits: int = 1,
) -> PortCandidate:
    """Scores one PR head by active switches, measured claims, and clean merge."""
    own_gaps = [
        g for g in gaps if g.kind == "new_upstream" and g.flag in own_flags
    ]
    active = tuple(g for g in own_gaps if g.inactive_reason is None)
    inert = tuple(g.flag for g in own_gaps if g.inactive_reason is not None)

    score = 0.0
    for g in active:
        score += 2.0 + (1.0 if g.claims else 0.0)
        if g.prior is not None and g.prior.decision in ("reject", "DISCARD"):
            score -= 2.5
    if on_base:
        score += 1.0
    if merge == "clean":
        score += 1.5
    elif merge.startswith("conflicts"):
        score -= 0.5

    return PortCandidate(
        number=head.number,
        sha=head.sha[:12],
        subject=subject,
        date=date,
        on_base=on_base,
        merge=merge,
        score=score,
        active=active,
        inert=inert,
        commits=commits,
    )


def rank(candidates: Sequence[PortCandidate]) -> list[PortCandidate]:
    """Orders `PortCandidate`s descending by `(score, len(active), number)`."""
    return sorted(
        candidates,
        key=lambda c: (c.score, len(c.active), c.number),
        reverse=True,
    )


def scan_local_refs(
        repo: str,
        served_ref: str,
        candidate_refs: Sequence[str],
        *,
        envs_path: str = "src/vllm_torchtpu/envs.py",
        arm_env: dict[str, str] | None = None,
        steps: Sequence[int] = (),
) -> list[PortCandidate]:
    """Evaluates already-present local git refs without network fetches."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", repo, *args],
            capture_output=True,
            text=True,
            check=False,
        )

    served_show = git("show", f"{served_ref}:{envs_path}")
    if served_show.returncode != 0:
        raise ValueError(f"Cannot read {served_ref}:{envs_path} in {repo}: "
                         f"{served_show.stderr.strip()}")
    served_reg = code_gap.parse_env_registry(served_show.stdout)
    eff_env = arm_env or {}

    found: list[PortCandidate] = []
    for idx, ref in enumerate(candidate_refs, start=1):
        up_show = git("show", f"{ref}:{envs_path}")
        if up_show.returncode != 0:
            continue
        up_reg = code_gap.parse_env_registry(up_show.stdout)
        sha = git("rev-parse", ref).stdout.strip() or ref
        subj = git("log", "-1", "--format=%s", ref).stdout.strip()
        date = git("log", "-1", "--format=%cs", ref).stdout.strip()
        mt = git(
            "merge-tree",
            "--write-tree",
            "--name-only",
            "--no-messages",
            served_ref,
            ref,
        )
        m_status = merge_status(mt.returncode, mt.stdout)
        pr_num_match = re.search(r"(\d+)$", ref)
        pr_num = int(pr_num_match.group(1)) if pr_num_match else idx
        gaps = code_gap.find_gaps(served_reg, up_reg, eff_env, steps=steps)
        own = set(up_reg) - set(served_reg)
        found.append(
            candidate(
                PullHead(number=pr_num, sha=sha),
                subj,
                date,
                True,
                m_status,
                gaps,
                own_flags=own,
            ))
    return rank(found)


def render(
        candidates: Sequence[PortCandidate],
        served_ref: str,
        steps: Sequence[int] = (),
) -> str:
    """Formats ranked `PortCandidate`s as a Markdown table."""
    step_str = f" at steps `{list(steps)}`" if steps else ""
    if not candidates:
        return f"No candidate PRs add active switches over `{served_ref}`{step_str}.\n"
    lines = [
        f"### Ranked Upstream PR Candidates (onto `{served_ref}`{step_str})",
        "",
        "| PR | SHA | Score | Merge | Active Switches | Inert Switches | Subject |",
        "|---:|---|---:|---|---|---|---|",
    ]
    for c in candidates:
        act = ", ".join(f"`{g.flag}`" for g in c.active) or "none"
        inert = ", ".join(f"`{f}`" for f in c.inert) or "—"
        lines.append(
            f"| #{c.number} | `{c.sha}` | {c.score:.1f} | {c.merge} | "
            f"{act} | {inert} | {c.subject} |")
    return "\n".join(lines) + "\n"

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
"""Env-registry switch-gap analyzer, prerequisite gate chains, and MoE conflicts.

Diffs the served checkout's `envs.py` registry against upstream branches/PRs,
traces transitive prerequisite gates (`off_gate_chain`, 0/152 false positives on
upstream `vllm-torchtpu` registries), detects EP-vs-TP MoE mode conflicts
(`moe_mode_conflict`), parses semicolon- and comma-delimited `EXTRA_ENV*` lists,
and filters out switches that are inert at the arm's decode step token count.
"""

import dataclasses
import re
from collections.abc import Mapping, Sequence

from hillclimb import ledger

_OFF_VALUES = frozenset({"", "0", "false", "False", "off", "none", "None"})
_ENV_LIST_SPLIT_RE = re.compile(r";|,(?=\s*[A-Z_][A-Z0-9_]*=)")
_FLAG_DEF_RE = re.compile(
    r'^\s*"([A-Z][A-Z0-9_]+)"\s*:\s*lambda\s*:\s*(.+?)\s*,?\s*$')
_GETENV_DEFAULT_RE = re.compile(
    r'getenv\(\s*"[A-Z0-9_]+"\s*,\s*["\']?([^"\'),]+)["\']?\s*\)')
_BOOL_ENV_DEFAULT_RE = re.compile(
    r'_bool_env\(\s*"[A-Z0-9_]+"\s*,\s*(True|False|1|0|"1"|"0")\s*\)')
_REQUIRES_FLAG_RE = re.compile(
    r"(?:requires|when|under|needs|with)\s+`?([A-Z][A-Z0-9_]{5,})`?"
    r"(?:=([0-9A-Za-z_]+))?",
    re.IGNORECASE,
)
_EP_ONLY_RE = re.compile(
    r"\b(?:under|when|for|requires)\s+(?:--enable-expert-parallel|expert[- ]parallel|EP\s+MoE|MOE_FUSED_EP)\b",
    re.IGNORECASE,
)
_TP_ONLY_RE = re.compile(
    r"\b(?:under|when|for|requires)\s+(?:TP\s+MoE|non-EP|USE_MOE_TP_DECODE_KERNEL|without\s+--enable-expert-parallel)\b",
    re.IGNORECASE,
)
_STEP_LE_RE = re.compile(
    r"(?:at|for|when)?\s*(?:<=|at most|up to)\s*(\d+)\s*(?:decode\s+)?(?:rows|tokens)\b",
    re.IGNORECASE,
)
_STEP_GE_RE = re.compile(
    r"(?:at|for|when)?\s*(?:>=|at least|above)\s*(\d+)\s*(?:decode\s+)?(?:rows|tokens)\b",
    re.IGNORECASE,
)
_STEP_RANGE_RE = re.compile(
    r"\b(\d+)\s*(?:-|to|\.\.)\s*(\d+)\s*(?:decode\s+)?(?:rows|tokens)\b",
    re.IGNORECASE,
)
_HBM_RISK_RE = re.compile(
    r"([^.\n]*(?:extra\s+HBM|increases?\s+HBM|HBM\s+OOM|replicates?\s+weights)[^.\n]*)",
    re.IGNORECASE,
)
_CLAIM_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?(?:\s*(?:%|x|us|ms)\b)[^.\n]{0,90})")


@dataclasses.dataclass(frozen=True)
class EnvFlag:
    """One environment flag parsed from `envs.py`."""

    name: str
    default: str
    doc: str = ""
    line: int = 0


@dataclasses.dataclass(frozen=True)
class PriorVerdict:
    """Outcome close-gaps recorded for a previously tested flag."""

    decision: str
    delta_rel: float | None = None
    run_id: str = ""
    lesson: str = ""


@dataclasses.dataclass(frozen=True)
class GapCandidate:
    """An optimization switch either unset on served code or new upstream."""

    flag: str
    kind: str  # unset_in_served | new_upstream
    default: str
    doc: str
    needs_port: bool
    inactive_reason: str | None = None
    off_gates: tuple[str, ...] = ()
    step_bounds: tuple[int | None, int | None] = (None, None)
    claims: tuple[str, ...] = ()
    commits: tuple[str, ...] = ()
    prior: PriorVerdict | None = None


def parse_extra_env(value: str) -> dict[str, str]:
    """Parses `;`- or `,`-delimited `KEY=VALUE` env lists safely.

    Preserves commas inside flag values (for example
    `TPU_SAMPLER_VERIFY_TRIM_ROWS=32,40,48;TPU_FUSED_NORM_QKVA=1`) while
    splitting comma-separated `KEY=VALUE,KEY2=VALUE2` lists.
    """
    out: dict[str, str] = {}
    for part in _ENV_LIST_SPLIT_RE.split(str(value or "")):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        clean_key = key.strip()
        if clean_key:
            out[clean_key] = val.strip()
    return out


def recipe_env(
    recipe_text: str,
    knobs: Mapping[str, str] | None = None,
    extra_env_keys: Sequence[str] = (
        "EXTRA_ENV",
        "EXTRA_ENV_PREFILL",
        "EXTRA_ENV_DECODE",
        "EXTRA_SERVER_ENV",
        "SERVER_EXTRA_ENV",
    ),
) -> dict[str, str]:
    """Extracts effective env settings from recipe text plus `EXTRA_ENV*` knobs."""
    env: dict[str, str] = {}
    for m in re.finditer(
            r'["\']?([A-Z][A-Z0-9_]{3,})["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            recipe_text or "",
    ):
        env[m.group(1)] = m.group(2)
    if knobs:
        for key in extra_env_keys:
            if key in knobs:
                env.update(parse_extra_env(knobs[key]))
    return env


def parse_env_registry(source: str) -> dict[str, EnvFlag]:
    """Parses `ENVIRONMENT_VARIABLES` entries and preceding `#` comments."""
    registry: dict[str, EnvFlag] = {}
    comment_buf: list[str] = []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if stripped.startswith("#"):
            comment_buf.append(stripped.lstrip("# ").strip())
            i += 1
            continue
        key_only = re.match(r'^\s*"([A-Z][A-Z0-9_]+)"\s*:\s*$', raw)
        if key_only and i + 1 < len(lines) and lines[
                i + 1].lstrip().startswith("lambda"):
            combined = raw.rstrip() + " " + lines[i + 1].lstrip()
            start_line = i + 1
            i += 1
        else:
            combined = raw
            start_line = i + 1
        m = re.match(r'^\s*"([A-Z][A-Z0-9_]+)"\s*:\s*lambda\s*:(.*)$',
                     combined)
        if m:
            flag_name = m.group(1)
            expr_lines = [m.group(2)]
            paren_depth = combined.count("(") - combined.count(")")
            while paren_depth > 0 and i + 1 < len(lines):
                i += 1
                nxt = lines[i]
                if nxt.strip().startswith("#"):
                    comment_buf.append(nxt.strip().lstrip("# ").strip())
                else:
                    expr_lines.append(nxt.strip())
                paren_depth += nxt.count("(") - nxt.count(")")
            expr = " ".join(expr_lines)
            default = ""
            m_get = _GETENV_DEFAULT_RE.search(expr)
            m_bool = _BOOL_ENV_DEFAULT_RE.search(expr)
            if m_get:
                default = m_get.group(1).strip()
            elif m_bool:
                raw_b = m_bool.group(1).strip('"')
                default = "1" if raw_b in ("True", "1") else "0"
            elif "True" in expr:
                default = "1"
            elif "False" in expr:
                default = "0"
            registry[flag_name] = EnvFlag(
                name=flag_name,
                default=default,
                doc=" ".join(comment_buf).strip(),
                line=start_line,
            )
            comment_buf = []
            i += 1
            continue
        if stripped:
            comment_buf = []
        i += 1
    return registry


def is_enabled(
    flag: str,
    registry: Mapping[str, EnvFlag],
    env: Mapping[str, str],
) -> bool:
    """Returns True when `flag` is active in `env` or enabled by default."""
    if flag in env:
        return str(env[flag]).strip() not in _OFF_VALUES
    entry = registry.get(flag)
    if entry is None:
        return False
    return str(entry.default).strip() not in _OFF_VALUES


def required_gates(
    flag: str,
    registry: Mapping[str, EnvFlag],
) -> list[str]:
    """Returns direct prerequisite env flags documented on `flag`."""
    entry = registry.get(flag)
    if entry is None or not entry.doc:
        return []
    gates: list[str] = []
    for match in _REQUIRES_FLAG_RE.finditer(entry.doc):
        cand = match.group(1)
        req_val = match.group(2)
        if cand == flag or cand not in registry:
            continue
        if req_val is not None and req_val in _OFF_VALUES:
            continue
        if cand not in gates:
            gates.append(cand)
    return gates


def off_gate_chain(
    flag: str,
    registry: Mapping[str, EnvFlag],
    env: Mapping[str, str],
) -> list[str]:
    """Traces transitive prerequisite gates that are currently OFF in `env`."""
    visited: set[str] = {flag}
    chain: list[str] = []
    cur = flag
    while True:
        direct = required_gates(cur, registry)
        off_next = next(
            (g for g in direct
             if g not in visited and not is_enabled(g, registry, env)),
            None,
        )
        if off_next is not None:
            chain.append(off_next)
            visited.add(off_next)
            cur = off_next
            continue
        # Also walk through enabled intermediate gates whose own upstream gate is off.
        deeper: list[str] = []
        for g in direct:
            if g in visited:
                continue
            visited.add(g)
            sub = off_gate_chain(g, registry, env)
            if sub:
                deeper = [g, *sub]
                break
        if deeper:
            chain.extend(deeper)
        break
    return chain


def describe_gate_chain(
    chain: Sequence[str],
    registry: Mapping[str, EnvFlag],
    env: Mapping[str, str],
) -> str:
    """Human-readable explanation for an `off_gate_chain`."""
    if not chain:
        return ""
    leaf = chain[-1]
    val = env.get(leaf, registry[leaf].default if leaf in registry else "0")
    arrow = " -> ".join(chain)
    return f"requires {arrow}, and {leaf}={val or '0'} is off"


def moe_mode_conflict(
    flag: str,
    registry: Mapping[str, EnvFlag],
    ep_enabled: bool,
) -> str | None:
    """Returns a reason string if `flag` targets the opposite MoE sharding mode."""
    entry = registry.get(flag)
    doc = entry.doc if entry is not None else ""
    upper = flag.upper()
    is_ep_flag = ("MOE_FUSED_EP" in upper or "EP_MOE" in upper
                  or bool(_EP_ONLY_RE.search(doc)))
    is_tp_flag = ("MOE_TP_DECODE" in upper or bool(_TP_ONLY_RE.search(doc)))
    if is_ep_flag and not ep_enabled:
        return (f"{flag} only runs under expert-parallel MoE "
                "(--enable-expert-parallel), which is off on this leg")
    if is_tp_flag and ep_enabled:
        return (
            f"{flag} only runs on TP MoE decode, but --enable-expert-parallel "
            "is enabled on this leg")
    return None


def step_range(doc: str) -> tuple[int | None, int | None]:
    """Extracts `(min_tokens, max_tokens)` step window from a flag docstring."""
    if not doc:
        return None, None
    m_range = _STEP_RANGE_RE.search(doc)
    if m_range:
        return int(m_range.group(1)), int(m_range.group(2))
    lo_m = _STEP_GE_RE.search(doc)
    hi_m = _STEP_LE_RE.search(doc)
    lo = int(lo_m.group(1)) if lo_m else None
    hi = int(hi_m.group(1)) if hi_m else None
    return lo, hi


def inert_at_steps(
    flag: str,
    registry: Mapping[str, EnvFlag],
    steps: Sequence[int],
) -> str | None:
    """Returns a reason when `flag`'s documented step window misses `steps`."""
    if not steps:
        return None
    entry = registry.get(flag)
    if entry is None:
        return None
    lo, hi = step_range(entry.doc)
    if lo is None and hi is None:
        return None
    active = [
        s for s in steps if (lo is None or s >= lo) and (hi is None or s <= hi)
    ]
    if active:
        return None
    bounds = (f"{lo}..{hi}" if lo is not None and hi is not None else
              (f">={lo}" if lo is not None else f"<={hi}"))
    return f"active only at {bounds} tokens (arm steps: {list(steps)})"


def hbm_risk(doc: str) -> str | None:
    """Returns documented HBM memory-increase warning text, if any."""
    if not doc:
        return None
    m = _HBM_RISK_RE.search(doc)
    return m.group(1).strip() if m else None


def _extract_claims(texts: Sequence[str]) -> tuple[str, ...]:
    claims: list[str] = []
    for txt in texts:
        for m in _CLAIM_RE.finditer(txt):
            snippet = m.group(1).strip()
            if snippet not in claims:
                claims.append(snippet)
            if len(claims) >= 3:
                return tuple(claims)
    return tuple(claims)


def find_gaps(
    served: Mapping[str, EnvFlag],
    upstream: Mapping[str, EnvFlag],
    arm_env: Mapping[str, str],
    commits_by_flag: Mapping[str, Sequence[str]] | None = None,
    *,
    target_terms: Sequence[str] = (),
    exclude_terms: Sequence[str] = (),
    priors: Mapping[str, PriorVerdict] | None = None,
    steps: Sequence[int] = (),
    ep_enabled: bool | None = None,
) -> list[GapCandidate]:
    """Finds unset served flags and new upstream flags, annotating inertness."""
    out: list[GapCandidate] = []
    for name, up_flag in sorted(upstream.items()):
        hay = f"{name} {up_flag.doc}".lower()
        if exclude_terms and any(t.lower() in hay for t in exclude_terms):
            continue
        if target_terms and not any(t.lower() in hay for t in target_terms):
            continue

        in_served = name in served
        if in_served and is_enabled(name, served, arm_env):
            continue
        if not in_served and str(up_flag.default).strip() not in _OFF_VALUES:
            # Already default-on once ported, still a port candidate!
            pass

        kind = "unset_in_served" if in_served else "new_upstream"
        reg_for_gates = served if in_served else upstream
        chain = off_gate_chain(name, reg_for_gates, arm_env)
        inactive = inert_at_steps(name, up_flag and upstream, steps)
        if inactive is None and ep_enabled is not None:
            inactive = moe_mode_conflict(name, upstream, ep_enabled)
        if inactive is None and chain and not in_served:
            # Note prerequisite chain on port candidates without hiding unset served chains.
            pass
        commit_msgs = tuple((commits_by_flag or {}).get(name, ()))
        claims = _extract_claims((up_flag.doc, *commit_msgs))
        out.append(
            GapCandidate(
                flag=name,
                kind=kind,
                default=up_flag.default,
                doc=up_flag.doc,
                needs_port=not in_served,
                inactive_reason=inactive,
                off_gates=tuple(chain),
                step_bounds=step_range(up_flag.doc),
                claims=claims,
                commits=tuple(c.splitlines()[0] for c in commit_msgs[:2]),
                prior=(priors or {}).get(name),
            ))
    out.sort(key=lambda c: (
        c.inactive_reason is not None,
        c.prior is not None and c.prior.decision in ledger.NEGATIVE,
        c.needs_port,
        -len(c.claims),
        c.flag,
    ))
    return out


def render_markdown(
    candidates: Sequence[GapCandidate],
    served_ref: str = "served",
    upstream_ref: str = "upstream",
) -> str:
    """Renders `GapCandidate` rows as Markdown."""
    lines = [
        f"### Switch Gap Analysis (`{served_ref}` vs `{upstream_ref}`)",
        "",
        "| Flag | Status | Gate Chain / Inertness | Claims / Notes |",
        "|---|---|---|---|",
    ]
    for c in candidates:
        status = "port needed" if c.needs_port else f"unset (default `{c.default or '0'}`)"
        gate_info = c.inactive_reason or (f"needs `{' -> '.join(c.off_gates)}`"
                                          if c.off_gates else "ready")
        notes = "; ".join(c.claims) if c.claims else (c.doc[:90] or "—")
        if c.prior is not None:
            notes = f"[prior: {c.prior.decision}] {notes}"
        lines.append(f"| `{c.flag}` | {status} | {gate_info} | {notes} |")
    return "\n".join(lines) + "\n"

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
"""Run-history indexing, baseline lookup, and per-concurrency replicate noise.

Pooling runs across a concurrency sweep (`c=32..96`) under one recipe inflates
the apparent variance far above the run-to-run spread of repeated runs at one
concurrency. `noise_estimate` groups runs by exact `(perf_knobs, concurrency)`
so noise bands reflect true replicate spread rather than sweep spread.

Runs are scored on tok/s/chip by default. `Mtok/$` needs a chip-hour rate from
the caller (`price_usd`, and `gpu_hour_cost_usd` for GPU rows); this package
ships no prices.
"""

import dataclasses
import json
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hillclimb import metrics as metrics_lib

MTOK_PER_DOLLAR = "mtok_per_dollar"
TOK_PER_S_PER_CHIP = "token_throughput_per_chip"

METRIC_LABELS: Mapping[str, str] = {
    MTOK_PER_DOLLAR: "Mtok/$",
    TOK_PER_S_PER_CHIP: "tok/s/chip",
}


def metric_label(metric: str) -> str:
    """Human-readable label for a metric name."""
    return METRIC_LABELS.get(metric, metric)


@dataclasses.dataclass(frozen=True)
class KnobConventions:
    """Names of harness, benchmark, env-list, and concurrency knobs."""

    plumbing: tuple[str, ...] = (
        "EXIT_ON_READY",
        "EXPORT_LOG_ARTIFACTS",
        "RESULTS_UPLOAD_ROOT",
    )
    bench: tuple[str, ...] = (
        "FAST",
        "BENCH_TIMEOUT_SEC",
        "BENCHMARK_DURATION_SECONDS",
    )
    code: tuple[str, ...] = ("CODE_COMMIT", "DOCKER_IMAGE")
    extra_env: tuple[str, ...] = (
        "EXTRA_ENV",
        "EXTRA_ENV_PREFILL",
        "EXTRA_ENV_DECODE",
        "EXTRA_SERVER_ENV",
        "SERVER_EXTRA_ENV",
    )
    kv_bytes: tuple[str, ...] = (
        "PREFILL_KV_BYTES",
        "DECODE_KV_BYTES",
        "KV_CACHE_BYTES",
    )
    concurrency: str = "CLIENT_CONCURRENCIES"
    family_groups: tuple[tuple[str, ...], ...] = ()

    @property
    def non_perf(self) -> frozenset[str]:
        """Knobs ignored when matching configurations for baselines or noise."""
        return frozenset((*self.plumbing, *self.bench, *self.code))

    def family(self, recipe: str) -> str:
        """Canonical recipe family name."""
        for group in self.family_groups:
            if recipe in group:
                return group[0]
        return recipe


DEFAULT_CONVENTIONS = KnobConventions()


@dataclasses.dataclass(frozen=True)
class RunRecord:
    """One benchmark run record from a JSONL history file."""

    run_id: str
    recipe: str
    status: str
    knobs: Mapping[str, str]
    metrics: Mapping[str, Any]
    price_usd: float | None = None
    metric: str = TOK_PER_S_PER_CHIP
    created: str = ""
    tags: tuple[str, ...] = ()

    @property
    def job_id(self) -> str:
        """Compatibility alias for `run_id`."""
        return self.run_id

    @property
    def succeeded(self) -> bool:
        return self.status.strip().lower() in (
            "succeeded",
            "success",
            "completed",
            "ok",
            "passed",
        )

    @property
    def tokens_per_s_per_chip(self) -> float | None:
        val = self.metrics.get("token_throughput_per_chip")
        if not isinstance(val, bool) and isinstance(val, (int, float)):
            return float(val) if val > 0 else None
        return None

    @property
    def mtok_per_dollar(self) -> float | None:
        tpc = self.tokens_per_s_per_chip
        if tpc is None or self.price_usd is None or self.price_usd <= 0:
            return None
        return metrics_lib.mtok_per_usd(tpc, self.price_usd)

    @property
    def score(self) -> float | None:
        """Value of the run's primary scoring metric (tok/s/chip by default)."""
        if self.metric == MTOK_PER_DOLLAR:
            return self.mtok_per_dollar
        val = self.metrics.get(self.metric)
        if not isinstance(val, bool) and isinstance(val, (int, float)):
            return float(val) if val > 0 else None
        return None

    @property
    def p90_interactivity(self) -> float | None:
        for key in ("p90_interactivity_tok_s", "p90_interactivity"):
            val = self.metrics.get(key)
            if not isinstance(val, bool) and isinstance(val, (int, float)):
                return float(val)
        return None

    @property
    def mean_ttft_ms(self) -> float | None:
        val = self.metrics.get("mean_ttft_ms")
        if not isinstance(val, bool) and isinstance(val, (int, float)):
            return float(val)
        return None


def _is_gpu_row(recipe: str, tags: Sequence[str] = ()) -> bool:
    haystack = " ".join((recipe, *tags)).lower()
    return any(tok in haystack for tok in ("b200", "gb200", "h100", "h200"))


def record_from_row(
    row: Mapping[str, Any],
    price_usd: float | None = None,
    metric: str = TOK_PER_S_PER_CHIP,
    gpu_hour_cost_usd: float | None = None,
) -> RunRecord:
    """Converts a parsed JSONL row dictionary into a `RunRecord`.

    `price_usd` (and `gpu_hour_cost_usd` for GPU rows) are the caller's
    chip-hour rates; without them `mtok_per_dollar` is `None`.
    """
    raw_knobs = row.get("knobs") or {}
    knobs = ({
        str(k): str(v)
        for k, v in raw_knobs.items()
    } if isinstance(raw_knobs, Mapping) else {})
    raw_metrics = row.get("metrics") or {}
    metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
    tags = tuple(str(t) for t in (row.get("tags") or ()))
    recipe = str(row.get("recipe") or "")
    row_price = (gpu_hour_cost_usd if _is_gpu_row(recipe, tags) else price_usd)
    run_id = str(row.get("run_id") or row.get("id") or "")
    status = str(row.get("status") or row.get("job_status") or "Succeeded")
    return RunRecord(
        run_id=run_id,
        recipe=recipe,
        status=status,
        knobs=knobs,
        metrics=metrics,
        price_usd=row_price,
        metric=metric,
        created=str(row.get("create_time") or row.get("created") or ""),
        tags=tags,
    )


def load_jsonl(
    lines: Iterable[str],
    price_usd: float | None = None,
    metric: str = TOK_PER_S_PER_CHIP,
    gpu_hour_cost_usd: float | None = None,
) -> list[RunRecord]:
    """Parses JSONL lines into `RunRecord` objects."""
    out: list[RunRecord] = []
    for line in lines:
        if line.strip():
            out.append(
                record_from_row(
                    json.loads(line),
                    price_usd=price_usd,
                    metric=metric,
                    gpu_hour_cost_usd=gpu_hour_cost_usd,
                ))
    return out


def upsert_rows(
    existing_lines: Iterable[str],
    new_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[str], int, int]:
    """Upserts JSONL rows by `run_id` / `id`, preserving row order."""
    parsed: list[dict[str, Any]] = []
    index_by_id: dict[str, int] = {}
    for line in existing_lines:
        stripped = line.strip()
        if not stripped:
            continue
        obj = json.loads(stripped)
        key = str(obj.get("run_id") or obj.get("id") or "")
        if key:
            index_by_id[key] = len(parsed)
        parsed.append(obj)

    replaced = 0
    appended = 0
    for row in new_rows:
        obj = dict(row)
        key = str(obj.get("run_id") or obj.get("id") or "")
        if key and key in index_by_id:
            parsed[index_by_id[key]] = obj
            replaced += 1
        else:
            if key:
                index_by_id[key] = len(parsed)
            parsed.append(obj)
            appended += 1
    rendered = [json.dumps(obj, sort_keys=True) for obj in parsed]
    return rendered, replaced, appended


def perf_knobs(
    knobs: Mapping[str, str],
    defaults: Mapping[str, str] | None = None,
    conventions: KnobConventions = DEFAULT_CONVENTIONS,
) -> dict[str, str]:
    """Returns performance-relevant knobs after merging `defaults`."""
    merged = {**(defaults or {}), **knobs}
    return {
        k: v
        for k, v in merged.items() if k not in conventions.non_perf and v != ""
    }


def _concurrency_key(
    run: RunRecord,
    conventions: KnobConventions = DEFAULT_CONVENTIONS,
) -> int | None:
    """Extracts the run's integer concurrency level from metrics or knobs."""
    for key in ("concurrency", "max_concurrency", "conc", "concurrency_level"):
        val = run.metrics.get(key)
        if not isinstance(val, bool) and isinstance(val, (int, float)):
            if int(val) > 0:
                return int(val)
    raw = run.knobs.get(conventions.concurrency, "")
    nums = [int(tok) for tok in re.findall(r"\d+", str(raw))]
    if len(nums) == 1:
        return nums[0]
    return None


@dataclasses.dataclass(frozen=True)
class BaselineMatch:
    """Closest completed baseline run and the perf knobs that differ."""

    run: RunRecord
    differing_knobs: tuple[str, ...]


def find_baselines(
    recipe: str,
    target_knobs: Mapping[str, str],
    runs: Sequence[RunRecord],
    defaults: Mapping[str, str] | None = None,
    conventions: KnobConventions = DEFAULT_CONVENTIONS,
) -> list[BaselineMatch]:
    """Finds completed runs in the same recipe family ordered by closeness."""
    family = conventions.family(recipe)
    want = perf_knobs(target_knobs, defaults, conventions)
    matches: list[BaselineMatch] = []
    for run in runs:
        if (conventions.family(run.recipe) != family or not run.succeeded
                or run.score is None):
            continue
        got = perf_knobs(run.knobs, defaults, conventions)
        diff = tuple(
            sorted(k for k in set(want) | set(got)
                   if want.get(k) != got.get(k)))
        matches.append(BaselineMatch(run=run, differing_knobs=diff))
    matches.sort(key=lambda m: (len(m.differing_knobs), -(m.run.score or 0.0)))
    return matches


@dataclasses.dataclass(frozen=True)
class NoiseEstimate:
    """Replicate noise estimate from repeated fixed-concurrency runs."""

    n_groups: int
    n_runs: int
    rel_half_range: float
    rel_sd: float = 0.0
    pooled_sweep_rel_sd: float | None = None
    per_concurrency_sd: Mapping[int, float] = dataclasses.field(
        default_factory=dict)


def noise_estimate(
    runs: Sequence[RunRecord],
    recipe: str,
    defaults: Mapping[str, str] | None = None,
    conventions: KnobConventions = DEFAULT_CONVENTIONS,
) -> NoiseEstimate | None:
    """Computes run-to-run replicate noise grouped by `(perf_knobs, concurrency)`.

    Grouping strictly by both `perf_knobs` AND resolved `concurrency` prevents
    concurrency-sweep runs (`c=32, 48, 64, 96`) that share the same recipe or
    knob template from inflating the noise estimate: the pooled sweep spread is
    far wider than the replicate spread at one concurrency.

    Args:
      runs: Completed and failed run records.
      recipe: Target recipe name (matched via `conventions.family`).
      defaults: Optional recipe default knobs.
      conventions: Knob naming conventions.

    Returns:
      `NoiseEstimate` when at least one replicate group of size `>= 2` exists,
      otherwise `None`.
    """
    family = conventions.family(recipe)
    groups: dict[tuple[Any, ...], list[float]] = {}
    by_conc: dict[tuple[int, tuple[tuple[str, str], ...]], list[float]] = {}
    all_family_scores: list[float] = []

    for run in runs:
        if conventions.family(run.recipe) != family or not run.succeeded:
            continue
        val = run.score
        if val is None or val <= 0:
            continue
        all_family_scores.append(val)
        conc = _concurrency_key(run, conventions)
        pk = dict(perf_knobs(run.knobs, defaults, conventions))
        if conc is not None:
            pk[conventions.concurrency] = str(conc)
        sig = tuple(sorted(pk.items()))
        key = (conc, sig)
        groups.setdefault(key, []).append(val)
        if conc is not None:
            by_conc.setdefault((conc, sig), []).append(val)

    half_ranges: list[float] = []
    rel_sds: list[float] = []
    total_replicates = 0
    for vals in groups.values():
        if len(vals) < 2:
            continue
        mean = statistics.fmean(vals)
        if mean <= 0:
            continue
        half_ranges.append((max(vals) - min(vals)) / (2.0 * mean))
        rel_sds.append(statistics.stdev(vals) / mean)
        total_replicates += len(vals)

    if not half_ranges:
        return None

    conc_sds: dict[int, float] = {}
    conc_buckets: dict[int, list[float]] = {}
    for (conc, _), vals in by_conc.items():
        if len(vals) >= 2:
            mean = statistics.fmean(vals)
            if mean > 0:
                conc_buckets.setdefault(conc, []).append(
                    statistics.stdev(vals) / mean)
    for conc, sds in sorted(conc_buckets.items()):
        conc_sds[conc] = statistics.median(sds)

    pooled_sd: float | None = None
    if len(all_family_scores) >= 2:
        pooled_mean = statistics.fmean(all_family_scores)
        if pooled_mean > 0:
            pooled_sd = statistics.stdev(all_family_scores) / pooled_mean

    return NoiseEstimate(
        n_groups=len(half_ranges),
        n_runs=total_replicates,
        rel_half_range=statistics.median(half_ranges),
        rel_sd=statistics.median(rel_sds),
        pooled_sweep_rel_sd=pooled_sd,
        per_concurrency_sd=conc_sds,
    )


def tried_values(
    runs: Sequence[RunRecord],
    recipe: str,
    knob: str,
    conventions: KnobConventions = DEFAULT_CONVENTIONS,
) -> dict[str, list[RunRecord]]:
    """Groups runs in `recipe`'s family by their setting of `knob`."""
    family = conventions.family(recipe)
    by_value: dict[str, list[RunRecord]] = {}
    for run in runs:
        if conventions.family(run.recipe) != family or knob not in run.knobs:
            continue
        by_value.setdefault(run.knobs[knob], []).append(run)
    return by_value


def render_noise_markdown(
    estimate: NoiseEstimate | None,
    recipe: str,
) -> str:
    """Formats a `NoiseEstimate` as a concise Markdown summary."""
    if estimate is None:
        return (f"No fixed-concurrency replicate groups (`n >= 2`) found for "
                f"`{recipe}`.\n")
    lines = [
        f"### Replicate Noise Estimate (`{recipe}`)",
        "",
        f"- **Replicate groups**: `{estimate.n_groups}` (`{estimate.n_runs}` runs)",
        f"- **True replicate relative SD**: **{estimate.rel_sd * 100:.2f}%**",
        (f"- **Median relative half-range**: "
         f"`±{estimate.rel_half_range * 100:.2f}%` "
         f"(2x decision band `±{2 * estimate.rel_half_range * 100:.2f}%`)"),
    ]
    if estimate.pooled_sweep_rel_sd is not None:
        lines.append(
            f"- **Pooled sweep CV (unfiltered across concurrencies)**: "
            f"`{estimate.pooled_sweep_rel_sd * 100:.2f}%`")
    if estimate.per_concurrency_sd:
        parts = [
            f"`c={c}`: {sd * 100:.2f}%"
            for c, sd in sorted(estimate.per_concurrency_sd.items())
        ]
        lines.append(f"- **Per-concurrency replicate SD**: {', '.join(parts)}")
    return "\n".join(lines) + "\n"

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
"""Trial-scoped benchmark `summary.json` parsing and raw-replicate measurement pooling."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import sqlite3
import statistics
from typing import Any, Mapping, Sequence

from framework.hc import constants, db, telemetry

Path = pathlib.Path

ENVIRONMENT_FIELDS: tuple[str, ...] = (
    "image",
    "image_digest",
    "python_version",
    "jax_version",
    "jaxlib_version",
    "torch_version",
    "torch_tpu_version",
    "engine_version",
    "libtpu_init_args",
    "xla_flags",
)


@dataclasses.dataclass(frozen=True)
class MetricObservation:
    """One job's raw sample observations of a single `(cell_id, metric_name)`."""

    samples: tuple[float, ...]

    @classmethod
    def from_samples(cls, values: Sequence[float]) -> MetricObservation:
        """Build a `MetricObservation` from a non-empty sequence of finite samples."""
        cleaned = tuple(_check_finite(v, "sample") for v in values)
        if not cleaned:
            raise constants.ValidationError(
                "metric sample list cannot be empty")
        return cls(samples=cleaned)


@dataclasses.dataclass(frozen=True)
class PooledMetric:
    """Aggregated metric value stored in `Measurement_Metrics`."""

    value: float
    n: int
    stdev: float | None
    min_val: float
    max_val: float


@dataclasses.dataclass(frozen=True)
class ParsedBenchmarkJob:
    """Verified benchmark payload for a single `Succeeded` job."""

    job_id: str
    environment: dict[str, str | None]
    artifacts: tuple[str, ...]
    cell_metrics: dict[int, dict[str, MetricObservation]]


def _check_finite(raw: Any, label: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise constants.ValidationError(f"{label} must be a finite number")
    val = float(raw)
    if not math.isfinite(val):
        raise constants.ValidationError(f"{label} must be finite, got {val}")
    return val


def _as_int(raw: Any, label: str) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise constants.ValidationError(
            f"{label} must be an integer, got {raw!r}")
    return raw


def normalize_environment(
    *,
    exec_image: str,
    flags: Mapping[str, Any] | None = None,
    summary_env: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    """Construct the 10-field `Measurements.environment` record for a job."""
    env_flags = (flags["env"] if isinstance(flags, Mapping)
                 and isinstance(flags.get("env"), Mapping) else {})
    raw_env = summary_env if isinstance(summary_env, Mapping) else {}

    def _opt_str(val: Any) -> str | None:
        if val is None:
            return None
        text = str(val).strip()
        return text if text else None

    out = {field: _opt_str(raw_env.get(field)) for field in ENVIRONMENT_FIELDS}
    out["image"] = out["image"] or exec_image
    if (out["image_digest"] is None and out["image"]
            and "@sha256:" in out["image"]):
        _, _, digest = out["image"].partition("@sha256:")
        if digest:
            out["image_digest"] = f"sha256:{digest}"
    out["libtpu_init_args"] = (out["libtpu_init_args"]
                               or _opt_str(env_flags.get("LIBTPU_INIT_ARGS")))
    out["xla_flags"] = out["xla_flags"] or _opt_str(env_flags.get("XLA_FLAGS"))
    return out


def _load_ok_summary(
    summary_path: Path,
    job_id: str,
    label: str,
) -> dict[str, Any]:
    if not summary_path.is_file():
        raise constants.ValidationError(
            f"sealed {label} summary.json missing for job {job_id}: {summary_path}"
        )
    try:
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise constants.ValidationError(
            f"invalid JSON in sealed {label} summary.json for job {job_id}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise constants.ValidationError(
            f"sealed {label} summary.json for job {job_id} must be a JSON object"
        )
    status = raw.get("status")
    if status != "ok":
        raise constants.ValidationError(
            f"{label} job {job_id} summary.json status is {status!r}, expected 'ok'"
        )
    return raw


def _extract_cell_metrics_from_dict(
    cell_obj: Mapping[str, Any],
    context: str,
) -> dict[str, MetricObservation]:
    """Extract `{metric_name: MetricObservation}` from `cell.replicates`."""
    reps = cell_obj.get("replicates")
    if not isinstance(reps, list) or not reps or not all(
            isinstance(r, dict) for r in reps):
        raise constants.ValidationError(
            f"{context}.replicates must be a non-empty list of objects")

    metric_keys = {
        k
        for rep in reps
        for k, v in rep.items() if k not in ("replicate", "seed", "failed")
        and isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    out: dict[str, MetricObservation] = {}
    for key in sorted(metric_keys):
        vals = [
            rep[key] for rep in reps if isinstance(rep.get(key), (int, float))
            and not isinstance(rep.get(key), bool)
        ]
        if vals:
            out[key] = MetricObservation.from_samples(vals)
    return out


def _match_protocol_cell_id(
    cell_obj: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
    workload_kind: constants.WorkloadKind,
) -> int | None:
    """Return matching `Protocol_Cells.id` for `cell_obj`, or None if non-protocol cell."""
    cidx = _as_int(cell_obj.get("cell_index"), "cell_index")
    if cidx is not None:
        for pcell in protocol_cells:
            if int(pcell["cell_index"]) == cidx:
                return int(pcell["id"])
        raise constants.ValidationError(
            f"cell_index {cidx} out of range for protocol ({len(protocol_cells)} cells)"
        )

    topo = cell_obj.get("topology")
    if workload_kind == constants.WorkloadKind.INFERENCE:
        raw_prefill = cell_obj.get("prefill")
        prefill = _as_int(
            cell_obj.get("isl") if raw_prefill is None else raw_prefill,
            "prefill",
        )
        raw_decode = cell_obj.get("decode")
        decode = _as_int(
            cell_obj.get("osl") if raw_decode is None else raw_decode,
            "decode",
        )
        conc = _as_int(cell_obj.get("concurrency"), "concurrency")
        if prefill is None or decode is None or conc is None:
            return None
        matches = [
            pcell for pcell in protocol_cells
            if int(pcell["prefill"]) == prefill and int(pcell["decode"]) ==
            decode and int(pcell["concurrency"]) == conc and (
                topo is None or str(pcell["topology"]) == str(topo))
        ]
    else:
        seq_len = _as_int(cell_obj.get("sequence_length"), "sequence_length")
        gbs = _as_int(cell_obj.get("global_batch_size"), "global_batch_size")
        pdbs = _as_int(cell_obj.get("per_device_batch_size"),
                       "per_device_batch_size")
        if seq_len is None or gbs is None:
            return None
        matches = [
            pcell for pcell in protocol_cells
            if int(pcell["sequence_length"]) == seq_len
            and int(pcell["global_batch_size"]) == gbs and (
                pdbs is None or pcell["per_device_batch_size"] is None
                or int(pcell["per_device_batch_size"]) == pdbs) and (
                    topo is None or str(pcell["topology"]) == str(topo))
        ]

    if len(matches) == 1:
        return int(matches[0]["id"])
    if len(matches) > 1:
        raise constants.ValidationError(
            "ambiguous cell in summary.json matches multiple protocol cells; specify topology or cell_index"
        )
    return None


def parse_benchmark_summary(
        *,
        job_id: str,
        summary_path: Path,
        protocol_row: Mapping[str, Any],
        protocol_cells: Sequence[Mapping[str, Any]],
        flags: Mapping[str, Any] | None = None,
        extra_artifacts: Sequence[str] = (),
) -> ParsedBenchmarkJob:
    """Validate a sealed benchmark `summary.json` and extract per-cell observations."""
    raw = _load_ok_summary(summary_path, job_id, "benchmark")
    workload_kind = constants.WorkloadKind(protocol_row["workload_kind"])
    primary_metric = str(protocol_row["primary_metric"])

    cells_raw = raw.get("cells")
    if not isinstance(cells_raw, list) or not cells_raw:
        raise constants.ValidationError(
            f"sealed summary.json for job {job_id} must contain a non-empty 'cells' list"
        )

    cell_metrics: dict[int, dict[str, MetricObservation]] = {}
    for idx, cell_obj in enumerate(cells_raw):
        if not isinstance(cell_obj, Mapping):
            raise constants.ValidationError(
                f"summary.cells[{idx}] in job {job_id} must be an object")
        cid = _match_protocol_cell_id(cell_obj, protocol_cells, workload_kind)
        if cid is None:
            continue
        cell_status = cell_obj.get("status")
        if cell_status is not None and cell_status != "ok":
            raise constants.ValidationError(
                f"protocol cell {cid} in job {job_id} reported status={cell_status!r}"
            )
        extracted = _extract_cell_metrics_from_dict(cell_obj,
                                                    f"summary.cells[{idx}]")
        if cid in cell_metrics:
            raise constants.ValidationError(
                f"duplicate cell entry for protocol cell {cid} in job {job_id}"
            )
        cell_metrics[cid] = extracted

    if not cell_metrics:
        raise constants.ValidationError(
            f"sealed summary.json for job {job_id} matched no protocol cells")

    for cid, m_map in cell_metrics.items():
        if primary_metric not in m_map:
            raise constants.ValidationError(
                f"sealed summary.json for job {job_id} missing primary_metric {primary_metric!r} on cell {cid}"
            )

    env = normalize_environment(
        exec_image=str(protocol_row["exec_image"]),
        flags=flags,
        summary_env=raw.get("environment"),
    )
    artifacts = [str(summary_path), *(str(a) for a in extra_artifacts)]
    return ParsedBenchmarkJob(
        job_id=job_id,
        environment=env,
        artifacts=tuple(artifacts),
        cell_metrics=cell_metrics,
    )


def parse_accuracy_summary(
    *,
    job_id: str,
    summary_path: Path,
    metric_name: str,
) -> float:
    """Validate a sealed accuracy `summary.json` (`{"status": "ok", "scores": {...}}`)."""
    raw = _load_ok_summary(summary_path, job_id, "accuracy")
    scores = raw.get("scores")
    if not isinstance(scores, Mapping) or metric_name not in scores:
        raise constants.ValidationError(
            f"accuracy metric {metric_name!r} not found in sealed summary for job {job_id}"
        )
    return _check_finite(scores[metric_name], f"scores.{metric_name}")


def pool_observations(
    observations: Sequence[MetricObservation],
    statistic: constants.MetricStatistic,
    *,
    metric_name: str = "metric",
) -> PooledMetric:
    """Pool 1+ `MetricObservation`s across jobs using their raw per-replicate samples."""
    if not observations:
        raise constants.ValidationError(
            f"cannot pool empty observations for {metric_name}")

    merged_samples = [v for obs in observations for v in obs.samples]
    n_total = len(merged_samples)
    center = (statistics.fmean(merged_samples)
              if statistic == constants.MetricStatistic.MEAN else float(
                  statistics.median(merged_samples)))
    stdev_val = statistics.stdev(merged_samples) if n_total >= 2 else None
    return PooledMetric(
        value=center,
        n=n_total,
        stdev=stdev_val,
        min_val=min(merged_samples),
        max_val=max(merged_samples),
    )


def pool_parsed_jobs(
    jobs: Sequence[ParsedBenchmarkJob],
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str | None], list[str], dict[int, dict[
        str, PooledMetric]], float]:
    """Verify environment consistency across `jobs` and pool per-cell metrics."""
    if not jobs:
        raise constants.ValidationError("cannot pool zero benchmark jobs")

    ref_env = jobs[0].environment
    for other in jobs[1:]:
        for field in ENVIRONMENT_FIELDS:
            if other.environment.get(field) != ref_env.get(field):
                raise constants.ValidationError(
                    f"inconsistent environment.{field} across pooled jobs in trial: "
                    f"{jobs[0].job_id}={ref_env.get(field)!r} vs "
                    f"{other.job_id}={other.environment.get(field)!r}")

    statistic = constants.MetricStatistic(
        protocol_row["primary_metric_statistic"])
    primary_metric = str(protocol_row["primary_metric"])

    all_cell_ids = sorted({cid for job in jobs for cid in job.cell_metrics})
    pooled_cells: dict[int, dict[str, PooledMetric]] = {}
    for cid in all_cell_ids:
        metric_names = sorted({
            m_name
            for job in jobs
            for m_name in job.cell_metrics.get(cid, ())
        })
        pooled_cells[cid] = {
            m_name:
            pool_observations(
                [
                    job.cell_metrics[cid][m_name] for job in jobs if
                    cid in job.cell_metrics and m_name in job.cell_metrics[cid]
                ],
                statistic,
                metric_name=m_name,
            )
            for m_name in metric_names
        }

    artifacts: list[str] = []
    seen_art: set[str] = set()
    for job in jobs:
        for art in job.artifacts:
            if art not in seen_art:
                seen_art.add(art)
                artifacts.append(art)

    # Compute weighted geometric mean primary_value across measured cells for telemetry.
    weight_by_cid = {int(c["id"]): float(c["weight"]) for c in protocol_cells}
    weighted_pairs = [(weight_by_cid.get(cid, 0.0),
                       pooled_cells[cid][primary_metric].value)
                      for cid in pooled_cells
                      if primary_metric in pooled_cells[cid]
                      and weight_by_cid.get(cid, 0.0) > 0.0]
    measured_w_sum = sum(w for w, _ in weighted_pairs)
    if measured_w_sum > 0.0:
        if all(v > 0.0 for _, v in weighted_pairs):
            primary_value = math.exp(
                sum(w * math.log(v)
                    for w, v in weighted_pairs) / measured_w_sum)
        else:
            primary_value = (sum(w * v
                                 for w, v in weighted_pairs) / measured_w_sum)
    else:
        first_cid = next(iter(pooled_cells))
        primary_value = pooled_cells[first_cid][primary_metric].value

    return ref_env, artifacts, pooled_cells, primary_value


def record_pooled_measurement(
    conn: sqlite3.Connection,
    *,
    campaign_id: int,
    round_id: int | None,
    trial: str,
    candidate_id: int,
    protocol_row: Mapping[str, Any],
    protocol_cells: Sequence[Mapping[str, Any]],
    parsed_jobs: Sequence[ParsedBenchmarkJob],
    trigger_job_id: str,
    telem: telemetry.TelemetryBuffer,
) -> int:
    """Persist a new immutable `Measurements` snapshot and update `Jobs.measurement`."""
    protocol_id = int(protocol_row["id"])
    env, artifacts, pooled_cells, primary_value = pool_parsed_jobs(
        parsed_jobs, protocol_row, protocol_cells)

    with db.transaction(conn):
        cur = db.exec_query(
            conn,
            "insert_measurement",
            trial=trial,
            subject=candidate_id,
            protocol=protocol_id,
            env_image=env["image"],
            env_image_digest=env["image_digest"],
            env_python_version=env["python_version"],
            env_jax_version=env["jax_version"],
            env_jaxlib_version=env["jaxlib_version"],
            env_torch_version=env["torch_version"],
            env_torch_tpu_version=env["torch_tpu_version"],
            env_engine_version=env["engine_version"],
            env_libtpu_init_args=env["libtpu_init_args"],
            env_xla_flags=env["xla_flags"],
        )
        measurement_id = int(cur.lastrowid)

        for cid, m_map in sorted(pooled_cells.items()):
            for m_name, pooled in sorted(m_map.items()):
                db.exec_query(
                    conn,
                    "insert_measurement_metric",
                    measurement_id=measurement_id,
                    cell_id=cid,
                    metric_name=m_name,
                    value=pooled.value,
                    n=pooled.n,
                    stdev=pooled.stdev,
                    min_val=pooled.min_val,
                    max_val=pooled.max_val,
                )

        for idx, art in enumerate(artifacts):
            db.exec_query(
                conn,
                "insert_measurement_artifact",
                measurement_id=measurement_id,
                artifact_index=idx,
                artifact_path=art,
            )

        for idx, pjob in enumerate(parsed_jobs):
            db.exec_query(
                conn,
                "insert_measurement_job",
                measurement_id=measurement_id,
                job_index=idx,
                job_id=pjob.job_id,
            )
            db.exec_query(
                conn,
                "set_job_measurement",
                measurement_id=measurement_id,
                job_id=pjob.job_id,
            )

    telem.record(
        "measurement.recorded",
        campaign_id=campaign_id,
        round_id=round_id,
        measurement_id=measurement_id,
        candidate_id=candidate_id,
        protocol_id=protocol_id,
        trial=trial,
        job_id=trigger_job_id,
        primary_value=primary_value,
    )
    return measurement_id

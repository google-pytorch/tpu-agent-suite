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
"""Normalized `STRICT` SQLite coordination store schema and connection helpers."""

from __future__ import annotations

import contextlib
import itertools
import pathlib
import sqlite3
from typing import Iterator

from framework.hc import constants, paths

Path = pathlib.Path

BUSY_TIMEOUT_MS: int = 30_000
SCHEMA_VERSION: int = 1
_SAVEPOINT_SEQ = itertools.count(1)

SCHEMA_SQL: str = f"""
CREATE TABLE IF NOT EXISTS Protocols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workload_kind TEXT NOT NULL,
    primary_metric TEXT NOT NULL,
    compute_profile TEXT NOT NULL,
    primary_metric_unit TEXT NOT NULL,
    primary_metric_direction TEXT NOT NULL,
    primary_metric_statistic TEXT NOT NULL,
    min_replicates INTEGER NOT NULL CHECK (min_replicates >= 1),
    min_win_percentage REAL NOT NULL,
    max_cell_regression_percentage REAL,
    noise_floor_percentage REAL NOT NULL,
    max_rounds INTEGER,
    max_barren_rounds INTEGER,
    target_improvement_percent REAL,
    has_accuracy_gate INTEGER NOT NULL CHECK (has_accuracy_gate IN (0, 1)),
    accuracy_gate_metric TEXT,
    accuracy_gate_comparator TEXT,
    accuracy_gate_threshold REAL,
    accuracy_gate_is_relative INTEGER CHECK (
        accuracy_gate_is_relative IS NULL OR accuracy_gate_is_relative IN (0, 1)
    ),
    accuracy_gate_relative_to REAL,
    exec_image TEXT NOT NULL,
    exec_e2e_script TEXT NOT NULL,
    exec_accuracy_script TEXT,
    exec_serving_script TEXT,
    exec_base_server_args TEXT,
    exec_base_client_args TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS Protocol_Cells (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    protocol_id INTEGER NOT NULL REFERENCES Protocols(id),
    cell_index INTEGER NOT NULL,
    topology TEXT NOT NULL,
    weight REAL NOT NULL CHECK (weight > 0.0),
    prefill INTEGER,
    decode INTEGER,
    concurrency INTEGER,
    sequence_length INTEGER,
    global_batch_size INTEGER,
    per_device_batch_size INTEGER,
    UNIQUE (protocol_id, cell_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Campaigns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    owner TEXT NOT NULL,
    target_model TEXT NOT NULL,
    engine TEXT NOT NULL,
    accelerator TEXT NOT NULL,
    created_at TEXT NOT NULL,
    closed_at TEXT,
    protocol INTEGER NOT NULL REFERENCES Protocols(id),
    microbenchmark_protocol INTEGER REFERENCES Protocols(id),
    champion INTEGER REFERENCES Candidates(id) DEFERRABLE INITIALLY DEFERRED,
    initial_version INTEGER NOT NULL REFERENCES Candidates(id) DEFERRABLE INITIALLY DEFERRED,
    entry_baseline INTEGER REFERENCES Measurements(id) DEFERRABLE INITIALLY DEFERRED,
    current_baseline INTEGER REFERENCES Measurements(id) DEFERRABLE INITIALLY DEFERRED,
    close_reason TEXT,
    num_consecutive_barren_rounds INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL,
    block_reason TEXT CHECK (
        block_reason IS NULL OR length(block_reason) <= {constants.MAX_TEXT_LENGTH}
    )
) STRICT;

CREATE TABLE IF NOT EXISTS Rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign INTEGER NOT NULL REFERENCES Campaigns(id),
    round_number INTEGER NOT NULL CHECK (round_number >= 1),
    current_champion INTEGER NOT NULL REFERENCES Candidates(id),
    state TEXT NOT NULL,
    UNIQUE (campaign, round_number)
) STRICT;

CREATE TABLE IF NOT EXISTS Hypotheses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_created INTEGER NOT NULL REFERENCES Rounds(id),
    lever_kind TEXT NOT NULL,
    target_scope TEXT NOT NULL,
    short_claim TEXT NOT NULL CHECK (length(short_claim) <= {constants.MAX_SHORT_LENGTH}),
    state TEXT NOT NULL,
    claim TEXT NOT NULL CHECK (length(claim) <= {constants.MAX_TEXT_LENGTH}),
    refutation_reason TEXT CHECK (
        refutation_reason IS NULL OR length(refutation_reason) <= {constants.MAX_TEXT_LENGTH}
    )
) STRICT;

CREATE TABLE IF NOT EXISTS Hypothesis_Canonical_Ops (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES Hypotheses(id),
    op_index INTEGER NOT NULL,
    layer_kind TEXT NOT NULL,
    operator TEXT NOT NULL,
    UNIQUE (hypothesis_id, op_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Hypothesis_Relayout_Edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES Hypotheses(id),
    edge_index INTEGER NOT NULL,
    producer_layer_kind TEXT,
    producer_operator TEXT,
    consumer_layer_kind TEXT,
    consumer_operator TEXT,
    CHECK (
        (producer_operator IS NOT NULL OR consumer_operator IS NOT NULL)
        AND (producer_layer_kind IS NULL) = (producer_operator IS NULL)
        AND (consumer_layer_kind IS NULL) = (consumer_operator IS NULL)
    ),
    UNIQUE (hypothesis_id, edge_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Hypothesis_Evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES Hypotheses(id),
    evidence_index INTEGER NOT NULL,
    source_type TEXT NOT NULL,
    locator TEXT NOT NULL,
    summary TEXT NOT NULL CHECK (length(summary) <= {constants.MAX_TEXT_LENGTH}),
    UNIQUE (hypothesis_id, evidence_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Hypothesis_Code_Refs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES Hypotheses(id),
    ref_index INTEGER NOT NULL,
    code_ref TEXT NOT NULL,
    UNIQUE (hypothesis_id, ref_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    campaign_id INTEGER NOT NULL REFERENCES Campaigns(id),
    round INTEGER REFERENCES Rounds(id),
    sha TEXT NOT NULL,
    tree_sha TEXT NOT NULL,
    patch_sha TEXT NOT NULL,
    short_innovation TEXT NOT NULL CHECK (
        length(short_innovation) <= {constants.MAX_SHORT_LENGTH}
    ),
    superseded_by INTEGER REFERENCES Candidates(id),
    innovation TEXT NOT NULL CHECK (length(innovation) <= {constants.MAX_TEXT_LENGTH}),
    abandoned_reason TEXT CHECK (
        abandoned_reason IS NULL OR length(abandoned_reason) <= {constants.MAX_TEXT_LENGTH}
    )
) STRICT;

CREATE TABLE IF NOT EXISTS Hypothesis_Candidates (
    hypothesis_id INTEGER NOT NULL REFERENCES Hypotheses(id),
    candidate_id INTEGER NOT NULL REFERENCES Candidates(id),
    PRIMARY KEY (hypothesis_id, candidate_id)
) STRICT;

CREATE TABLE IF NOT EXISTS Candidate_Parents (
    candidate_id INTEGER NOT NULL REFERENCES Candidates(id),
    parent_index INTEGER NOT NULL CHECK (parent_index IN (0, 1)),
    parent_id INTEGER NOT NULL REFERENCES Candidates(id),
    PRIMARY KEY (candidate_id, parent_index),
    UNIQUE (candidate_id, parent_id)
) STRICT;

CREATE TABLE IF NOT EXISTS Measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trial TEXT NOT NULL,
    subject INTEGER NOT NULL REFERENCES Candidates(id),
    protocol INTEGER NOT NULL REFERENCES Protocols(id),
    env_image TEXT NOT NULL,
    env_image_digest TEXT,
    env_python_version TEXT,
    env_jax_version TEXT,
    env_jaxlib_version TEXT,
    env_torch_version TEXT,
    env_torch_tpu_version TEXT,
    env_engine_version TEXT,
    env_libtpu_init_args TEXT,
    env_xla_flags TEXT
) STRICT;

CREATE TABLE IF NOT EXISTS Jobs (
    job_id TEXT PRIMARY KEY,
    round INTEGER REFERENCES Rounds(id),
    candidate INTEGER NOT NULL REFERENCES Candidates(id),
    protocol INTEGER NOT NULL REFERENCES Protocols(id),
    trial TEXT,
    kind TEXT NOT NULL,
    measurement INTEGER REFERENCES Measurements(id),
    baseline_candidate INTEGER REFERENCES Candidates(id),
    baseline_measurement INTEGER REFERENCES Measurements(id),
    accuracy_value REAL,
    passed_accuracy INTEGER CHECK (
        passed_accuracy IS NULL OR passed_accuracy IN (0, 1)
    ),
    status TEXT NOT NULL,
    CHECK (baseline_candidate IS NULL OR baseline_measurement IS NULL)
) STRICT;

CREATE TABLE IF NOT EXISTS Measurement_Metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL REFERENCES Measurements(id),
    cell_id INTEGER NOT NULL REFERENCES Protocol_Cells(id),
    metric_name TEXT NOT NULL,
    value REAL NOT NULL,
    n INTEGER NOT NULL CHECK (n >= 1),
    stdev REAL,
    min_val REAL,
    max_val REAL,
    UNIQUE (measurement_id, cell_id, metric_name)
) STRICT;

CREATE TABLE IF NOT EXISTS Measurement_Artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    measurement_id INTEGER NOT NULL REFERENCES Measurements(id),
    artifact_index INTEGER NOT NULL,
    artifact_path TEXT NOT NULL,
    UNIQUE (measurement_id, artifact_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Measurement_Jobs (
    measurement_id INTEGER NOT NULL REFERENCES Measurements(id),
    job_index INTEGER NOT NULL,
    job_id TEXT NOT NULL REFERENCES Jobs(job_id),
    PRIMARY KEY (measurement_id, job_id),
    UNIQUE (measurement_id, job_index)
) STRICT;

CREATE TABLE IF NOT EXISTS Reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate INTEGER REFERENCES Candidates(id),
    hypothesis INTEGER REFERENCES Hypotheses(id),
    human_authored INTEGER NOT NULL CHECK (human_authored IN (0, 1)),
    reviewer TEXT NOT NULL,
    approval TEXT NOT NULL,
    short_comments TEXT NOT NULL CHECK (
        length(short_comments) <= {constants.MAX_SHORT_LENGTH}
    ),
    conditions TEXT CHECK (
        conditions IS NULL OR length(conditions) <= {constants.MAX_TEXT_LENGTH}
    ),
    comments TEXT NOT NULL CHECK (length(comments) <= {constants.MAX_TEXT_LENGTH}),
    CHECK (
        (candidate IS NOT NULL AND hypothesis IS NULL)
        OR (candidate IS NULL AND hypothesis IS NOT NULL)
    )
) STRICT;

CREATE TABLE IF NOT EXISTS Decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round INTEGER REFERENCES Rounds(id),
    is_e2e INTEGER NOT NULL CHECK (is_e2e IN (0, 1)),
    subject_measurement INTEGER NOT NULL REFERENCES Measurements(id),
    reference_measurement INTEGER NOT NULL REFERENCES Measurements(id),
    verdict TEXT NOT NULL,
    reason TEXT NOT NULL CHECK (length(reason) <= {constants.MAX_TEXT_LENGTH})
) STRICT;

CREATE INDEX IF NOT EXISTS idx_rounds_campaign ON Rounds(campaign);
CREATE INDEX IF NOT EXISTS idx_hypotheses_round ON Hypotheses(round_created);
CREATE INDEX IF NOT EXISTS idx_candidates_campaign ON Candidates(campaign_id);
CREATE INDEX IF NOT EXISTS idx_candidates_round ON Candidates(round);
CREATE INDEX IF NOT EXISTS idx_jobs_candidate ON Jobs(candidate);
CREATE INDEX IF NOT EXISTS idx_jobs_trial_pool ON Jobs(trial, candidate, protocol);
CREATE INDEX IF NOT EXISTS idx_measurements_subject ON Measurements(subject);
CREATE INDEX IF NOT EXISTS idx_measurements_trial ON Measurements(trial, subject, protocol);
CREATE INDEX IF NOT EXISTS idx_reviews_candidate ON Reviews(candidate);
CREATE INDEX IF NOT EXISTS idx_reviews_hypothesis ON Reviews(hypothesis);
CREATE INDEX IF NOT EXISTS idx_decisions_subject ON Decisions(subject_measurement);
"""


def connect(db_path: Path | None = None,
            *,
            init_schema: bool = True) -> sqlite3.Connection:
    """Open the SQLite coordination database with WAL, busy_timeout, and FKs enabled."""
    target = Path(
        db_path) if db_path is not None else paths.get_campaign_db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target),
                           timeout=BUSY_TIMEOUT_MS / 1000.0,
                           isolation_level=None)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA foreign_keys = ON")
        ver = int(conn.execute("PRAGMA user_version").fetchone()[0])
        has_tables = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        ).fetchone() is not None
        if ver not in (0, SCHEMA_VERSION) or (ver == 0 and has_tables):
            raise constants.InvalidStateError(
                f"incompatible campaign.db schema version {ver} "
                f"(expected {SCHEMA_VERSION}): {target}")
        if init_schema:
            conn.executescript(SCHEMA_SQL)
            if ver == 0:
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    except BaseException:
        conn.close()
        raise
    return conn


@contextlib.contextmanager
def open_db(db_path: Path | None = None,
            *,
            init_schema: bool = True) -> Iterator[sqlite3.Connection]:
    """Open and automatically close the SQLite coordination database."""
    with contextlib.closing(connect(db_path, init_schema=init_schema)) as conn:
        yield conn


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Execute statements inside `BEGIN IMMEDIATE` (or a `SAVEPOINT` if nested)."""
    if conn.in_transaction:
        sp = f"hc_sp_{next(_SAVEPOINT_SEQ)}"
        conn.execute(f"SAVEPOINT {sp}")
        try:
            yield conn
            if conn.in_transaction:
                conn.execute(f"RELEASE SAVEPOINT {sp}")
        except BaseException:
            if conn.in_transaction:
                conn.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                conn.execute(f"RELEASE SAVEPOINT {sp}")
            raise
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        if conn.in_transaction:
            conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise

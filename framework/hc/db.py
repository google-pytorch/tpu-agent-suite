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
"""Normalized `STRICT` SQLite coordination store connection and named-query helpers."""

from __future__ import annotations

import contextlib
import functools
import itertools
import pathlib
import re
import sqlite3
from typing import Any, Iterator

from framework.hc import constants, paths

Path = pathlib.Path

BUSY_TIMEOUT_MS: int = 30_000
SCHEMA_VERSION: int = 1
_SQL_DIR = Path(__file__).resolve().parent
_SAVEPOINT_SEQ = itertools.count(1)
_SINGLE_QUOTED_RE = re.compile(r"'(?:''|[^'])*'")
_PARAM_RE = re.compile(r"(?<![:\w]):([A-Za-z_][A-Za-z0-9_]*)")

SCHEMA_SQL: str = (_SQL_DIR / "schema.sql").read_text(encoding="utf-8")


@functools.cache
def _load_queries() -> dict[str, tuple[str, frozenset[str]]]:
    """Load and index `-- name: <query>` blocks from `queries.sql`."""
    queries: dict[str, tuple[str, frozenset[str]]] = {}
    current_name: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        if current_name is None:
            return
        sql = "\n".join(buf).strip()
        if not sql:
            raise ValueError(f"empty SQL body for query {current_name!r}")
        unquoted = _SINGLE_QUOTED_RE.sub("''", sql)
        if "?" in unquoted:
            raise ValueError(
                f"positional '?' forbidden in queries.sql ({current_name!r})")
        queries[current_name] = (sql, frozenset(_PARAM_RE.findall(unquoted)))

    for raw_line in (_SQL_DIR /
                     "queries.sql").read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("-- name:"):
            _flush()
            name = stripped[len("-- name:"):].strip()
            if not name or name in queries:
                raise ValueError(f"invalid or duplicate query name: {name!r}")
            current_name = name
            buf = []
        elif not stripped.startswith("--") and current_name is not None:
            buf.append(raw_line)
    _flush()
    return queries


def query(name: str) -> str:
    """Return the SQL string for named query `name` in `queries.sql`."""
    return _load_queries()[name][0]


def exec_query(
    conn: sqlite3.Connection,
    query_name: str,
    /,
    **params: Any,
) -> sqlite3.Cursor:
    """Execute named query `query_name` from `queries.sql` with exact `:param` bindings."""
    sql, expected = _load_queries()[query_name]
    if set(params) != expected:
        missing = sorted(expected - set(params))
        extra = sorted(set(params) - expected)
        raise ValueError(
            f"parameter mismatch for query {query_name!r}: missing={missing}, extra={extra}"
        )
    return conn.execute(sql, params)


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
        has_tables = exec_query(conn, "has_user_tables").fetchone() is not None
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

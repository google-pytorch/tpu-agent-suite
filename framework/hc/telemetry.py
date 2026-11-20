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
"""Append-only JSONL telemetry logger for `hc.py`, guarded by an OS file lock (`fcntl.flock`)."""

from __future__ import annotations

import os
import pathlib
from typing import Any, Mapping, Sequence

from framework import jsonio, utils
from framework.hc import constants, paths

Path = pathlib.Path

RESERVED_KEYS: frozenset[str] = frozenset({
    "timestamp",
    "run_id",
    "round_id",
    "event_name",
})


def _check_reserved_keys(fields: Mapping[str, Any]) -> None:
    collision = RESERVED_KEYS.intersection(fields.keys())
    if collision:
        raise ValueError(
            f"telemetry event_fields cannot overwrite reserved keys: {sorted(collision)}"
        )


def event_log_path(campaign_id: int | str | None) -> Path:
    """Return the telemetry JSONL path for `campaign_id` (or unscoped fallback)."""
    if campaign_id is not None:
        return paths.get_campaign_events_log(campaign_id)
    return paths.get_unscoped_events_log()


def append_event(
    *,
    campaign_id: int | str | None,
    run_id: str,
    round_id: int | None,
    event_name: str,
    event_fields: Mapping[str, Any] | None = None,
    timestamp: str | None = None,
) -> Path:
    """Append one flat JSONL telemetry event under `jsonio.append_jsonl`'s file lock."""
    fields = event_fields or {}
    _check_reserved_keys(fields)
    record: dict[str, Any] = {
        "timestamp": timestamp or utils.utc_now_iso(),
        "run_id": run_id,
        "round_id": round_id,
        "event_name": event_name,
        **fields,
    }
    target = event_log_path(campaign_id)
    jsonio.append_jsonl(target, record)
    return target


class TelemetryBuffer:
    """Queues events in-memory so they are written strictly after DB commits."""

    def __init__(
        self,
        *,
        run_id: str,
        campaign_id: int | None = None,
        round_id: int | None = None,
    ) -> None:
        self.run_id = run_id
        self.campaign_id = campaign_id
        self.round_id = round_id
        self._pending: list[tuple[str, dict[str, Any], int | None,
                                  int | None]] = []
        self.emitted_count: int = 0

    def record(
        self,
        event_name: str,
        *,
        round_id: int | None = None,
        campaign_id: int | None = None,
        **event_fields: Any,
    ) -> None:
        """Queue a domain lifecycle event to flush after transaction commit."""
        _check_reserved_keys(event_fields)
        self._pending.append((
            event_name,
            dict(event_fields),
            self.round_id if round_id is None else round_id,
            self.campaign_id if campaign_id is None else campaign_id,
        ))

    def discard_pending(self) -> None:
        """Drop queued events if a database transaction rolled back."""
        self._pending.clear()

    def flush(self) -> int:
        """Flush all queued events to disk under an OS file lock (`fcntl.flock`)."""
        events = list(self._pending)
        self._pending.clear()
        for event_name, fields, rid, cid in events:
            append_event(
                campaign_id=(cid if cid is not None else self.campaign_id),
                run_id=self.run_id,
                round_id=(rid if rid is not None else self.round_id),
                event_name=event_name,
                event_fields=fields,
            )
            self.emitted_count += 1
        return len(events)

    def record_rejection(
        self,
        *,
        args: Sequence[str],
        error_code: str,
        message: str,
    ) -> Path:
        """Immediately emit `cmd.rejected` on any `hc.py` nonzero exit."""
        self.discard_pending()
        path = append_event(
            campaign_id=self.campaign_id,
            run_id=self.run_id,
            round_id=self.round_id,
            event_name="cmd.rejected",
            event_fields={
                "args": list(args),
                "error_code": error_code,
                "message": message,
            },
        )
        self.emitted_count += 1
        return path

    def maybe_record_call_fallback(
        self,
        *,
        subcommands: Sequence[str],
        args: Sequence[str],
    ) -> bool:
        """When `TPU_AGENT_SUITE_HC_LOG_ALL_CALLS` is enabled and no event fired yet, log the call."""
        enabled = os.environ.get(constants.LOG_ALL_CALLS_ENV,
                                 "").strip().lower() in ("1", "true", "yes")
        if not enabled or self.emitted_count > 0 or self._pending:
            return False
        name = ".".join(s for s in subcommands if s) or "hc"
        append_event(
            campaign_id=self.campaign_id,
            run_id=self.run_id,
            round_id=self.round_id,
            event_name=name,
            event_fields={"args": list(args)},
        )
        self.emitted_count += 1
        return True

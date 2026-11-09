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
"""Reading JSON and JSONL files, and writing files atomically."""

from __future__ import annotations

import json
import os
import pathlib
import uuid
from typing import Any

from framework import errors

Path = pathlib.Path


def load_json(path: Path) -> Any:
    """Load JSON content from a file with descriptive error handling."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise errors.TpuAgentSuiteError(
            f"cannot read JSON {path}: {exc}") from exc


def load_json_or_empty(path: Path) -> dict[str, Any]:
    """Load a JSON object, yielding an empty mapping when absent or unusable.

    Unlike load_json, this never raises and always returns a mapping, so callers
    reading optional state files need no guard of their own.
    """
    if not path.is_file():
        return {}
    try:
        data = load_json(path)
    except (errors.TpuAgentSuiteError, OSError, ValueError):
        # ValueError also covers a file that is not valid UTF-8, which
        # load_json lets through.
        return {}
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path,
               limit: int = 0,
               *,
               errors: str = "strict") -> list[dict[str, Any]]:
    """Read JSONL, skipping unparseable lines rather than failing.

    A torn or hand-edited line must not blind a monitor to every event around
    it, so a bad line is dropped and the rest are returned. A missing file
    reads as empty; a positive `limit` keeps only the last `limit` records.
    `errors` is the decoding error handler, as for open(): a reader of
    files another program writes (a harness transcript) may pass "ignore".
    """
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors=errors) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out[-limit:] if limit > 0 else out


def format_json(value: Any) -> str:
    """Serialize value to canonical deterministic JSON string."""
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=3,
        separators=(",", " : "),
        sort_keys=True,
    )


def write_text_atomic(path: Path, text: str, *, mkdir: bool = True) -> None:
    """Replace a file's contents with `text` so no reader sees a partial write."""
    if mkdir:
        path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename: readers see either the old document or the new one, never
    # the truncated middle of a write. PID and UUID keep concurrent writers
    # (across processes and threads) off each other's scratch file.
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        try:
            # Carry the mode over: renaming would otherwise hand the file
            # whatever the umask gave the scratch file. Masked to permission
            # bits, since st_mode also carries the file type.
            os.chmod(tmp_path, path.stat().st_mode & 0o7777)
        except FileNotFoundError:
            pass  # First write; the umask default is correct.
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_json(path: Path, value: Any) -> None:
    """Write value to file as formatted deterministic JSON with a trailing newline."""
    write_text_atomic(path, format_json(value) + "\n")

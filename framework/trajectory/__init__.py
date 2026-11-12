#!/usr/bin/env python3
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
"""On-demand export of a session's trajectory to a bucket or Harbor hub.

Collects the entrypoint agent's transcript plus every subagent transcript
beneath it, scoped to the most recent entrypoint activation, and uploads them
as one gzipped tar alongside an uncompressed manifest, to a user-owned GCS
bucket or a Harbor hub.
"""

from __future__ import annotations

from .bundle import (ARTIFACT_BYTE_CAP, BUNDLE_NAME, MANIFEST_NAME,
                     PREFIX_ROOT, build_manifest, collect_run_files,
                     export_to_harbor, export_trajectory, harbor_missing,
                     object_prefix, resolve_for_export, slice_root_records,
                     stage_session_dir, verify_bucket_exists, write_bundle)
from .resolvers import (RESOLVERS, Cut, SessionBundle, TranscriptRef,
                        resolve_claude_code, resolve_google_harness,
                        resolve_session, supported, unsupported_message)

__all__ = [
    "ARTIFACT_BYTE_CAP",
    "BUNDLE_NAME",
    "MANIFEST_NAME",
    "PREFIX_ROOT",
    "RESOLVERS",
    "Cut",
    "SessionBundle",
    "TranscriptRef",
    "build_manifest",
    "collect_run_files",
    "object_prefix",
    "export_to_harbor",
    "harbor_missing",
    "resolve_for_export",
    "stage_session_dir",
    "export_trajectory",
    "resolve_claude_code",
    "resolve_google_harness",
    "resolve_session",
    "slice_root_records",
    "supported",
    "unsupported_message",
    "verify_bucket_exists",
    "write_bundle",
]

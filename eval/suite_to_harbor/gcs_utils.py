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
"""Utilities for fetching and caching GCS assets."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

DEFAULT_GCS_CACHE_DIR = Path.home(
) / ".cache" / "tpu-agent-suite" / "eval" / "gcs"


def get_gcs_cache_root() -> Path:
    """Returns root directory for cached GCS assets."""
    override = os.environ.get("EVAL_GCS_CACHE_DIR")
    if override and override.strip():
        return Path(override.strip()).resolve()
    return DEFAULT_GCS_CACHE_DIR.resolve()


def fetch_gcs_asset(
    gcs_uri: str,
    cache_root: Path | None = None,
) -> Path:
    """Downloads a versioned GCS asset into local cache.

    Args:
        gcs_uri: Full URI (e.g. 'gs://bucket/path/to/asset').
        cache_root: Optional directory override for storing cached downloads.

    Returns:
        Path to the local cached file or directory.

    Raises:
        ValueError: If gcs_uri does not start with 'gs://'.
        FileNotFoundError: If the asset is not cached and 'gcloud' CLI is not
            found in PATH.
        RuntimeError: If gcloud storage download fails.
    """
    clean_uri = gcs_uri.strip()
    if not clean_uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {gcs_uri}")

    root = cache_root or get_gcs_cache_root()
    rel_path = clean_uri.removeprefix("gs://").strip("/")
    cached_path = root / rel_path

    if cached_path.exists():
        logging.info("GCS asset cache hit: %s -> %s", clean_uri, cached_path)
        return cached_path

    gcloud_bin = shutil.which("gcloud")
    if not gcloud_bin:
        raise FileNotFoundError(
            f"Cannot fetch GCS asset '{clean_uri}': 'gcloud' CLI not found in PATH. "
            "Please install the Google Cloud SDK or pre-download the asset locally."
        )

    logging.info("GCS asset cache miss: Downloading %s...", clean_uri)
    cached_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_dest = cached_path.parent / f".tmp_{cached_path.name}_{os.getpid()}"
    if tmp_dest.exists():
        if tmp_dest.is_dir():
            shutil.rmtree(tmp_dest)
        else:
            tmp_dest.unlink()
    tmp_dest.mkdir(parents=True, exist_ok=True)

    cmd = [gcloud_bin, "storage", "cp", "-r", clean_uri, str(tmp_dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if tmp_dest.exists():
            shutil.rmtree(tmp_dest, ignore_errors=True)
        err_msg = e.stderr.strip() if e.stderr else str(e)
        raise RuntimeError(
            f"Failed to fetch GCS asset '{clean_uri}' via 'gcloud storage': {err_msg}"
        ) from e

    downloaded_item = tmp_dest / cached_path.name
    if not downloaded_item.exists():
        items = list(tmp_dest.iterdir())
        if len(items) == 1:
            downloaded_item = items[0]
        else:
            downloaded_item = tmp_dest

    shutil.move(str(downloaded_item), str(cached_path))
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest, ignore_errors=True)
    return cached_path

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
"""Interval arithmetic helpers for timeline (trace viewer) analysis.

All intervals are half-open `[start, end)` pairs of integers (nanoseconds).
Functions operate on numpy arrays for speed on multi-million event traces.
"""

from collections.abc import Sequence

import numpy as np


def _as_array(intervals: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(intervals, dtype=np.int64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return arr.reshape(-1, 2)


def union(intervals: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Merges overlapping/adjacent intervals. Returns sorted (N, 2) array."""
    arr = _as_array(intervals)
    if arr.shape[0] == 0:
        return arr
    arr = arr[arr[:, 1] > arr[:, 0]]
    if arr.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    arr = arr[np.argsort(arr[:, 0], kind='stable')]
    starts = arr[:, 0]
    ends = np.maximum.accumulate(arr[:, 1])
    # A new merged block starts where start > running max end of previous rows.
    new_block = np.empty(arr.shape[0], dtype=bool)
    new_block[0] = True
    new_block[1:] = starts[1:] > ends[:-1]
    block_ids = np.cumsum(new_block) - 1
    n_blocks = block_ids[-1] + 1
    out = np.empty((n_blocks, 2), dtype=np.int64)
    out[:, 0] = starts[new_block]
    out[:, 1] = np.maximum.reduceat(arr[:, 1], np.flatnonzero(new_block))
    return out


def total_length(intervals: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Total covered length of the union of intervals."""
    u = union(intervals)
    return int((u[:, 1] - u[:, 0]).sum()) if u.shape[0] else 0


def intersect(
    a: Sequence[Sequence[int]] | np.ndarray,
    b: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Intersection of two interval sets (each is unioned first)."""
    ua, ub = union(a), union(b)
    out = []
    i = j = 0
    while i < ua.shape[0] and j < ub.shape[0]:
        lo = max(ua[i, 0], ub[j, 0])
        hi = min(ua[i, 1], ub[j, 1])
        if lo < hi:
            out.append((lo, hi))
        if ua[i, 1] < ub[j, 1]:
            i += 1
        else:
            j += 1
    return _as_array(out)


def subtract(
    a: Sequence[Sequence[int]] | np.ndarray,
    b: Sequence[Sequence[int]] | np.ndarray,
) -> np.ndarray:
    """Parts of `a` not covered by `b`."""
    ua, ub = union(a), union(b)
    out = []
    j = 0
    for s, e in ua:
        cur = s
        while j < ub.shape[0] and ub[j, 1] <= cur:
            j += 1
        k = j
        while k < ub.shape[0] and ub[k, 0] < e:
            if ub[k, 0] > cur:
                out.append((cur, ub[k, 0]))
            cur = max(cur, ub[k, 1])
            if cur >= e:
                break
            k += 1
        if cur < e:
            out.append((cur, e))
    return _as_array(out)


def gaps(
    intervals: Sequence[Sequence[int]] | np.ndarray,
    window: tuple[int, int] | None = None,
) -> np.ndarray:
    """Uncovered sub-intervals between busy intervals (optionally in a window)."""
    u = union(intervals)
    if window is None:
        if u.shape[0] < 2:
            return np.zeros((0, 2), dtype=np.int64)
        window = (int(u[0, 0]), int(u[-1, 1]))
    return subtract([window], u)

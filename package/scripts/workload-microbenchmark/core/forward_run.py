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
"""Run ON THE TPU HOST, in the image under test: time and profile the twin's steps for a points
plan. No server, no scheduler.

    python3 forward_run.py --steps <plan>.steps.json [--iters 10] [--profile] \
        [--cache-in <tgz|dir>] [--save-cache] [--out <dir>]

`cli.py forward-cmd` writes a recipe that runs this beside its inputs and packs `--out`.

The plan (`cli.py points`) is self-contained: the twin (engine args incl. sharding, env,
context scale), the checkpoint config, and per point its mixed steps (each full chunk at its
prefix) and its decode step as batches of (query_len, seq_len) rows in REAL tokens.

Each batch runs through `_run_batch` on every rank via collective_rpc, with a serving step's
inputs built the way `_prepare_inputs` builds them: seeded random token ids, true positions,
distinct non-null KV pages and recurrent-state slots per request (so attention reads each
request's own pages and experts route per token); wall ms once the outputs materialize. Only the plan's buckets are
compiled. The MTP drafter runs only for plans made with `points --mtp` (then its propose replay
is timed after every step; that replay has no context, so the drafter's step-0 top-k is not in
it). Contexts are scaled by the twin's
context_scale, so KV per rank = context / real DCP.

The profile window is one mixed step + one decode step per point, in plan order, 30 ms apart
(`profile_order` in result.json).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time

T0 = time.time()
TIMES: dict[str, float] = {}


def mark(name: str, since: float) -> float:
    dt = round(time.time() - since, 2)
    TIMES[name] = dt
    print(f"[t+{time.time() - T0:7.1f}s] {name}: {dt:.1f} s", flush=True)
    return dt


#: Engine start-up lasts from the line that starts with ENGINE_START_LINE to the line
#: that contains ENGINE_READY_MARK.
ENGINE_START_LINE = "ENGINE ARGS:"
ENGINE_READY_MARK = "engine init"
#: Messages that vLLM logs (in vllm/v1/engine and vllm/v1/executor) when engine
#: start-up fails.
STARTUP_ERRORS = ("EngineCore failed to start", "WorkerProc failed to start",
                  "Worker failed with error",
                  "Engine core initialization failed")
#: Matches a Python exception or a failed native check. The first match is usually the
#: root cause; the STARTUP_ERRORS come later.
CAUSE_RE = r"\b[A-Z]\w*(?:Error|Exception): |Check failed: "
#: The exit code of `supervise` when it stops the run.
STOPPED_RC = 3


def _proc_state(pid: int) -> str | None:
    """Reads the state of a process from /proc.

    Args:
        pid: The process ID.

    Returns:
        The one-letter state from /proc/<pid>/stat, such as "R" or "Z", or None if the
        process no longer exists.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return None


def _descendants(pid: int) -> list[int]:
    """Lists the descendants of a process.

    The function reads the parent ID of every process from /proc. For forward_run.py,
    the descendants are vLLM's engine process and its worker processes.

    Args:
        pid: The ID of the top process.

    Returns:
        The IDs of the children of `pid`, of their children, and so on.
    """
    children: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                ppid = int(f.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(d))
    found, stack = [], [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            found.append(c)
            stack.append(c)
    return found


def _stop_tree(proc, wait_s: float = 10.0) -> None:
    """Stops a process and all of its descendants.

    The function sends SIGTERM, and then SIGKILL to any process that is still running
    after `wait_s` seconds. It lists the descendants before it sends any signal,
    because a process whose parent has exited can no longer be found from `proc`.

    Args:
        proc: The `subprocess.Popen` object of the top process. The process must have
            been started in a new session, so that it leads its own process group.
        wait_s: How many seconds to wait after each signal.
    """
    pids = [proc.pid] + _descendants(proc.pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            # Signal the whole process group. The top process leads the group,
            # because it was started in a new session.
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        for p in pids:
            try:
                os.kill(p, sig)
            except OSError:
                pass
        deadline = time.time() + wait_s
        while time.time() < deadline:
            proc.poll()  # Reap the top process.
            if all(_proc_state(p) in (None, "Z") for p in pids):
                return
            time.sleep(0.2)


def _exit_on_signal(signum, frame) -> None:
    """Raises SystemExit(128 + signum), so that `finally` blocks run on a signal.

    Args:
        signum: The number of the signal.
        frame: The stack frame that the signal interrupted. Unused.
    """
    del frame
    raise SystemExit(128 + signum)


def supervise(args: list[str],
              out: str,
              start_timeout_s: float,
              grace_s: float = 60.0,
              cmd: list[str] | None = None) -> int:
    """Runs forward_run.py and stops it if vLLM's engine fails but does not exit.

    The run is forward_run.py together with the vLLM processes that it starts. The
    engine is vLLM's engine process. An engine can log a fatal error during start-up
    and keep running. vLLM's own wait for the engine does not cover every stage of
    start-up, so such an engine can hold the TPUs until someone cancels the job.

    Engine start-up lasts from ENGINE_START_LINE to ENGINE_READY_MARK. During
    start-up, this function stops the run in two cases: vLLM logs one of the
    STARTUP_ERRORS and the run is still running `grace_s` seconds later, or start-up
    lasts longer than `start_timeout_s` seconds. A compile that cannot reuse cached
    results can print nothing for many minutes, so silence alone does not stop the
    run. After start-up, this function sets no time limit.

    The standard output and the standard error of the run share one pipe, and the run
    has PYTHONUNBUFFERED set, so this function reads the lines in the order in which
    they were written. It copies them to its own standard output.

    If the run ends without result.json, this function writes failure.json to `out`.
    failure.json contains these fields:

    - stage: "before engine start-up", "engine start-up" or "after engine start-up".
    - reason: "start_up_error" or "start_up_timeout" if this function stopped the
      run, and "exited" otherwise.
    - detail: the first line that matches CAUSE_RE. If no line matches, it is the
      first line that contains one of the STARTUP_ERRORS, or else the exit code. After
      a timeout, it states the time limit. When the engine becomes ready, this
      function forgets the earlier lines, because they did not stop start-up.
    - cause_context: the detail line and the four lines that follow it. The list is
      empty if no line matched, and after a timeout.
    - exit_code: the exit code that this function returns.
    - start_up_s: how long engine start-up lasted, or had lasted when the run ended.
      It is None if start-up never began.
    - start_up_limit_s: the start-up limit that this function applied.
    - log_tail: the last 60 lines of the output of the run.

    The run leads its own process group, so `compute job cancel` does not signal it.
    On SIGTERM or SIGINT, this function stops the run, writes no failure.json, and
    exits with code 128 plus the signal number.

    Args:
        args: The command-line arguments for forward_run.py.
        out: The output directory of forward_run.py, which is the value of its
            `--out` argument.
        start_timeout_s: The longest time that engine start-up may last, in seconds.
        grace_s: How long the run may take to exit by itself after vLLM logs one of
            the STARTUP_ERRORS, in seconds.
        cmd: A command to run instead of forward_run.py. Tests use it.

    Returns:
        The exit code of forward_run.py, or STOPPED_RC if this function stopped the
        run.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    lock = threading.Lock()
    tail: collections.deque[str] = collections.deque(maxlen=60)
    st: dict = {
        "phase": "before engine start-up",
        "start": None,
        "error": None,
        "error_at": None,
        "cause": None,
        "cause_context": [],
        "ready_at": None
    }

    def update_state_from_line(text: str) -> None:
        # The function looks for a cause in every phase, so that a run that fails
        # before, during or after engine start-up reports a cause. It looks for the
        # STARTUP_ERRORS at any time before the engine is ready, because they can
        # make this function stop the run.
        with lock:
            tail.append(text)
            if st["phase"] == "before engine start-up" and text.startswith(
                    ENGINE_START_LINE):
                st["phase"], st["start"] = "engine start-up", time.time()
            elif (st["phase"] == "engine start-up"
                  and f"] {ENGINE_READY_MARK} (" in text):
                # The engine became ready even though vLLM may have logged errors, so
                # those errors did not stop start-up and are unlikely to explain a
                # later failure. Forget them, so that the run is not stopped, and so
                # that a later failure reports its own cause.
                st.update(phase="after engine start-up",
                          ready_at=time.time(),
                          error=None,
                          error_at=None,
                          cause=None,
                          cause_context=[])
            else:
                if (st["phase"] != "after engine start-up"
                        and st["error"] is None
                        and any(e in text for e in STARTUP_ERRORS)):
                    st["error"], st["error_at"] = text, time.time()
                if st["cause"] is None and re.search(CAUSE_RE, text):
                    st["cause"] = text
                # Keep the cause line and the four lines that follow it, because a
                # cause can continue onto the next lines. For example, the line after
                # an XLA compile error names the operation that failed.
                if st["cause"] is not None and len(st["cause_context"]) < 5:
                    st["cause_context"].append(text)

    def copy_output(src) -> None:
        dst = sys.stdout.buffer
        for raw in iter(src.readline, b""):
            dst.write(raw)
            dst.flush()
            update_state_from_line(raw.decode(errors="replace").rstrip("\n"))

    # Install the handlers before the run starts, so that no signal can arrive
    # between the start of the run and the installation. Python can install signal
    # handlers only in the main thread.
    signals = (signal.SIGTERM, signal.SIGINT)
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {s: signal.signal(s, _exit_on_signal) for s in signals}
    proc = None
    reason = None
    try:
        proc = subprocess.Popen(
            cmd
            or [sys.executable,
                os.path.join(here, "forward_run.py"), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
            start_new_session=True)
        reader = threading.Thread(target=copy_output,
                                  args=(proc.stdout, ),
                                  daemon=True)
        reader.start()
        while True:
            # wait() returns as soon as the run exits. Otherwise, check the limits
            # every 0.1 seconds.
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                pass
            with lock:
                now = time.time()
                if (st["error_at"] is not None
                        and now - st["error_at"] > grace_s):
                    reason = "start_up_error"
                elif (st["phase"] == "engine start-up"
                      and now - st["start"] > start_timeout_s):
                    reason = "start_up_timeout"
            if reason:
                _stop_tree(proc)
                break
        proc.wait()
    finally:
        if previous:
            # Ignore further signals while the run stops, so that a second SIGTERM
            # cannot interrupt _stop_tree.
            for s in signals:
                signal.signal(s, signal.SIG_IGN)
        try:
            if proc is not None and proc.poll() is None:
                _stop_tree(proc)
        finally:
            for s, handler in previous.items():
                if handler is not None:
                    signal.signal(s, handler)
    reader.join(timeout=5)
    rc = STOPPED_RC if reason else proc.returncode
    if rc != 0 and not os.path.exists(os.path.join(out, "result.json")):
        with lock:
            start = st["start"]
            failure = {
                "stage":
                st["phase"],
                "reason":
                reason or "exited",
                # The detail is the first line that matches CAUSE_RE. If no line
                # matches, it is the first vLLM error message or the exit code.
                "detail":
                (f"no engine after {start_timeout_s:.0f} s of start-up"
                 if reason == "start_up_timeout" else st["cause"]
                 or st["error"] or f"exit code {rc}"),
                # After a timeout, a line that matched CAUSE_RE can be many minutes
                # old. Leave cause_context empty, so that `collect` prints the end
                # of log_tail, which shows where start-up stopped.
                "cause_context":
                [] if reason == "start_up_timeout" else st["cause_context"],
                "exit_code":
                rc,
                "start_up_s":
                round((st["ready_at"] or time.time()) -
                      start, 1) if start else None,
                "start_up_limit_s":
                start_timeout_s,
                "log_tail":
                list(tail)
            }
        os.makedirs(out, exist_ok=True)
        json.dump(failure,
                  open(os.path.join(out, "failure.json"), "w"),
                  indent=1)
        print(
            f"WMB FAILED during {failure['stage']}. Reason: "
            f"{failure['reason']}. Cause: {failure['detail']}",
            file=sys.stderr,
            flush=True)
    return rc


HARNESS_SALT = "dummy-kv-scale-sentinel-1"


def engine_argv(twin: dict, model_dir: str, buckets: list[int],
                profile_dir: str) -> list[str]:
    # server-only flags: no scheduler, cache manager or tokenizer in this path. The drafter is
    # built only when the plan asks for it (`points --mtp` sets twin.run_drafter).
    skip = {"async-scheduling", "enable-prefix-caching", "skip-tokenizer-init"}
    if not twin.get("run_drafter"):
        skip.add("speculative-config")
    # HARNESS_SALT: an inert config key that changes vLLM's config hash (and so the compile-cache
    # dirs and the local cache key) whenever the harness changes what gets compiled in: a graph
    # cached before such a change would carry stale constants. Bump it with any such change.
    hf = {**twin["hf_overrides"], "wmb_harness": HARNESS_SALT}
    argv = [
        model_dir, "--hf-overrides",
        json.dumps(hf), "--skip-tokenizer-init", "--compilation-config",
        json.dumps({"compile_sizes": sorted(buckets)})
    ]
    for k, v in twin["server_args"].items():
        if k in skip:
            continue
        argv += [f"--{k}"] if v is True else [f"--{k}", str(v)]
    # Always pass the profiler flags with a FIXED dir: vLLM hashes the engine config into the
    # compile-cache key, so a job-specific dir (or flags present only when profiling) makes
    # every job miss the cache. Profiling happens only between start_profile/stop_profile.
    return argv + [
        "--profiler-config.profiler=torch",
        f"--profiler-config.torch_profiler_dir={profile_dir}",
        "--profiler-config.torch_profiler_with_stack=false"
    ]


def _page_plan(rows: list,
               groups: list[dict],
               num_blocks: int,
               dcp: int = 1,
               alloc_block: int | None = None) -> dict:
    """Distinct, non-overlapping, non-null KV pages for every request of a batch, per KV-cache
    group, as the scheduler's block pool hands them out. Pure: no torch, unit-tested.

    KV blocks count up from 1 and recurrent-state blocks down from the top (see below). The
    pool hands out blocks of `alloc_block` tokens (the cache block size, e.g. 1152 on the
    hybrid Qwen3.5) from ONE id space shared by every group (block 0 is the null block), so no
    block is shared across groups or requests. A group whose table counts smaller kernel pages
    (`block_size`, e.g. 128 for attention) stores each block b as the `ratio = alloc_block /
    block_size` kernel ids b x ratio .. b x ratio + ratio - 1 (vLLM BlockTable's kernel-block
    mapping), so its ids never land inside another group's block.

    rows    [[query_len, seq_len], ...]
    groups  [{"gid", "block_size", "is_mamba", "width", "ckpt_window"}]: `width` is the row width
            the builder slices (max_num_blocks_per_req when use_max_model_len)
    An attention row holds cdiv(seq_len, block_size x dcp) kernel pages (vLLM's DCP virtual block
    is block_size x dcp tokens). A mamba row holds only its state entry at (seq_len - 1) //
    block_size and the ckpt_window - 1 checkpoint entries after it (align mode: earlier entries
    are null), which is what the unified-pool builder gathers; the compact pool ignores the table.
    Returns {"tables": {gid: [row, ...]} (each row `width` long, 0 = null), "used": blocks}."""
    nxt, top, tables = 1, num_blocks - 1, {}
    for g in groups:
        bs, w = int(g["block_size"]), int(g["width"])
        ab = int(alloc_block or bs)
        ratio = ab // bs if ab >= bs and ab % bs == 0 else 1
        out = []
        for _, s in rows:
            row = [0] * w
            if g.get("is_mamba"):
                so = min(max(int(s) - 1, 0) // bs, w - 1)
                idx = list(
                    range(so,
                          min(w, so + max(1, int(g.get("ckpt_window") or 1)))))
            else:
                n = -(-int(s) // (bs * max(1, dcp)))
                if n > w:
                    raise ValueError(
                        f"seq_len {s} needs {n} pages of {bs} x dcp {dcp}; "
                        f"the row holds {w}")
                idx = list(range(n))
            # consecutive entries share a pool block, `ratio` kernel pages per block. KV blocks
            # count up from 1, state blocks down from the top of the pool: every batch of a plan
            # then keeps its KV pages out of every other batch's state blocks (a GDN state write
            # read back as fp8 KV is garbage, often NaN)
            for j0 in range(0, len(idx), ratio):
                if g.get("is_mamba"):
                    b, top = top, top - 1
                else:
                    b, nxt = nxt, nxt + 1
                for t, j in enumerate(idx[j0:j0 + ratio]):
                    row[j] = b * ratio + t
            out.append(row)
        tables[g["gid"]] = out
    if nxt > top + 1:
        raise ValueError(
            f"batch needs {nxt - 1 + num_blocks - 1 - top} KV blocks, the pool has "
            f"{num_blocks - 1} usable")
    return {"tables": tables, "used": nxt - 1 + num_blocks - 1 - top}


def _positions(rows: list, num_tokens: int) -> list[int]:
    """True token positions of a padded batch: a row with query len q at seq len s holds tokens
    s-q .. s-1; padding rows past the last request are 0 (as `_prepare_inputs` pads)."""
    pos = []
    for q, s in rows:
        pos += list(range(int(s) - int(q), int(s)))
    return pos + [0] * (num_tokens - len(pos))


def _mamba_slots(n_rows: int, stride: int) -> list[int]:
    """Compact-pool recurrent-state slots, as `_init_mamba_slot_pool` hands them out: slot 0 is
    null; request i owns the group of `stride` (= 1 + spec tokens) slots starting at 1 + i x stride."""
    return [1 + i * max(1, stride) for i in range(n_rows)]


#: Profile markers around the sampling stage: a tiny device program whose output is s32[N], N
#: one of these, run (and synchronized) right before and right after the stage. `xprof_steps`
#: finds the modules between them on `XLA Modules` and books them as the step's sampling, apart
#: from the backbone and the drafter. Primes, so no model tensor has this shape by accident.
#: torch-tpu may run both through one program (the trace then shows s32[7919] twice), so the
#: reader takes them as a toggle (first opens, second closes), not by size.
SAMPLING_MARKS = (7919, 7907
                  )  # (start, end); xprof_steps.SAMPLING_MARKS must match


def _num_draft_tokens(rows: list, verify: int) -> list[int]:
    """Per request, the draft tokens its row carries: a verify window of 2..verify tokens (the
    rows `_run_batch` counts as MTP windows) holds q - 1 drafts; a 1-token decode or a prefill /
    mixed chunk holds none. Pure, unit-tested."""
    return [
        int(q) - 1 if verify > 1 and 1 < int(q) <= verify else 0
        for q, _ in rows
    ]


def _sampling_layout(rows: list,
                     verify: int,
                     token_paddings: list[int],
                     max_num_reqs: int,
                     min_reqs: int = 8) -> dict:
    """The shapes the runner's sampling stage sees for this batch, as `_prepare_inputs` /
    `get_spec_decode_metadata` pad them (pure; the pod uses the runner's own functions, this is
    the record and the unit-tested expectation):
    - spec verify (some row carries drafts): logits rows = sum(drafts + 1) padded to the token
      bucket (`padded_logits_length`); bonus rows = padded num reqs; target rows = the same
      padded_logits_length;
    - otherwise: one logits row per request, padded to padded num reqs."""
    nd = _num_draft_tokens(rows, verify)
    n = len(rows)
    preqs = min(min_reqs if n <= min_reqs else 1 << (n - 1).bit_length(),
                max_num_reqs)
    if any(nd):
        rows_needed = sum(d + 1 for d in nd)
        plen = next((p for p in sorted(token_paddings) if p >= rows_needed),
                    None)
        return {
            "path": "spec_verify",
            "num_draft_tokens": nd,
            "logits_rows": plen,
            "bonus_rows": preqs,
            "target_rows": plen,
            "padded_num_reqs": preqs
        }
    return {
        "path": "sample_from_logits",
        "num_draft_tokens": nd,
        "logits_rows": preqs,
        "bonus_rows": None,
        "target_rows": None,
        "padded_num_reqs": preqs
    }


def _mark(r, code: int) -> None:
    """A tiny device program with output s32[code] (SAMPLING_MARKS), synchronized so it is its
    own module on `XLA Modules`. Its input lives on device (built once per code), so XLA cannot
    fold it to a constant."""
    import torch
    from vllm_torchtpu.runner import tpu_runner as tr
    base = r.__dict__.setdefault("_wmb_mark_base", {})
    if code not in base:
        base[code] = torch.arange(code, dtype=torch.int32).to(r.device)
        tr.synchronize_tensors([base[code]])
    tr.synchronize_tensors([base[code] * 3 + 1])


def _prepare_sampling(r, rows: list, ids, verify: int, sampling: dict) -> dict:
    """Executed on a rank, once per batch and outside any timed or profiled region: the state the
    runner's sampling stage reads, built with the runner's own code.
    - sampling params: the source job's (temperature, top_k, top_p) in the persistent input
      batch rows 0..B-1, where `_build_padded_sampling_params` reads them;
    - spec verify metadata (MTP): `spec_decode_manager.get_spec_decode_metadata(num_draft_tokens,
      cu_num_scheduled_tokens, padded_num_reqs)`, the call `_prepare_inputs` makes, with the
      batch's token ids in `input_ids_cpu` (the draft ids it gathers). Its H2D copies happen here,
      as in serving they happen in `_prepare_inputs`, not in the sample stage."""
    import numpy as np
    import torch
    from vllm_torchtpu.runner import tpu_runner as tr
    q = [int(a) for a, _ in rows]
    n, real = len(q), sum(q)
    ib = r.input_batch
    ib.temperature_cpu_tensor[:n] = float(sampling["temperature"])
    # as vLLM's InputBatch stores it: a disabled top-k (0 / -1) is the vocab size
    k = int(sampling.get("top_k") or 0)
    ib.top_k_cpu_tensor[:n] = k if 0 < k < int(r.vocab_size) else int(
        r.vocab_size)
    ib.top_p_cpu_tensor[:n] = float(
        sampling["top_p"] if sampling.get("top_p") is not None else 1.0)
    preqs = tr._get_padded_num_reqs_with_upper_limit(n, int(r.max_num_reqs))
    nd = np.array(_num_draft_tokens(rows, verify), dtype=np.int32)
    rec = {
        "temperature": float(sampling["temperature"]),
        "top_k": int(sampling["top_k"] or 0),
        "top_p": float(sampling.get("top_p") or 1.0),
        "padded_num_reqs": int(preqs),
        "num_draft_tokens": nd.tolist()
    }
    md = None
    cu = np.cumsum(np.array(q, dtype=np.int32)).astype(np.int32)
    if nd.any() and getattr(r, "rejection_sampler", None) is not None:
        r.input_ids_cpu[:real] = ids[:real].to(r.input_ids_cpu.dtype)
        md = r.spec_decode_manager.get_spec_decode_metadata(nd, cu, preqs)
        idx = md.final_logits_indices
        rec.update(
            path=
            "spec verify: compute_selected_logits -> _sample_spec_verify_chunk "
            "(spec_gather_bonus_and_target_logits, sample_from_logits on the bonus "
            "rows, rejection_sampler do_sampling=True)",
            logits_rows=int(idx.shape[0]),
            bonus_rows=int(md.bonus_logits_indices.shape[0]),
            target_rows=int(md.target_logits_indices.shape[0]))
    else:
        li = np.full((preqs, ), real - 1, dtype=np.int32)
        li[:n] = cu - 1
        idx = torch.from_numpy(li).to(r.device)
        rec.update(
            path="compute_selected_logits -> _build_padded_sampling_params -> "
            "sample_from_logits",
            logits_rows=int(preqs))
    # what this image's _build_padded_sampling_params returns: 3 tensors (older images) or 3 +
    # the static top-k bound (the bounded top-k/top-p path); recorded once, on dummy logits
    probe = r._build_padded_sampling_params(
        0, n, torch.zeros((preqs, 1), dtype=torch.float32))
    rec["image_returns_k_bound"] = len(probe) == 4
    rec["k_bound"] = int(probe[3]) if len(probe) == 4 else None
    return {"md": md, "idx": idx, "rec": rec, "n": n}


def _sample_stage(r, hidden, plan, st: dict):
    """Executed on a rank: one serving step's sample stage on `hidden` (the backbone output),
    through the runner's own compiled functions, the calls `execute_model` + `sample_tokens` make:
    the target LM head on the sampled rows (`maybe_select_logits_hidden_states` /
    `compute_selected_logits`), then for an MTP verify step `_sample_spec_verify_chunk` (bonus
    token: `_build_padded_sampling_params` + Gumbel `sample_from_logits`; drafts: the rejection
    sampler), else `_build_padded_sampling_params` + `sample_from_logits`. Not included: logprobs
    (the job requested none), structured output, the mamba read-offset update after the verify.
    Returns the sampled token tensor (not synchronized)."""
    import torch
    from vllm_torchtpu.runner import tpu_runner as tr
    planner = tr._get_sequence_layout_planner_for_runner(r)
    sel = planner.maybe_select_logits_hidden_states(hidden, plan, st["idx"])
    if sel is not None:
        logits = r.compute_logits_from_hidden_states(sel)
    else:
        logits = r.compute_selected_logits(
            planner.finalize_hidden_states(hidden, plan), st["idx"])
    gen = r._get_sampling_generator()
    if st["md"] is not None:
        return r._sample_spec_verify_chunk(logits, st["md"], None, None, 0,
                                           st["n"], False, gen)
    p = r._build_padded_sampling_params(0, st["n"], logits)
    kw = {"k_bound": p[3]} if len(p) == 4 else {}
    u = torch.rand_like(logits, generator=gen)
    return r.sample_from_logits_func(logits,
                                     p[0],
                                     u,
                                     p[1],
                                     p[2],
                                     all_greedy=False,
                                     **kw)


def _last_sampling(worker):
    """Executed on every rank: the sample stage's wall ms per iteration of the last batch."""
    return getattr(worker.model_runner, "_wmb_sample_ms", None)


def _run_batch(worker,
               rows: list,
               num_tokens: int,
               iters: int,
               verify: int = 1,
               sampling: dict | None = None,
               mark: bool = False):
    """Executed on every rank. A serving step's inputs for an arbitrary batch, built the way
    `_prepare_inputs` builds them (the image's own copies are saved with every run: result.json
    `image_source`), then `forward_model` timed `iters` times. rows = [[query_len, seq_len], ...],
    padded to the compiled bucket `num_tokens` (tokens past the last request belong to none).

    Real, distinct inputs per request (all built once, outside the timed loop):
    - token ids: seeded random over the vocabulary (identical on every rank), padding 0;
    - positions: s-q .. s-1 per row (`_positions`); mRoPE models get all 3 streams = the text
      position, as `_calc_mrope_positions` does for text;
    - KV pages: distinct, non-null blocks per request and group (`_page_plan`), written into the
      runner's persistent block table and staged by the runner's own `stage_block_table_uploads`
      (position_ids_override stays None: that override keeps the all-zero dummy table). KV writes
      derive their slots from this table, so slot mappings are consistent with it;
    - recurrent state (linear attention / mamba): distinct non-null compact-pool slots
      (`_mamba_slots`); on a unified pool the builder derives them from the mamba rows.

    request_distribution = [decode_end, prefill_end, mixed_end], split as `_prepare_inputs` splits
    a batch ordered [1-token decodes][spec verify windows][prefill/mixed]: `verify` = 1 + k is the
    width of an MTP verify window (rows of 2..verify tokens after the 1-token rows). Ragged paged
    attention counts the windows as decode when the runner's `reorder_batch_threshold` > 1; the
    GDN (mamba) distribution always counts decode + windows. Returns wall ms per iteration, after
    the outputs materialize; what the inputs were is kept for `_last_inputs`. PP stages,
    multimodal inputs and LoRA are not carried: asserted absent.

    `sampling` (the plan's `sampling` block, when it runs): after each forward, the serving
    step's sample stage on its output (`_sample_stage`: LM head, bonus-token sampling, rejection
    sampler), timed apart from the forward (`_last_sampling`). `mark` (the profile window) wraps
    the stage in SAMPLING_MARKS so the profile can book it separately."""
    import time as _t

    import torch
    from vllm.forward_context import set_forward_context
    from vllm_torchtpu.runner import tpu_runner as tr

    r = worker.model_runner
    assert getattr(r, "_pp_is_first", True) and not getattr(r, "supports_mm_inputs", False) \
        and r.lora_config is None, "twin batch runner: PP / multimodal / LoRA not supported"
    num_reqs = r.num_reqs_max_model_len
    q = [int(a) for a, _ in rows]
    assert len(q) <= num_reqs, f"{len(q)} requests > batch capacity {num_reqs}"
    real = sum(q)
    assert real <= num_tokens, f"{real} tokens > bucket {num_tokens}"
    n_dec = 0
    while n_dec < len(q) and q[n_dec] == 1:
        n_dec += 1
    n_win = n_dec
    while verify > 1 and n_win < len(q) and 1 < q[n_win] <= verify:
        n_win += 1
    rpa_dec = n_win if getattr(r, "reorder_batch_threshold", 1) > 1 else n_dec

    gen = torch.Generator().manual_seed(1234 + 7919 * len(q) + real)
    ids = torch.zeros((num_tokens, ), dtype=torch.int32)
    ids[:real] = torch.randint(0,
                               int(r.vocab_size), (real, ),
                               generator=gen,
                               dtype=torch.int32)
    input_ids = ids.to(r.device)
    pos = torch.tensor(_positions(rows, num_tokens), dtype=torch.int32)
    uses_mrope = getattr(r, "uses_mrope", False)
    position_ids = (pos.unsqueeze(0).expand(3, num_tokens).contiguous()
                    if uses_mrope else pos).to(r.device)

    # KV pages: distinct blocks in the runner's persistent block table, one table per group
    groups, seen = [], set()
    for group_list in r.attn_groups:
        for group in group_list:
            bld = group.metadata_builders[0]
            gid = bld.kv_cache_group_id
            if gid in seen:
                continue
            seen.add(gid)
            bt = r.input_batch.block_table[gid]
            groups.append({
                "gid":
                gid,
                "block_size":
                int(
                    getattr(bld, "target_block_size", 0)
                    or bld.kv_cache_spec.block_size),
                "is_mamba":
                bool(getattr(bld, "is_mamba_group", False)),
                "width":
                int(bt.max_num_blocks_per_req),
                "ckpt_window":
                int(getattr(r, "_mamba_ckpt_window", 1) or 1)
            })
    dcp = int(
        getattr(r.parallel_config, "decode_context_parallel_size", 1) or 1)
    alloc = int(getattr(r.cache_config, "block_size", 0)
                or 0) or None  # the pool's block
    pp = _page_plan(rows, groups, int(r.kv_cache_config.num_blocks), dcp,
                    alloc)
    for g in groups:
        cpu = r.input_batch.block_table[g["gid"]].get_cpu_tensor()
        cpu[:len(q)] = 0
        cpu[:len(q), :g["width"]] = torch.tensor(pp["tables"][g["gid"]],
                                                 dtype=cpu.dtype)
    r.position_ids = position_ids  # what the builder reads with no override
    r._wmb_attn_pages = sorted({
        x
        for g in groups if not g["is_mamba"] for row in pp["tables"][g["gid"]]
        for x in row if x
    })
    # the context a served request reads was written by its own prefill; a never-written page
    # holds whatever the pool's memory held (j-41c5ff8e: non-finite fp8 bytes), and attention over
    # it makes the forward NaN and collapses MoE routing. Fill the batch's KV pages with finite
    # values once per batch shape, before anything is timed or profiled (the fill's eager ops
    # would otherwise sit between profiled steps).
    filled = r.__dict__.setdefault("_wmb_filled", {})
    key = tuple(tuple(int(v) for v in x) for x in rows)
    if key not in filled:
        filled[key] = _fill_pages(r, r._wmb_attn_pages)

    query_lens = q + [0] * (num_reqs - len(q))
    query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                dtype=torch.int32),
                                   dim=0,
                                   dtype=torch.int32).to(r.device)
    seq_lens_cpu = torch.zeros(
        (num_reqs, ), dtype=torch.int32)  # padded requests: 0, as served
    for i, (_, sl) in enumerate(rows):
        seq_lens_cpu[i] = int(sl)
    seq_lens = seq_lens_cpu.to(r.device)
    dist = [rpa_dec, rpa_dec, len(q)]
    request_distribution = torch.tensor(dist, dtype=torch.int32).to(r.device)
    slots = None
    mamba_idx = None
    if getattr(r, "_has_mamba_state", False):
        slots = _mamba_slots(len(q),
                             int(getattr(r, "_mamba_slot_stride", 1) or 1))
        mi = torch.zeros((num_reqs, ), dtype=torch.int32)
        mi[:len(q)] = torch.tensor(slots, dtype=torch.int32)
        mamba_idx = mi.to(r.device)
    mamba_dist = None
    if getattr(r, "mamba_slot_read_offsets", None) is not None:
        combined = torch.tensor(dist + [n_win, n_win, len(q)],
                                dtype=torch.int32).to(r.device)
        request_distribution, mamba_dist = combined[0:3], combined[3:6]
    plan = tr._get_sequence_layout_planner_for_runner(r).prepare_dummy(
        num_tokens=num_tokens, num_reqs=num_reqs, kv_cache_initialized=True)
    r._attn_metadata_builder_ctx = tr.AttentionMetadataBuilderContext(
        num_reqs=len(q),
        start_index=0,
        use_max_model_len=True,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        request_distribution=request_distribution,
        position_ids_override=None,
        mamba_state_indices=mamba_idx,
        mamba_slot_read_offsets=getattr(r, "mamba_slot_read_offsets", None),
        mamba_ckpt_window=getattr(r, "_mamba_ckpt_window", 1),
        mamba_request_distribution=mamba_dist,
        sequence_layout_descriptor=plan.descriptor)
    per_layer, _ = r._build_attention_metadata(
        num_tokens=real,
        num_reqs=num_reqs,
        max_query_len=max(q),
        num_tokens_padded=num_tokens,
        num_reqs_padded=num_reqs,
        slot_mappings=r.empty_slot_mappings)
    ps = getattr(r, "_token_padding_state", None)
    if ps is not None:
        ps.update(real, num_tokens)
    st = _prepare_sampling(r, rows, ids, verify,
                           sampling) if sampling else None
    out_ms, samp_ms = [], []
    with set_forward_context(per_layer, r.vllm_config, num_tokens=0, num_tokens_across_dp=None), \
            tr.set_vllm_model_wrapper_context(mesh=r.mesh, vllm_config=r.vllm_config,
                                              kv_cache_bundle=r._kv_cache_bundle):
        tr.synchronize_tensors([input_ids])
        for _ in range(iters):
            t0 = _t.perf_counter()
            out, _ = r.forward_model(input_ids=input_ids,
                                     positions=position_ids,
                                     inputs_embeds=None,
                                     intermediate_tensors=None)
            tr.synchronize_tensors(out)
            out_ms.append((_t.perf_counter() - t0) * 1000.0)
            if st is not None:
                if mark:
                    _mark(r, SAMPLING_MARKS[0])
                t1 = _t.perf_counter()
                tok = _sample_stage(r, out, plan, st)
                tr.synchronize_tensors(tok)
                samp_ms.append((_t.perf_counter() - t1) * 1000.0)
                if mark:
                    _mark(r, SAMPLING_MARKS[1])
    r._wmb_sample_ms = samp_ms if st is not None else None
    r._hidden_states_dtype = out.dtype  # read by the drafter's dummy draft
    try:  # after the timed loop: a NaN would collapse routing
        finite = bool(torch.isfinite(out.float()).all().item())
    except Exception as e:  # noqa: BLE001
        finite = repr(e)[:80]
    r._wmb_inputs = {
        "path":
        "persistent block table + stage_block_table_uploads; position_ids_override=None",
        "ids_unique":
        int(torch.unique(ids[:real]).numel()),
        "ids_real":
        real,
        "positions": [[int(s) - int(a), int(s) - 1] for a, s in rows],
        "mrope_streams":
        3 if uses_mrope else 0,
        "pages": {
            str(g["gid"]): {
                "block_size":
                g["block_size"],
                "mamba":
                g["is_mamba"],
                "width":
                g["width"],
                "per_row": [[
                    min(x for x in row if x),
                    max(x for x in row if x),
                    sum(1 for x in row if x)
                ] if any(row) else [] for row in pp["tables"][g["gid"]]]
            }
            for g in groups
        },
        "blocks_used":
        pp["used"],
        "pool_blocks":
        int(r.kv_cache_config.num_blocks),
        "alloc_block":
        alloc,
        "dcp":
        dcp,
        "mamba_slots":
        slots,
        "kv_fill":
        filled.get(key),
        "mamba_path": ("compact slots"
                       if slots is not None else "unified: from the mamba rows"
                       if getattr(r, "_unified_kv_layout", False) else "none"),
        "out_finite":
        finite,
        "sampling":
        st["rec"] if st is not None else None
    }
    return out_ms


def _fill_pages(r, pages: list[int], chunk: int = 64) -> dict:
    """Executed on a rank: write finite, seeded random values into kernel pages `pages` of every
    KV-cache tensor large enough to hold them (all are views of the one pool; the pages belong to
    the attention groups by construction of `_page_plan`). Returns what was written, for
    result.json.

    Every page gets its own values, and the seed differs per global rank. Integer caches are filled
    too: fp8 KV caches are stored as u8 (indexer K) or u32 (SparseCore MLA layout), and an unfilled
    or repeated cache gives tied indexer scores, so the top-k breaks ties by index and one DCP rank
    owns nearly every slot (a twin-only skew; served CUMF took its overflow fallback 0/1458 times).
    Integer caches get random bytes with the e4m3fn NaN patterns (0x7F, 0xFF) replaced."""
    import torch
    if not pages:
        return {"pages": 0}
    try:
        from vllm.distributed import get_world_group
        rank = int(get_world_group().rank)
    except Exception:  # noqa: BLE001
        rank = 0
    caches = r.kv_caches if isinstance(r.kv_caches,
                                       (list,
                                        tuple)) else list(r.kv_caches.values())
    done, kinds = [], {}
    try:
        for i, t in enumerate(caches):
            if not torch.is_tensor(t) or t.dim() < 2 or t.shape[0] <= max(
                    pages):
                continue
            g = torch.Generator().manual_seed(97 + 1009 * rank + 31 * i)
            row = tuple(t.shape[1:])
            for c in range(0, len(pages), chunk):
                ids = pages[c:c + chunk]
                where = torch.tensor(ids, dtype=torch.long).to(t.device)
                if torch.is_floating_point(t):
                    src = (torch.randn(
                        (len(ids), ) + row, generator=g) * 0.5).to(t.dtype)
                    if t.element_size() == 1:  # fp8: write as bytes
                        t.view(torch.uint8)[where] = src.view(torch.uint8).to(
                            t.device)
                    else:
                        t[where] = src.to(t.device)
                else:
                    nbytes = t.element_size()
                    b = torch.randint(0,
                                      256, (len(ids), ) + row[:-1] +
                                      (row[-1] * nbytes, ),
                                      generator=g,
                                      dtype=torch.uint8)
                    b[(b == 0x7F) | (b == 0xFF)] = 0x7E
                    t.view(torch.uint8)[where] = b.to(t.device)
            done.append(i)
            kinds[i] = str(t.dtype)
        return {
            "pages": len(pages),
            "tensors": done,
            "dtypes": kinds,
            "rank_seed": rank
        }
    except Exception as e:  # noqa: BLE001
        return {"pages": len(pages), "tensors": done, "error": repr(e)[:200]}


def _last_inputs(worker) -> dict:
    """Executed on every rank: what the last `_run_batch` fed the model (ids, positions, pages,
    state slots, whether its output was finite)."""
    return getattr(worker.model_runner, "_wmb_inputs", None)


def _run_draft(worker, bucket: int, iters: int, spec: dict):
    """Executed on every rank: the drafter's propose trace for one step at `bucket` tokens, via
    the replay the model names (`spec` = model.py DRAFTER: `replay` method, and the `gate`
    method that limits it to DP-lockstep runs, lifted on this worker only). Whatever that replay
    leaves out (e.g. context in its attention metadata) is not included. Returns wall ms per
    iteration."""
    import time as _t

    from vllm_torchtpu.utils import synchronize_device
    r = worker.model_runner
    d = r.drafter
    if d is None:
        raise RuntimeError(
            "no drafter: the plan was not made with `points --mtp`")
    if spec.get("gate"):
        setattr(d, spec["gate"], lambda: True)
    r._dp_target_bucket = bucket
    replay = getattr(d, spec["replay"])
    # non-zero hidden-state inputs for the replay (it builds them with torch.zeros): see
    # _NonZeroHidden. Built inside the timed call, as the image builds them; host-side only.
    import sys as _s

    import torch as _torch
    mod = _s.modules[type(d).__module__]
    hs = getattr(d, "_draft_hidden_size", None)
    if not hasattr(r, "_wmb_rand_cache"):
        r._wmb_rand_cache = {}
    proxy = _NonZeroHidden(_torch, int(hs()), r._wmb_rand_cache) if callable(hs) and getattr(mod, "torch", None) \
        is _torch else None
    out = []
    try:
        if proxy is not None:
            mod.torch = proxy
        for _ in range(iters):
            synchronize_device()
            t0 = _t.perf_counter()
            replay(1)
            synchronize_device()
            out.append((_t.perf_counter() - t0) * 1000.0)
    finally:
        if proxy is not None:
            mod.torch = _torch
    r._wmb_draft_inputs = {
        "nonzero_hidden_tensors": proxy.replaced if proxy else 0
    }
    return out


def _diagnose(worker) -> dict:
    """Executed on every rank, after the plan: what could make the forward non-finite or wildly
    scaled. Every attention layer's compiled-in fp8 KV scales, a scan
    of the model's floating parameters and buffers (non-finite, largest and smallest |x|; tensors
    over 16M elements are sampled), and whether the KV pages the last batch wrote are finite."""
    import torch
    r = worker.model_runner
    out: dict = {}
    d = getattr(r, "drafter", None)
    models = [("target", r.model)]
    dm = getattr(d, "model", None) or getattr(d, "draft_model", None)
    if dm is not None and dm is not r.model:
        models.append(("drafter", dm))
    scales, stats, seen = [], [], set()
    for tag, m in models:
        for name, mod in m.named_modules():
            if hasattr(mod, "_k_scale_float"):
                scales.append({
                    "layer": f"{tag}:{name}",
                    **{
                        k: getattr(mod, k, None)
                        for k in ("_k_scale_float", "_v_scale_float", "_q_scale_float")
                    }
                })
        for name, t in list(m.named_parameters()) + list(m.named_buffers()):
            if id(t) in seen or not torch.is_floating_point(t) or t.numel(
            ) == 0:
                continue
            seen.add(id(t))
            try:
                x = t.detach().reshape(-1)
                if x.numel() > (1 << 24):
                    x = x[:1 << 20]
                x = x.float()
                fin = bool(torch.isfinite(x).all().item())
                ax = x.abs()
                stats.append(
                    (f"{tag}:{name}", str(t.dtype).replace("torch.", ""), fin,
                     float(ax.max().item()), float(ax.mean().item()),
                     float(x.min().item())))
            except Exception as e:  # noqa: BLE001
                stats.append((f"{tag}:{name}", str(t.dtype), None,
                              repr(e)[:60], 0.0, 0.0))
    num = [x for x in stats if isinstance(x[3], float)]
    out["attention_scales"] = scales
    out["params_scanned"] = len(stats)
    out["params_nonfinite"] = [x[:2] for x in stats if x[2] is False]
    out["params_largest"] = sorted(num, key=lambda x: -x[3])[:8]
    out["scale_params"] = [x for x in num if "scale" in x[0].lower()][:24]
    out["norm_and_gdn_params"] = [
        x for x in num if any(k in x[0] for k in ("norm", "A_log", "dt_bias"))
    ][:24]
    pages = getattr(r, "_wmb_attn_pages", None)
    kv = []
    try:
        caches = r.kv_caches if isinstance(
            r.kv_caches, (list, tuple)) else list(r.kv_caches.values())
        for i, t in enumerate(caches[:8]):
            if pages and t.dim() >= 1 and t.shape[0] > max(pages):
                sub = t[torch.tensor(pages[:64],
                                     dtype=torch.long).to(t.device)].float()
                kv.append({
                    "cache": i,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype),
                    "finite": bool(torch.isfinite(sub).all().item()),
                    "absmax": float(sub.abs().max().item())
                })
    except Exception as e:  # noqa: BLE001
        kv.append({"error": repr(e)[:160]})
    out["kv_pages_written"] = kv
    return out


class _NonZeroHidden:
    """Stand-in for the `torch` module inside the drafter's module while its replay runs: a
    floating `torch.zeros` shaped like a hidden state (last dim = the draft hidden size) comes back
    as seeded N(0, 1) instead, so the replayed drafter's tokens differ and its MoE routes per token
    as a real propose does; everything else is torch itself. The replay's code is the image's."""

    def __init__(self, torch_mod, hidden: int, cache: dict, seed: int = 4321):
        self._t, self._h, self._c = torch_mod, hidden, cache
        self._g = torch_mod.Generator().manual_seed(seed)
        self.replaced = 0

    def __getattr__(self, name):
        return getattr(self._t, name)

    def zeros(self, *shape, dtype=None, device=None, **kw):
        t = self._t
        dims = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, t.Size)) \
            else tuple(shape)
        if dtype is not None and dtype.is_floating_point and dims and dims[
                -1] == self._h:
            self.replaced += 1
            # built once per shape and kept on device (cache on the runner): no host draw or
            # H2D copy inside a timed replay after the warm-up
            key = (dims, str(dtype), str(device))
            if key not in self._c:
                self._c[key] = t.randn(dims,
                                       generator=self._g).to(dtype).to(device)
            return self._c[key]
        return t.zeros(*shape, dtype=dtype, device=device, **kw)


def _key_factors(worker) -> dict:
    """Executed on every rank after boot: the inputs of vLLM's torch.compile cache dir
    (backends.py: sha256([env_hash, config_hash, code_hash, compiler_hash])[:10]), except
    code_hash (traced files are consumed at compile). Two boots whose dirs differ while these
    agree differ in code_hash. `tpu_env` is the TPU compiler's env factor dict, for diffing."""
    import hashlib

    from vllm import envs
    from vllm.compilation.backends import hash_factors
    from vllm_torchtpu.compilation import tpu_compiler as tc
    cfg = worker.model_runner.vllm_config
    env = tc._tpu_compile_env_factors()
    return {
        "env_hash":
        hash_factors(envs.compile_factors()),
        "config_hash":
        cfg.compute_hash(),
        "compiler_hash":
        tc.compute_tpu_compilation_hash(cfg),
        "cache_dir":
        cfg.compilation_config.cache_dir,
        "num_gpu_blocks":
        cfg.cache_config.num_gpu_blocks,
        "tpu_env_hash":
        hashlib.sha256(repr(sorted(env.items())).encode()).hexdigest()[:10],
        "tpu_env": {
            k: str(v)[:120]
            for k, v in env.items()
        }
    }


def _randomize_router(worker, spec: dict, seed: int = 0):
    """Dummy weights make every router emit ~equal logits (plus any dummy bias), so tokens
    collapse onto a few experts and a rank's MoE mostly skips work. Re-draw the router weights
    the model names (`spec` = model.py ROUTER: `weight_suffix`, `zero_suffixes`, `std`) so
    routing is ~uniform: in the target model and, when one is built, the drafter's own model
    (an MTP layer has its own MoE router). Returns the tensors reset per model."""
    import torch
    g = torch.Generator().manual_seed(seed)
    zero = tuple(spec.get("zero_suffixes") or ())
    r = worker.model_runner
    models = [("target", r.model)]
    d = getattr(r, "drafter", None)
    dm = getattr(d, "model", None) or getattr(
        d, "draft_model", None)  # Eagle3Proposer: draft_model
    if dm is not None and dm is not r.model:
        models.append(("drafter", dm))
    seen, out = set(), {}
    for tag, m in models:
        n = 0
        for name, p in m.named_parameters():
            if id(p) in seen:  # a weight the drafter shares with the target
                continue
            seen.add(id(p))
            if name.endswith(spec["weight_suffix"]):
                p.data.copy_((torch.randn(p.shape, generator=g) *
                              spec["std"]).to(p.dtype))
                n += 1
            elif zero and name.endswith(zero):
                p.data.zero_()
                n += 1
        out[tag] = n
    return out


def _twin_rows(batch: list[dict], scale: float) -> list[list[int]]:
    """REAL (q, seq) -> twin: the cached part (seq - q) scales, the query tokens do not."""
    return [[b["q"], b["q"] + max(16, int((b["seq"] - b["q"]) * scale))]
            for b in batch]


DRAFT: dict | None = None  # the plan's drafter spec when it runs the drafter
VERIFY = 1  # MTP verify-window width, 1 + k (the plan's deploy)
SAMPLING: dict | None = None  # the plan's sampling block when its sample stage runs


def _time(engine, rows: list, bucket: int, iters: int) -> dict:
    """One batch, `iters` times; a step waits for its slowest rank. A first iteration far
    slower than the rest means the bucket compiled on first use: that step's first number
    includes compile time, and a bucket is missing from the plan."""
    ms = engine.collective_rpc(_run_batch,
                               args=(rows, bucket, iters, VERIFY, SAMPLING))
    step = [max(r[i] for r in ms) for i in range(iters)]
    med = _med(step)
    samp = None
    if SAMPLING:
        sm = [x for x in engine.collective_rpc(_last_sampling, args=()) if x]
        if sm:
            samp = [max(r[i] for r in sm) for i in range(iters)]
    inputs = engine.collective_rpc(_last_inputs,
                                   args=())[0]  # rank 0's record of the batch
    if inputs and inputs.get("out_finite") is False:
        print(
            f"! non-finite forward output ({len(rows)} reqs, bucket {bucket}): MoE routing "
            "after the bad layer collapses onto one rank; see result.json inputs",
            flush=True)
    out = {
        "reqs": len(rows),
        "tokens": sum(q for q, _ in rows),
        "bucket": bucket,
        "all": [round(v, 3) for v in step],
        "median_after_first": med,
        "compiled_on_first_use": iters > 1 and step[0] > 3 * med + 50,
        "inputs": inputs
    }
    if samp:
        # the sample stage, timed right after its forward (wall; device busy is in the profile)
        out["sampling_ms"] = {
            "all": [round(v, 3) for v in samp],
            "median_after_first": _med(samp)
        }
    if DRAFT:
        n = max(iters,
                5)  # host-side Python in the replay: occasional 2-5x outliers
        dm = engine.collective_rpc(_run_draft, args=(bucket, n, DRAFT))
        dstep = [max(r[i] for r in dm) for i in range(n)]
        out["draft_ms"] = {
            "all": [round(v, 3) for v in dstep],
            "median_after_first": _med(dstep)
        }
        out["with_draft_ms"] = round(med + _med(dstep), 3)
    return out


def _time_points(engine, plan: dict, scale: float, a, rep: int):
    """One pass over the plan: every step timed `a.iters` times. Returns the per-entry records
    and the profile window: one mixed + one decode step per point, and for step-mix cells
    only those the plan marks `profile` (the heaviest per class)."""
    out, window = [], []
    for s in plan["points"]:
        name = s["point"]["name"]
        prof = s.get("profile", True)
        rec = {
            "point": s["point"],
            "mixed_ms": [],
            "decode_ms": None,
            "last_chunk_ms": None
        }
        mb = s["mixed"]["bucket"]
        for j, batch in enumerate(s["mixed"]["batches"]):
            rows = _twin_rows(batch, scale)
            t = time.time()
            rec["mixed_ms"].append({
                "chunk": j,
                "prefix_twin": rows[-1][1] - rows[-1][0],
                **_time(engine, rows, mb, a.iters)
            })
            mark(
                f"repeat {rep}: {name} mixed chunk {j} ({len(rows)} reqs, {mb} bucket)",
                t)
            if prof and j == s["mixed"]["profile_index"]:
                window.append((f"{name}/mixed/chunk{j}", rows, mb))
        lc = s.get("last_chunk")
        if lc and lc.get("run"):
            rows = _twin_rows(lc["batch"], scale)
            t = time.time()
            rec["last_chunk_ms"] = _time(engine, rows, lc["bucket"], a.iters)
            mark(f"repeat {rep}: {name} last chunk ({lc['bucket']} bucket)", t)
            if prof and s["mixed"]["profile_index"] is None:
                window.append((f"{name}/mixed/last", rows, lc["bucket"]))
        if s.get("decode"):
            rows, db = _twin_rows(s["decode"]["batch"],
                                  scale), s["decode"]["bucket"]
            t = time.time()
            rec["decode_ms"] = {
                "seq_twin": rows[0][1],
                **_time(engine, rows, db, a.iters)
            }
            mark(
                f"repeat {rep}: {name} decode ({len(rows)} reqs x {rows[0][0]} tokens, {db} bucket)",
                t)
            if prof:
                window.append((f"{name}/decode", rows, db))
        rec["twin_model"] = _cell_model(
            s, rec) if s.get("cell") else _twin_model(s, rec)
        out.append(rec)
        print(
            f"POINT {name} (repeat {rep}): mixed {[m['median_after_first'] for m in rec['mixed_ms']]} ms, "
            f"decode {(rec['decode_ms'] or {}).get('median_after_first')} ms",
            flush=True)
    return out, window


def _plan_shapes(plan: dict, scale: float) -> list[tuple[str, list, int]]:
    """Every batch `_time_points` times, as (label, twin rows, bucket), in plan order."""
    out = []
    for s in plan["points"]:
        name = s["point"]["name"]
        for j, batch in enumerate(s["mixed"]["batches"]):
            out.append((f"{name}/mixed/chunk{j}", _twin_rows(batch, scale),
                        s["mixed"]["bucket"]))
        lc = s.get("last_chunk")
        if lc and lc.get("run"):
            out.append((f"{name}/mixed/last", _twin_rows(lc["batch"],
                                                         scale), lc["bucket"]))
        if s.get("decode"):
            out.append(
                (f"{name}/decode", _twin_rows(s["decode"]["batch"],
                                              scale), s["decode"]["bucket"]))
    return out


def _merge_repeats(reps: list[list[dict]]) -> list[dict]:
    """Per step: the median over repeats of each repeat's median, and every repeat's value
    (`by_repeat`), so the spread across repeats is visible next to the number used."""

    def merge(xs: list[dict]) -> dict:
        m = dict(xs[0])
        vals = [x["median_after_first"] for x in xs]
        m["by_repeat"] = vals
        m["all"] = [x["all"] for x in xs
                    ]  # every repeat's iterations, not just repeat 0
        m["median_after_first"] = round(statistics.median(vals), 3)
        m["spread_pct"] = round(
            100 * (max(vals) - min(vals)) /
            statistics.median(vals), 2) if vals[0] else None
        m["compiled_on_first_use"] = any(
            x.get("compiled_on_first_use") for x in xs)
        if "draft_ms" in xs[0]:
            dv = [x["draft_ms"]["median_after_first"] for x in xs]
            m["draft_ms"] = {
                "by_repeat": dv,
                "median_after_first": round(statistics.median(dv), 3),
                "all": [x["draft_ms"]["all"] for x in xs]
            }
            m["with_draft_ms"] = round(
                m["median_after_first"] + m["draft_ms"]["median_after_first"],
                3)
        if all("sampling_ms" in x for x in xs):
            sv = [x["sampling_ms"]["median_after_first"] for x in xs]
            m["sampling_ms"] = {
                "by_repeat": sv,
                "median_after_first": round(statistics.median(sv), 3),
                "all": [x["sampling_ms"]["all"] for x in xs]
            }
        return m

    out = []
    for i, first in enumerate(reps[0]):
        rec = {
            "point":
            first["point"],
            "mixed_ms": [
                merge([r[i]["mixed_ms"][j] for r in reps])
                for j in range(len(first["mixed_ms"]))
            ],
            "decode_ms":
            merge([r[i]["decode_ms"]
                   for r in reps]) if first["decode_ms"] else None,
            "last_chunk_ms":
            merge([r[i]["last_chunk_ms"]
                   for r in reps]) if first["last_chunk_ms"] else None
        }
        rec["twin_model"] = first["twin_model"]
        out.append(rec)
    return out


def _mix_rollup(plan: dict, points: list[dict]) -> dict | None:
    """The step mix at twin ms: sum over cells of steps x the cell's step (median over
    repeats), per class and in total, beside the job's measured busy time. Twin ms: the
    twin's layers, per-rank exact -- not the served model."""
    cells = [(s, r) for s, r in zip(plan["points"], points) if s.get("cell")]
    if not cells:
        return None
    by: dict[str, float] = {}
    for s, r in cells:
        c = s["cell"]
        t = (r["mixed_ms"][0]
             if r["mixed_ms"] else r["decode_ms"])["median_after_first"]
        by[c["class"]] = by.get(c["class"], 0.0) + c["steps"] * t
    mx = plan.get("mix") or {}
    return {
        "busy_twin_s":
        round(sum(by.values()) / 1e3, 2),
        "by_class_twin_s": {
            k: round(v / 1e3, 2)
            for k, v in by.items()
        },
        # the cells THIS plan ran (a split plan holds one), not the whole mix's kept share
        "share_of_busy_covered":
        round(sum(s["cell"]["share"] for s, _ in cells), 4),
        "job_busy_s_measured":
        mx.get("busy_s"),
        "note":
        "twin ms x the job's step counts; the job's busy time is the served model"
    }


def run_plan(engine, plan: dict, scale: float, a, prof_dir: str,
             result: dict) -> None:
    """Time every step of the plan `a.repeats` times in this one engine, then profile one mixed +
    one decode step per point."""
    if DRAFT:
        # The runner's own spec-decode warmup does not necessarily warm the replay's eager
        # programs, which would then compile inside timed steps. Warm each bucket, untimed,
        # after one backbone batch at that bucket (the replay reads its hidden-state dtype).
        t = time.time()
        for b in plan["compile_buckets"]:
            engine.collective_rpc(_run_batch, args=([[1, 16]], b, 1))
            engine.collective_rpc(_run_draft, args=(b, 2, DRAFT))
        mark(f"drafter warm-up, buckets {plan['compile_buckets']}", t)
    if SAMPLING:
        # the sample stage's programs (LM head, sampler, rejection sampler, the eager RNG / index
        # ops, the markers) compile on first use, per shape: several seconds, and on j-3e69c2dd
        # they slowed the repeat-0 FORWARDS of the next iterations too (b2: 275, 17, 15 ms vs
        # 1.5). Run every timed shape through forward + stage, untimed, before repeat 0
        t = time.time()
        shapes = _plan_shapes(plan, scale)
        for _, rows, b in shapes:
            engine.collective_rpc(_run_batch,
                                  args=(rows, b, 3, VERIFY, SAMPLING, True))
        mark(f"sample-stage warm-up ({len(shapes)} shapes)", t)
    reps = []
    for rep in range(a.repeats):
        recs, window = _time_points(engine, plan, scale, a, rep)
        reps.append(recs)
    result["repeats"] = reps
    result["points"] = _merge_repeats(reps)
    result["mix"] = _mix_rollup(plan, result["points"])
    if a.profile:
        t = time.time()
        engine.start_profile()
        order = []
        # each step `--profile-iters` times, consecutively and under the same label: the cost
        # model pools same-label steps (one profile is one step; its per-layer noise is
        # multiplied by the served layer count)
        for label, rows, b in window:
            for _ in range(a.profile_iters):
                # the sample stage (when the plan runs it) follows its forward between the
                # SAMPLING_MARKS programs, then the drafter's pass, as in serving
                ms = engine.collective_rpc(_run_batch,
                                           args=(rows, b, 1, VERIFY, SAMPLING,
                                                 True))
                if DRAFT:
                    engine.collective_rpc(_run_draft, args=(b, 1, DRAFT))
                order.append({
                    "label": label,
                    "ms": round(max(r[0] for r in ms), 3),
                    "reqs": len(rows),
                    "tokens": sum(q for q, _ in rows),
                    "bucket": b,
                    "sampling": bool(SAMPLING)
                })
                time.sleep(0.03)  # a visible gap between steps in the trace
        engine.stop_profile()
        os.system(
            f"mkdir -p {a.out}/xprof && cp -a {prof_dir}/. {a.out}/xprof/")
        mark(f"profile window ({len(window)} steps) + merge", t)
        result["profile_dir"] = f"{a.out}/xprof"
        result["profile_order"] = order


def _med(step: list[float]):
    return round(statistics.median(step[1:]), 3) if len(step) > 1 else round(
        step[0], 3)


def _twin_model(s: dict, rec: dict) -> dict:
    """The point's TWIN-ms model (the twin's layers, not the served model): TTFT-like prefill time and
    steady output rate from the step mix. Partial last chunk counted at the full-chunk time
    unless it was run."""
    mixed = [m["median_after_first"] for m in rec["mixed_ms"]]
    t_mixed = statistics.mean(mixed) if mixed else (
        rec["last_chunk_ms"] or {}).get("median_after_first")
    t_dec = rec["decode_ms"]["median_after_first"]
    last = (rec["last_chunk_ms"] or {}).get(
        "median_after_first", t_mixed) if s.get("last_chunk") else 0
    prefill = sum(mixed) + (last or 0)
    mx = s["step_mix"]
    C, osl = s["point"]["concurrency"], s["point"]["output"]
    T = mx["mixed_steps"] * (t_mixed or 0) + mx["decode_steps"] * t_dec
    return {
        "prefill_ms":
        round(prefill, 2),
        "t_mixed_ms":
        t_mixed and round(t_mixed, 3),
        "t_decode_ms":
        t_dec,
        "ms_per_C_turns":
        round(T, 1),
        "output_tok_per_s":
        round(C * osl / (T / 1e3), 1) if T else None,
        "note":
        "twin ms (the twin's layers, per-rank exact); not the served model's ms"
    }


def _cell_model(s: dict, rec: dict) -> dict:
    """A step-mix cell at twin ms: its step, the chunk's own cost (a mixed step minus the same
    decode rows alone), and its weight in the job (steps x step)."""
    c = s["cell"]
    t_mix = rec["mixed_ms"][0]["median_after_first"] if rec[
        "mixed_ms"] else None
    t_dec = (rec["decode_ms"] or {}).get("median_after_first")
    t = t_mix if t_mix is not None else t_dec
    return {
        "step_ms":
        t,
        "decode_rows_alone_ms":
        t_dec if t_mix is not None else None,
        "chunk_ms":
        round(t_mix - t_dec, 3) if t_mix is not None and t_dec else None,
        "class":
        c["class"],
        "share":
        c["share"],
        "steps":
        c["steps"],
        "weighted_s":
        round(c["steps"] * t / 1e3, 3) if t else None,
        "measured_step_ms_served":
        c["measured_step_ms"],
        "note":
        "twin ms (the twin's layers, per-rank exact); not the served model's ms"
    }


#: Flags a plan may carry from its source job's image that a newer vLLM replaced with env.
#: flag -> (env name, value). Newer vLLM reads only VLLM_DCP_Q_REPLICATE (envs.py, default 0).
_FLAG_TO_ENV = {"--no-dcp-q-replicate": ("VLLM_DCP_Q_REPLICATE", "0")}

#: Env worth auditing: what selects kernels, collectives and compile behaviour. Everything else
#: in the pod (credentials, paths, k8s) is left out of result.json.
_AUDIT_PREFIXES = ("VLLM_", "TPU_", "LIBTPU", "XLA_", "JAX_", "PJRT_", "USE_",
                   "ENABLE_", "ONEHOT_", "TORCH", "MEGASCALE_")


def _compat(argv: list[str], parser) -> tuple[list[str], list[str]]:
    """Translate flags this image's vLLM does not know into their env equivalent; report it."""
    known, notes, out = parser._option_string_actions, [], []
    for tok in argv:
        if tok in _FLAG_TO_ENV and tok not in known:
            name, val = _FLAG_TO_ENV[tok]
            os.environ[name] = val
            notes.append(
                f"{tok} not in this image's vLLM: set {name}={val} instead")
        else:
            out.append(tok)
    for n in notes:
        print("COMPAT", n, flush=True)
    return out, notes


def _effective_env(plan_env: dict) -> list[dict]:
    """The env the engine actually starts with, and where each key came from: the plan (the
    source job's recipe), forward_run itself, or the image / pod (not set by the plan)."""
    ours = {"VLLM_CACHE_ROOT", "VLLM_XLA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR",
            "VLLM_DISABLE_COMPILE_CACHE", "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
            "VLLM_ALLOW_INSECURE_SERIALIZATION", "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK"} \
        | {e for e, _ in _FLAG_TO_ENV.values()}
    return [{
        "name":
        k,
        "value":
        v,
        "source":
        "plan" if plan_env.get(k) == v else
        "forward_run" if k in ours else "image/pod"
    } for k, v in sorted(os.environ.items()) if k.startswith(_AUDIT_PREFIXES)]


PARTS = 16  # parallel gzip streams (the image has no pigz/zstd; pods have many CPUs)


def _save_parts(cache_dir: str, out: str) -> None:
    """Packs the compile cache into one file, compressing it in parallel.

    The output file is a plain tar archive that holds PARTS gzip archives.
    The function spreads the cache files over the parts by size, so the parts
    compress at the same time. One `tar | gzip` stream is much slower.
    `cachelocal.py` and the forward-cmd recipe read the output file, and
    `_restore_parts` unpacks it.

    Args:
        cache_dir: The compile cache directory to pack.
        out: The output file. A relative path is relative to the current
            working directory.

    Raises:
        RuntimeError: A tar command failed, so the output file is missing or
            incomplete.
    """
    import subprocess
    import tempfile

    # The final tar runs in the temporary directory, so resolve a relative
    # output path against the caller's working directory first.
    out = os.path.abspath(out)
    files = []
    for root, _, names in os.walk(cache_dir):
        for n in names:
            fp = os.path.join(root, n)
            files.append((os.path.getsize(fp), os.path.relpath(fp, cache_dir)))
    groups = [[0, []] for _ in range(PARTS)]
    for size, rel in sorted(
            files, reverse=True):  # greedy: largest file to lightest part
        g = min(groups, key=lambda x: x[0])
        g[0] += size
        g[1].append(rel)
    # A private directory per call: a fixed /tmp path fails with
    # PermissionError when another user on a shared VM already owns it.
    with tempfile.TemporaryDirectory(prefix="wmb_parts_out_") as tmp:
        procs, parts = [], []
        for i, (_, rels) in enumerate(groups):
            if not rels:
                continue
            lst = os.path.join(tmp, f"list{i}")
            with open(lst, "w") as f:
                f.write("\n".join(rels) + "\n")
            parts.append(f"part{i}.tgz")
            procs.append(
                subprocess.Popen([
                    "tar", "-I", "gzip -1", "-cf",
                    os.path.join(tmp, parts[-1]), "-C", cache_dir, "-T", lst
                ]))
        failed = [p.args for p in procs if p.wait() != 0]
        # GNU tar refuses to create an empty archive from an empty file list,
        # so an empty cache reads its list from /dev/null instead.
        members = parts or ["-T", os.devnull]
        if not failed and subprocess.run(
            ["tar", "-cf", out, "-C", tmp, *members],
                check=False).returncode != 0:
            failed.append(["tar", "-cf", out])
    if failed:
        raise RuntimeError(
            f"could not save the compile cache to {out}. These tar commands "
            f"failed: {failed}")


def _restore_parts(src: str, cache_dir: str) -> None:
    """Undo `_save_parts` in parallel; a plain gzip tarball (older entries) still works."""
    import subprocess
    import tempfile

    if open(src, "rb").read(2) == b"\x1f\x8b":
        os.system(f"tar -xzf {src} -C {cache_dir}")
        return
    # A private directory per call, for the same reason as in `_save_parts`.
    with tempfile.TemporaryDirectory(prefix="wmb_parts_in_") as tmp:
        subprocess.run(["tar", "-xf", src, "-C", tmp], check=False)
        procs = [
            subprocess.Popen(
                ["tar", "-xzf",
                 os.path.join(tmp, f), "-C", cache_dir])
            for f in sorted(os.listdir(tmp)) if f.endswith(".tgz")
        ]
        for p in procs:
            p.wait()


MANIFEST = "wmb_manifest.json"  # torch.compile dir -> "cold" | "restored" (the boot that made it)
TIER3_KEEP_BYTES = 16 << 20


def _compile_dirs(cache_dir: str) -> set[str]:
    tcc = os.path.join(cache_dir, "torch_compile_cache")
    return {d
            for d in os.listdir(tcc)
            if d != "torch_aot_compile"} if os.path.isdir(tcc) else set()


def _used_compile_dirs(worker) -> list[str]:
    """Executed on every rank: the torch.compile dirs this process's compiled models use
    (target, and the drafter's own config when it has one)."""
    cfgs = [worker.model_runner.vllm_config]
    d = getattr(worker.model_runner, "drafter", None)
    for attr in ("vllm_config", "draft_vllm_config",
                 "speculative_vllm_config"):
        c = getattr(d, attr, None) if d is not None else None
        if c is not None:
            cfgs.append(c)
    out = set()
    for c in cfgs:
        for p in (getattr(c.compilation_config, "cache_dir", None),
                  getattr(c.compilation_config, "local_cache_dir", None)):
            if p and "torch_compile_cache/" in p:
                out.add(p.split("torch_compile_cache/", 1)[1].split("/", 1)[0])
    return sorted(out)


def _prune_for_save(cache_dir: str,
                    before: set[str],
                    restored: bool,
                    used: set[str] | None = None) -> dict:
    """Keep the tarball under 2 GiB, so it uploads quickly as the next job's input, dropping
    only what the NEXT run cannot use.

    A boot with no cache and a boot that restored one compute different torch.compile dirs
    (vLLM code_hash; design-notes P31). So:
    - Cold save (no cache restored): this boot's own torch.compile dirs are never loaded by the
      next run, which is a restored boot and computes the other key. Drop them. KEEP the large
      Tier-3 binaries (the compiled XLA executables): the next run recompiles the same programs
      under its own key, and Tier-3 can serve them instead of compiling.
    - Restored save: its torch.compile dirs are what every later run loads directly, so the
      large Tier-3 binaries are duplicates: drop them. Also drop "cold" dirs left from a cold
      save, but only when this boot flipped keys (made dirs of its own) and never a dir the
      workers report using (a dir with the same hash in both kinds of boot).
    - torch_aot_compile entries are loaded by both kinds of boot: never pruned.
    Dirs are tagged with the kind of boot that made them in a manifest; untagged dirs (older
    tarballs) are kept.
    """
    path = os.path.join(cache_dir, MANIFEST)
    man = json.load(open(path)) if os.path.isfile(path) else {}
    new = _compile_dirs(cache_dir) - before
    for d in new:
        man[d] = "restored" if restored else "cold"
    tcc = os.path.join(cache_dir, "torch_compile_cache")
    dropped = []

    def drop(d: str) -> None:
        if os.path.isdir(os.path.join(tcc, d)):
            os.system(f"rm -rf {os.path.join(tcc, d)}")
            dropped.append(d)

    big = []
    if not restored:
        for d in sorted(new):
            drop(d)  # tag stays in the manifest: "cold", already dropped
    else:
        if new:
            for d, kind in list(man.items()):
                if kind == "cold" and d not in (used or set()):
                    drop(d)
                    del man[d]
        for root, _, files in os.walk(os.path.join(cache_dir, "tier3")):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.getsize(fp) > TIER3_KEEP_BYTES:
                    big.append(f)
                    os.remove(fp)
    json.dump(man, open(path, "w"), indent=1)
    return {
        "compile_dirs_dropped": dropped,
        "tier3_large_dropped": len(big),
        "manifest": man
    }


# What _run_batch relies on in the vllm-torchtpu runner, by (file under vllm_torchtpu,
# qualified name); the plan's drafter spec adds the drafter's. Their source is saved with every
# run so the runner can be checked against what the image actually contains.
IMAGE_SOURCE_FUNCS = [
    ("runner/tpu_runner.py", "TPUModelRunner._dummy_run"),
    ("runner/tpu_runner.py", "TPUModelRunner._build_attention_metadata"),
    ("runner/tpu_runner.py", "TPUModelRunner._prepare_inputs"),
    ("runner/tpu_runner.py", "TPUModelRunner._dp_lockstep_enabled"),
    ("layers/core/sequence_layout.py",
     "AllSequenceLayoutPlanner.prepare_real"),
    # the sample stage (`_sample_stage`, `_prepare_sampling`)
    ("runner/tpu_runner.py", "TPUModelRunner.sample_tokens"),
    ("runner/tpu_runner.py", "TPUModelRunner._sample_spec_verify_chunk"),
    ("runner/tpu_runner.py", "TPUModelRunner._build_padded_sampling_params"),
    ("runner/tpu_runner.py", "TPUModelRunner.sample_from_logits"),
    ("runner/tpu_runner.py", "TPUModelRunner.compute_selected_logits"),
    ("runner/tpu_runner.py",
     "TPUModelRunner.spec_gather_bonus_and_target_logits"),
    ("runner/speculative_decoding_manager.py",
     "SpeculativeDecodingManager.get_spec_decode_metadata"),
    ("layers/adapter/sample/rejection_sampler.py", "RejectionSampler.forward"),
]


def _image_source(out_dir: str, extra: list | None = None) -> dict:
    """Package version, git commit (if the image kept .git) and the source + line of each
    function in IMAGE_SOURCE_FUNCS, read as text via AST (nothing imported)."""
    import ast
    import hashlib
    import importlib.metadata as md
    import importlib.util
    import subprocess
    rec: dict = {}
    spec = importlib.util.find_spec("vllm_torchtpu")
    if not spec or not spec.submodule_search_locations:
        return {"error": "vllm_torchtpu not found"}
    pkg = list(spec.submodule_search_locations)[0]
    rec["package_dir"] = pkg
    try:
        rec["version"] = md.version("vllm-torchtpu")
    except Exception:
        rec["version"] = None
    repo = os.path.dirname(os.path.dirname(pkg))
    g = subprocess.run(["git", "-C", repo, "log", "-1", "--format=%H %cI %s"],
                       capture_output=True,
                       text=True)
    rec["git"] = g.stdout.strip(
    ) or f"no git metadata ({g.stderr.strip()[:80]})"
    d = os.path.join(out_dir, "image_source")
    os.makedirs(d, exist_ok=True)
    funcs = {}
    funcs_wanted = IMAGE_SOURCE_FUNCS + list(extra or [])
    for rel, qual in funcs_wanted:
        path = os.path.join(pkg, rel)
        try:
            text = open(path).read()
            tree = ast.parse(text)
        except Exception as e:
            funcs[qual] = {"error": repr(e)[:120]}
            continue
        cls, fn = qual.split(".")
        node = next(
            (f for c in tree.body
             if isinstance(c, ast.ClassDef) and c.name == cls for f in c.body
             if isinstance(f, (ast.FunctionDef,
                               ast.AsyncFunctionDef)) and f.name == fn), None)
        if node is None:
            funcs[qual] = {"file": rel, "error": "not found"}
            continue
        src = "\n".join(text.splitlines()[node.lineno - 1:node.end_lineno])
        open(os.path.join(d,
                          qual.replace(".", "__") + ".py"),
             "w").write(src + "\n")
        funcs[qual] = {
            "file": rel,
            "lines": [node.lineno, node.end_lineno],
            "sha1": hashlib.sha1(src.encode()).hexdigest()[:12]
        }
    rec["functions"] = funcs
    rec["file_sha1"] = {
        rel:
        hashlib.sha1(open(os.path.join(pkg, rel),
                          "rb").read()).hexdigest()[:12]
        for rel in sorted({r
                           for r, _ in funcs_wanted})
        if os.path.isfile(os.path.join(pkg, rel))
    }
    return rec


CACHE_DIR = "/tmp/wmb_cache"  # every compile cache inside one saved dir


def _run_env(twin: dict, cache_dir: str = CACHE_DIR) -> dict:
    """The env every process of the run needs: the plan's env, then the cache paths. Applied
    in-process on one host; on two hosts `--node-prep` exports it on each pod before `ray start`,
    because Ray workers inherit their env from the raylet, not from the driver."""
    return {
        **twin["env"],
        **{
            "VLLM_CACHE_ROOT": cache_dir,
            "VLLM_XLA_CACHE_PATH": f"{cache_dir}/xla",
            # torch's inductor cache defaults to /tmp/torchinductor_root: keep it inside
            "TORCHINDUCTOR_CACHE_DIR": f"{cache_dir}/inductor",
            # torch_tpu's persistent PJRT-binary cache (vllm_torchtpu env_override.py):
            # without it every boot recompiles ~140 small eager programs (dummy weight
            # init, copies). Excluded from the compile-cache key
            # (tpu_compiler.py _TPU_COMPILE_ENV_IGNORED), so turning it on moves no key.
            "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": f"{cache_dir}/tier3",
            # a Tier-3 read otherwise also starts a local "backup" compile of the same
            # program (torch_tpu UsesLocalBackupTaskForTier3Read). Not in any key.
            "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_LOCAL_BACKUP_TASK": "0",
            # vLLM and vllm-torchtpu run from source checkouts in the image: without
            # a kept bytecode cache every pod recompiles them. Workers read the env
            # var; this process is already running, so sys.pycache_prefix is set below.
            "PYTHONPYCACHEPREFIX": f"{cache_dir}/pycache",
            "VLLM_DISABLE_COMPILE_CACHE": "0",
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1"
        }
    }  # collective_rpc(callable)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",
                    required=True,
                    help="a points plan from `cli.py points`")
    ap.add_argument(
        "--iters",
        type=int,
        default=10,
        help=
        "timed iterations per batch; the median drops the first (10: one slow "
        "host iteration moved a 3-iteration repeat by 20-45%%)")
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=
        "time the whole plan this many times in the same engine (report shows each)"
    )
    ap.add_argument("--profile", action="store_true")
    ap.add_argument(
        "--profile-iters",
        type=int,
        default=1,
        help="profile each window step N times (same label; pooled downstream)"
    )
    ap.add_argument(
        "--no-sampling",
        action="store_true",
        help=
        "skip the plan's sample stage (LM head + sampler) even if it asks for it"
    )
    ap.add_argument("--cache-in", default="")
    ap.add_argument("--save-cache", action="store_true")
    ap.add_argument(
        "--node-prep",
        action="store_true",
        help="two-host plans: run on EVERY host before `ray start`. "
        "Restores --cache-in on this pod and prints `export` lines for the run's "
        "env on stdout, to eval; nothing else goes to stdout")
    ap.add_argument(
        "--out",
        default=os.path.join("wmb_out", "wmb", "forward"),
        help="output directory, relative to the working directory "
        "(default wmb_out/wmb/forward; the forward-cmd recipe packs wmb_out)")
    a = ap.parse_args()
    plan = json.load(open(a.steps))
    if a.node_prep:
        import contextlib
        import shlex
        if a.cache_in and not os.path.isdir(CACHE_DIR):
            os.makedirs(CACHE_DIR, exist_ok=True)
            with contextlib.redirect_stdout(
                    sys.stderr):  # stdout carries only the exports
                if os.path.isdir(a.cache_in):
                    os.system(f"cp -a {a.cache_in}/. {CACHE_DIR}/ 1>&2")
                else:
                    _restore_parts(a.cache_in, CACHE_DIR)
        for k, v in {
                **_run_env(plan["twin"]), "WMB_NODE_PREPPED": "1"
        }.items():
            print(f"export {k}={shlex.quote(str(v))}")
        return 0
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(a.steps,
                os.path.join(a.out,
                             "plan.steps.json"))  # the plan this job ran
    twin, buckets = plan["twin"], plan["compile_buckets"]
    os.makedirs(a.out, exist_ok=True)
    result: dict = {
        "steps": a.steps,
        "compile_buckets": buckets,
        "iters": a.iters,
        "sharding": plan.get("sharding")
    }
    dr = twin.get("drafter") or {}
    result["image_source"] = _image_source(
        a.out,
        [(dr["file"], q)
         for q in dr.get("source_funcs", [])] if dr.get("file") else None)
    print("IMAGE SOURCE:",
          result["image_source"].get("version"),
          result["image_source"].get("git"),
          flush=True)

    model_dir = "/tmp/wmb_model"  # config.json only (dummy weights)
    os.makedirs(model_dir, exist_ok=True)
    json.dump(plan["checkpoint_config"],
              open(os.path.join(model_dir, "config.json"), "w"))

    cache_dir = CACHE_DIR
    os.environ.update(_run_env(twin, cache_dir))
    if a.cache_in and os.environ.get("WMB_NODE_PREPPED") == "1":
        mark("compile cache restored per pod by --node-prep", time.time())
    elif a.cache_in:
        t = time.time()
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.isdir(a.cache_in):
            os.system(f"cp -a {a.cache_in}/. {cache_dir}/")
        else:
            _restore_parts(a.cache_in, cache_dir)
        mark("restore compile cache", t)
    restored_dirs = _compile_dirs(cache_dir)

    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    t = time.time()
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.engine.llm_engine import LLMEngine
    mark("import vllm", t)

    prof_dir = "/tmp/wmb_xprof"
    argv = engine_argv(twin, model_dir, buckets, prof_dir)
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.add_argument("model_tag")
    plan_argv = list(argv)  # the compile-cache key (cachelocal) is on these
    if twin.get("multihost"):
        # two hosts: vLLM's multi-host TPU executor is Ray (a launcher brought the cluster up before this ran).
        # Added after plan_argv, so the compile-cache key does not move with it.
        argv += ["--distributed-executor-backend", "ray"]
    argv, result["compat"] = _compat(argv, parser)
    result["effective_env"] = _effective_env(twin["env"])
    result["plan_env"] = twin[
        "env"]  # part of the compile-cache key (cachelocal)
    ns = parser.parse_args(argv)
    ns.model = ns.model_tag
    print(ENGINE_START_LINE, " ".join(argv), flush=True)
    result["engine_argv"] = plan_argv
    result["engine_argv_run"] = argv
    t = time.time()
    engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(ns))
    result["boot_s"] = mark(
        f"{ENGINE_READY_MARK} (weights + KV + compile of buckets {buckets})",
        t)
    t = time.time()
    router = twin.get("router")
    result["router_tensors_reset"] = (engine.collective_rpc(
        _randomize_router, args=(router, ))[0] if router else {})
    mark(f"randomize routers ({result['router_tensors_reset']} tensors)", t)
    try:  # the compile-cache key's inputs, rank 0 (see _key_factors)
        result["compile_key_factors"] = engine.collective_rpc(_key_factors)[0]
        print("COMPILE KEY:", {
            k: v
            for k, v in result["compile_key_factors"].items() if k != "tpu_env"
        },
              flush=True)
    except Exception as e:
        result["compile_key_factors"] = {"error": repr(e)[:300]}
    used = os.path.basename(
        (result["compile_key_factors"] or {}).get("cache_dir") or "")
    result["cache"] = {
        "restored": bool(a.cache_in),
        "backbone_compile_dir": used,
        # warm = the dir this boot's key names was in the restored tarball
        "warm": bool(used) and used in restored_dirs
    }

    global DRAFT, VERIFY, SAMPLING
    DRAFT = twin.get("drafter") if twin.get("run_drafter") else None
    VERIFY = 1 + int((plan.get("deploy") or {}).get("spec_tokens") or 0)
    sp = plan.get("sampling") or {}
    SAMPLING = sp if sp.get("run") and not a.no_sampling else None
    result["sampling"] = {
        **sp, "run": bool(SAMPLING),
        "marks": list(SAMPLING_MARKS),
        "disabled_by":
        "--no-sampling" if sp.get("run") and a.no_sampling else None
    }
    print("SAMPLING:", json.dumps(result["sampling"]), flush=True)
    if twin.get("run_drafter") and not DRAFT:
        raise SystemExit(
            "the plan runs the drafter but carries no drafter spec (model.py DRAFTER)"
        )
    result[
        "drafter"] = "run: dummy-draft replay after every step" if DRAFT else "not run"
    run_plan(engine, plan, plan["context_scale"], a, prof_dir, result)
    try:  # rank 0: why a forward could be non-finite
        result["diagnose"] = engine.collective_rpc(_diagnose)[0]
        result["drafter_inputs"] = engine.collective_rpc(lambda w: getattr(
            w.model_runner, "_wmb_draft_inputs", None))[0] if DRAFT else None
        print("DIAGNOSE:",
              json.dumps({
                  k: result["diagnose"].get(k)
                  for k in ("attention_scales", "params_nonfinite",
                            "kv_pages_written")
              })[:3000],
              flush=True)
    except Exception as e:  # noqa: BLE001
        result["diagnose"] = {"error": repr(e)[:300]}

    if a.save_cache:
        t = time.time()
        try:
            used = {
                d
                for r_ in engine.collective_rpc(_used_compile_dirs)
                for d in r_
            }
        except Exception as e:
            used = None  # unknown: prune nothing rather than guess
            result["compile_dirs_used_error"] = repr(e)[:200]
        result["compile_dirs_used"] = sorted(
            used) if used is not None else None
        # without the used-dir list, keep every compile dir (restored=False skips the drop)
        result["compile_cache_pruned"] = _prune_for_save(
            cache_dir,
            restored_dirs,
            restored=bool(a.cache_in) and used is not None,
            used=used)
        # The measurements are already in `result`. A failed cache save is
        # recorded there, so that it does not stop result.json from being
        # written.
        try:
            _save_parts(cache_dir, f"{a.out}/compile_cache.tgz")
            result["compile_cache_bytes"] = os.path.getsize(
                f"{a.out}/compile_cache.tgz")
        except (OSError, RuntimeError) as e:
            result["compile_cache_error"] = repr(e)[:500]
            print("save compile cache failed:", e, flush=True)
        mark("save compile cache", t)
    result["times_s"] = TIMES
    result["total_s"] = round(time.time() - T0, 1)
    json.dump(result, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("TIMES", json.dumps(TIMES), flush=True)
    try:  # clean shutdown: no teardown core dumps
        engine.engine_core.shutdown()
    except Exception as e:
        print("shutdown:", e, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

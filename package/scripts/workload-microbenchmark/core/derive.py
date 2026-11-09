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
"""snapshot (+ overrides) -> one L1 config. The single entry point `onboard` and `derive` share."""

from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import asdict, replace
from typing import Any

from . import cells as cells_mod
from .deploy import Deploy, Facts, apply_overrides, deploy_from_args, per_rank

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Where model directories (`<MODEL_DIR>/model.py`, `config/`, `examples/`) are looked up, in
#: order. The package ships the reviewed models beside `core/`; `cli.py` puts the work
#: directory's `models/` first, where a model being onboarded is written.
MODEL_ROOTS = [SCRIPTS]


def model_path(model_dir: str) -> str:
    """The directory of `model_dir`: the first root that holds its model.py, else the first
    root (where a new model is written)."""
    for root in MODEL_ROOTS:
        if os.path.exists(os.path.join(root, model_dir, "model.py")):
            return os.path.join(root, model_dir)
    return os.path.join(MODEL_ROOTS[0], model_dir)


#: Overrides that change the deployment or the scheduler. L1 can price their per-layer
#: effect, but they also change step counts and collectives, so only L2 decides them.
SCHEDULING_KEYS = {
    "tp", "dp", "pp", "dcp", "pcp", "ep", "block_size", "mnbt", "mns",
    "max_model_len", "prefix_caching", "spec_method", "spec_tokens",
    "interleave", "kv_cache_memory_bytes", "dcp_q_replicate"
}

#: Recipe env that describes the job's plumbing rather than the server's behaviour:
#: multi-host wiring, paths, caches, the job platform's own bookkeeping (see
#: `_platform_prefixes`). Everything else is carried into the twin, including debug logging,
#: because it was part of what the job measured.
_ENV_DROP = ("HF_", "RAY_", "MEGASCALE_", "TPU_PROCESS_", "DUMP_HLO", "JS_")
_ENV_DROP_EXACT = {
    "MODEL_PATH", "SERVED_MODEL_NAME", "TPU_MULTIHOST_BACKEND",
    "VLLM_CACHE_ROOT", "POD_NAMESPACE", "JOBSET_JOB_INDEX", "GET_HOST_METRICS",
    "TPU_NODE_ID", "TPU_WORKER_ID"
}  # multi-host rank ids set by the script

#: Env that selects how the sharding communicates (vllm-torchtpu / libtpu names, the same for
#: every model on this runtime). Recipe scripts compute these FROM the
#: sharding (e.g. hierarchical all-reduce only when TP > 1, the async row gather for DP+EP), so
#: a sharding override must not silently carry the job's values.
SHARDING_ENV = {
    "LIBTPU_INIT_ARGS", "XLA_FLAGS",
    "TPU_TP_HIERARCHICAL_ALL_REDUCE_MIN_TOKENS", "USE_MOE_FUSED_EP_KERNEL",
    "TPU_MOE_FUSED_EP_ASYNC_ROW_GATHER", "TPU_MOE_FUSED_EP_STEP_MIN_TOKENS",
    "VLLM_DCP_Q_REPLICATE"
}
SHARDING_KEYS = {
    "tp", "dp", "pp", "dcp", "pcp", "ep", "interleave", "dcp_q_replicate"
}


def env_provenance(rec: dict[str, Any],
                   env: dict[str, str],
                   overrides: dict[str, str],
                   model_env: dict[str, str] | None = None) -> list[dict]:
    """Find out where each environment variable in `env` came from.

    Args:
        rec: The job's recipe from `jobdata_agentx.read_recipe`. This function reads
            three keys:
            - "env": {variable name: value} the job ran with.
            - "env_from_script": {variable name: value} that the recipe script set.
            - "serve_parse": a string. "env_from_script" is only used when it is
              "bash-eval".
        env: {variable name: value} of the microbenchmark.
        overrides: {key: value} from --override. A key is a deploy field such as "tp"
            or "dcp", or "env.<variable name>".
        model_env: {variable name: value} from the model's TWIN_ENV. None when the
            caller does not pass it. cli.py always passes it, and passes {} when the
            model has no TWIN_ENV.

    Returns:
        One dict per variable in `env`, sorted by name, with these keys:
        - "name", "value": the variable.
        - "source": "override", "model", "script", "container", or
          "container (recipe script NOT evaluated)".
        - "sharding": True if the variable is in SHARDING_ENV.
        - "review": a note to check the value. Only present when --override changed
          the sharding but not this variable.
    """
    ov_env = {k[4:] for k in overrides if k.startswith("env.")}
    script = rec.get("env_from_script") if rec.get(
        "serve_parse") == "bash-eval" else None
    resharded = bool(set(overrides) & SHARDING_KEYS)
    out = []
    for k, v in sorted(env.items()):
        src = ("override" if k in ov_env else "model"
               if k in (model_env or {}) and k not in rec.get("env", {}) else
               "script" if script and k in script else "container" if script
               is not None else "container (recipe script NOT evaluated)")
        row = {
            "name": k,
            "value": v,
            "source": src,
            "sharding": k in SHARDING_ENV
        }
        if row["sharding"] and resharded and src != "override":
            row["review"] = (
                "carried from the job's sharding; confirm for the new one "
                "or pass --override env.%s=..." % k)
        out.append(row)
    return out


def _is_path(v: Any) -> bool:
    """A bare absolute path (`/root/vllm-torchtpu-cache/tmp`): a directory the recipe script
    created in its own pod, which the twin's pod does not have. forward_run sets the cache
    paths it needs itself."""
    s = str(v)
    return s.startswith("/") and not any(c in s for c in " ,;=")


def _platform_prefixes(env: dict) -> tuple[str, ...]:
    """Prefixes of the job platform's own variables. The platform names the job in a
    `<PREFIX>_JOB_ID` variable, so every `<PREFIX>_*` beside it is its bookkeeping."""
    return tuple(k[:-len("JOB_ID")] for k in env if k.endswith("_JOB_ID"))


def server_env(env: dict) -> dict:
    drop = _ENV_DROP + _platform_prefixes(env)
    return {
        k: v
        for k, v in sorted(env.items())
        if v is not None and "$(" not in str(v) and not _is_path(v)
        and not k.startswith(drop) and k not in _ENV_DROP_EXACT
    }


def load_model(model_dir: str):
    path = os.path.join(model_path(model_dir), "model.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: onboard the model first "
                                f"(`cli.py onboard {model_dir} --job <id>`)")
    spec = importlib.util.spec_from_file_location(f"wmb_model_{model_dir}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def feasibility(f: Facts,
                d: Deploy,
                source: Deploy,
                server: dict,
                pool_tokens: int | None,
                model_checks=None,
                state_bytes_rank: float = 0.0,
                kv_rank=None) -> dict[str, Any]:
    """Generic vLLM divisibility rules, the model's own checks (`model_checks(f, d)` -> list of
    problems), and the KV budget: per-token KV per rank (`kv_rank`, the model's hook, else
    / DCP), plus recurrent state per running sequence (`state_bytes_rank`, from the model) for
    models that have it."""
    problems, notes = [], []
    if model_checks:
        problems += list(model_checks(f, d))
    if f.heads % d.tp:
        problems.append(f"heads {f.heads} % tp {d.tp} != 0")
    if d.pcp == 1 and d.dcp > 1 and d.tp % d.dcp:
        problems.append(f"DCP reuses TP ranks: tp {d.tp} % dcp {d.dcp} != 0")
    if f.experts and d.ep and f.experts % d.ep_group:
        problems.append(f"experts {f.experts} % EP group {d.ep_group} != 0")
    if d.world != source.world:
        notes.append(
            f"world {d.world} != recorded {source.world}: needs a different slice"
        )
    from .deploy import kv_per_token_rank
    per_tok = kv_per_token_rank(f, d, kv_rank)
    budget = d.kv_cache_memory_bytes
    kv = {
        "kv_bytes_per_token_per_rank": round(per_tok, 1),
        "kv_budget_bytes_per_rank": budget,
        "state_bytes_per_seq_per_rank": round(state_bytes_rank, 1)
    }
    if budget and state_bytes_rank:
        # recurrent state is held per running sequence, up to max-num-seqs
        budget = budget - d.mns * state_bytes_rank
        kv["kv_budget_after_state_bytes_per_rank"] = int(budget)
    if budget:
        one = d.max_model_len * per_tok
        kv["one_max_len_request_bytes"] = int(one)
        if d.max_model_len and one > budget:
            problems.append(
                f"one max-model-len request needs {one/1e9:.1f} GB KV per rank "
                f"> budget {budget/1e9:.1f} GB (engine start fails)")
        usage = (server.get("kv_usage") or {}).get("max")
        if usage and pool_tokens:
            peak = usage * pool_tokens  # tokens in flight at the recorded peak
            need = peak * per_tok
            kv.update(peak_tokens_in_flight=int(peak),
                      peak_need_bytes=int(need))
            if need > budget:
                problems.append(
                    f"recorded peak {peak/1e6:.2f}M tokens in flight needs "
                    f"{need/1e9:.1f} GB KV per rank > budget {budget/1e9:.1f} GB"
                )
            min_dcp = math.ceil(peak * per_tok * d.dcp /
                                budget) if budget else None
            if min_dcp and d.tp:
                ok = [
                    x for x in range(1, d.tp + 1)
                    if d.tp % x == 0 and x >= min_dcp
                ]
                kv["min_dcp_for_recorded_peak"] = ok[0] if ok else None
        kv["max_model_len_that_fits"] = int(budget // per_tok)
    return {
        "feasible": not problems,
        "problems": problems,
        "notes": notes,
        "kv": kv
    }


def derive(snap: dict[str, Any],
           model_dir: str,
           overrides: dict[str, str] | None = None,
           twin_world: int = 8) -> dict[str, Any]:
    overrides = overrides or {}
    mod = load_model(model_dir)
    rec = snap["recipe"]
    source = deploy_from_args(rec["server_args"], rec["env"])
    log = snap.get("log") or {}
    if log.get("block_size_resolved"
               ) and log["block_size_resolved"] != source.block_size:
        # the TPU runtime resized the page (hybrid models: one page must hold a recurrent
        # state); the served page is what the prefix cache and the KV pool are built on
        source = replace(source, block_size=log["block_size_resolved"])
    d = apply_overrides(source, overrides)
    # env the model always needs (model.py TWIN_ENV), under the job's own env and overrides: a
    # server read from its vLLM log has no env record, and a recipe may not export them all
    env = {
        **getattr(mod, "TWIN_ENV", {}),
        **rec["env"],
        **{
            k[4:]: v
            for k, v in overrides.items() if k.startswith("env.")
        }
    }
    f = mod.facts(snap["checkpoint_config"], d.kv_cache_dtype)
    pool = log.get("kv_pool_tokens")
    # the pool scales with DCP (each rank holds 1/DCP of each token)
    pool_d = int(pool * d.dcp / source.dcp) if pool else None
    state = mod.state_bytes_per_seq_per_rank(f, d) if hasattr(mod, "state_bytes_per_seq_per_rank") \
        else 0.0
    kv_rank = getattr(mod, "kv_bytes_per_token_per_rank", None)
    feas = feasibility(f, d, source, snap["server"], pool,
                       getattr(mod, "feasibility_checks", None), state,
                       kv_rank)
    out: dict[str, Any] = {
        "_what":
        "workload-microbenchmark L1 config (generated; do not hand-edit)",
        "job_id":
        snap["job_id"],
        "model_dir":
        model_dir,
        "overrides":
        overrides,
        # a label for file names: the recipe's benchmark name when it has one
        "workload":
        str((rec.get("bench_env") or {}).get("BENCH_NAME") or "bench").lower(),
        "decidable_at": ("L1" if not (set(overrides) & SCHEDULING_KEYS) else
                         "L1 per-layer delta only; verdict needs L2 "
                         "(step counts and collectives change)"),
        "source": {
            "image": rec.get("image"),
            "owner": rec.get("owner"),
            "checkpoint_config": snap["checkpoint_config"].get("_source")
        },
        "deploy":
        asdict(d),
        "facts": {
            **asdict(f), "layer_kinds": f.layer_kinds,
            "kv_bytes_per_token": f.kv_bytes_per_token
        },
        "kernels":
        mod.kernels(env, d),
        "server_env":
        server_env(env),
        "feasibility":
        feas,
    }
    if not feas["feasible"]:
        return out
    if source.block_size != int(rec["server_args"].get("block-size",
                                                       source.block_size)):
        out["block_size_resolved"] = {
            "served_arg": int(rec["server_args"]["block-size"]),
            "resolved": source.block_size,
            "from": log.get("log_path")
        }
    out["per_rank"] = per_rank(f, d, pool_d, getattr(mod, "per_rank_rows",
                                                     None), kv_rank)
    c = cells_mod.derive(snap["requests"],
                         d,
                         snap["server"],
                         cache_hit=getattr(mod, "cache_hit", None))
    out["cells"] = c
    if not overrides:
        out["closure_G0"] = cells_mod.closure(c, snap["server"],
                                              snap.get("log") or {})
    out["components"] = mod.COMPONENTS
    return out

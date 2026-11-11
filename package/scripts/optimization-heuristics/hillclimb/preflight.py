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
"""Pre-launch hill-climbing gates (`preflight.py`).

Catches configuration traps deterministically before multi-slice TPU runs:
  * Asymmetric vs symmetric P/D DCP (`dcp_size_prefill_kv_halved`: setting
    symmetric `DCP_SIZE=8` halves the prefill HBM prefix cache from `11.89M ->
    5.95M` tokens and loses `24.8%` Mtok/$ at `c=48`; require
    `PREFILL_DCP_SIZE=16` + `DECODE_DCP_SIZE=8`, plus `asym_dcp_indexer_layout`
    and `asym_dcp_interleave` checks).
  * Decode step bucket divisibility (`decode_bucket_dcp`), padding waste
    (`decode_bucket_padding`), and fused TP MoE decode kernel `ROW_BUCKETS`
    admission (`decode_tp_kernel`).
  * `tpu7x` 64 MiB VMEM compile OOM (`vmem_absorb_q_head`: `68 MiB > 64 MiB` at
    `DCP >= 16` with replicated Q heads).
  * EP-vs-TP MoE mode conflicts (`moe_mode`), off prerequisite gate chains
    (`gate_off`), decode-blind prefill-only arms on decode-limited baselines
    (`off_leg`), and settings that close-gaps already recorded as `not a win`
    or `crash` in its ledger or gap files (`prior_outcome`).
"""

import dataclasses
import pathlib
import re
from collections.abc import Mapping, Sequence

from hillclimb import code_gap, code_limits, ledger
from hillclimb import legs as legs_lib
from hillclimb import lessons, run_history

try:
    from serving_analysis import limits as _serving_limits
except ImportError:  # Standalone hillclimb install without serving_analysis.limits
    _serving_limits = None  # type: ignore[assignment]

ERROR = "ERROR"
WARNING = "WARNING"
INFO = "INFO"

_OFF = frozenset({"", "0", "false", "False", "off", "none", "None"})
_POWER_OF_TWO_BUCKETS = frozenset(16 << i for i in range(12))
_TP_DECODE_KERNEL_TOKENS_DEFAULT: tuple[int, ...] = tuple(
    getattr(_serving_limits, "MOE_TP_DECODE_ROW_BUCKETS", (16, 32, 64)))
_MIN_ASYMMETRIC_PREFILL_DCP = int(
    getattr(_serving_limits, "MIN_ASYMMETRIC_PREFILL_DCP", 16))
_SMALL_BATCH_DECODE_TARGET_DCP = int(
    getattr(_serving_limits, "SMALL_BATCH_DECODE_TARGET_DCP", 8))
_A2A_MIN_TOKENS = 64
_MIN_PADDING_WASTE = 0.10
_PACKED_ROW_TOKENS = 8
# When neither the arm nor the recipe defaults set DECODE_INTERLEAVE, the
# engine runs with vLLM's default interleave (1), which is not a whole packed
# row; assuming a passing value here would let a crashing arm launch.
_DEFAULT_DCP_INTERLEAVE = int(
    getattr(_serving_limits, "DEFAULT_DCP_INTERLEAVE", 1))
_NUMERICS_RE = re.compile(
    r"\b(?:w4a8|fp4|mxfp4|w8a16|quant|approx|lossy|bf16\s+accum)\b",
    re.IGNORECASE,
)
_CLI_DCP_RE = re.compile(r"--decode-context-parallel-size(?:=|\s+)(\d+)")
_CLI_TP_RE = re.compile(r"(?:--tensor-parallel-size|-tp)(?:=|\s+)(\d+)")


@dataclasses.dataclass(frozen=True)
class Finding:
    """One pre-launch or post-run gate finding."""

    severity: str  # ERROR | WARNING | INFO
    check: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "check": self.check,
            "message": self.message,
        }


@dataclasses.dataclass(frozen=True)
class KnobMap:
    """Maps model-independent decode-step quantities onto recipe knob names."""

    name: str
    decode_cap: str = "DECODE_BATCH_CAP"
    decode_cap_floor: int = 16
    spec_label: str = "MTP"
    spec_on: str = "MTP"
    spec_default_on: bool = True
    spec_tokens: str = "MTP_NUM_SPECULATIVE_TOKENS"
    spec_tokens_default: int = 3
    decode_dcp: tuple[str, ...] = ("DECODE_DCP_SIZE", "DCP_SIZE")
    decode_dcp_default: int = 16
    bucket_extra: str = "DECODE_TOKEN_BUCKET_EXTRA"
    buckets_without_extra: tuple[int, ...] = (4, 8)
    ep_buckets_from: int = 80
    decode_moe: str = "DECODE_MOE"
    moe_auto_ep_at: int = 24
    decode_env: tuple[str, ...] = ("EXTRA_ENV_DECODE", )
    concurrency: str = "CLIENT_CONCURRENCIES"
    tp_decode_kernel: bool = True
    fused_moe_decode_max_tokens: int = 128
    compile_sizes: str = ""
    token_budget: str = ""
    # Highest client concurrency at which a second prefill leg (xP1D, x > 1)
    # was measured to cost more chips than it returns. `None`: not calibrated
    # for this map, so `extra_prefill_leg` stays silent.
    extra_prefill_leg_max_concurrency: int | None = None

    def knob_label(self, source: str) -> str:
        if source.startswith("env:"):
            where = self.decode_env[0] if self.decode_env else "decode env"
            return f"{source[4:]} in {where}"
        return source


GLM5_PD_KNOB_MAP = KnobMap(name="glm5_mla_moe",
                           extra_prefill_leg_max_concurrency=96)

QWEN35_SINGLE_KNOB_MAP = KnobMap(
    name="qwen3_5_hybrid_moe",
    decode_cap="MAX_NUM_SEQS",
    decode_cap_floor=0,
    spec_on="MTP",
    spec_tokens="MTP_NUM_SPECULATIVE_TOKENS",
    spec_tokens_default=3,
    decode_dcp=("DECODE_CONTEXT_PARALLEL_SIZE", ),
    decode_dcp_default=1,
    bucket_extra="TPU_TOKEN_BUCKET_EXTRA",
    buckets_without_extra=(),
    ep_buckets_from=0,
    decode_moe="ENABLE_EXPERT_PARALLEL",
    moe_auto_ep_at=0,
    decode_env=("EXTRA_SERVER_ENV", ),
    tp_decode_kernel=False,
    compile_sizes="COMPILE_SIZES",
    token_budget="MAX_NUM_BATCHED_TOKENS",
)

DEEPSEEK_PD_KNOB_MAP = KnobMap(
    name="deepseek_mla_moe",
    decode_dcp_default=16,
)

KNOB_MAPS: Mapping[str, KnobMap] = {
    "glm5_mla_moe": GLM5_PD_KNOB_MAP,
    "glm52_pd": GLM5_PD_KNOB_MAP,
    "1p1d_long_context": GLM5_PD_KNOB_MAP,
    "qwen3_5_hybrid_moe": QWEN35_SINGLE_KNOB_MAP,
    "qwen35_single": QWEN35_SINGLE_KNOB_MAP,
    "deepseek_mla_moe": DEEPSEEK_PD_KNOB_MAP,
    "deepseek_r1_pd": DEEPSEEK_PD_KNOB_MAP,
}


def _int_val(val: str | None) -> int:
    try:
        return int(str(val))
    except (TypeError, ValueError):
        return 0


def _ints_val(val: str | None) -> list[int]:
    return [int(t) for t in re.findall(r"\d+", str(val or ""))]


def bucket_gate_edge(min_tokens: int) -> int:
    """`vllm-torchtpu` gate threshold edge (`tokens > edge`)."""
    if min_tokens <= 1:
        return 0
    if min_tokens & (min_tokens - 1) == 0:
        return min_tokens // 2
    return min_tokens - 1


@dataclasses.dataclass(frozen=True)
class DecodeShape:
    """Resolved decode step geometry for an arm."""

    cap: int
    spec_tokens: int
    dcp: int
    extra_buckets: frozenset[int] | None
    moe_mode: str
    env: Mapping[str, str]
    a2a_edge: int
    a2a_on: bool
    compile_sizes: frozenset[int] | None = None
    budget: int = 0

    @property
    def step_tokens(self) -> int:
        per_req = 1 + self.spec_tokens
        full = self.cap * per_req
        if self.budget > 0:
            full = min(full, (self.budget // per_req) * per_req)
        return full

    def legal_bucket(self, tokens: int) -> bool:
        return (not self.a2a_on or self.dcp <= 1 or tokens <= self.a2a_edge
                or tokens % self.dcp == 0)

    def legal_round_up(self, tokens: int) -> int:
        if self.legal_bucket(tokens):
            return tokens
        return -(-tokens // self.dcp) * self.dcp


def _get_source(knobs: Mapping[str, str], env: Mapping[str, str],
                source: str) -> str | None:
    if not source:
        return None
    if source.startswith("env:"):
        return env.get(source[4:])
    return knobs.get(source)


def decode_shape(
    knobs: Mapping[str, str],
    knob_map: KnobMap = GLM5_PD_KNOB_MAP,
) -> DecodeShape | None:
    """Resolves `DecodeShape` from `knobs` using `knob_map`."""
    if not knob_map.decode_cap:
        return None
    env: dict[str, str] = {}
    for key in knob_map.decode_env:
        env.update(code_gap.parse_extra_env(knobs.get(key, "")))
    cap = _int_val(_get_source(knobs, env, knob_map.decode_cap))
    spec_raw = _get_source(knobs, env, knob_map.spec_on)
    spec_on = (knob_map.spec_default_on
               if spec_raw is None else str(spec_raw).lower() not in _OFF)
    spec_tokens = (_int_val(_get_source(knobs, env, knob_map.spec_tokens))
                   or knob_map.spec_tokens_default if spec_on else 0)
    dcp = 0
    for src in knob_map.decode_dcp:
        dcp = _int_val(_get_source(knobs, env, src))
        if dcp:
            break
    extra_raw = _get_source(knobs, env, knob_map.bucket_extra)
    extra = None if extra_raw is None else frozenset(_ints_val(extra_raw))
    compile_raw = _get_source(knobs, env, knob_map.compile_sizes)
    moe_raw = str(_get_source(knobs, env, knob_map.decode_moe) or "").lower()
    if moe_raw in ("true", "1"):
        moe_mode = "ep"
    elif moe_raw in ("false", "0"):
        moe_mode = "tp"
    else:
        moe_mode = moe_raw
    if not moe_mode and knob_map.moe_auto_ep_at and cap > 0:
        max_seqs = max(knob_map.decode_cap_floor, cap)
        lanes = _ints_val(_get_source(knobs, env, knob_map.concurrency))
        moe_mode = ("ep" if min(max(lanes or [max_seqs]), max_seqs)
                    >= knob_map.moe_auto_ep_at else "tp")
    return DecodeShape(
        cap=cap,
        spec_tokens=spec_tokens,
        dcp=dcp or knob_map.decode_dcp_default,
        extra_buckets=extra,
        moe_mode=moe_mode or "tp",
        env=env,
        a2a_edge=bucket_gate_edge(
            _int_val(env.get("TPU_DCP_INDEXER_A2A_MIN_TOKENS"))
            or _A2A_MIN_TOKENS),
        a2a_on=env.get("TPU_DCP_INDEXER_A2A_MERGE", "1").lower() not in _OFF,
        compile_sizes=frozenset(_ints_val(compile_raw)) or None,
        budget=_int_val(_get_source(knobs, env, knob_map.token_budget)),
    )


def compiled_decode_buckets(
    shape: DecodeShape,
    knob_map: KnobMap = GLM5_PD_KNOB_MAP,
) -> frozenset[int]:
    """Returns the compiled decode token buckets for `shape`."""
    if shape.compile_sizes is not None:
        buckets = set(shape.compile_sizes)
    else:
        buckets = set(_POWER_OF_TWO_BUCKETS)
        if shape.extra_buckets is not None:
            buckets |= {
                b
                for b in shape.extra_buckets if shape.legal_bucket(b)
            }
        else:
            buckets |= set(knob_map.buckets_without_extra)
            if shape.moe_mode == "ep" and knob_map.ep_buckets_from:
                buckets |= set(
                    range(knob_map.ep_buckets_from, shape.step_tokens + 16,
                          16))
    if shape.budget > 0:
        buckets = {b for b in buckets if b <= shape.budget} | {shape.budget}
    return frozenset(buckets)


def tp_decode_kernel_tokens(
    repo_dir: str | pathlib.Path | None, ) -> tuple[int, ...]:
    """Reads live `ROW_BUCKETS` from `repo_dir` first, else shared static fallback.

    Candidate branches in `/close-gaps` can extend `ROW_BUCKETS` in the
    checked-out `vllm-torchtpu` tree (for example to `(16, 32, 64, 80)`), so
    the live AST in `repo_dir` takes precedence over `serving_analysis.limits`.
    """
    if repo_dir is None:
        return _TP_DECODE_KERNEL_TOKENS_DEFAULT
    limits = code_limits.scan(repo_dir)
    for lim in limits:
        if lim.kind == "table" and lim.name in (
                "ROW_BUCKETS",
                "SUPPORTED_ROW_BUCKETS",
                "TP_DECODE_ROW_BUCKETS",
        ):
            return lim.values
    return _TP_DECODE_KERNEL_TOKENS_DEFAULT


def _decode_step_findings(
    knobs: Mapping[str, str],
    knob_map: KnobMap,
    tp_kernel_tokens: Sequence[int],
) -> list[Finding]:
    shape = decode_shape(knobs, knob_map)
    if shape is None:
        return []
    out: list[Finding] = []

    illegal = sorted(b for b in (shape.extra_buckets or ())
                     if not shape.legal_bucket(b))
    if illegal:
        fixed = sorted({shape.legal_round_up(b) for b in illegal})
        out.append(
            Finding(
                ERROR,
                "decode_bucket_dcp",
                (f"{knob_map.knob_label(knob_map.bucket_extra)} has "
                 f"{illegal}: above {shape.a2a_edge} tokens, DCP indexer "
                 f"all-to-all splits decode buckets across {shape.dcp} DCP "
                 f"ranks, so buckets not divisible by {shape.dcp} fail "
                 f"precompile. Use {fixed}."),
            ))

    if shape.cap > 0:
        step = shape.step_tokens
        buckets = compiled_decode_buckets(shape, knob_map)
        if step not in buckets:
            padded = min((b for b in buckets if b >= step), default=step)
            per_req = 1 + shape.spec_tokens
            fits = max((b // per_req for b in buckets if b < step), default=0)
            want = shape.legal_round_up(step)
            if want < padded and padded > step * (1.0 + _MIN_PADDING_WASTE):
                bsrc = (knob_map.compile_sizes
                        if shape.compile_sizes is not None
                        and knob_map.compile_sizes else knob_map.bucket_extra)
                out.append(
                    Finding(
                        WARNING,
                        "decode_bucket_padding",
                        (f"{knob_map.knob_label(knob_map.decode_cap)}="
                         f"{shape.cap} puts {step} tokens in a full decode "
                         f"step ({knob_map.spec_label} k={shape.spec_tokens}), "
                         f"which pads up to the {padded}-token bucket. Add "
                         f"{want} to {knob_map.knob_label(bsrc)} or cap at "
                         f"{fits}."),
                    ))

        # A step pads up to the next ROW_BUCKETS entry (48 runs in the 64
        # bucket), so only steps above the largest bucket leave the kernel.
        if (knob_map.tp_decode_kernel and shape.moe_mode == "tp"
                and tp_kernel_tokens and step > max(tp_kernel_tokens)):
            per_req = 1 + shape.spec_tokens
            max_fused_cap = max(tp_kernel_tokens) // per_req
            prefill_above = (_int_val(
                shape.env.get("FUSED_MOE_DECODE_KERNEL_MAX_TOKENS"))
                             or knob_map.fused_moe_decode_max_tokens)
            if step <= prefill_above:
                out.append(
                    Finding(
                        WARNING,
                        "decode_tp_kernel",
                        (f"{knob_map.knob_label(knob_map.decode_cap)}="
                         f"{shape.cap} with TP MoE puts {step} tokens in a "
                         f"full decode step, above the largest bucket in "
                         f"ROW_BUCKETS={tuple(sorted(tp_kernel_tokens))} while "
                         f"FUSED_MOE_DECODE_KERNEL_MAX_TOKENS={prefill_above} "
                         "blocks the prefill MoE kernel, causing a 2.5x "
                         "slower unfused GMM+psum fallback (-28% Mtok/$ 11 "
                         "min into one run). "
                         f"Extend ROW_BUCKETS to {step} (and verify via "
                         f"1-host microbench), keep cap <= {max_fused_cap}, "
                         "or switch to EP MoE."),
                    ))
    return out


def check_disagg_knobs(
    knobs: Mapping[str, str],
    tp_kernel_tokens: Sequence[int] = _TP_DECODE_KERNEL_TOKENS_DEFAULT,
    knob_map: KnobMap = GLM5_PD_KNOB_MAP,
) -> list[Finding]:
    """Checks disaggregated P/D knobs against known configuration hazards."""
    findings: list[Finding] = []
    offload_on = str(knobs.get("PREFILL_KV_OFFLOAD", "")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    try:
        offload_gib = float(knobs.get("PREFILL_KV_OFFLOAD_GIB", "0"))
    except ValueError:
        offload_gib = 0.0

    if offload_on and offload_gib >= 600:
        small_buckets = [
            tok for tok in _ints_val(knobs.get("PREFILL_COMPILE_SIZES", ""))
            if tok < 512
        ]
        if small_buckets:
            findings.append(
                Finding(
                    ERROR,
                    "host_ram_oom",
                    (f"PREFILL_COMPILE_SIZES includes buckets < 512 "
                     f"({small_buckets}) alongside "
                     f"PREFILL_KV_OFFLOAD_GIB={offload_gib:.0f}: host DRAM "
                     "OOMs. Use PREFILL_COMPILE_SIZES=512,1024,2048,4096."),
                ))
        for lease_key in ("PREFILL_STAGING_LEASES", "DECODE_STAGING_LEASES"):
            leases = _int_val(knobs.get(lease_key))
            if leases > 2:
                findings.append(
                    Finding(
                        WARNING,
                        "host_ram_oom",
                        (f"{lease_key}={leases} reserves >2 max-length KV "
                         "staging leases alongside a >=600 GiB CPU tier; "
                         "keep PREFILL_STAGING_LEASES=1 and "
                         "DECODE_STAGING_LEASES<=2."),
                    ))

    findings.extend(_decode_step_findings(knobs, knob_map, tp_kernel_tokens))

    batch_cap = _int_val(knobs.get("DECODE_BATCH_CAP"))
    decode_moe = knobs.get("DECODE_MOE", "").lower()
    if 0 < batch_cap <= 16 and decode_moe == "ep":
        findings.append(
            Finding(
                WARNING,
                "decode_moe_crossover",
                (f"DECODE_BATCH_CAP={batch_cap} with DECODE_MOE=ep: at caps "
                 "<=16 (<24 sequences), DECODE_MOE=tp is faster than ep."),
            ))
    elif batch_cap >= 32 and decode_moe == "tp":
        findings.append(
            Finding(
                WARNING,
                "decode_moe_crossover",
                (f"DECODE_BATCH_CAP={batch_cap} with DECODE_MOE=tp: at caps "
                 ">=32, DECODE_MOE=ep beats tp on throughput."),
            ))

    extra_args_decode = str(knobs.get("EXTRA_ARGS_DECODE", ""))
    max_inflight = _int_val(knobs.get("PROXY_DECODE_MAX_INFLIGHT"))
    if 0 < batch_cap < 16 and ("KvPrefetchAsyncScheduler" in extra_args_decode
                               or max_inflight > batch_cap):
        findings.append(
            Finding(
                ERROR,
                "kv_prefetch_cap_mismatch",
                (f"DECODE_BATCH_CAP={batch_cap} (<16) with "
                 f"PROXY_DECODE_MAX_INFLIGHT={max_inflight}: decode "
                 "--max-num-seqs floors at 16, overflowing target batch cap."),
            ))

    concurrencies = _ints_val(knobs.get("CLIENT_CONCURRENCIES", ""))
    num_p = max(
        1,
        _int_val(knobs.get("NUM_PREFILLS") or knobs.get("NUM_PREFILL") or "1"))
    num_d = max(
        1,
        _int_val(knobs.get("NUM_DECODES") or knobs.get("NUM_DECODE") or "1"))
    per_prefill_c = ((max(concurrencies) / num_p) if concurrencies else 48.0 /
                     num_p)

    dcp_size = _int_val(knobs.get("DCP_SIZE"))
    prefill_dcp = _int_val(knobs.get("PREFILL_DCP_SIZE")) or dcp_size
    decode_dcp = _int_val(knobs.get("DECODE_DCP_SIZE")) or dcp_size

    if 0 < prefill_dcp <= _SMALL_BATCH_DECODE_TARGET_DCP and per_prefill_c >= 32:
        sym_error = "PREFILL_DCP_SIZE" not in knobs and dcp_size > 0
        findings.append(
            Finding(
                ERROR if sym_error else WARNING,
                "dcp_size_prefill_kv_halved",
                (f"Prefill DCP={prefill_dcp} at {per_prefill_c:.0f} "
                 "sessions/prefill leg halves the prefill HBM prefix cache "
                 "(11.89M -> 5.95M tokens, HBM hit 91.9% -> 80.2%) and "
                 "lost 24.8% Mtok/$ at c=48 cap 16 (1.7% at c=32 cap 8) "
                 f"despite faster decode TPOT. Keep prefill at DCP "
                 f"{_MIN_ASYMMETRIC_PREFILL_DCP} and lower only decode: "
                 f"PREFILL_DCP_SIZE={_MIN_ASYMMETRIC_PREFILL_DCP} "
                 f"DECODE_DCP_SIZE={_SMALL_BATCH_DECODE_TARGET_DCP}."),
            ))

    if 0 < decode_dcp < prefill_dcp:
        merged_env = ";".join(
            str(knobs.get(k, ""))
            for k in ("EXTRA_ENV", "EXTRA_ENV_PREFILL", "EXTRA_ENV_DECODE"))
        if str(knobs.get("INDEXER_KV_CACHE_MODE", "")).lower() == "sal":
            findings.append(
                Finding(
                    ERROR,
                    "asym_dcp_indexer_layout",
                    (f"PREFILL_DCP_SIZE={prefill_dcp} DECODE_DCP_SIZE="
                     f"{decode_dcp} with INDEXER_KV_CACHE_MODE=sal: "
                     "SEQ_ALONG_LANE indexer cache lacks asymmetric byte "
                     "ranges. Use INDEXER_KV_CACHE_MODE=has."),
                ))
        if "TPU_KV_STRIPED_LAYOUT=1" in merged_env:
            findings.append(
                Finding(
                    ERROR,
                    "asym_dcp_striped_layout",
                    (f"PREFILL_DCP_SIZE={prefill_dcp} DECODE_DCP_SIZE="
                     f"{decode_dcp} with TPU_KV_STRIPED_LAYOUT=1: striped "
                     "KV layout conflicts with asymmetric P/D KV byte "
                     "ranges. Use TPU_KV_STRIPED_LAYOUT=0."),
                ))
        interleave = (_int_val(knobs.get("DECODE_INTERLEAVE"))
                      or _DEFAULT_DCP_INTERLEAVE)
        if interleave % _PACKED_ROW_TOKENS != 0:
            findings.append(
                Finding(
                    ERROR,
                    "asym_dcp_interleave",
                    (f"PREFILL_DCP_SIZE={prefill_dcp} DECODE_DCP_SIZE="
                     f"{decode_dcp} with DECODE_INTERLEAVE={interleave}: "
                     "asymmetric KV transfer requires whole packed rows of "
                     f"{_PACKED_ROW_TOKENS} tokens."),
                ))

    leg_cap = knob_map.extra_prefill_leg_max_concurrency
    if (leg_cap is not None and num_p > num_d and concurrencies
            and max(concurrencies) <= leg_cap):
        findings.append(
            Finding(
                WARNING,
                "extra_prefill_leg",
                (f"{num_p}P{num_d}D at CLIENT_CONCURRENCIES="
                 f"{max(concurrencies)}: adding a second prefill leg costs "
                 "50% more chips while a single decode leg caps throughput."),
            ))
    return findings


def _leg_dcp_and_tp(
    knob: str,
    knobs: Mapping[str, str],
    leg_commands: Mapping[str, str] | None,
) -> tuple[int, int]:
    cmd = leg_commands.get(knob, "") if leg_commands is not None else ""
    m_dcp = _CLI_DCP_RE.search(cmd)
    m_tp = _CLI_TP_RE.search(cmd)
    dcp = int(m_dcp.group(1)) if m_dcp else 0
    tp = int(m_tp.group(1)) if m_tp else 0
    if dcp == 0:
        if "PREFILL" in knob:
            dcp = _int_val(knobs.get("PREFILL_DCP_SIZE")) or _int_val(
                knobs.get("DCP_SIZE"))
        elif "DECODE" in knob:
            dcp = _int_val(knobs.get("DECODE_DCP_SIZE")) or _int_val(
                knobs.get("DCP_SIZE"))
        else:
            dcp = (_int_val(knobs.get("DCP_SIZE"))
                   or _int_val(knobs.get("DECODE_DCP_SIZE"))
                   or _int_val(knobs.get("PREFILL_DCP_SIZE")))
    if tp == 0:
        tp = _int_val(knobs.get("DECODE_TP_SIZE")) or _int_val(
            knobs.get("TP_SIZE"))
    return dcp, tp


def _vmem_absorb_q_findings(
    arm_knobs: Mapping[str, str],
    knobs: Mapping[str, str],
    conventions: run_history.KnobConventions,
    code_flags: Mapping[str, code_gap.EnvFlag] | None,
    leg_envs: Mapping[str, Mapping[str, str]] | None,
    leg_commands: Mapping[str, str] | None,
) -> list[Finding]:
    """Blocks `TPU_MLA_ABSORB_Q_FROM_HEAD=1` when 68 MiB > 64 MiB tpu7x VMEM."""
    out: list[Finding] = []
    for knob in conventions.extra_env:
        if knob not in arm_knobs:
            continue
        turned_on = code_gap.parse_extra_env(arm_knobs[knob])
        val = turned_on.get("TPU_MLA_ABSORB_Q_FROM_HEAD", "")
        if not val or val in _OFF:
            continue
        base_leg_env = leg_envs.get(knob) if leg_envs is not None else None
        eff_env = ({
            **base_leg_env,
            **turned_on
        } if base_leg_env is not None else dict(turned_on))
        if code_flags is not None and base_leg_env is not None:
            if code_gap.off_gate_chain("TPU_MLA_ABSORB_Q_FROM_HEAD",
                                       code_flags, eff_env):
                continue
        if ("TPU_MLA_DECODE_PROLOGUE" in eff_env
                and eff_env["TPU_MLA_DECODE_PROLOGUE"] in _OFF):
            continue
        q_rep = eff_env.get("VLLM_DCP_Q_REPLICATE")
        if q_rep is not None and q_rep in _OFF:
            continue
        dcp, tp = _leg_dcp_and_tp(knob, knobs, leg_commands)
        if dcp == 0:
            dcp = 16
        if dcp >= 16 and (tp == 0 or dcp >= tp):
            out.append(
                Finding(
                    ERROR,
                    "vmem_absorb_q_head",
                    (f"{knob}: TPU_MLA_ABSORB_Q_FROM_HEAD={val} at DCP={dcp} "
                     "(>=64 replicated local Q heads/rank) allocates 68 MiB "
                     "of Pallas windows in absorb_q_head, exceeding 64 MiB "
                     "tpu7x per-core VMEM at XLA compile time. Drop "
                     "TPU_MLA_ABSORB_Q_FROM_HEAD or run DECODE_DCP_SIZE <= 8."
                     ),
                ))
    return out


def changed_knobs(
    arm_knobs: Mapping[str, str],
    defaults: Mapping[str, str] | None = None,
    conventions: run_history.KnobConventions = run_history.DEFAULT_CONVENTIONS,
) -> list[str]:
    """Returns the performance knobs in `arm_knobs` that differ from `defaults`."""
    base = defaults or {}
    return sorted(
        k for k in arm_knobs
        if k not in conventions.non_perf and arm_knobs[k] != base.get(k))


def check_launch(
    recipe: str,
    arm_knobs: Mapping[str, str],
    history: Sequence[run_history.RunRecord] = (),
    code_flags: Mapping[str, code_gap.EnvFlag] | None = None,
    recipe_defaults: Mapping[str, str] | None = None,
    *,
    conventions: run_history.KnobConventions = run_history.DEFAULT_CONVENTIONS,
    repo_dir: str | pathlib.Path | None = None,
    ledger_entries: Sequence[ledger.Entry] = (),
    knob_map: KnobMap = GLM5_PD_KNOB_MAP,
    leg_envs: Mapping[str, Mapping[str, str]] | None = None,
    leg_commands: Mapping[str, str] | None = None,
    leg_verdict: legs_lib.Verdict | None = None,
) -> list[Finding]:
    """Runs all deterministic pre-launch hill-climbing checks on `arm_knobs`."""
    findings: list[Finding] = []
    defaults = dict(recipe_defaults or {})
    knobs = {**defaults, **arm_knobs}

    findings.extend(
        check_disagg_knobs(
            knobs,
            tp_kernel_tokens=tp_decode_kernel_tokens(repo_dir),
            knob_map=knob_map,
        ))

    if leg_verdict is not None:
        off_leg_msg = legs_lib.warn_off_leg_arm(
            leg_verdict, changed_knobs(arm_knobs, defaults, conventions))
        if off_leg_msg:
            findings.append(Finding(ERROR, "off_leg", off_leg_msg))

    arm_env: dict[str, str] = {}
    for knob in conventions.extra_env:
        if knob in arm_knobs:
            arm_env.update(code_gap.parse_extra_env(arm_knobs[knob]))

    if code_flags is not None:
        for flag in sorted(arm_env):
            if flag not in code_flags:
                is_serving = flag.startswith(("TPU_", "VLLM_"))
                findings.append(
                    Finding(
                        ERROR if is_serving else INFO,
                        "unknown_env_flag",
                        (f"{flag} is not in the served env registry: " +
                         ("typo or unported upstream flag."
                          if is_serving else "runtime/harness variable.")),
                    ))

    risky: list[tuple[str, str]] = []
    if code_flags is not None:
        for flag, val in sorted(arm_env.items()):
            entry = code_flags.get(flag)
            if entry is None or val in _OFF:
                continue
            warn = code_gap.hbm_risk(entry.doc)
            if warn:
                risky.append((flag, warn))
            if _NUMERICS_RE.search(entry.doc):
                findings.append(
                    Finding(
                        INFO,
                        "numerics",
                        f"{flag} may alter numerics; attach an accuracy gate.",
                    ))

    if code_flags is not None:
        shape = decode_shape(knobs, knob_map)
        inferred_ep = shape.moe_mode == "ep" if shape is not None else False
        for knob in conventions.extra_env:
            if knob not in arm_knobs:
                continue
            turned_on = code_gap.parse_extra_env(arm_knobs[knob])
            base_leg_env = (leg_envs.get(knob) if leg_envs is not None
                            and knob in leg_envs else {})
            eff_leg_env = {**base_leg_env, **turned_on}
            cmd = leg_commands.get(knob,
                                   "") if leg_commands is not None else ""
            if cmd:
                ep_on = "--enable-expert-parallel" in cmd
                check_moe = "vllm serve" in cmd or "--enable-expert-parallel" in cmd
            else:
                ep_on = inferred_ep
                check_moe = "DECODE" in knob.upper(
                ) or knob in knob_map.decode_env
            for flag, val in sorted(turned_on.items()):
                if val in _OFF or flag not in code_flags:
                    continue
                chain = code_gap.off_gate_chain(flag, code_flags, eff_leg_env)
                if chain:
                    why = code_gap.describe_gate_chain(chain, code_flags,
                                                       eff_leg_env)
                    findings.append(
                        Finding(
                            WARNING,
                            "gate_off",
                            f"{knob}: {flag}={val} is inert on this leg ({why}).",
                        ))
                if check_moe:
                    conflict = code_gap.moe_mode_conflict(
                        flag, code_flags, ep_on)
                    if conflict is not None:
                        findings.append(
                            Finding(
                                WARNING,
                                "moe_mode",
                                f"{knob}: {flag}={val} is inert ({conflict}).",
                            ))

    findings.extend(
        _vmem_absorb_q_findings(arm_knobs, knobs, conventions, code_flags,
                                leg_envs, leg_commands))

    for flag, warn in risky:
        findings.append(
            Finding(WARNING, "hbm_risk", f'{flag} is enabled: "{warn}"'))

    family = conventions.family(recipe)
    same_family = [
        r for r in history if conventions.family(r.recipe) == family
    ]
    for kv in conventions.kv_bytes:
        raw_val = knobs.get(kv)
        if raw_val is None:
            continue
        try:
            want = float(raw_val)
        except ValueError:
            findings.append(
                Finding(ERROR, "kv_budget", f"{kv}={raw_val} is not numeric."))
            continue
        ok_vals: list[float] = []
        failed_ids: list[str] = []
        for r in same_family:
            got_raw = r.knobs.get(kv, defaults.get(kv))
            try:
                got = float(got_raw) if got_raw is not None else None
            except ValueError:
                got = None
            if got is None:
                continue
            if r.succeeded:
                ok_vals.append(got)
            elif got >= want:
                failed_ids.append(r.run_id)
        if ok_vals and want > max(ok_vals):
            sev = ERROR if (failed_ids or risky) else WARNING
            findings.append(
                Finding(
                    sev,
                    "kv_budget",
                    (f"{kv}={raw_val} exceeds max succeeded run of {family} "
                     f"({max(ok_vals):.0f})."),
                ))

    base_knobs: Mapping[str, str] = {}
    if history:
        matches = run_history.find_baselines(recipe,
                                             arm_knobs,
                                             history,
                                             defaults=defaults,
                                             conventions=conventions)
        if matches:
            base_knobs = matches[0].run.knobs
    if ledger_entries:
        added = [
            s for s in lessons.settings_diff(arm_knobs, base_knobs, defaults,
                                             conventions)
            if not s.endswith("=")
        ]
        for prior in lessons.prior_outcomes(ledger_entries, added)[:3]:
            delta = (f" ({prior.delta_rel:+.1%})"
                     if prior.delta_rel is not None else "")
            neg = prior.decision in ledger.NEGATIVE
            findings.append(
                Finding(
                    WARNING if neg else INFO,
                    "prior_outcome",
                    (f"{', '.join(prior.settings)} was previously tested in "
                     f"{prior.run_id}: {prior.decision}{delta}" +
                     (f", as one of {prior.bundle_size} settings changed "
                      "together, so that outcome is not this setting's alone."
                      if prior.bundle_size > 1 else ".")),
                ))

    return findings


def render_findings(findings: Sequence[Finding]) -> str:
    """Formats `Finding` items as aligned text lines."""
    if not findings:
        return "PREFLIGHT PASS (0 findings)\n"
    has_err = any(f.severity == ERROR for f in findings)
    header = "PREFLIGHT BLOCKED" if has_err else "PREFLIGHT OK (with warnings/info)"
    lines = [header]
    for f in findings:
        lines.append(f"{f.severity:7s} {f.check:28s} {f.message}")
    return "\n".join(lines) + "\n"

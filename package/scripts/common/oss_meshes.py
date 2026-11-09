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
"""Enumerates and ranks vLLM serving meshes for OSS models.

Given an HF config.json, the device count and per-device HBM, this does the
mesh arithmetic the sharding skill used to leave to the LLM:

  * every TP x PP x PCP x DP with TP * PP * PCP * DP == devices, plus the
    decode-context-parallel (DCP) variants and, for MoE models, the EP
    variant. vLLM semantics (vllm/config/parallel.py): the
    engine world size is PP * TP * PCP per DP replica; DCP does not add
    devices, it re-uses the TP ranks (TP % DCP == 0) to shard the decode KV
    cache by token instead of replicating KV heads; the expert-parallel size
    is TP * PCP * DP when `--enable-expert-parallel` is set, so EP is not an
    independent axis either;
  * head divisibility for query, KV and global-KV heads (KV heads may be
    replicated when TP > KV heads) and vLLM's DCP rules for GQA models
    (TP > KV heads, DCP <= TP // KV heads, query heads per KV head % DCP);
  * an HBM estimate: weight bytes per device plus how many KV-cache tokens fit
    per engine replica versus what the benchmark needs;
  * a hand-curated table of kernel/engine constraints observed on real images
    (each entry records the image digest it was seen on, since it will go
    stale);
  * a diagnosis-driven ranking: a collective-bound prefill (serving detector
    D7) favors a smaller TP all-reduce group for prefill-heavy traffic
    (lower TP with more DP, or lower TP with PP so per-device weight bytes do
    not grow), decode-heavy traffic favors keeping TP; PP pays pipeline
    bubbles at low concurrency, DCP only helps when the KV cache (long
    context) is the limit, PCP only long prefills. PP / DCP / PCP have never
    been run on torch_tpu, so their candidates carry an "unverified" risk;
  * optional multi-point scoring (`points=[{isl, osl, c}, ...]`): each
    candidate gets a `per_point_assessment` (regime, score, expected
    direction, rationale) per benchmark point, and is flagged
    `workload_dependent` when the directions disagree. The D7 credit only
    applies to prefill-heavy points; at balanced / decode-heavy points a
    mesh that grows per-device weight bytes pays for the slower HBM-bound
    decode step (Gemma-4 TP4xDP2 -> TP2xDP4 measured +27.8% at 8k1k c128 and
    -24.9% at 1k1k c32);
  * a determinism risk on every candidate that changes TP or DP: the
    reduction order changes, so greedy outputs may differ from the baseline.
"""

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

# Per-device runtime reserve (activations, compile scratch). An estimate, not
# a vLLM setting; with it the free HBM matches a Gemma-3-27B TP 8 server log
# within 3% (oss_meshes_test).
DEFAULT_RESERVE_GIB = 5.0

# vLLM's default --gpu-memory-utilization (vllm/config/cache.py, CacheConfig),
# which onboarding does not override.
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92
# Ratio input_len / output_len at or above which traffic is prefill-heavy.
PREFILL_HEAVY_RATIO = 4.0
MAX_CANDIDATES = 3
# Sequences in flight per DP replica per pipeline stage below which PP's
# pipeline bubbles dominate (vLLM PP keeps up to PP batches in flight).
PP_MIN_SEQS_PER_STAGE = 8
# Tokens per sequence at or above which the decode KV read / KV capacity is
# treated as the limit DCP / PCP address.
LONG_CONTEXT_TOKENS = 32768
# Sequences per DP replica at or below which decode is treated as HBM-bound
# on weight reads (per-device weight bytes set the decode step time).
DECODE_HBM_BOUND_SEQS = 32

# Exact vLLM CLI flags (vllm/engine/arg_utils.py).
FLAG_TP = '--tensor-parallel-size'
FLAG_DP = '--data-parallel-size'
FLAG_PP = '--pipeline-parallel-size'
FLAG_DCP = '--decode-context-parallel-size'
FLAG_PCP = '--prefill-context-parallel-size'
FLAG_EP = '--enable-expert-parallel'
# Axes never run on torch_tpu (see `Mesh.unverified_axes`).
UNVERIFIED_AXES = ('pp', 'dcp', 'pcp')

_BYTES = {'bfloat16': 2, 'float16': 2, 'float32': 4, 'fp8': 1, 'int8': 1}


@dataclasses.dataclass(frozen=True)
class ModelShape:
    """The config.json fields the enumerator needs."""

    model_type: str = ''
    hidden_size: int = 0
    intermediate_size: int = 0
    num_layers: int = 0
    num_heads: int = 0
    num_kv_heads: int = 0
    head_dim: int = 0
    num_global_kv_heads: int | None = None
    global_head_dim: int | None = None
    vocab_size: int = 0
    tie_word_embeddings: bool = True
    num_experts: int | None = None
    moe_intermediate_size: int | None = None
    linear_num_value_heads: int | None = None
    # Multi-head latent attention (DeepSeek-style `kv_lora_rank`).
    use_mla: bool = False
    # Multimodal wrapper (vision / audio towers): vLLM PCP does not support it.
    is_multimodal: bool = False
    # Per-layer attention type: 'full_attention' | 'sliding_attention' |
    # 'linear_attention'.
    layer_types: tuple[str, ...] = ()
    # Tokens a 'sliding_attention' layer keeps per sequence.
    sliding_window: int | None = None
    dtype_bytes: int = 2

    @classmethod
    def from_hf_config(cls, cfg: Mapping[str, Any]) -> 'ModelShape':
        """Reads an HF config (multimodal configs use their `text_config`)."""
        t = dict(cfg.get('text_config') or cfg)

        def g(*names: str) -> Any:
            for n in names:
                if t.get(n) is not None:
                    return t[n]
            return None

        heads = int(g('num_attention_heads') or 0)
        hidden = int(g('hidden_size') or 0)
        num_layers = int(g('num_hidden_layers') or 0)
        layer_types = tuple(g('layer_types') or ())
        interval = g('full_attention_interval')
        if not layer_types and interval and g('linear_num_value_heads'):
            layer_types = tuple('full_attention' if (i + 1) %
                                int(interval) == 0 else 'linear_attention'
                                for i in range(num_layers))
        # transformers 5.17 moved Gemma 4's full-attention heads into
        # per_layer_config ({"05": {"num_key_value_heads": 4, ...}, ...}).
        full = [
            v for k, v in (g('per_layer_config') or {}).items()
            if str(k).isdigit() and int(k) < len(layer_types) and layer_types[
                int(k)] == 'full_attention' and isinstance(v, dict)
        ]
        dtype = str(g('torch_dtype', 'dtype') or cfg.get('torch_dtype') or '')
        experts = g('num_experts', 'num_local_experts', 'n_routed_experts')
        return cls(
            model_type=str(g('model_type') or cfg.get('model_type') or ''),
            hidden_size=hidden,
            intermediate_size=int(g('intermediate_size') or 0),
            num_layers=num_layers,
            num_heads=heads,
            num_kv_heads=int(g('num_key_value_heads') or heads),
            head_dim=int(g('head_dim') or (hidden // heads if heads else 0)),
            num_global_kv_heads=g('num_global_key_value_heads')
            or (full[0].get('num_key_value_heads') if full else None),
            global_head_dim=g('global_head_dim')
            or (full[0].get('head_dim') if full else None),
            # Gemma 3 text configs omit vocab_size; HF's default is 262208.
            vocab_size=int(g('vocab_size') or 262208),
            tie_word_embeddings=bool(
                cfg.get('tie_word_embeddings',
                        t.get('tie_word_embeddings', True))),
            num_experts=int(experts) if experts else None,
            moe_intermediate_size=g('moe_intermediate_size',
                                    'expert_intermediate_size'),
            linear_num_value_heads=g('linear_num_value_heads'),
            use_mla=bool(g('kv_lora_rank')),
            is_multimodal=any(
                cfg.get(k) for k in ('vision_config', 'audio_config')),
            layer_types=layer_types,
            sliding_window=int(g('sliding_window') or 0) or None,
            dtype_bytes=_BYTES.get(dtype, 2),
        )

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts and self.num_experts > 1)


@dataclasses.dataclass(frozen=True)
class Mesh:
    """A vLLM serving mesh.

  Devices = tp * pp * pcp * dp. `dcp` re-uses the TP ranks (tp % dcp == 0)
  and `ep` is tp * pcp * dp when set, so neither adds devices.
  """

    tp: int
    dp: int
    ep: int = 1
    pp: int = 1
    dcp: int = 1
    pcp: int = 1

    @property
    def devices(self) -> int:
        return self.tp * self.pp * self.pcp * self.dp

    @property
    def label(self) -> str:
        s = f'TP{self.tp}'
        s += f'xPCP{self.pcp}' if self.pcp > 1 else ''
        s += f'xPP{self.pp}' if self.pp > 1 else ''
        s += f'xDP{self.dp}'
        s += f'+DCP{self.dcp}' if self.dcp > 1 else ''
        return s + (f'+EP{self.ep}' if self.ep > 1 else '')

    @property
    def unverified_axes(self) -> list[str]:
        return [a for a in UNVERIFIED_AXES if getattr(self, a) > 1]

    def to_dict(self) -> dict[str, int]:
        return {
            'tp': self.tp,
            'dp': self.dp,
            'ep': self.ep,
            'pp': self.pp,
            'dcp': self.dcp,
            'pcp': self.pcp,
        }

    def vllm_args(self, baseline: 'Mesh | None' = None) -> dict[str, Any]:
        """vLLM CLI args; PP / DCP / PCP only when > 1 here or in `baseline`."""
        args: dict[str, Any] = {FLAG_TP: self.tp, FLAG_DP: self.dp}
        for axis, flag in (('pp', FLAG_PP), ('dcp', FLAG_DCP), ('pcp',
                                                                FLAG_PCP)):
            if getattr(self, axis) > 1 or (baseline
                                           and getattr(baseline, axis) > 1):
                args[flag] = getattr(self, axis)
        if self.ep > 1:
            args[FLAG_EP] = True
        return args

    def argv_set(self, baseline: 'Mesh | None' = None) -> dict[str, str]:
        """`candidates.json` argv_set (value flags; EP is an add-flag)."""
        return {
            k: str(v)
            for k, v in self.vllm_args(baseline).items()
            if not isinstance(v, bool)
        }


# ----------------------------------------------------------------------------
# Kernel / engine constraints observed on real images.
# ----------------------------------------------------------------------------
def _gdn_triangular_block_size(local_v_heads: int) -> int:
    # vllm_torchtpu GDN v3 kernel.
    return max(2, min(16, 16 // math.ceil(local_v_heads / 8)))


def _gdn_v3_block(m: ModelShape, mesh: Mesh) -> str | None:
    v = m.linear_num_value_heads
    if not v:
        return None
    if v % mesh.tp:
        return f'linear_num_value_heads={v} not divisible by tp={mesh.tp}'
    tbs = _gdn_triangular_block_size(v // mesh.tp)
    if 16 % tbs:
        return (f'GDN v3 triangular_block_size={tbs} (local value heads'
                f' {v // mesh.tp}) does not divide 16')
    return None


def _is_qwen38_27b(m: ModelShape) -> bool:
    return (m.model_type.startswith('qwen3_5')
            and m.linear_num_value_heads == 48 and m.hidden_size == 5120)


def _qwen38_tp8(m: ModelShape, mesh: Mesh) -> str | None:
    if _is_qwen38_27b(m) and mesh.tp == 8:
        return 'Qwen3.8-27B at TP8 fails engine init'
    return None


def _torchtpu_pp_with_tp(m: ModelShape, mesh: Mesh) -> str | None:
    del m
    if mesh.pp > 1 and mesh.tp > 1:
        return 'Pipeline parallelism on TPU does not support tensor parallelism inside a stage yet'
    return None


def _torchtpu_tp1(m: ModelShape, mesh: Mesh) -> str | None:
    del m
    if mesh.tp == 1:
        return 'torch_tpu has no device layout for TP1'
    return None


def _torchtpu_ep_row_partitions(m: ModelShape, mesh: Mesh) -> str | None:
    if mesh.ep > 1 and m.num_experts and m.num_experts > 16:
        return (
            f'EP with {m.num_experts} experts: num_row_partitions={m.num_experts}'
            ' > num_simd_lanes=16')
    return None


def _dp_hang(m: ModelShape, mesh: Mesh) -> str | None:
    del m
    if mesh.dp > 1:
        return f'DP={mesh.dp} engines hang on first requests'
    return None


def _vllm_pcp_limits(m: ModelShape, mesh: Mesh) -> str | None:
    """vLLM PCP runtime limits (Model Runner V2 pcp_manager / ParallelConfig)."""
    if mesh.pcp <= 1:
        return None
    why = []
    if not m.use_mla:
        why.append('PCP supports MLA models only')
    if m.is_multimodal:
        why.append('PCP does not support multimodal inputs')
    if mesh.pp > 1:
        why.append('PCP does not support PP')
    if mesh.dp > 1:
        why.append('PCP does not support DP')
    return '; '.join(why) or None


@dataclasses.dataclass(frozen=True)
class KernelConstraint:
    """A mesh limit observed on a real image (hand-curated; may go stale)."""

    id: str
    framework: str
    description: str
    check: Any  # Callable[[ModelShape, Mesh], str | None]
    failure_signature: str
    observed_on: tuple[str, ...]  # Image digests (prefixes) it was seen on.
    # True: the limit comes from the kernel/engine code and is assumed to hold
    # on any image until re-verified. False: only applies when the image
    # matches `observed_on`; otherwise it is a warning.
    structural: bool = True
    source: str = ''
    # TPU generations it holds on (e.g. ('v7',)); empty means every one. When
    # the caller names a generation outside this list, the constraint is
    # skipped; an unknown generation keeps it.
    accelerators: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop('check')
        return d


KERNEL_CONSTRAINTS: tuple[KernelConstraint, ...] = (
    KernelConstraint(
        id='gdn_v3_triangular_block',
        framework='vllm_torchtpu',
        description=(
            'Qwen3.5/3.8 Gated DeltaNet v3: TP must leave a local value-head'
            ' count whose triangular_block_size = max(2, min(16, 16 //'
            ' cdiv(v_local, 8))) divides 16 (TP2 of 48 heads fails).'),
        check=_gdn_v3_block,
        failure_signature='array split does not result in an equal division',
        observed_on=('sha256:107753fc', ),
        source='Qwen3.8-27B TP2 serve, image sha256:107753fc, 2026-10-14',
    ),
    KernelConstraint(
        id='qwen38_tp8_engine_init',
        framework='vllm_torchtpu',
        description='Qwen3.8-27B at TP8 crashes during engine init.',
        check=_qwen38_tp8,
        failure_signature="Worker failed with error '(384, 256)'",
        observed_on=('sha256:107753fc', 'sha256:8e22f91d'),
        # An engine-init crash on two images, not a rule of the kernel code.
        structural=False,
        source=(
            'Qwen3.8-27B TP8xDP1 serves, image sha256:107753fc, 2026-10-14,'
            ' and sha256:8e22f91d with --attention-backend'
            ' CUSTOM, 2026-10-16'),
    ),
    KernelConstraint(
        id='torchtpu_pp_with_tp_unsupported',
        framework='vllm_torchtpu',
        description=(
            'Pipeline parallelism on TPU does not support tensor parallelism '
            'inside a stage yet.'),
        check=_torchtpu_pp_with_tp,
        failure_signature=
        'NotImplementedError: Pipeline parallelism on TPU does not support tensor parallelism inside a stage yet',
        observed_on=('sha256:107753fc', ),
        source='Qwen3.8-27B TP4xPP2 serve, image sha256:107753fc, 2026-10-15',
    ),
    KernelConstraint(
        id='torchtpu_tp1_no_layout',
        framework='vllm_torchtpu',
        description=(
            'On tpu7x, torch_tpu has no single-device (TP1) layout: its v7'
            ' topology table starts at 2 devices (one chip), so every model'
            ' fails worker init at TP1 (Qwen3.8-27B, Gemma-4-31B). v5e, v5p'
            ' and v6e have a 1-device topology.'),
        check=_torchtpu_tp1,
        failure_signature='No TPU topology found for count: 1',
        observed_on=('sha256:107753fc', ),
        # Chip-specific (torch_tpu's per-generation topology tables): a
        # single-device v5e / v6e slice has a layout.
        accelerators=('v7', ),
        source=('Qwen3.8-27B and Gemma-4-31B TP1 serves on tpu7x, image'
                ' sha256:107753fc, 2026-10-13 and 2026-10-14; torch_tpu'
                ' _internal/utils/hardware.py, the per-generation topologies'),
    ),
    KernelConstraint(
        id='torchtpu_ep_row_partitions',
        framework='vllm_torchtpu',
        description=(
            'Expert parallelism with more than 16 experts trips the MoE'
            ' kernel row-partition limit (gpt-oss-20b, 32 experts).'),
        check=_torchtpu_ep_row_partitions,
        failure_signature=(
            'AssertionError: num_row_partitions=32 must be <= num_simd_lanes=16'
        ),
        observed_on=('sha256:ec855bc5', ),
        # Seen on one image only; TP8 + EP8 served on sha256:581bcde5.
        structural=False,
        source=
        'gpt-oss-20b TP2xDP4 + EP serve, image sha256:ec855bc5, 2026-10-14',
    ),
    KernelConstraint(
        id='dp_engine_hang',
        framework='vllm_torchtpu',
        description=(
            'With data parallelism (DP>1) the server starts healthy, then the'
            ' first requests hang.'),
        check=_dp_hang,
        failure_signature='TimeoutError: RPC call ... timed out',
        observed_on=('sha256:581bcde5', 'sha256:ec855bc5'),
        structural=False,
        source=('Gemma-3-27B TP4xDP2 on vLLM 0.22.1 (sha256:581bcde5) and'
                ' gpt-oss-20b (sha256:ec855bc5) serves, 2026-10-14'),
    ),
    KernelConstraint(
        id='vllm_pcp_limits',
        framework='vllm',
        description=(
            'Prefill context parallelism runs only on Model Runner V2 for'
            ' MLA models, without PP, DP or multimodal inputs.'),
        check=_vllm_pcp_limits,
        failure_signature='NotImplementedError: MRV2 PCP currently supports'
        ' MLA models only.',
        observed_on=(),
        source=('vLLM v0.26.1rc0:'
                ' v1/worker/gpu/pcp_manager.py, config/parallel.py,'
                ' config/vllm.py _get_v2_model_runner_unsupported_features;'
                ' re-check on newer images'),
    ),
)


def _constraint_applies(c: KernelConstraint, framework: str) -> bool:
    # 'vllm' constraints are engine semantics shared by every vLLM backend.
    return (not framework or c.framework == framework
            or framework.startswith(c.framework + '_'))


def generation(accelerator: str | None) -> str | None:
    """'tpu7x' / 'v7x' / 'tpu-v6e-slice' / 'v5litepod' -> 'v7' / 'v6e' / 'v5e'."""
    if not accelerator:
        return None
    a = accelerator.lower()
    if 'litepod' in a or 'v5e' in a:
        return 'v5e'
    for gen in ('v7', 'v6e', 'v5p', 'v4'):
        if gen in a:
            return gen
    if re.search(r'tpu7|7x', a):
        return 'v7'
    return None


def _image_matches(image: str | None, digests: Sequence[str]) -> bool:
    if not image:
        return False
    img = image.lower()
    if '@' in img:
        img = img.split('@', 1)[1]
    if not img.startswith('sha256:') and re.fullmatch(r'[0-9a-f]{8,64}', img):
        img = 'sha256:' + img
    # The table records at least 8 hex characters; a shorter value would
    # match digests it does not name.
    if not re.fullmatch(r'sha256:[0-9a-f]{8,64}', img):
        return False
    return any(img.startswith(d) for d in digests)


# ----------------------------------------------------------------------------
# Divisibility and HBM.
# ----------------------------------------------------------------------------
def head_divisibility(m: ModelShape, mesh: Mesh) -> list[str]:
    """Errors for query / KV / global-KV heads and experts under `mesh`."""
    errors = []
    if m.num_heads and m.num_heads % mesh.tp:
        errors.append(f'num_attention_heads={m.num_heads} % tp={mesh.tp} != 0')
    for name, kv in (
        ('num_key_value_heads', m.num_kv_heads),
        ('num_global_key_value_heads', m.num_global_kv_heads),
    ):
        if kv and kv % mesh.tp and mesh.tp % kv:
            errors.append(
                f'{name}={kv} neither divides nor is divisible by tp')
    if mesh.ep > 1 and m.num_experts and m.num_experts % mesh.ep:
        errors.append(f'num_experts={m.num_experts} % ep={mesh.ep} != 0')
    errors += _dcp_errors(m, mesh)
    return errors


def _dcp_errors(m: ModelShape, mesh: Mesh) -> list[str]:
    """vLLM decode-context-parallel rules (config/parallel.py, config/model.py).

  Without PCP, DCP re-uses the TP ranks; for GQA/MQA it shards the KV heads
  TP would otherwise replicate, so it needs TP > KV heads.

  Args:
    m: Model shape.
    mesh: Candidate mesh.
  """
    if mesh.dcp <= 1:
        return []
    errors = []
    if mesh.pcp == 1 and mesh.tp % mesh.dcp:
        errors.append(f'tp={mesh.tp} must be divisible by dcp={mesh.dcp}')
    elif mesh.pcp > 1 and mesh.dcp not in (mesh.pcp, mesh.tp * mesh.pcp):
        errors.append(f'with pcp={mesh.pcp}, dcp must be pcp or tp*pcp')
    if m.use_mla or not m.num_kv_heads:
        return errors
    kv = m.num_kv_heads
    if mesh.tp <= kv:
        errors.append(
            f'DCP for GQA needs tp={mesh.tp} > num_key_value_heads={kv}')
    elif mesh.dcp > mesh.tp // kv:
        errors.append(
            f'dcp={mesh.dcp} > tp // num_key_value_heads={mesh.tp // kv}')
    if m.num_heads and (m.num_heads // kv) % mesh.dcp:
        errors.append(
            f'query heads per KV head ({m.num_heads // kv}) % dcp={mesh.dcp} != 0'
        )
    return errors


def _attn_params_per_layer(m: ModelShape, kv_heads: int, head_dim: int) -> int:
    q = m.num_heads * head_dim
    return 2 * m.hidden_size * q + 2 * m.hidden_size * kv_heads * head_dim


def weight_bytes(m: ModelShape) -> tuple[float, float]:
    """(dense bytes, expert bytes) from config shapes (no quantization)."""
    layer_types = m.layer_types or ('full_attention', ) * m.num_layers
    attn = 0
    for lt in layer_types:
        if lt == 'full_attention' and m.num_global_kv_heads:
            attn += _attn_params_per_layer(m, m.num_global_kv_heads,
                                           m.global_head_dim or m.head_dim)
        else:
            # Linear-attention (GDN) layers are approximated with the full
            # attention projection count.
            attn += _attn_params_per_layer(m, m.num_kv_heads, m.head_dim)
    emb = m.vocab_size * m.hidden_size * (1 if m.tie_word_embeddings else 2)
    experts = 0
    mlp = 3 * m.hidden_size * m.intermediate_size * m.num_layers
    if m.is_moe:
        inter = m.moe_intermediate_size or m.intermediate_size
        experts = 3 * m.hidden_size * int(inter) * int(m.num_experts or 0)
        experts *= m.num_layers
        mlp = 0
    b = m.dtype_bytes
    return float((attn + mlp + emb) * b), float(experts * b)


def kv_bytes_per_token_per_device(m: ModelShape, mesh: Mesh) -> float:
    """KV-cache bytes per token on one device, as if every layer kept it.

  PP splits the layers over the stages; DCP shards the (otherwise
  replicated) KV heads by token; PCP does not shard the KV cache.

  Args:
    m: Model shape.
    mesh: Candidate mesh.
  """
    layer_types = m.layer_types or ('full_attention', ) * m.num_layers
    total = 0.0
    for lt in layer_types:
        if lt == 'linear_attention':
            continue  # Fixed-size recurrent state, not per token.
        if lt == 'full_attention' and m.num_global_kv_heads:
            kv, hd = m.num_global_kv_heads, m.global_head_dim or m.head_dim
        else:
            kv, hd = m.num_kv_heads, m.head_dim
        local = max(1.0, kv / mesh.tp)
        total += 2 * local * hd * m.dtype_bytes
    return total / (mesh.pp * mesh.dcp)


def kv_bytes_per_sequence(m: ModelShape, mesh: Mesh, seq_len: int) -> float:
    """KV-cache bytes one sequence of `seq_len` tokens takes on one device.

    A sliding-window layer keeps only its window, as vLLM's hybrid KV-cache
    manager allocates it; every other attention layer keeps the whole
    sequence. Linear-attention layers hold a fixed-size state instead.
    """
    layer_types = m.layer_types or ('full_attention', ) * m.num_layers
    total = 0.0
    for lt in layer_types:
        if lt == 'linear_attention':
            continue
        if lt == 'full_attention' and m.num_global_kv_heads:
            kv, hd = m.num_global_kv_heads, m.global_head_dim or m.head_dim
        else:
            kv, hd = m.num_kv_heads, m.head_dim
        tokens = seq_len
        if lt == 'sliding_attention' and m.sliding_window:
            tokens = min(seq_len, m.sliding_window)
        total += 2 * max(1.0, kv / mesh.tp) * hd * m.dtype_bytes * tokens
    return total / (mesh.pp * mesh.dcp)


@dataclasses.dataclass
class HbmEstimate:
    weights_gib_per_device: float
    kv_bytes_per_token_per_device: float
    kv_tokens_per_replica: int
    kv_tokens_needed_per_replica: int | None
    fits_weights: bool
    kv_ok: bool | None
    # HBM per device left for the KV cache, and what the target point's
    # sequences take of it (sliding windows counted).
    kv_gib_free: float = 0.0
    kv_gib_needed: float | None = None


def hbm_estimate(
    m: ModelShape,
    mesh: Mesh,
    hbm_gib: float,
    *,
    weights_gib: float | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    max_len: int | None = None,
    concurrency: int | None = None,
) -> HbmEstimate:
    """Weight bytes per device and KV tokens per replica vs. the need.

  Args:
    m: Model shape.
    mesh: Candidate mesh.
    hbm_gib: HBM per device.
    weights_gib: Measured total weight size (overrides the config estimate).
    gpu_memory_utilization: vLLM --gpu-memory-utilization.
    reserve_gib: Per-device runtime reserve.
    max_len: Tokens per sequence (input + output) at the target point.
    concurrency: Concurrent sequences at the target point (all replicas).
  """
    dense, experts = weight_bytes(m)
    if weights_gib is not None:
        if dense + experts > 0:
            scale = weights_gib * 2**30 / (dense + experts)
            dense, experts = dense * scale, experts * scale
        else:
            # No shape estimate (e.g. a config without hidden_size): split the
            # measured weights as dense, over TP.
            dense, experts = weights_gib * 2**30, 0.0
    # PP splits the layers (embedding / LM head approximated as evenly split).
    per_dev = (dense / mesh.tp + experts /
               (mesh.ep if mesh.ep > 1 else mesh.tp)) / mesh.pp
    w_gib = per_dev / 2**30
    free = hbm_gib * gpu_memory_utilization - w_gib - reserve_gib
    kv_tok = kv_bytes_per_token_per_device(m, mesh)
    cap = int(max(free, 0.0) * 2**30 / kv_tok) if kv_tok else 0
    need = need_gib = None
    if max_len and concurrency:
        seqs = math.ceil(concurrency / mesh.dp)
        need = seqs * max_len
        need_gib = seqs * kv_bytes_per_sequence(m, mesh, max_len) / 2**30
    return HbmEstimate(
        weights_gib_per_device=round(w_gib, 2),
        kv_bytes_per_token_per_device=kv_tok,
        kv_tokens_per_replica=cap,
        kv_tokens_needed_per_replica=need,
        fits_weights=free > 0,
        kv_ok=None if need_gib is None else max(free, 0.0) >= need_gib,
        kv_gib_free=round(max(free, 0.0), 2),
        kv_gib_needed=None if need_gib is None else round(need_gib, 2),
    )


# ----------------------------------------------------------------------------
# Enumeration and ranking.
# ----------------------------------------------------------------------------
def regime_from_lengths(input_len: int | None, output_len: int | None) -> str:
    if not input_len or not output_len:
        return 'balanced'
    r = input_len / output_len
    if r >= PREFILL_HEAVY_RATIO:
        return 'prefill_heavy'
    if r <= 1 / PREFILL_HEAVY_RATIO:
        return 'decode_heavy'
    return 'balanced'


def prefill_collective_bound(serving: Mapping[str, Any] | None) -> bool:
    """True if detector D7 flagged a prefill / mixed step in the serving output.

  Accepts a single-profile `serving_analyze_main` summary or its `--case`
  output.

  Args:
    serving: Parsed serving_analyze_main JSON.
  """
    if not serving:
        return False
    results = [p.get('result') or {} for p in serving.get('profiles') or []]
    results = results or [serving]
    for r in results:
        for f in r.get('findings') or []:
            if str(f.get('detector_id', '')).startswith('D7_') and (
                    'prefill' in str(f.get('phase', ''))
                    or 'mixed' in str(f.get('phase', ''))):
                return True
    return False


def kv_head_padding(serving: Mapping[str, Any] | None) -> int | None:
    """The KV packing factor if detector D8 flagged padded KV heads, else None.

  D8 fires when a device's KV-cache rows (K and V interleaved: two per KV
  head, one at head_dim 64) do not fill the packing factor (2 for bf16, 4 for
  fp8), or TP exceeds the KV heads, so attention reads padded or replicated
  rows.

  Args:
    serving: Parsed serving_analyze_main JSON (single or --case).
  """
    if not serving:
        return None
    results = [p.get('result') or {} for p in serving.get('profiles') or []]
    results = results or [serving]
    for r in results:
        for f in r.get('findings') or []:
            if not str(f.get('detector_id', '')).startswith('D8_'):
                continue
            for e in f.get('evidence') or []:
                # findings.json writes the evidence label as `metric`.
                label = e.get('metric') or e.get('name')
                if label == 'KV packing factor' and e.get('value'):
                    return int(e['value'])
            # No packing in the evidence: unknown (it depends on the KV
            # dtype), so the KV-row term is not scored rather than guessed.
            return None
    return None


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def all_meshes(m: ModelShape, devices: int) -> list[Mesh]:
    """Every TP x PP x PCP x DP == devices, with DCP and EP variants."""
    out = []
    for tp in _divisors(devices):
        for pp in _divisors(devices // tp):
            for pcp in _divisors(devices // (tp * pp)):
                dp = devices // (tp * pp * pcp)
                # DCP re-uses TP ranks (PCP=1); DCP over PCP is left out.
                dcps = [d for d in _divisors(tp) if d > 1] if pcp == 1 else []
                for dcp in [1] + dcps:
                    out.append(Mesh(tp, dp, pp=pp, dcp=dcp, pcp=pcp))
                    ep = tp * pcp * dp
                    if m.is_moe and ep > 1:
                        out.append(Mesh(tp, dp, ep, pp=pp, dcp=dcp, pcp=pcp))
    return out


def evaluate_mesh(
    m: ModelShape,
    mesh: Mesh,
    *,
    hbm_gib: float,
    framework: str,
    image: str | None,
    ignore: Sequence[str] = (),
    accelerator: str | None = None,
    **hbm_kwargs: Any,
) -> dict[str, Any]:
    """Feasibility record for one mesh."""
    gen = generation(accelerator)
    rejected, warnings = [], []
    for err in head_divisibility(m, mesh):
        rejected.append({'id': 'head_divisibility', 'reason': err})
    for c in KERNEL_CONSTRAINTS:
        if c.id in ignore or not _constraint_applies(c, framework):
            continue
        if c.accelerators and gen and gen not in c.accelerators:
            continue
        why = c.check(m, mesh)
        if not why:
            continue
        entry = {
            'id': c.id,
            'reason': why,
            'failure_signature': c.failure_signature,
            'observed_on': list(c.observed_on),
        }
        if c.structural or _image_matches(image, c.observed_on):
            rejected.append(entry)
        else:
            entry['reason'] += ' (seen on other images; verify on this one)'
            warnings.append(entry)
    hbm = hbm_estimate(m, mesh, hbm_gib, **hbm_kwargs)
    if not hbm.fits_weights:
        rejected.append({
            'id':
            'hbm',
            'reason':
            (f'weights {hbm.weights_gib_per_device} GiB/device leave no KV'
             ' cache room'),
        })
    elif hbm.kv_ok is False:
        warnings.append({
            'id':
            'kv_capacity',
            'reason':
            (f'KV cache needs {hbm.kv_gib_needed:g} GiB per device at this'
             f' point, {hbm.kv_gib_free:g} GiB free: preemption risk'),
        })
    return {
        'mesh': mesh.to_dict(),
        'label': mesh.label,
        'feasible': not rejected,
        'rejected_by': rejected,
        'warnings': warnings,
        'risks': _unverified_risks(mesh),
        'hbm': dataclasses.asdict(hbm),
    }


def _unverified_risks(mesh: Mesh) -> list[str]:
    return [
        f'{a.upper()}={getattr(mesh, a)} unverified on torch_tpu: never run on'
        ' this stack (engine init, compile and accuracy untested)'
        for a in mesh.unverified_axes
    ]


@dataclasses.dataclass(frozen=True)
class _Context:
    """Inputs to the ranking that do not depend on the candidate."""

    baseline: Mesh
    regime: str
    prefill_bound: bool
    # Whether the KV estimate is trusted (the baseline itself fits); only then
    # does a KV-capacity warning lower the score.
    trust_kv: bool
    # The baseline is short of KV cache (kv_capacity warning).
    baseline_kv_short: bool
    # No feasible non-baseline TP x DP (x EP) mesh exists.
    no_tpdp_alternative: bool
    max_len: int | None = None
    input_len: int | None = None
    concurrency: int | None = None
    # D8: the KV packing factor when the trace shows padded KV heads, and the
    # model's KV head count.
    kv_packing: int | None = None
    kv_heads: int = 0
    head_dim: int = 0
    # Full-attention layers' own KV heads (Gemma-3/4), when the model has them.
    global_kv_heads: int = 0
    global_head_dim: int = 0


def _kv_row_waste(kv_heads: int, tp: int, packing: int,
                  head_dim: int) -> float:
    """Share of the KV bytes a device reads that are padding or replicas.

    The RPA KV cache interleaves K and V: two rows per KV head, padded up to
    the dtype packing (tpu-inference / vllm-torchtpu RPA v3
    `get_kv_cache_shape`: `align_to(num_kv_heads * 2, packing)`), and one row
    per head at head_dim 64 (vllm-torchtpu `kernel_hd64.py` keeps K and V
    side by side in one 128-wide row); at least one head per device
    (replicated when TP exceeds the KV heads). Same rule as detector D8.
    """
    per_dev = kv_heads / tp
    rows = max(per_dev, 1.0) * (1 if head_dim == 64 else 2)
    padded = math.ceil(rows / packing) * packing
    return 1.0 - rows / padded * min(per_dev, 1.0)


def _kv_waste(ctx: _Context, tp: int) -> float:
    """The worse KV-row waste of the local heads and, if any, the global heads.

    Sliding / local layers and the global (full-attention) layers' own KV heads
    (Gemma-3/4) are separate caches: a layout fills the packing only if both
    do.
    """
    groups = [(ctx.kv_heads, ctx.head_dim)]
    if ctx.global_kv_heads:
        groups.append((ctx.global_kv_heads, ctx.global_head_dim
                       or ctx.head_dim))
    return max(_kv_row_waste(h, tp, ctx.kv_packing or 2, d) for h, d in groups)


def _score(
        mesh: Mesh, ctx: _Context,
        warnings: Sequence[Mapping[str, Any]]) -> tuple[float, str, list[str]]:
    """Ranks a feasible non-baseline mesh; returns (score, rationale, risks).

  TP x DP (x EP) terms:
    * -0.25 per TP doubling or halving vs. the baseline, and -0.25 for
      turning expert parallelism on or off (one flag; EP = TP x PCP x DP
      follows from the mesh) (change cost);
    * D7 (collective-bound prefill) on prefill-heavy / balanced traffic:
      +1 for one TP halving (smaller all-reduce group), -1 per TP doubling;
    * decode-heavy: -1 per TP step; otherwise -0.5 per TP step;
    * D8 (padded KV rows): +1 when the mesh leaves no KV-cache padding or
      replicas per device (K and V interleaved, two rows per head, one at
      head_dim 64) and the baseline does not; the per-step cost keeps the
      largest such TP first;
    * +0.5 for adding EP; -1 for a KV-capacity warning (if trusted).
  PP / DCP / PCP terms (all "unverified on torch_tpu": -0.25 flat):
    * PP: -0.25 per PP doubling (pipeline bubbles, stage-hop latency); -1
      more when fewer than PP_MIN_SEQS_PER_STAGE sequences per stage are in
      flight per replica; +0.75 when it lets TP shrink without growing the
      per-device weight bytes (TP x PP >= baseline TP x PP), i.e. the D7
      gain without the decode regression of lower TP alone; +0.75 when it
      only replaces DP at the same TP but the diagnosis (D7 or a short KV
      cache) asks for a change and no TP x DP alternative is feasible.
    * DCP: +0.75 when the KV cache is the limit (baseline KV warning or
      sequences >= LONG_CONTEXT_TOKENS) on decode-heavy / balanced traffic.
    * PCP: +0.75 for prefill-heavy traffic with inputs >= LONG_CONTEXT_TOKENS.

  Args:
    mesh: Candidate mesh.
    ctx: Baseline, regime and diagnosis.
    warnings: Non-fatal warnings for the mesh.
  """
    base = ctx.baseline
    diff = math.log2(mesh.tp / base.tp)  # <0: lower TP.
    score = -0.25 * abs(diff) - 0.25 * (mesh.ep != base.ep)
    risks = []
    whys = []
    if ctx.prefill_bound and ctx.regime in ('prefill_heavy', 'balanced'):
        # Fewer TP ranks shrink the prefill all-reduce; more DP replicas add
        # prefill capacity. One halving captures most of it.
        score += min(-diff, 1.0) if diff < 0 else -diff
        if diff < 0:
            whys.append(
                'D7: prefill is collective-bound; lower TP shortens the all-reduce'
                ' group' +
                (' and more DP replicas add prefill capacity' if mesh.dp >
                 base.dp else ''))
        else:
            whys.append(
                'D7: prefill is collective-bound; this mesh does not shrink the TP'
                ' all-reduce group')
    elif ctx.regime == 'decode_heavy':
        score -= abs(diff)
        whys.append(
            'decode-heavy traffic: keep TP unless memory forces a change')
    else:
        score -= 0.5 * abs(diff)
        whys.append('no collective-bound prefill; mesh change is exploratory')
    if ctx.kv_packing and ctx.kv_heads:
        # Sliding / local layers and, when the model has them, the global
        # (full-attention) layers' own KV heads: a layout fills the packing
        # only if both do.
        base_waste = _kv_waste(ctx, base.tp)
        waste = _kv_waste(ctx, mesh.tp)
        if base_waste > 0 >= waste:
            score += 1.0
            whys.append(
                f'D8: {ctx.kv_heads / mesh.tp:g} KV heads per device fill the'
                f' KV packing of {ctx.kv_packing} (baseline'
                f' {ctx.kv_heads / base.tp:g} heads, {base_waste:.0%} padding'
                ' or replicas), so attention reads no padded KV rows')
    # Per-device weight bytes scale with 1 / (TP x PP).
    weight_ratio = (base.tp * base.pp) / (mesh.tp * mesh.pp)
    if weight_ratio > 1:
        risks.append(
            f'decode-regressing: per-device weight bytes x{weight_ratio:.0f} on'
            ' an HBM-bound decode step; low-concurrency and decode-heavy points'
            ' may slow down')
    if mesh.ep > base.ep:
        score += 0.5
        whys.append('expert parallelism removes per-expert TP splits')

    if mesh.unverified_axes:
        score -= 0.25
    if mesh.pp != base.pp:
        score -= 0.25 * abs(math.log2(mesh.pp / base.pp))
    if mesh.pp > 1:
        per_stage = (ctx.concurrency / mesh.dp /
                     mesh.pp if ctx.concurrency else None)
        if per_stage is not None and per_stage < PP_MIN_SEQS_PER_STAGE:
            score -= 1.0
            risks.append(
                f'PP{mesh.pp}: only {per_stage:.1f} sequences per stage in flight;'
                ' pipeline bubbles dominate at this concurrency')
        else:
            risks.append(
                f'PP{mesh.pp}: pipeline bubbles at low concurrency and one stage'
                ' hop per layer group on every decode step')
    if mesh.pp > base.pp:
        if diff < 0 and weight_ratio <= 1:
            score += 0.75
            whys.append(
                f'PP{mesh.pp} keeps per-device weight bytes at or below the'
                ' baseline while TP shrinks, so the smaller all-reduce group comes'
                ' without the decode regression of lower TP alone')
        elif (diff == 0 and (ctx.prefill_bound or ctx.baseline_kv_short)
              and ctx.no_tpdp_alternative):
            score += 0.75
            whys.append(
                f'exploratory: every other TP x DP mesh is infeasible; PP{mesh.pp}'
                f' replaces DP at TP{mesh.tp} (same all-reduce group), halving'
                ' per-device weights for more KV cache and a single engine'
                ' instead of DP replicas')
    long_ctx = bool(ctx.max_len and ctx.max_len >= LONG_CONTEXT_TOKENS)
    if mesh.dcp > base.dcp:
        if ctx.regime in ('decode_heavy',
                          'balanced') and (ctx.baseline_kv_short or long_ctx):
            score += 0.75
            whys.append(
                f'DCP{mesh.dcp} shards the replicated KV heads by token: more KV'
                ' cache and smaller per-device KV reads for long-context decode'
            )
        risks.append(
            f'DCP{mesh.dcp}: extra all-gather / reduce-scatter per attention'
            ' layer on decode')
    if mesh.pcp > base.pcp:
        if (ctx.regime == 'prefill_heavy' and ctx.input_len
                and ctx.input_len >= LONG_CONTEXT_TOKENS):
            score += 0.75
            whys.append(f'PCP{mesh.pcp} splits long prefills across ranks')
    risks += _unverified_risks(mesh)
    for w in warnings:
        if w['id'] == 'kv_capacity' and ctx.trust_kv:
            score -= 1.0
        risks.append(w['reason'])
    return score, '; '.join(whys), risks


def _len_label(n: int) -> str:
    return f'{n // 1024}k' if n % 1024 == 0 else str(n)


@dataclasses.dataclass(frozen=True)
class Point:
    """One benchmark point (input / output length, concurrency)."""

    isl: int
    osl: int
    c: int
    label: str = ''

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'Point':
        """Accepts {isl, osl, c} or {input_len, output_len, concurrency}."""

        def get(*names: str) -> int:
            for n in names:
                if d.get(n) is not None:
                    return int(d[n])
            raise ValueError(f'benchmark point {dict(d)} lacks one of {names}')

        isl = get('isl', 'input_len')
        osl = get('osl', 'output_len')
        c = get('c', 'concurrency')
        if min(isl, osl, c) <= 0:
            raise ValueError(f'benchmark point {dict(d)} must be positive')
        a, b = _len_label(isl), _len_label(osl)
        sep = '' if a.endswith('k') and b.endswith('k') else '/'
        label = str(d.get('label') or f'{a}{sep}{b} c{c}')
        return cls(isl, osl, c, label)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def determinism_risk(mesh: Mesh, baseline: Mesh) -> str | None:
    """Risk line for meshes that change TP or DP (greedy output drift)."""
    if (mesh.tp, mesh.dp) == (baseline.tp, baseline.dp):
        return None
    return (f'reduction order changes (TP{baseline.tp}xDP{baseline.dp} ->'
            f' TP{mesh.tp}xDP{mesh.dp}): greedy outputs may differ from the'
            ' baseline; check greedy parity before comparing quality (Gemma-4'
            ' TP4xDP2 -> TP2xDP4 changed 3/8 greedy prompts)')


def _direction(score: float) -> str:
    if score > 0:
        return 'positive'
    return 'negative' if score < 0 else 'neutral'


def _assess_point(
    m: ModelShape,
    mesh: Mesh,
    base: Mesh,
    point: Point,
    *,
    prefill_bound: bool,
    no_tpdp_alternative: bool,
    eval_kwargs: Mapping[str, Any],
    kv_packing: int | None = None,
) -> dict[str, Any]:
    """Regime, score, expected direction and rationale of `mesh` at `point`.

  Same terms as `_score`, with two point-specific changes: the D7 credit
  (collective-bound prefill) only counts at prefill-heavy points, and at
  balanced / decode-heavy points a mesh that grows per-device weight bytes
  (lower TP x PP) loses log2(growth) (half of that above
  DECODE_HBM_BOUND_SEQS sequences per replica), since the HBM-bound decode
  step dominates there.

  Args:
    m: Model shape.
    mesh: Candidate mesh.
    base: Baseline mesh.
    point: Benchmark point.
    prefill_bound: D7 fired on a prefill / mixed profile.
    no_tpdp_alternative: No feasible non-baseline TP x DP mesh exists.
    eval_kwargs: evaluate_mesh keyword arguments (HBM, framework, image).
    kv_packing: D8's KV packing factor when it fired, else None.
  """
    regime = regime_from_lengths(point.isl, point.osl)
    max_len = point.isl + point.osl
    rec = evaluate_mesh(m,
                        mesh,
                        max_len=max_len,
                        concurrency=point.c,
                        **eval_kwargs)
    base_rec = evaluate_mesh(m,
                             base,
                             max_len=max_len,
                             concurrency=point.c,
                             **eval_kwargs)
    applies_d7 = prefill_bound and regime == 'prefill_heavy'
    ctx = _Context(
        baseline=base,
        regime=regime,
        prefill_bound=applies_d7,
        trust_kv=base_rec['hbm']['kv_ok'] is not False,
        baseline_kv_short=base_rec['hbm']['kv_ok'] is False,
        no_tpdp_alternative=no_tpdp_alternative,
        max_len=max_len,
        input_len=point.isl,
        concurrency=point.c,
        kv_packing=kv_packing,
        kv_heads=m.num_kv_heads,
        head_dim=m.head_dim,
        global_kv_heads=int(m.num_global_kv_heads or 0),
        global_head_dim=int(m.global_head_dim or 0),
    )
    score, why, _ = _score(mesh, ctx, rec['warnings'])
    whys = [why] if why else []
    if prefill_bound and not applies_d7:
        whys.insert(
            0,
            f'D7 not credited: a {regime} point spends most step time in decode',
        )
    weight_ratio = (base.tp * base.pp) / (mesh.tp * mesh.pp)
    if regime != 'prefill_heavy' and weight_ratio > 1:
        per_replica = point.c / mesh.dp
        hbm_bound = per_replica <= DECODE_HBM_BOUND_SEQS
        score -= math.log2(weight_ratio) * (1.0 if hbm_bound else 0.5)
        whys.append(f'per-device weight bytes x{weight_ratio:.0f} slow the'
                    f' {"HBM-bound " if hbm_bound else ""}decode step'
                    f' ({per_replica:g} sequences per replica)')
    return {
        'isl': point.isl,
        'osl': point.osl,
        'c': point.c,
        'regime': regime,
        'score': round(score, 3),
        'expected_direction': _direction(score),
        'rationale': '; '.join(whys),
        'feasible': rec['feasible'],
        'warnings': [w['reason'] for w in rec['warnings']],
    }


def enumerate_oss_meshes(
    config: Mapping[str, Any],
    *,
    devices: int,
    hbm_gib: float,
    framework: str = 'vllm_torchtpu',
    image: str | None = None,
    baseline: Mapping[str, int] | None = None,
    serving: Mapping[str, Any] | None = None,
    regime: str | None = None,
    input_len: int | None = None,
    output_len: int | None = None,
    concurrency: int | None = None,
    weights_gib: float | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    reserve_gib: float = DEFAULT_RESERVE_GIB,
    ignore_constraints: Sequence[str] = (),
    points: Sequence[Mapping[str, Any]] | None = None,
    accelerator: str | None = None,
) -> dict[str, Any]:
    """Enumerates every vLLM mesh (TP/PP/PCP/DP, DCP, EP); proposes up to 3.

  Args:
    config: HF config.json.
    devices: Devices per engine group (e.g. 8 TensorCores on tpu7x-8).
    hbm_gib: HBM per device in GiB.
    framework: Serving framework (constraints are framework-specific).
    image: Serving image digest (enables image-scoped constraints).
    baseline: Baseline mesh {'tp', 'dp'[, 'ep', 'pp', 'dcp', 'pcp']}.
    serving: serving_analyze_main output (single or --case) for diagnosis.
    regime: 'prefill_heavy' | 'decode_heavy' | 'balanced' (default: from the
      input/output lengths).
    input_len: Benchmark input length.
    output_len: Benchmark output length.
    concurrency: Benchmark concurrency (all replicas).
    weights_gib: Measured total weight size to override the estimate.
    gpu_memory_utilization: vLLM --gpu-memory-utilization.
    reserve_gib: Per-device runtime reserve.
    ignore_constraints: Constraint ids to skip (after re-verification).
    points: Benchmark points [{isl, osl, c[, label]}, ...]. When set, every
      candidate gets `per_point_assessment` and `workload_dependent`, a mesh is
      a candidate if it scores positive at the main point or at any point, and
      `score` is the best of those. The main point is input_len / output_len /
      concurrency, or the first point if unset.
    accelerator: TPU type the deployment runs on (e.g. tpu7x); scopes
      chip-specific constraints. Unknown keeps them.

  Returns:
    JSON-serializable dict with `meshes`, `candidates` and the constraint
    table.
  """
    m = ModelShape.from_hf_config(config)
    pts = [Point.from_dict(p) for p in points or []]
    if pts and not (input_len or output_len or concurrency):
        input_len, output_len, concurrency = pts[0].isl, pts[0].osl, pts[0].c
    regime = regime or regime_from_lengths(input_len, output_len)
    prefill_bound = prefill_collective_bound(serving)
    kv_packing = kv_head_padding(serving)
    max_len = input_len + output_len if input_len and output_len else None
    b = baseline or {}
    base = Mesh(
        int(b.get('tp', devices)),
        int(b.get('dp', 1)),
        int(b.get('ep', 1)),
        pp=int(b.get('pp', 1)),
        dcp=int(b.get('dcp', 1)),
        pcp=int(b.get('pcp', 1)),
    )
    records = []
    for mesh in all_meshes(m, devices):
        rec = evaluate_mesh(
            m,
            mesh,
            hbm_gib=hbm_gib,
            framework=framework,
            image=image,
            ignore=ignore_constraints,
            accelerator=accelerator,
            weights_gib=weights_gib,
            gpu_memory_utilization=gpu_memory_utilization,
            reserve_gib=reserve_gib,
            max_len=max_len,
            concurrency=concurrency,
        )
        rec['is_baseline'] = mesh == base
        records.append(rec)
    base_rec = next((r for r in records if r['is_baseline']), None)
    ctx = _Context(
        baseline=base,
        regime=regime,
        prefill_bound=prefill_bound,
        trust_kv=bool(base_rec and base_rec['hbm']['kv_ok'] is not False),
        baseline_kv_short=bool(base_rec and base_rec['hbm']['kv_ok'] is False),
        no_tpdp_alternative=not any(r['feasible'] and not r['is_baseline']
                                    and not Mesh(**r['mesh']).unverified_axes
                                    for r in records),
        max_len=max_len,
        input_len=input_len,
        concurrency=concurrency,
        kv_packing=kv_packing,
        kv_heads=m.num_kv_heads,
        head_dim=m.head_dim,
        global_kv_heads=int(m.num_global_kv_heads or 0),
        global_head_dim=int(m.global_head_dim or 0),
    )
    eval_kwargs = dict(
        hbm_gib=hbm_gib,
        framework=framework,
        image=image,
        ignore=ignore_constraints,
        accelerator=accelerator,
        weights_gib=weights_gib,
        gpu_memory_utilization=gpu_memory_utilization,
        reserve_gib=reserve_gib,
    )
    scored, not_argued = [], []
    for rec in records:
        if not rec['feasible'] or rec['is_baseline']:
            continue
        mesh = Mesh(**rec['mesh'])
        score, why, risks = _score(mesh, ctx, rec['warnings'])
        per_point = {
            p.label:
            _assess_point(
                m,
                mesh,
                base,
                p,
                prefill_bound=prefill_bound,
                no_tpdp_alternative=ctx.no_tpdp_alternative,
                eval_kwargs=eval_kwargs,
                kv_packing=kv_packing,
            )
            for p in pts
        }
        best = max([score] + [a['score'] for a in per_point.values()])
        if best > 0:  # Only meshes the diagnosis argues for (at some point).
            scored.append((best, mesh, why, risks, per_point))
        else:
            not_argued.append({
                'label': mesh.label,
                'score': round(best, 3),
                'rationale': why,
            })
    # Ties go to the mesh with fewer unverified (PP / DCP / PCP) axes.
    scored.sort(key=lambda x: (-x[0], len(x[1].unverified_axes)))
    candidates = []
    for rank, (score, mesh, why, risks,
               per_point) in enumerate(scored[:MAX_CANDIDATES], 1):
        det = determinism_risk(mesh, base)
        cand = {
            'rank': rank,
            'mesh': mesh.to_dict(),
            'label': mesh.label,
            'vllm_args': mesh.vllm_args(base),
            'argv_set': mesh.argv_set(base),
            'argv_add_flags': [FLAG_EP] if mesh.ep > 1 else [],
            'target_regime': regime,
            'score': round(score, 3),
            'rationale': why,
            'risks': risks + ([det] if det else []),
            'unverified_on_torch_tpu': mesh.unverified_axes,
            'determinism_risk': det,
        }
        if pts:
            dirs = {a['expected_direction'] for a in per_point.values()}
            cand['per_point_assessment'] = per_point
            cand['workload_dependent'] = {'positive', 'negative'} <= dirs
        candidates.append(cand)
    return {
        'model_type':
        m.model_type,
        'devices':
        devices,
        'hbm_gib_per_device':
        hbm_gib,
        'framework':
        framework,
        'image':
        image,
        'baseline':
        base.to_dict(),
        'regime':
        regime,
        'prefill_collective_bound':
        prefill_bound,
        'kv_head_padding':
        kv_packing,
        'points': [p.to_dict() for p in pts],
        'meshes':
        records,
        'candidates':
        candidates,
        # Feasible meshes the diagnosis does not argue for, with why.
        'not_argued_for':
        sorted(not_argued, key=lambda x: -x['score']),
        'notes': [
            ('HBM: weights from config shapes (or --weights_gib); the KV'
             ' need counts only the window of sliding-window layers, as'
             " vLLM's hybrid KV-cache manager allocates them;"
             ' kv_tokens_per_replica is a full-attention figure.'),
            ('Candidates are meshes with a positive score: the diagnosis'
             ' argues for the change. Empty means keep the baseline mesh.'),
            ('With points: per_point_assessment scores each candidate per'
             ' benchmark point (D7 credit only at prefill-heavy points;'
             ' lower TP x PP is penalized at balanced / decode-heavy points);'
             ' workload_dependent means the expected directions disagree, so'
             ' measure every point before adopting the mesh.'),
            ('Devices = TP x PP x PCP x DP (vLLM world size x DP); DCP re-uses'
             ' TP ranks and EP = TP x PCP x DP, so neither adds devices. PP /'
             ' DCP / PCP have never run on torch_tpu (see each `risks`).'),
            'vLLM flags: ' + ', '.join(
                (FLAG_TP, FLAG_DP, FLAG_PP, FLAG_DCP, FLAG_PCP, FLAG_EP)) +
            '.',
        ],
        'kernel_constraints': [c.to_dict() for c in KERNEL_CONSTRAINTS],
    }

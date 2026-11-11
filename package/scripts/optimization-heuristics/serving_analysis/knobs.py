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
"""Registry of serving knobs that detectors may recommend.

Each knob names the file that defines it (`source_path`, relative to the
installed package) and a string that proves it is there. In the round, the
image check runs in the serving image itself: `knob_probe.py` reads its
installed source for these strings and `propose.py --knobs-present` reports a
knob the image lacks instead of proposing it. `ImageSource` and
`drop_knobs_absent_from_image` do the same against a local copy of an image's
source (a checkout or an extracted site-packages). Detectors may only emit
`KnobChange`s built via `make_change`.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import os
from collections.abc import Sequence
from typing import Mapping

from serving_analysis import findings

# Where each framework defines its knobs, relative to its installed package
# (site-packages). The image check below looks for them in a serving image.
TPU_INFERENCE_ENVS = 'tpu_inference/envs.py'
TPU_INFERENCE_SHARDING = 'tpu_inference/layers/common/sharding.py'
TPU_INFERENCE_RUNNER = 'tpu_inference/runner/tpu_runner.py'
MAXTEXT_BASE_YML = 'maxtext/configs/base.yml'
VLLM_ARG_UTILS = 'vllm/engine/arg_utils.py'
VLLM_TORCHTPU_ENVS = 'vllm_torchtpu/envs.py'


@dataclasses.dataclass(frozen=True)
class Knob:
    key: str  # Unique registry key.
    framework: (
        str  # 'vllm' | 'vllm_tpu' | 'vllm_torchtpu' | 'maxtext'
    )
    # 'env' | 'cli' | 'additional_config' | 'maxtext_yml'.
    kind: str
    name: str  # Literal name the user sets.
    default: str
    description: str
    source_path: str  # Defining file, relative to the installed package.
    source_token: str  # String that must appear in source_path.
    # Non-empty when the knob only exists with an out-of-tree patch applied to
    # the framework; recommending one means the image must carry the patch.
    # The image check names the patch when the knob is missing.
    requires_patch: str = ''


def _env(name: str, default: str, description: str) -> Knob:
    return Knob(
        key=f'vllm_tpu.env.{name}',
        framework='vllm_tpu',
        kind='env',
        name=name,
        default=default,
        description=description,
        source_path=TPU_INFERENCE_ENVS,
        source_token=f'"{name}"',
    )


def _mt(name: str, default: str, description: str) -> Knob:
    return Knob(
        key=f'maxtext.yml.{name}',
        framework='maxtext',
        kind='maxtext_yml',
        name=name,
        default=default,
        description=description,
        source_path=MAXTEXT_BASE_YML,
        source_token=f'\n{name}:',
    )


def _tt(name: str,
        default: str,
        description: str,
        requires_patch: str = '') -> Knob:
    return Knob(
        key=f'vllm_torchtpu.env.{name}',
        framework='vllm_torchtpu',
        kind='env',
        name=name,
        default=default,
        description=description,
        source_path=VLLM_TORCHTPU_ENVS,
        source_token=f'"{name}"',
        requires_patch=requires_patch,
    )


# Exposes the RPA v3 decode block sizes as TPU_RPA_DECODE_BLOCK_SIZES on the
# vLLM 0.31.0 plugin; carries no sizes itself (relative to the package root).
RPA_DECODE_BLOCK_OVERRIDE_PATCH = (
    'scripts/common/vllm_patches/rpa_decode_block_override.patch')

_KNOBS: tuple[Knob, ...] = (
    # ---- vLLM-TorchTPU env vars ------------------------------------------
    _tt(
        'USE_MOE_FUSED_EP_KERNEL',
        '0',
        'Run the whole EP MoE layer (top-k, dispatch, expert matmuls, '
        'combine, transport) as one SPMD Pallas program instead of '
        'all-gather + grouped matmul + exposed reduce-scatter.',
    ),
    _tt(
        'MOE_FUSED_EP_KERNEL_MIN_TOKENS',
        '1024',
        'Node-wide token cap at/above which the fused EP kernel is armed.',
    ),
    _tt(
        'TPU_MOE_COLLECTION_CHUNK_SIZE',
        '0',
        'Post-gather MoE token chunk size for comm/compute pipelining '
        '(0 disables).',
    ),
    _tt(
        'TPU_SPARSE_MLA_NOPE_LAYOUT',
        'tensorcore',
        'Sparse-MLA NoPE KV cache layout (tensorcore|sparsecore).',
    ),
    _tt(
        'TPU_SPARSE_MLA_ROPE_LAYOUT',
        'tensorcore',
        'Sparse-MLA RoPE KV cache layout (tensorcore|sparsecore).',
    ),
    _tt(
        'TPU_MOE_SKIP_PADDED_TOKENS',
        '0',
        'Skip padded tokens in MoE instead of routing them to experts.',
    ),
    _tt(
        'TPU_MOE_HIERARCHICAL_EP',
        '0',
        'EP across chips, TP across the two chiplets of a chip; helps at low '
        'concurrency when MoE is weight-bandwidth bound and imbalanced.',
    ),
    _tt(
        'TPU_MOE_ROUTER_TOPK',
        'rowmax',
        'Router top-k implementation (rowmax is sort-free; sort=torch.topk).',
    ),
    _tt(
        'TPU_RPA_DECODE_BLOCK_SIZES',
        '',
        'RPA v3 decode KV block and compute chunk per layer kind, as'
        ' sliding=<bkv>:<bkv_c>,global=<bkv>:<bkv_c> (tokens); unset keeps'
        ' the kernel default. Read only by the non-batched RPA v3 kernel'
        ' (no --attention-backend CUSTOM).',
        requires_patch=RPA_DECODE_BLOCK_OVERRIDE_PATCH,
    ),
    _tt(
        'TPU_ROPE_CACHE_TRUNCATE',
        '0',
        'Slice rotary cos/sin caches to max model len at load to remove the '
        'per-step layout copy.',
    ),
    _tt(
        'TPU_MOE_LOWTOK_MAX_TOKENS',
        '0',
        'Run the few-token MoE kernel (kernels/moe_lowtok) for EP calls up to '
        'this many tokens, walking hit experts with routing weights as the row '
        'mask without sort/permute/pad/unpermute.',
    ),
    _tt(
        'TPU_SMALL_ALL_REDUCE_MAX_ROWS',
        '0',
        'Row cap of the Pallas small all-reduce kernel '
        '(kernels/collectives/small_all_reduce.py) on the collectives it '
        'serves; needs an expert-parallel group. envs.py says 0 keeps XLA, '
        'but current layers/vllm/small_all_reduce_op.py reads 0 as 64 rows.',
    ),
    _tt(
        'TPU_TOKEN_BUCKET_STEP',
        '0',
        'Add linear token buckets every STEP rows from 1/4 of the token budget '
        'up to max prompt pass length on top of the power-of-two ladder.',
    ),
    _tt(
        'TPU_ROPE_CACHE_ROW_MAJOR',
        '0',
        'Materialize rotary cos/sin caches in row-major {1, 0} device layout '
        'at load (avoids the per-step layout copy; may cost more HBM due to '
        'padding).',
    ),
    _tt(
        'TPU_FUSED_QK_NORM_ROPE',
        '0',
        'Qwen3-Next attention (text-only, gated, neox RoPE): one Pallas call '
        'for the q/gate split, Q/K RMSNorm, RoPE and gate on steps of <= 64 '
        'rows; larger steps keep the unfused path.',
    ),
    _tt(
        'TPU_LATENT_PROJ_INTRA_CHIP_TP',
        '0',
        "Shard Kimi K3's latent-MoE down/up projections column-parallel "
        "across each chip's cores (halves weight reads per core; one "
        'on-package all-gather per projection, two per layer).',
    ),
    _tt(
        'TPU_SMALL_ALL_REDUCE_FORM',
        '',
        'Small decode all-reduce form (oneshot=all-gather+sum, '
        'twoshot=reduce-scatter+all-gather; empty = chosen by block size).',
    ),
    # ---- Upstream vLLM engine CLI (both TPU backends) ---------------------
    Knob(
        key='vllm.cli.data_parallel_size',
        framework='vllm',
        kind='cli',
        name='--data-parallel-size',
        default='1',
        description='Attention data parallelism across engine ranks.',
        source_path=VLLM_ARG_UTILS,
        source_token='data_parallel_size',
    ),
    Knob(
        key='vllm.cli.tensor_parallel_size',
        framework='vllm',
        kind='cli',
        name='--tensor-parallel-size',
        default='1',
        description='Tensor parallelism within an engine rank.',
        source_path=VLLM_ARG_UTILS,
        source_token='tensor_parallel_size',
    ),
    Knob(
        key='vllm.cli.enable_expert_parallel',
        framework='vllm',
        kind='cli',
        name='--enable-expert-parallel',
        default='false',
        description='Shard MoE experts over the TP x DP group.',
        source_path=VLLM_ARG_UTILS,
        source_token='enable_expert_parallel',
    ),
    Knob(
        key='vllm.cli.async_scheduling',
        framework='vllm',
        kind='cli',
        name='--async-scheduling',
        default='',
        description=('Overlap scheduling of step N+1 with execution of step '
                     'N. Unset = auto: recent vLLM (config/vllm.py) enables '
                     'it unless spec-decode/executor is incompatible; older '
                     'releases default to off.'),
        source_path=VLLM_ARG_UTILS,
        source_token='async_scheduling',
    ),
    Knob(
        key='vllm.cli.max_num_seqs',
        framework='vllm',
        kind='cli',
        name='--max-num-seqs',
        default='',
        description='Max concurrent sequences; sizes per-step buffers.',
        source_path=VLLM_ARG_UTILS,
        source_token='max_num_seqs',
    ),
    Knob(
        key='vllm.cli.max_num_batched_tokens',
        framework='vllm',
        kind='cli',
        name='--max-num-batched-tokens',
        default='',
        description='Max tokens per engine step (prefill chunk budget).',
        source_path=VLLM_ARG_UTILS,
        source_token='max_num_batched_tokens',
    ),
    Knob(
        key='vllm.cli.attention_backend',
        framework='vllm',
        kind='cli',
        name='--attention-backend',
        default='',
        description=('Attention backend. On vLLM-TorchTPU, CUSTOM runs the'
                     ' batched RPA kernel and FLASH_ATTN the RPA v3 kernel,'
                     ' whose decode blocks TPU_RPA_DECODE_BLOCK_SIZES sets.'),
        source_path=VLLM_ARG_UTILS,
        source_token='attention_backend',
    ),
    Knob(
        key='vllm.cli.block_size',
        framework='vllm',
        kind='cli',
        name='--block-size',
        default='',
        description='KV cache page size in tokens (16..256 on TPU).',
        source_path=VLLM_ARG_UTILS,
        source_token='--block-size',
    ),
    Knob(
        key='vllm.cli.gpu_memory_utilization',
        framework='vllm',
        kind='cli',
        name='--gpu-memory-utilization',
        default='0.92',
        description='Fraction of device memory reserved for model + KV cache.',
        source_path=VLLM_ARG_UTILS,
        source_token='--gpu-memory-utilization',
    ),
    Knob(
        key='vllm.cli.decode_context_parallel_size',
        framework='vllm',
        kind='cli',
        name='--decode-context-parallel-size',
        default='1',
        description=(
            'Ranks that shard the decode KV cache (no extra processes); '
            'without PCP it reuses TP ranks, so tensor_parallel_size must be '
            'divisible by it.'),
        source_path=VLLM_ARG_UTILS,
        source_token='decode_context_parallel_size',
    ),
    Knob(
        key='vllm.cli.prefill_context_parallel_size',
        framework='vllm',
        kind='cli',
        name='--prefill-context-parallel-size',
        default='1',
        description=(
            'Ranks that split prefill sequence computation (expands the world '
            'size). vLLM-TPU (tpu_inference sharding.py) also requires an even '
            'size, NEW_MODEL_DESIGN=1, decode CP 1 and no SPMD data '
            'parallelism.'),
        source_path=VLLM_ARG_UTILS,
        source_token='prefill_context_parallel_size',
    ),
    Knob(
        key='vllm.cli.kv_cache_memory_bytes',
        framework='vllm',
        kind='cli',
        name='--kv-cache-memory-bytes',
        default='',
        description=(
            'Explicit per-rank HBM KV cache pool size in bytes (overrides '
            '--gpu-memory-utilization when set).'),
        source_path=VLLM_ARG_UTILS,
        source_token='kv_cache_memory_bytes',
    ),
    Knob(
        key='vllm.cli.enable_prefix_caching',
        framework='vllm',
        kind='cli',
        name='--enable-prefix-caching',
        default='',
        description=('Enable automatic prefix caching across requests (or '
                     '--no-enable-prefix-caching to disable).'),
        source_path=VLLM_ARG_UTILS,
        source_token='enable_prefix_caching',
    ),
    Knob(
        key='vllm.cli.cp_kv_cache_interleave_size',
        framework='vllm',
        kind='cli',
        name='--cp-kv-cache-interleave-size',
        default='1',
        description=(
            'Token interleave granularity across context-parallel (DCP/PCP) '
            'KV cache shards.'),
        source_path=VLLM_ARG_UTILS,
        source_token='cp_kv_cache_interleave_size',
    ),
    Knob(
        key='vllm.cli.speculative_config',
        framework='vllm',
        kind='cli',
        name='--speculative-config',
        default='',
        description=(
            'JSON speculative decoding config (method, num_speculative_tokens, '
            'synthetic_acceptance_length).'),
        source_path=VLLM_ARG_UTILS,
        source_token='speculative_config',
    ),
    # ---- vLLM-TPU (tpu_inference) env vars -------------------------------
    _env(
        'MOE_APPROX_TOPK',
        'False',
        'Approximate top-k for MoE expert selection (quality-gated).',
    ),
    _env(
        'USE_BATCHED_RPA_KERNEL',
        'False',
        'Batched ragged paged attention kernel (several sequences per '
        'kernel invocation).',
    ),
    _env(
        'USE_MOE_EP_KERNEL',
        'False',
        'Fused expert-parallel MoE kernel (requires expert parallelism).',
    ),
    _env(
        'MOE_APPROX_TOPK_RECALL_TARGET',
        '0.9',
        'Recall target for approximate top-k.',
    ),
    _env(
        'ONEHOT_MOE_PERMUTE_THRESHOLD',
        '0',
        'Use one-hot+matmul permute/unpermute when batch <= threshold.',
    ),
    _env(
        'MOE_ROUTE_PADDING_TO_EXPERT0',
        'False',
        'Route padding tokens to expert 0 to avoid activating extra experts '
        '(only when attention DP size is 1).',
    ),
    _env(
        'MOE_ALL_GATHER_ACTIVATION_DTYPE',
        '',
        'Dtype for the MoE dispatch all-gather (e.g. fp8).',
    ),
    _env(
        'ENABLE_RS_KERNEL',
        'False',
        'Hierarchical reduce-scatter kernel for MoE.',
    ),
    _env(
        'SC_ALLREDUCE_ALLGATHER_OFFLOAD_MIN_BYTES',
        'auto',
        'Min bytes before AR/AG is offloaded to SparseCore.',
    ),
    _env(
        'DP_SCHED_BATCH_PREFILL',
        'True',
        'DP scheduler: hold and batch prefills to avoid prefill/decode mixing.',
    ),
    _env(
        'DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS',
        '30000',
        'DP scheduler flush timeout for batched prefills.',
    ),
    _env('MIN_TOKEN_BUCKET', '16', 'Smallest token padding bucket.'),
    _env(
        'VLLM_TPU_BUCKET_PADDING_GAP',
        '0',
        'Linear gap between token buckets past the power-of-two ramp.',
    ),
    _env(
        'ATTN_BUCKETIZED_NUM_REQS',
        'False',
        'Precompile attention for power-of-two num_reqs buckets.',
    ),
    _env(
        'ATTN_CUSTOM_NUM_REQS_BUCKETS',
        '',
        'Custom num_reqs buckets for attention.',
    ),
    _env(
        'SLICE_ROPE_CACHE',
        'False',
        'Slice rotary cos/sin cache to max_model_len at load; removes the '
        'per-step layout copy of the full max_position_embeddings table.',
    ),
    _env(
        'ROPE_CACHE_ROW_MAJOR',
        'False',
        'Store rope cache row-major to avoid per-step transpose copy.',
    ),
    _env('RPA_V3_DECODE_BLOCK_SIZES', '',
         'Override RPA v3 decode block sizes.'),
    _env('RPA_V3_PREFILL_BLOCK_SIZES', '',
         'Override RPA v3 prefill block sizes.'),
    _env('RPA_V3_MIXED_BLOCK_SIZES', '', 'Override RPA v3 mixed block sizes.'),
    _env(
        'CONTINUE_DECODE_EOS_CHECK_INTERVAL',
        '1',
        'Check EOS early-exit every N decode steps to amortize dispatch.',
    ),
    _env(
        'SKIP_JAX_PRECOMPILE',
        'False',
        'Skip precompilation (causes runtime compiles when True).',
    ),
    _env(
        'VLLM_XLA_CHECK_RECOMPILATION',
        'False',
        'Assert on runtime recompilation.',
    ),
    _env(
        'PYTHON_TRACER_LEVEL',
        '1',
        'Python tracer level for profiling host code.',
    ),
    _env(
        'PHASED_PROFILING_DIR',
        '',
        'Directory for phased (prefill/decode) profiling.',
    ),
    _env(
        'FORCE_MOE_RANDOM_ROUTING',
        'False',
        'Random expert routing (benchmark/diagnosis only).',
    ),
    _env(
        'REQUANTIZE_WEIGHT_DTYPE',
        'float8_e4m3fn',
        'Dtype for requantized linear weights.',
    ),
    _env('MOE_REQUANTIZE_WEIGHT_DTYPE', '',
         'Dtype for requantized MoE weights.'),
    _env(
        'MOE_REQUANTIZE_BLOCK_SIZE',
        '',
        'Quantization block size for requantized MoE weights (512 keeps fp4 '
        'quality-neutral with low TPU overhead).',
    ),
    _env(
        'VLLM_MOE_CHUNK_SIZE',
        '0',
        'Token chunk for the MoE gather-reduce + combine collective so each '
        "chunk's collective overlaps the next chunk (0 disables).",
    ),
    _env(
        'MLA_TRANSPOSE_KV_CACHE',
        'False',
        'Transposed MLA KV-cache layout (smaller footprint, larger batch).',
    ),
    _env(
        'MLA_KV_PACKING_SIZE',
        '32',
        'MLA KV-cache packing factor (MLA v2 tuned tables use 32).',
    ),
    _env(
        'DISABLE_MLA_Q_ACTIVATION_QUANTIZATION',
        'False',
        'Keep MLA Q activations in bf16 instead of fp8.',
    ),
    # ---- vLLM-TPU additional_config (sharding strategy) ------------------
    Knob(
        key='vllm_tpu.additional_config.enable_dp_attention',
        framework='vllm_tpu',
        kind='additional_config',
        name='sharding.sharding_strategy.enable_dp_attention',
        default='false',
        description=('Attention data parallelism (avoids KV-head padding and '
                     'replication when num_kv_heads < TP).'),
        source_path=TPU_INFERENCE_SHARDING,
        source_token='"enable_dp_attention"',
    ),
    Knob(
        key='vllm_tpu.additional_config.enable_continue_decode',
        framework='vllm_tpu',
        kind='additional_config',
        name='enable_continue_decode',
        default='false',
        description=(
            'Fused on-device multi-step decode loop (decode_loop.py); '
            'amortizes per-step host dispatch / scheduling across up to '
            'max_decode_steps tokens.'),
        source_path=TPU_INFERENCE_RUNNER,
        source_token='"enable_continue_decode"',
    ),
    Knob(
        key='vllm_tpu.additional_config.max_decode_steps',
        framework='vllm_tpu',
        kind='additional_config',
        name='max_decode_steps',
        default='10',
        description='Max fused decode steps per continue_decode dispatch.',
        source_path=TPU_INFERENCE_RUNNER,
        source_token='"max_decode_steps"',
    ),
    # ---- vLLM engine CLI (upstream vLLM; verified by name in tpu_runner) --
    Knob(
        key='vllm_tpu.cli.max_num_batched_tokens',
        framework='vllm_tpu',
        kind='cli',
        name='--max-num-batched-tokens',
        default='(engine default)',
        description='Token budget per step; controls prefill chunk size.',
        source_path=TPU_INFERENCE_RUNNER,
        source_token='max_num_batched_tokens',
    ),
    Knob(
        key='vllm_tpu.cli.max_num_seqs',
        framework='vllm_tpu',
        kind='cli',
        name='--max-num-seqs',
        default='(engine default)',
        description='Max concurrent sequences per step (decode batch).',
        source_path=TPU_INFERENCE_RUNNER,
        source_token='max_num_seqs',
    ),
    Knob(
        key='vllm_tpu.cli.kv_cache_dtype',
        framework='vllm_tpu',
        kind='cli',
        name='--kv-cache-dtype',
        default='auto',
        description='KV cache dtype (fp8 halves KV bandwidth/capacity).',
        source_path='tpu_inference/runner/kv_cache.py',
        source_token='cache_dtype',
    ),
    # ---- MaxText ----------------------------------------------------------
    _mt('use_ragged_attention', 'False', 'Ragged attention for decode.'),
    _mt('ragged_block_size', '256', 'Ragged attention block size.'),
    _mt('use_chunked_prefill', 'False', 'Chunked prefill.'),
    _mt('prefill_chunk_size', '256', 'Prefill chunk size.'),
    _mt('megablox', 'True', 'Megablox grouped matmul for MoE.'),
    _mt('sparse_matmul', 'True', 'Sparse (dropless) MoE matmul.'),
    _mt('capacity_factor', '-1.0', 'Expert capacity factor (dropping MoE).'),
    _mt('use_ragged_sort', 'False', 'Ragged sort for MoE routing.'),
    _mt(
        'moe_quantize_token_all_gather',
        'False',
        'Quantize MoE token all-gather.',
    ),
    _mt('quantize_kvcache', 'False', 'Quantize the KV cache.'),
    _mt(
        'ici_fsdp_parallelism',
        '-1',
        'ICI FSDP axis. Shards weights; each step all-gathers them.',
    ),
    _mt('ici_tensor_parallelism', '1', 'ICI tensor-parallel axis.'),
    _mt('ici_expert_parallelism', '1', 'ICI expert-parallel axis.'),
    _mt('kv_quant_dtype', 'int8', 'KV cache quantization dtype.'),
    _mt('quantization', '', 'Weight/activation quantization recipe.'),
)

REGISTRY: Mapping[str, Knob] = {k.key: k for k in _KNOBS}


def get(key: str) -> Knob:
    if key not in REGISTRY:
        raise KeyError(
            f'Unknown knob {key!r}; add it to knobs.py with a source.')
    return REGISTRY[key]


def current_value(key: str,
                  serving_config: Mapping[str, str] | None) -> str | None:
    """Looks up a knob's current value in a flat serving config dict.

  The serving config is a flat mapping of literal names (env var names, CLI
  flags, dotted additional_config paths, MaxText yml keys) to string values.
  Returns None if no serving config was provided; returns the knob default
  if a config was provided but the knob is absent (i.e. it is unset).
  """
    if serving_config is None:
        return None
    knob = get(key)
    for name in (
            knob.name,
            knob.name.lstrip('-'),
            knob.name.lstrip('-').replace('-', '_'),
    ):
        if name in serving_config:
            return str(serving_config[name])
    return knob.default


def make_change(
    key: str,
    suggested_value: str,
    serving_config: Mapping[str, str] | None,
) -> findings.KnobChange:
    """Builds a KnobChange for a registered knob (only way detectors do so)."""
    knob = get(key)
    return findings.KnobChange(
        knob=key,
        kind=knob.kind,
        name=knob.name,
        suggested_value=suggested_value,
        current_value=current_value(key, serving_config),
        requires_patch=knob.requires_patch,
    )


# Knob frameworks that live in a vLLM serving image (vLLM + its TPU plugin).
_IMAGE_FRAMEWORKS = {
    'vllm_torchtpu': ('vllm', 'vllm_torchtpu'),
    'vllm_tpu': ('vllm', 'vllm_tpu'),
}


def image_tokens(k: Knob) -> tuple[str, ...]:
    """Strings, any of which proves `k` exists in an image's Python source.

  Env vars and additional_config keys must appear as a quoted literal; CLI
  flags as their argparse dest (vLLM builds the CLI from EngineArgs fields)
  or the literal flag.

  Args:
    k: Registry knob.
  """
    if k.kind in ('env', 'additional_config'):
        leaf = k.name.rsplit('.', 1)[-1]
        return (f'"{leaf}"', f"'{leaf}'")
    if k.kind == 'cli':
        dest = k.name.lstrip('-').replace('-', '_')
        return (dest, k.name)
    return (k.name, )


class ImageSource:
    """Text index over a serving image's Python source tree.

  Point it at a directory extracted from the image (e.g. `docker cp
  <container>:/usr/local/lib/python3.12/site-packages <dir>`, or the
  `vllm` / `vllm_torchtpu` package dirs). This is the pre-launch G0 check from
  the overnight A/B harness, run before a knob is recommended.
  """

    def __init__(self, root: str):
        if not os.path.isdir(root):
            raise ValueError(f'image source dir not found: {root}')
        self.root = root
        parts = []
        for dirpath, _, files in os.walk(root):
            for name in files:
                if name.endswith('.py'):
                    try:
                        with open(os.path.join(dirpath, name),
                                  errors='replace') as f:
                            parts.append(f.read())
                    except OSError:
                        continue
        self._text = '\n'.join(parts)
        if not self._text:
            raise ValueError(f'no Python source under {root}')

    def has_knob(self, key: str) -> bool:
        return any(t in self._text for t in image_tokens(get(key)))


def verify_image(source: ImageSource,
                 framework: str = 'vllm_torchtpu') -> list[str]:
    """Registry knobs for `framework` that are absent from the image source."""
    fws = _IMAGE_FRAMEWORKS.get(framework, (framework, ))
    missing = []
    for k in _KNOBS:
        if k.framework in fws and not source.has_knob(k.key):
            patch = f' (needs {k.requires_patch})' if k.requires_patch else ''
            missing.append(f'{k.key}: `{k.name}` not in {source.root}{patch}')
    return missing


def drop_knobs_absent_from_image(
        found: Sequence[findings.Finding],
        source: ImageSource) -> tuple[list[findings.Finding], list[str]]:
    """Removes knob changes the image does not have; returns (findings, notes).

  A finding left with no knob change and no kernel follow-up is dropped.

  Args:
    found: Detector findings.
    source: Image source index.
  """
    out, notes = [], []
    for f in found:
        kept = []
        for kc in f.knob_changes:
            if kc.knob in REGISTRY and not source.has_knob(kc.knob):
                patch = (f'; it needs {kc.requires_patch}'
                         if kc.requires_patch else '')
                notes.append(
                    f'Image check: dropped `{kc.name}={kc.suggested_value}` from'
                    f' {f.detector_id}: not in the image source {source.root}{patch}.'
                )
                continue
            kept.append(kc)
        if len(kept) != len(f.knob_changes):
            f.knob_changes = kept
            if not kept and not f.kernel_followup:
                continue
        out.append(f)
    return out, notes

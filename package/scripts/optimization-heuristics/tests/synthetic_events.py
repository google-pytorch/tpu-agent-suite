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
"""Synthetic Events DB frames for serving_analysis unit tests."""

import dataclasses
import os
import sys
from collections.abc import Sequence
from typing import Any

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from serving_analysis import context  # noqa: E402
from serving_analysis import features  # noqa: E402
from serving_analysis import op_classes  # noqa: E402
from serving_analysis import segmenter  # noqa: E402
from serving_analysis import trace_table  # noqa: E402

US = 1000  # Nanoseconds per microsecond.
HIDDEN = 1024
N_DEVICES = 8
MOE_SCOPE = 'jit(step_fun)/SharedFusedMoE/jit(fused_moe_func)/shard_map'
DECODE_MODULE = 'jit_step_fun(111)'
PREFILL_MODULE = 'jit_step_fun(222)'
LOGITS_MODULE = 'jit_compute_logits_func(333)'
DECODE_BUCKET = 64
PREFILL_BUCKET = 2048
SOURCE_ROOT = '/home/user/tpu-inference/'
MOE_FILE = 'tpu_inference/layers/common/fused_moe_gmm.py'
MOE_COMBINE_LINE = 188


@dataclasses.dataclass(frozen=True)
class _Op:
    name: str
    dtype: str
    dims: tuple[int, ...]
    opcode: str
    scope: str
    tf_type: str
    category: str
    us: float
    source: str = ''


def _step_ops(
    tokens: int,
    device: int,
    scale: float,
    barrier_us: float,
    gmm_skew: float,
    attention_kernel: str,
    attention_us: float,
) -> Sequence[_Op]:
    """One step's ops: QKV matmul, RPA, TP psum, MoE top-k/gather/GMM/psum."""
    return (
        _Op(
            'fusion.1',
            'bf16',
            (tokens, HIDDEN),
            'fusion',
            'jit(step_fun)/QKVParallelLinear/dot_general',
            'dot_general',
            'convolution fusion',
            200 * scale,
            'tpu_inference/kernels/quantized_matmul/util.py:89',
        ),
        _Op(
            f'{attention_kernel}.1',
            'bf16',
            (tokens, 8, 128),
            'custom-call',
            'jit(step_fun)/jit(_jax_attn_func)/shard_map/'
            f'jit(ragged_paged_attention)/{attention_kernel}',
            'pallas_call',
            'custom-call',
            attention_us * scale,
            'tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:1584',
        ),
        _Op(
            'psum.1',
            'bf16',
            (tokens, HIDDEN),
            'all-reduce',
            'jit(step_fun)/RowParallelLinear/shard_map/psum',
            'psum',
            'all-reduce',
            100 * scale,
            'tpu_inference/layers/common/linear.py:100',
        ),
        _Op(
            'top_k.1',
            's32',
            (tokens, 8),
            'custom-call',
            f'{MOE_SCOPE}/top_k',
            'top_k',
            'sort',
            60 * scale,
            f'{MOE_FILE}:411',
        ),
        _Op(
            'gather.1',
            'bf16',
            (tokens * 8, HIDDEN),
            'gather',
            f'{MOE_SCOPE}/gather',
            'gather',
            'async-done',
            150 * scale,
            f'{MOE_FILE}:487',
        ),
        _Op(
            'gmm-g_20.1',
            'bf16',
            (tokens * 8, HIDDEN),
            'custom-call',
            f'{MOE_SCOPE}/jit(gmm_pipeline)/pallas_call',
            'pallas_call',
            'custom-call',
            300 * scale * (gmm_skew if device == 3 else 1.0),
            'tpu_inference/kernels/megablox/gmm.py:622',
        ),
        _Op(
            'psum.2',
            'bf16',
            (tokens, HIDDEN),
            'all-reduce',
            f'{MOE_SCOPE}/psum',
            'psum',
            'all-reduce',
            (100 + barrier_us) * scale,
            f'{MOE_FILE}:{MOE_COMBINE_LINE}',
        ),
    )


def row(*, start_ns: int, end_ns: int, **columns: object) -> dict[str, object]:
    """One Events DB row."""
    base: dict[str, object] = dict(
        host='h0',
        category='',
        kernel_name='',
        hlo_module='',
        hlo_op='',
        tf_op_name='',
        tf_op_type='',
        trace_args='',
        thread_name='',
        source_line='',
        self_time_ns=end_ns - start_ns,
    )
    base.update(columns)
    base['start_ns'] = start_ns
    base['end_ns'] = end_ns
    return base


def make_events(
    pattern: str = 'DDDPDDPDDD',
    idle_us: float = 400,
    barrier_us: float = 150,
    gmm_skew: float = 1.6,
    decode_reqs: int = 32,
    attention_kernel: str = 'RPA-bq_32',
    attention_us: float = 300,
) -> pd.DataFrame:
    """Builds a synthetic vLLM-TPU MoE Events DB frame."""
    rows = []
    modules = {
        'D': (DECODE_MODULE, DECODE_BUCKET, 1.0),
        'P': (PREFILL_MODULE, PREFILL_BUCKET, 20.0),
    }
    t0 = 1_000 * US
    for kind in pattern:
        module, bucket, scale = modules[kind]
        reqs, toks = (decode_reqs, decode_reqs) if kind == 'D' else (4, bucket)
        rows.append(
            row(
                device='cpu:0',
                category='host',
                thread_name='VLLM::EngineCor',
                kernel_name=f'execute_model: {reqs} reqs, {toks} toks',
                start_ns=t0 - 100 * US,
                end_ns=t0 + 20 * US,
            ))
        step_end = t0
        for device in range(N_DEVICES):
            cur = t0
            for op in _step_ops(
                    bucket,
                    device,
                    scale,
                    barrier_us,
                    gmm_skew,
                    attention_kernel,
                    attention_us,
            ):
                duration = int(op.us * US)
                dims = ','.join(map(str, op.dims))
                rows.append(
                    row(
                        device=f'TPU:{device}',
                        category='XLA Ops',
                        hlo_module=module,
                        hlo_op=op.name,
                        kernel_name=(f'%{op.name} = {op.dtype}[{dims}]{{1,0}}'
                                     f' {op.opcode}(%p.0)'),
                        tf_op_name=op.scope,
                        tf_op_type=op.tf_type,
                        trace_args=f'hlo_category={op.category},program_id=1',
                        source_line=(SOURCE_ROOT +
                                     op.source if op.source else ''),
                        start_ns=cur,
                        end_ns=cur + duration,
                    ))
                cur += duration
            rows.append(
                row(
                    device=f'TPU:{device}',
                    category='XLA Modules',
                    kernel_name=f'HLO Module:{module}',
                    start_ns=t0,
                    end_ns=cur,
                ))
            rows.append(
                row(
                    device=f'TPU:{device}',
                    category='XLA Modules',
                    kernel_name=f'HLO Module:{LOGITS_MODULE}',
                    start_ns=cur,
                    end_ns=cur + 50 * US,
                ))
            step_end = max(step_end, cur + 50 * US)
        rows.append(
            row(
                device='cpu:0',
                category='host',
                thread_name='VLLM::EngineCor',
                kernel_name='$tpu_runner.py:683 _execute_model',
                start_ns=step_end,
                end_ns=step_end + int(idle_us * US),
            ))
        t0 = step_end + int(idle_us * US)
    return pd.DataFrame(rows)


def serving_context(**overrides: Any) -> context.ServingContext:
    """Workload context matching `make_events`."""
    overrides.setdefault('tp', 8)
    overrides.setdefault('serving_config', {})
    return context.ServingContext(
        model=context.ModelConfig(
            hidden_size=HIDDEN,
            num_kv_heads=8,
            num_experts=160,
            num_experts_per_tok=8,
            max_position_embeddings=262144,
        ),
        **overrides,
    )


def compute_features(
    events: pd.DataFrame | None = None,
    ctx: context.ServingContext | None = None,
    session_id: str = '',
) -> features.Features:
    """Classifies, segments and computes features, as `analyze` does."""
    active_ctx = (ctx or serving_context()).finalize()
    table = trace_table.TraceTable.from_events_frame(
        make_events() if events is None else events, session_id)
    meta = op_classes.classify(table.op_meta)
    seg = segmenter.segment(table, active_ctx, meta)
    return features.compute(table, seg, meta)

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
"""Workload context for serving analysis: model config + serving config.

All fields are optional. Detectors degrade gracefully (lower confidence,
"current value unknown") when context is missing.
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import json
import re
import shlex
from typing import Any, Mapping

from serving_analysis import limits

_EMBEDDING_DIM_ALIASES = (
    'hidden_size',
    'd_model',
    'n_embd',
    'emb_dim',
    'base_emb_dim',
)
_LAYERS_ALIASES = (
    'num_hidden_layers',
    'num_layers',
    'n_layer',
    'base_num_decoder_layers',
)
_HEADS_ALIASES = (
    'num_attention_heads',
    'num_heads',
    'n_head',
    'num_query_heads',
    'base_num_query_heads',
)
_KV_HEADS_ALIASES = (
    'num_key_value_heads',
    'num_kv_heads',
    'n_head_kv',
    'base_num_kv_heads',
)
_HEAD_DIM_ALIASES = ('head_dim', 'kv_channels')
_NUM_EXPERTS_ALIASES = (
    'num_experts',
    'n_routed_experts',
    'num_local_experts',
    'moe_num_experts',
)
_EXPERTS_PER_TOK_ALIASES = (
    'num_experts_per_tok',
    'num_selected_experts',
    'moe_top_k',
    'moe_topk',
    'num_experts_per_token',
)
_MOE_FFN_DIM_ALIASES = (
    'moe_intermediate_size',
    'expert_ffn_dim',
    'ffn_hidden_size',
    'base_moe_mlp_dim',
)
_MAX_POSITION_ALIASES = ('max_position_embeddings', 'max_target_length')


def _get_config_int(cfg: Mapping[str, Any], aliases: tuple[str,
                                                           ...]) -> int | None:
    for k in aliases:
        v = cfg.get(k)
        if v is not None and not isinstance(v, bool):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


@dataclasses.dataclass
class ModelConfig:
    """Subset of HF / MaxText model config fields used by detectors."""

    name: str = ''
    hidden_size: int | None = None
    num_layers: int | None = None
    num_attention_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    max_position_embeddings: int | None = None
    sliding_window: int | None = None  # Window of the sliding layers, if any.
    kv_lora_rank: int | None = None  # MLA.
    qk_rope_head_dim: int | None = None  # MLA rotary head width (defaults to 64).
    rope_type: str = ''

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts and self.num_experts > 1)

    @property
    def is_mla(self) -> bool:
        return bool(self.kv_lora_rank)

    @property
    def mla_kv_dim(self) -> int | None:
        """Per-token MLA KV cache width (`kv_lora_rank + qk_rope_head_dim`)."""
        if not self.kv_lora_rank:
            return None
        rope_dim = (self.qk_rope_head_dim if self.qk_rope_head_dim is not None
                    else limits.DEFAULT_MLA_ROPE_DIM)
        return self.kv_lora_rank + rope_dim

    @classmethod
    def from_hf_dict(cls, d: Mapping[str, Any]) -> 'ModelConfig':
        """Parses a HuggingFace config.json dict (handles text_config nesting)."""
        cfg = dict(d)
        if isinstance(cfg.get('text_config'), dict):
            name = cfg.get('_name_or_path', '')
            cfg = dict(cfg['text_config'])
            cfg.setdefault('_name_or_path', name)

        def num(aliases: tuple[str, ...]) -> int | None:
            return _get_config_int(cfg, aliases)

        hidden = num(_EMBEDDING_DIM_ALIASES)
        heads = num(_HEADS_ALIASES)
        kv_lora_rank = num(('kv_lora_rank', ))
        qk_rope_head_dim = num(('qk_rope_head_dim', ))
        head_dim = num(_HEAD_DIM_ALIASES)
        if head_dim is None and not kv_lora_rank and hidden and heads:
            head_dim = hidden // heads

        moe_width = num(_MOE_FFN_DIM_ALIASES)
        if moe_width is None and cfg.get('num_local_experts'):
            moe_width = num(('intermediate_size', ))
        sliding_window = (None if cfg.get('use_sliding_window') is False else
                          num(('sliding_window', )))
        rope = cfg.get('rope_scaling')
        return cls(
            name=str(cfg.get('_name_or_path') or cfg.get('model_type') or ''),
            hidden_size=hidden,
            num_layers=num(_LAYERS_ALIASES),
            num_attention_heads=heads,
            num_kv_heads=num(_KV_HEADS_ALIASES),
            head_dim=head_dim,
            num_experts=num(_NUM_EXPERTS_ALIASES),
            num_experts_per_tok=num(_EXPERTS_PER_TOK_ALIASES),
            moe_intermediate_size=moe_width,
            max_position_embeddings=num(_MAX_POSITION_ALIASES),
            sliding_window=sliding_window,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            rope_type=str((rope.get('rope_type') or rope.get('type') or ''
                           ) if isinstance(rope, dict) else ''),
        )

    @classmethod
    def from_json_file(cls, path: str) -> 'ModelConfig':
        with open(path) as f:
            return cls.from_hf_dict(json.load(f))


@dataclasses.dataclass
class ServingContext:
    """Everything known about the workload besides the trace."""

    # 'auto' | 'vllm_tpu' | 'vllm_torchtpu' | 'maxtext' | 'jax' | 'generic'
    framework: str = 'auto'
    platform: str = ''  # e.g. 'v7x 2x2x1', 'v6e'
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    # Flat map of literal knob names -> values (env vars, CLI flags without
    # leading dashes, dotted additional_config paths, MaxText keys).
    # None means "serving config not provided" (knob state unknown).
    serving_config: dict[str, str] | None = None
    serving_command: str | None = None
    tp: int | None = None
    dp: int | None = None
    ep: int | None = None
    dcp: int | None = None
    pcp: int | None = None
    cp_kv_cache_interleave_size: int | None = None
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
    kv_cache_memory_bytes: int | None = None
    enable_prefix_caching: bool | None = None
    speculative_method: str | None = None
    num_speculative_tokens: int | None = None
    synthetic_acceptance_length: float | None = None
    # `kv_role` of `--kv-transfer-config` in a disaggregated prefill/decode
    # deployment: 'kv_producer' (prefill engine), 'kv_consumer' (decode engine)
    # or 'kv_both'. None for a unified server.
    kv_role: str | None = None
    input_len: int | None = None
    output_len: int | None = None
    concurrency: int | None = None
    # Every concurrency the benchmark measured (all replicas), so detectors can
    # judge loads the profile was not taken at.
    benchmark_concurrencies: tuple[int, ...] = ()
    # Engine/server log text (vLLM startup log), when available.
    server_log: str | None = None
    # TTFT matters for this workload (interactive serving). Gates
    # recommendations that trade TTFT for throughput (see guardrails.py).
    ttft_sensitive: bool = True

    @property
    def decode_tokens_per_req(self) -> int:
        """Tokens per running request per decode step (1 + speculative tokens)."""
        return 1 + max(0, self.num_speculative_tokens or 0)

    @property
    def is_pd_decode(self) -> bool:
        """Whether this is the decode engine of a disaggregated P/D pair."""
        return self.kv_role == 'kv_consumer'

    def config_get(self, *names: str) -> str | None:
        if self.serving_config is None:
            return None
        for n in names:
            if n in self.serving_config:
                return self.serving_config[n]
        return None

    def finalize(self) -> 'ServingContext':
        """Fills derived integer fields from the serving config when absent."""
        if self.serving_command and self.serving_config is None:
            self.serving_config = parse_serving_command(self.serving_command)

        def as_int(*names):
            v = self.config_get(*names)
            try:
                return int(v) if v is not None else None
            except ValueError:
                return None

        def as_float(*names):
            v = self.config_get(*names)
            try:
                return float(v) if v is not None else None
            except ValueError:
                return None

        def as_bool(*names):
            v = self.config_get(*names)
            if v is None:
                return None
            low = str(v).strip().lower()
            if low in ('true', '1', 'yes', 'on'):
                return True
            if low in ('false', '0', 'no', 'off'):
                return False
            return None

        self.max_num_seqs = self.max_num_seqs or as_int(
            'max-num-seqs', 'max_num_seqs')
        self.max_num_batched_tokens = self.max_num_batched_tokens or as_int(
            'max-num-batched-tokens', 'max_num_batched_tokens')
        self.max_model_len = self.max_model_len or as_int(
            'max-model-len', 'max_model_len', 'max_target_length')
        self.kv_cache_memory_bytes = self.kv_cache_memory_bytes or as_int(
            'kv-cache-memory-bytes', 'kv_cache_memory_bytes')
        if self.enable_prefix_caching is None:
            self.enable_prefix_caching = as_bool('enable-prefix-caching',
                                                 'enable_prefix_caching')
        self.tp = self.tp or as_int(
            'tensor-parallel-size',
            'tensor_parallel_size',
            'ici_tensor_parallelism',
        )
        self.dp = self.dp or as_int('data-parallel-size', 'data_parallel_size',
                                    'ici_data_parallelism')
        self.dcp = self.dcp or as_int('decode-context-parallel-size',
                                      'decode_context_parallel_size')
        self.pcp = self.pcp or as_int('prefill-context-parallel-size',
                                      'prefill_context_parallel_size')
        self.cp_kv_cache_interleave_size = (self.cp_kv_cache_interleave_size
                                            or as_int(
                                                'cp-kv-cache-interleave-size',
                                                'cp_kv_cache_interleave_size',
                                                'dcp-kv-cache-interleave-size',
                                                'dcp_kv_cache_interleave_size',
                                            ))
        ep_flag = as_bool('enable-expert-parallel', 'enable_expert_parallel')
        if not self.ep and ep_flag:
            # vLLM: experts are sharded over the whole TP x DP group.
            world = (self.tp or 1) * (self.dp or 1) * max(1, self.pcp or 1)
            moe_tp = as_int('TPU_MOE_EP_TP_SIZE', 'tpu_moe_ep_tp_size')
            if moe_tp and moe_tp > 1 and world % moe_tp == 0:
                self.ep = world // moe_tp
            else:
                self.ep = world
        ici_ep = as_int('ici_expert_parallelism')
        if not self.ep and ici_ep and ici_ep > 1:
            self.ep = ici_ep
        self._parse_speculative_config(as_int, as_float)
        if self.kv_role is None:
            raw = self.config_get('kv-transfer-config', 'kv_transfer_config')
            try:
                kv_transfer = json.loads(raw) if raw else None
            except json.JSONDecodeError:
                kv_transfer = None
            if isinstance(kv_transfer, dict) and kv_transfer.get('kv_role'):
                self.kv_role = str(kv_transfer['kv_role'])
        return self

    def _parse_speculative_config(self, as_int, as_float) -> None:
        """Extracts speculative decoding fields from --speculative-config JSON."""
        raw = self.config_get('speculative-config', 'speculative_config')
        spec_dict: dict[str, Any] = {}
        if raw:
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, dict):
                    spec_dict = loaded
            except json.JSONDecodeError:
                m_tok = re.search(
                    r'num_speculative_tokens[\'"]?\s*[:=]\s*(\d+)', raw)
                if m_tok:
                    spec_dict['num_speculative_tokens'] = int(m_tok.group(1))
                m_method = re.search(
                    r'method[\'"]?\s*[:=]\s*[\'"]([^\'"]+)[\'"]', raw)
                if m_method:
                    spec_dict['method'] = m_method.group(1)
                m_acc = re.search(
                    r'synthetic_acceptance_length[\'"]?\s*[:=]\s*(\d+(?:\.\d+)?)',
                    raw,
                )
                if m_acc:
                    spec_dict['synthetic_acceptance_length'] = float(
                        m_acc.group(1))
        if self.speculative_method is None:
            method = spec_dict.get('method') or self.config_get(
                'speculative_config.method')
            if method:
                self.speculative_method = str(method)
        if self.num_speculative_tokens is None:
            ntok = spec_dict.get('num_speculative_tokens')
            if ntok is not None:
                try:
                    self.num_speculative_tokens = int(ntok)
                except (TypeError, ValueError):
                    pass
            if self.num_speculative_tokens is None:
                self.num_speculative_tokens = as_int(
                    'num_speculative_tokens',
                    'speculative_config.num_speculative_tokens',
                )
        if self.synthetic_acceptance_length is None:
            syn = spec_dict.get('synthetic_acceptance_length')
            if syn is not None:
                try:
                    self.synthetic_acceptance_length = float(syn)
                except (TypeError, ValueError):
                    pass
            if self.synthetic_acceptance_length is None:
                self.synthetic_acceptance_length = as_float(
                    'synthetic_acceptance_length',
                    'speculative_config.synthetic_acceptance_length',
                )


_ASYNC_ON_LOG = re.compile(
    r'Asynchronous scheduling is enabled|DPScheduler \(Async = True\)'
    r'|Disabling NCCL for DP synchronization when using async scheduling'
    r'|async_scheduling=True',
    re.I,
)
_ASYNC_OFF_LOG = re.compile(
    r'Asynchronous scheduling is disabled|DPScheduler \(Async = False\)'
    r'|Disabling asynchronous scheduling|async_scheduling=False',
    re.I,
)
_ASYNC_BLOCKED_LOG = re.compile(
    r'Async scheduling\b.*\b(?:will be|is) disabled', re.I)


@dataclasses.dataclass(frozen=True)
class AsyncSchedulingLog:
    """What a vLLM server log says about async scheduling."""

    enabled: bool
    line: str
    blocked: bool = False

    def __bool__(self) -> bool:
        return self.enabled


def async_scheduling_from_log(log: str | None) -> AsyncSchedulingLog | None:
    """Whether the vLLM server log says async scheduling is on (None: unknown).

  vLLM logs "Asynchronous scheduling is enabled." at engine start (and the
  resolved `async_scheduling=` in its config dump), including when it was
  auto-enabled without the CLI flag.

  Args:
    log: Server log text.
  """
    enabled, line, blocker = None, '', ''
    for raw in (log or '').splitlines():
        if 'sync' not in raw.lower():
            continue
        if _ASYNC_BLOCKED_LOG.search(raw):
            enabled, line = False, raw.strip()
            blocker = line
        elif _ASYNC_OFF_LOG.search(raw):
            enabled, line = False, raw.strip()
        elif _ASYNC_ON_LOG.search(raw):
            enabled, line = True, raw.strip()
    if enabled is None:
        return None
    if enabled:
        return AsyncSchedulingLog(enabled=True, line=line)
    return AsyncSchedulingLog(enabled=False,
                              line=blocker or line,
                              blocked=bool(blocker))


_ENV_ASSIGN = re.compile(r'^([A-Z][A-Z0-9_]*)=(.*)$')
_YML_OVERRIDE = re.compile(r'^([a-z][a-z0-9_]*)=(.*)$')
_MAXTEXT_FLAG = 'maxtext_flag'
_SHELL_SEPARATORS = ('&&', ';', '||', '|', '\\', '&')
_REDIRECT_OP = re.compile(r'^[0-9&]?(?:>>?|<)$')
_REDIRECT = re.compile(r'^[0-9&]?(?:>>?|<)(?:&[0-9]+|[^<>&]+)?$')
_SHORT_FLAGS = {
    '-tp': 'tensor-parallel-size',
    '-dp': 'data-parallel-size',
    '-pp': 'pipeline-parallel-size',
    '-ep': 'enable-expert-parallel',
}
_BOOLEAN_FLAGS = frozenset({
    'async-scheduling',
    'disable-cascade-attn',
    'disable-custom-all-reduce',
    'disable-log-stats',
    'disable-sliding-window',
    'enable-chunked-prefill',
    'enable-dbo',
    'enable-ep-weight-filter',
    'enable-eplb',
    'enable-expert-parallel',
    'enable-log-requests',
    'enable-lora',
    'enable-prefix-caching',
    'enforce-eager',
    'language-model-only',
    'skip-tokenizer-init',
    'trust-remote-code',
})
_NEGATIVE_NUMBER = re.compile(r'^-\d+(?:\.\d+)?$')


def _shell_tokens(command: str) -> list[str]:
    """Shell words of `command`; a `;` that ends a word is its own token."""
    try:
        toks = shlex.split(command.replace('\\\n', ' '))
    except ValueError:
        toks = command.split()
    out = []
    for t in toks:
        if len(t) > 1 and t.endswith(';'):
            out += [t[:-1], ';']
        else:
            out.append(t)
    return out


def _takes_value(key: str) -> bool:
    name = key.replace('_', '-')
    return name not in _BOOLEAN_FLAGS and not name.startswith('no-')


def is_boolean_flag(flag: str) -> bool:
    """Whether a vLLM CLI flag takes no value."""
    short = '-' + flag.lstrip('-')
    key = _SHORT_FLAGS.get(flag) or _SHORT_FLAGS.get(short) or flag.lstrip('-')
    return not _takes_value(key)


def short_flags(flag: str) -> list[str]:
    """vLLM short aliases of a long flag."""
    key = flag.lstrip('-').replace('_', '-')
    return [short for short, long in _SHORT_FLAGS.items() if long == key]


def _is_value(token: str) -> bool:
    """Whether `token` can be a flag's value rather than an option or operator."""
    return (token not in _SHELL_SEPARATORS and not _REDIRECT.match(token) and
            (not token.startswith('-') or bool(_NEGATIVE_NUMBER.match(token))))


def parse_serving_command(command: str) -> dict[str, str]:
    """Parses a serving launch command line into a flat dict.

  Handles env assignments (`A=1 vllm serve ...`), `export A=1 && ...` chains,
  `\\` line continuations, `--flag value` / `--flag=value` / bare boolean
  flags, and `--additional-config '{"sharding": {...}}'` JSON (flattened to
  dotted keys such as 'sharding.sharding_strategy.enable_dp_attention').

  Args:
    command: A shell command line (env assignments + CLI).

  Returns:
    Flat mapping of knob names to string values.
  """
    out: dict[str, str] = {}
    toks = _shell_tokens(command)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _SHELL_SEPARATORS or t == 'export':
            i += 1
            continue
        if _REDIRECT_OP.match(t):
            i += 2
            continue
        m = _ENV_ASSIGN.match(t) or _YML_OVERRIDE.match(t)
        if m:
            out[m.group(1)] = m.group(2)
            i += 1
            continue
        flag, eq, val = t.partition('=')
        if flag in _SHORT_FLAGS:
            key = _SHORT_FLAGS[flag]
        elif flag.startswith('--'):
            key = flag[2:]
        else:
            i += 1
            continue
        if not eq:
            nxt = toks[i + 1] if i + 1 < len(toks) else None
            if nxt is not None and _takes_value(key) and _is_value(nxt):
                val = nxt
                i += 1
            else:
                val = 'true'
        if key.replace('_', '-') == 'additional-config':
            try:
                _flatten('', json.loads(val), out)
            except json.JSONDecodeError:
                out[key] = val
        elif key.replace('-', '_') == _MAXTEXT_FLAG and '=' in val:
            name, _, value = val.partition('=')
            out[name] = value
        else:
            out[key] = val
            out[key.replace('-', '_')] = val
            if (key.startswith('no-')
                    or key.startswith('no_')) and val == 'true':
                pos = key[3:]
                out.setdefault(pos, 'false')
                out.setdefault(pos.replace('-', '_'), 'false')
        i += 1
    return out


def _flatten(prefix: str, obj: Any, out: dict[str, str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f'{prefix}.{k}' if prefix else str(k), v, out)
    else:
        out[prefix] = json.dumps(obj) if not isinstance(obj, str) else obj


def _coerce_json_value(val: str) -> Any:
    low = str(val).strip().lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return val


def _set_dotted(root: dict[str, Any], dotted_key: str, val: Any) -> None:
    parts = dotted_key.split('.')
    cur = root
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = val


_BOOL_FLAGS = frozenset({
    '--enable-expert-parallel',
    '--async-scheduling',
    '--enable-prefix-caching',
    '--enable-ep-weight-filter',
})


def apply_knob_changes(
    command: str | None,
    changes: Any,
    framework: str = 'auto',
) -> str:
    """Synthesizes a ready-to-run command or config patch from KnobChanges.

  If `command` is provided, updates existing env vars, CLI flags, and
  `--additional-config` JSON in place and appends new ones. If `command` is
  None/empty, formats a copy-pasteable override snippet for the framework.
  Knobs that cannot be spliced into a launch command (config-file fields)
  and out-of-tree patch requirements are appended as `#` comment
  lines so they are not silently dropped.
  """
    del framework
    seen: set[tuple[str, str]] = set()
    active = []
    for c in changes:
        if getattr(c, 'already_set', False):
            continue
        key = (c.kind, c.name)
        if key not in seen:
            seen.add(key)
            active.append(c)
    if not active:
        return command or ''

    envs = [c for c in active if c.kind == 'env']
    clis = [c for c in active if c.kind == 'cli']
    add_cfgs = [c for c in active if c.kind == 'additional_config']
    ymls = [c for c in active if c.kind == 'maxtext_yml']
    notes = _unspliced_knob_notes(active)

    if not command:
        parts: list[str] = []
        if envs:
            parts.append(' '.join(
                f'{c.name}={shlex.quote(str(c.suggested_value))}'
                for c in envs))
        cli_parts: list[str] = []
        for c in clis:
            flag = c.name
            val = str(c.suggested_value)
            if flag in _BOOL_FLAGS and val.lower() in ('true', '1'):
                cli_parts.append(flag)
            elif flag in _BOOL_FLAGS and val.lower() in ('false', '0'):
                cli_parts.append(f'--no-{flag[2:]}')
            else:
                cli_parts.append(f'{flag}={shlex.quote(val)}')
        if add_cfgs:
            ac_dict: dict[str, Any] = {}
            for c in add_cfgs:
                _set_dotted(ac_dict, c.name,
                            _coerce_json_value(c.suggested_value))
            cli_parts.append(f"--additional-config '{json.dumps(ac_dict)}'")
        if cli_parts:
            parts.append(' '.join(cli_parts))
        if ymls:
            parts.append('\n'.join(f'{c.name}: {c.suggested_value}'
                                   for c in ymls))
        snippet = (' \\\n  '.join(parts) if len(parts) > 1 else
                   (parts[0] if parts else ''))
        return '\n'.join(([snippet] if snippet else []) + notes)

    cmd = command.strip()
    uses_export = cmd.lstrip().startswith('export ')
    new_envs: list[str] = []
    for c in envs:
        name = c.name
        val = shlex.quote(str(c.suggested_value))
        pat = re.compile(rf'(?<![A-Z0-9_]){re.escape(name)}=(\S+)')
        if pat.search(cmd):
            cmd = pat.sub(f'{name}={val}', cmd, count=1)
        else:
            new_envs.append(f'{name}={val}')
    if new_envs:
        if uses_export:
            prefix = ' && '.join(f'export {e}' for e in new_envs) + ' && '
            cmd = prefix + cmd
        else:
            cmd = ' '.join(new_envs) + ' ' + cmd

    for c in clis:
        flag = c.name
        alt_flag = '--' + flag[2:].replace('-', '_')
        neg_flag = f'--no-{flag[2:]}'
        val = str(c.suggested_value)
        if flag in _BOOL_FLAGS:
            if val.lower() in ('true', '1'):
                if re.search(rf'(?<!\S){re.escape(neg_flag)}(?!\S)', cmd):
                    cmd = re.sub(
                        rf'(?<!\S){re.escape(neg_flag)}(?!\S)',
                        flag,
                        cmd,
                        count=1,
                    )
                elif not re.search(
                        rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?!\S)',
                        cmd,
                ):
                    cmd = f'{cmd} {flag}'
            else:
                if re.search(
                        rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?!\S)',
                        cmd,
                ):
                    cmd = re.sub(
                        rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?!\S)',
                        neg_flag,
                        cmd,
                        count=1,
                    )
                elif not re.search(rf'(?<!\S){re.escape(neg_flag)}(?!\S)',
                                   cmd):
                    cmd = f'{cmd} {neg_flag}'
            continue
        val_q = shlex.quote(val)
        pat = re.compile(
            rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?:=|\s+)(\S+)'
        )
        if pat.search(cmd):
            cmd = pat.sub(f'{flag}={val_q}', cmd, count=1)
        else:
            cmd = f'{cmd} {flag}={val_q}'

    if add_cfgs:
        ac_pat = re.compile(
            r"--additional[-_]config(?:=|\s+)(?:'([^']*)'|\"([^\"]*)\"|(\S+))")
        m = ac_pat.search(cmd)
        ac_dict = {}
        if m:
            raw_json = m.group(1) or m.group(2) or m.group(3) or '{}'
            try:
                ac_dict = json.loads(raw_json)
            except json.JSONDecodeError:
                ac_dict = {}
        for c in add_cfgs:
            _set_dotted(ac_dict, c.name, _coerce_json_value(c.suggested_value))
        new_ac = f"--additional-config '{json.dumps(ac_dict)}'"
        if m:
            cmd = cmd[:m.start()] + new_ac + cmd[m.end():]
        else:
            cmd = f'{cmd} {new_ac}'

    for c in ymls:
        key = c.name
        val = str(c.suggested_value)
        pat = re.compile(rf'(?<!\S){re.escape(key)}=\S+')
        if pat.search(cmd):
            cmd = pat.sub(f'{key}={val}', cmd, count=1)
        else:
            cmd = f'{cmd} {key}={val}'

    return '\n'.join([cmd] + notes)


_SPLICED_KINDS = frozenset({'env', 'cli', 'additional_config', 'maxtext_yml'})


def _unspliced_knob_notes(changes: Any) -> list[str]:
    """`#` lines for knobs not spliced into the command and required patches."""
    notes = [
        f'# Also set ({c.kind}): {c.name}={c.suggested_value}' for c in changes
        if c.kind not in _SPLICED_KINDS
    ]
    notes += [
        f'# {c.name} requires {c.requires_patch}' for c in changes
        if getattr(c, 'requires_patch', '')
    ]
    return notes

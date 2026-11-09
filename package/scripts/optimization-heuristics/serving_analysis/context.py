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
    rope_type: str = ''

    @property
    def is_moe(self) -> bool:
        return bool(self.num_experts and self.num_experts > 1)

    @property
    def is_mla(self) -> bool:
        return bool(self.kv_lora_rank)

    @classmethod
    def from_hf_dict(cls, d: Mapping[str, Any]) -> 'ModelConfig':
        """Parses a HuggingFace config.json dict (handles text_config nesting)."""
        if 'text_config' in d and isinstance(d['text_config'], dict):
            merged = dict(d['text_config'])
            merged.setdefault('_name_or_path', d.get('_name_or_path', ''))
            d = merged

        def g(*keys):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None

        rope = g('rope_scaling') or {}
        return cls(
            name=str(g('_name_or_path') or g('model_type') or ''),
            hidden_size=g('hidden_size', 'd_model', 'emb_dim', 'base_emb_dim'),
            num_layers=g('num_hidden_layers', 'num_layers',
                         'base_num_decoder_layers'),
            num_attention_heads=g('num_attention_heads', 'num_query_heads',
                                  'base_num_query_heads'),
            num_kv_heads=g('num_key_value_heads', 'num_kv_heads',
                           'base_num_kv_heads'),
            head_dim=g('head_dim'),
            num_experts=g('num_experts', 'n_routed_experts',
                          'num_local_experts'),
            num_experts_per_tok=g('num_experts_per_tok', 'moe_topk',
                                  'num_experts_per_token'),
            moe_intermediate_size=g('moe_intermediate_size',
                                    'base_moe_mlp_dim')
            or (g('intermediate_size') if g('num_local_experts') else None),
            max_position_embeddings=g('max_position_embeddings',
                                      'max_target_length'),
            sliding_window=g('sliding_window'),
            kv_lora_rank=g('kv_lora_rank'),
            rope_type=str(
                rope.get('rope_type', rope.get('type', '')
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
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    max_model_len: int | None = None
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

        self.max_num_seqs = self.max_num_seqs or as_int(
            'max-num-seqs', 'max_num_seqs')
        self.max_num_batched_tokens = self.max_num_batched_tokens or as_int(
            'max-num-batched-tokens', 'max_num_batched_tokens')
        self.max_model_len = self.max_model_len or as_int(
            'max-model-len', 'max_model_len', 'max_target_length')
        self.tp = self.tp or as_int('tensor-parallel-size',
                                    'tensor_parallel_size',
                                    'ici_tensor_parallelism')
        self.dp = self.dp or as_int('data-parallel-size', 'data_parallel_size')
        ep_flag = self.config_get('enable-expert-parallel',
                                  'enable_expert_parallel')
        if (not self.ep and ep_flag is not None
                and str(ep_flag).lower() in ('true', '1')):
            # vLLM: experts are sharded over the whole TP x DP group.
            self.ep = (self.tp or 1) * (self.dp or 1)
        return self


_ASYNC_ON = re.compile(
    r'asynchronous scheduling is enabled|async_scheduling=True', re.I)
_ASYNC_OFF = re.compile(
    r'asynchronous scheduling is disabled|async_scheduling=False', re.I)


def async_scheduling_from_log(log: str | None) -> bool | None:
    """Whether the vLLM server log says async scheduling is on (None: unknown).

  vLLM logs "Asynchronous scheduling is enabled." at engine start (and the
  resolved `async_scheduling=` in its config dump), including when it was
  auto-enabled without the CLI flag.

  Args:
    log: Server log text.
  """
    if not log:
        return None
    if _ASYNC_ON.search(log):
        return True
    if _ASYNC_OFF.search(log):
        return False
    return None


_ENV_ASSIGN = re.compile(r'^([A-Z][A-Z0-9_]*)=(.*)$')
_SHELL_SEPARATORS = ('&&', ';', '||', '|', '\\')


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
    try:
        toks = shlex.split(command.replace('\\\n', ' '))
    except ValueError:
        toks = command.split()
    i = 0
    while i < len(toks):
        t = toks[i]
        if t in _SHELL_SEPARATORS or t == 'export':
            i += 1
            continue
        m = _ENV_ASSIGN.match(t)
        if m:
            out[m.group(1)] = m.group(2)
            i += 1
            continue
        if t.startswith('--'):
            key, eq, val = t[2:].partition('=')
            if not eq:
                nxt = toks[i + 1] if i + 1 < len(toks) else None
                if (nxt is not None and not nxt.startswith('--')
                        and nxt not in _SHELL_SEPARATORS):
                    val = nxt
                    i += 1
                else:
                    val = 'true'
            if key.replace('_', '-') == 'additional-config':
                try:
                    _flatten('', json.loads(val), out)
                except json.JSONDecodeError:
                    out[key] = val
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
                    cmd = re.sub(rf'(?<!\S){re.escape(neg_flag)}(?!\S)',
                                 flag,
                                 cmd,
                                 count=1)
                elif not re.search(
                        rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?!\S)',
                        cmd):
                    cmd = f'{cmd} {flag}'
            else:
                if re.search(
                        rf'(?<!\S)(?:{re.escape(flag)}|{re.escape(alt_flag)})(?!\S)',
                        cmd):
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

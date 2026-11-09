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
"""Unified, framework-agnostic timeline table for XProf profiles.

Loads trace-viewer-level events (all TPU devices + host threads) into compact
pandas frames from either:
  * an Events DB parquet, as tpu-diagnostics exports it from an XProf trace
    (XProf's `create_events_db`), or
  * a cached directory of parquet files written by `TraceTable.save()`.

Frames (all times are absolute nanoseconds):
  ops      : one row per device XLA op execution
             [host, device, module, op, start_ns, end_ns, self_ns]
  op_meta  : one row per (module, op)
             [module, op, kernel_text, tf_op_name, tf_op_type, hlo_category,
              source_line]
  modules  : one row per device XLA module (program) execution
             [host, device, module, start_ns, end_ns]
  host     : one row per host (CPU) trace event
             [host, thread_name, name, start_ns, end_ns]
"""

# pylint: disable=missing-function-docstring,missing-class-docstring,g-doc-args,g-doc-return-or-yield,g-docstring-has-escape,g-long-ternary,g-explicit-length-test,g-complex-comprehension,protected-access,cell-var-from-loop,comparison-with-itself,unused-argument

import dataclasses
import os
import re

import pandas as pd


def read_parquet(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """A parquet file as a DataFrame, through duckdb when it is installed.

    pandas needs pyarrow (or fastparquet) to read parquet; duckdb does not,
    and the package's host requirements already include it.
    """
    try:
        import duckdb  # pylint: disable=g-import-not-at-top
    except ImportError:
        return pd.read_parquet(path, columns=columns)
    cols = ', '.join(f'"{c}"' for c in columns) if columns else '*'
    with duckdb.connect() as con:
        return con.execute(f'SELECT {cols} FROM read_parquet(?)', [path]).df()


_OPS_COLS = ['host', 'device', 'module', 'op', 'start_ns', 'end_ns', 'self_ns']
_META_COLS = [
    'module',
    'op',
    'kernel_text',
    'tf_op_name',
    'tf_op_type',
    'hlo_category',
    'source_line',
]
_MOD_COLS = ['host', 'device', 'module', 'start_ns', 'end_ns']
_HOST_COLS = ['host', 'thread_name', 'name', 'start_ns', 'end_ns']

_KERNEL_TEXT_CHARS = 400
_HOST_NAME_CHARS = 200
_MODULE_PREFIX = 'HLO Module:'
_HLO_CATEGORY_RE = re.compile(r'hlo_category=([^,]*)')
_HOST_FROM_PARTITION_RE = r'([^/]+)\.events_db$'


def _empty(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype='object') for c in cols})


@dataclasses.dataclass
class TraceTable:
    """Timeline frames for one profile session."""

    ops: pd.DataFrame
    op_meta: pd.DataFrame
    modules: pd.DataFrame
    host: pd.DataFrame
    session_id: str = ''
    source: str = ''

    # ---------------------------------------------------------------------------
    # Derived helpers
    # ---------------------------------------------------------------------------
    @property
    def devices(self) -> list[str]:
        """Sorted list of TensorCore device keys ('host/device')."""
        if self.modules.empty:
            return []
        hosts = self.modules['host'].fillna('').astype(str)
        devs = self.modules['device'].fillna('').astype(str)
        keys = (hosts + '/' + devs).unique()
        return sorted((str(k) for k in keys if k and k != '/'),
                      key=_device_sort_key)

    def ops_with_meta(self) -> pd.DataFrame:
        """Ops joined with per-op metadata (cached)."""
        cached = getattr(self, '_ops_meta_cache', None)
        if cached is not None:
            return cached
        df = self.ops.merge(self.op_meta, on=['module', 'op'], how='left')
        for c in (
                'kernel_text',
                'tf_op_name',
                'tf_op_type',
                'hlo_category',
                'source_line',
        ):
            df[c] = df[c].fillna('')
        object.__setattr__(self, '_ops_meta_cache', df)
        return df

    # ---------------------------------------------------------------------------
    # Persistence (hermetic tests / eval fixtures)
    # ---------------------------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        for name in ('ops', 'op_meta', 'modules', 'host'):
            getattr(self,
                    name).to_parquet(os.path.join(directory,
                                                  f'{name}.parquet'))
        safe_sid = str(self.session_id).replace('\r', '').replace('\n', '')
        safe_src = str(self.source).replace('\r', '').replace('\n', '')
        with open(os.path.join(directory, 'META'), 'w') as f:
            f.write(f'session_id={safe_sid}\nsource={safe_src}\n')

    @classmethod
    def load(cls, directory: str) -> 'TraceTable':
        frames = {
            name: read_parquet(os.path.join(directory, f'{name}.parquet'))
            for name in ('ops', 'op_meta', 'modules', 'host')
        }
        meta = {}
        meta_path = os.path.join(directory, 'META')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        meta[k] = v
        return cls(
            **frames,
            session_id=meta.get('session_id', ''),
            source=f'cache:{directory}',
        )

    # ---------------------------------------------------------------------------
    # Loaders
    # ---------------------------------------------------------------------------
    @classmethod
    def from_events_frame(cls,
                          events: pd.DataFrame,
                          session_id: str = '',
                          source: str = 'frame') -> 'TraceTable':
        """Builds a TraceTable from a raw Events DB frame (any backend).

    Args:
      events: DataFrame with Events DB columns (device, category, kernel_name,
        hlo_module, hlo_op, tf_op_name, tf_op_type, start_ns, end_ns,
        self_time_ns, trace_args, source_line, thread_name, optional 'host').
      session_id: Session identifier for links.
      source: Free-form provenance string.

    Returns:
      A normalized TraceTable.
    """
        ev = events.copy()
        for c in (
                'device',
                'category',
                'kernel_name',
                'hlo_module',
                'hlo_op',
                'tf_op_name',
                'tf_op_type',
                'trace_args',
                'source_line',
                'thread_name',
        ):
            if c not in ev.columns:
                ev[c] = ''
            ev[c] = ev[c].fillna('').astype(str)
        if 'host' not in ev.columns:
            ev['host'] = 'host0'
        for tc in ('start_ns', 'end_ns'):
            if tc not in ev.columns:
                ev[tc] = 0
            ev[tc] = pd.to_numeric(ev[tc],
                                   errors='coerce').fillna(0).astype('int64')
        if 'self_time_ns' not in ev.columns:
            ev['self_time_ns'] = ev['end_ns'] - ev['start_ns']
        else:
            ev['self_time_ns'] = (pd.to_numeric(
                ev['self_time_ns'], errors='coerce').fillna(0).astype('int64'))
        dev_lower = ev['device'].str.lower()
        is_tpu = dev_lower.str.startswith('tpu')
        is_cpu = dev_lower.str.startswith('cpu') | (ev['category'] == 'host')

        op_rows = ev[is_tpu & (ev['category'] == 'XLA Ops')]
        ops = pd.DataFrame({
            'host': op_rows['host'],
            'device': op_rows['device'],
            'module': op_rows['hlo_module'],
            'op': op_rows['hlo_op'],
            'start_ns': op_rows['start_ns'].astype('int64'),
            'end_ns': op_rows['end_ns'].astype('int64'),
            'self_ns': op_rows['self_time_ns'].astype('int64'),
        })
        meta_src = op_rows.drop_duplicates(['hlo_module', 'hlo_op'])
        op_meta = pd.DataFrame({
            'module':
            meta_src['hlo_module'],
            'op':
            meta_src['hlo_op'],
            'kernel_text':
            meta_src['kernel_name'].str.slice(0, _KERNEL_TEXT_CHARS),
            'tf_op_name':
            meta_src['tf_op_name'],
            'tf_op_type':
            meta_src['tf_op_type'],
            'hlo_category':
            (meta_src['trace_args'].str.extract(_HLO_CATEGORY_RE,
                                                expand=False).fillna('')),
            'source_line':
            meta_src['source_line'],
        })
        mod_rows = ev[is_tpu & (ev['category'] == 'XLA Modules')]
        modules = pd.DataFrame({
            'host':
            mod_rows['host'],
            'device':
            mod_rows['device'],
            'module': (mod_rows['kernel_name'].str.replace(_MODULE_PREFIX,
                                                           '',
                                                           regex=False)),
            'start_ns':
            mod_rows['start_ns'].astype('int64'),
            'end_ns':
            mod_rows['end_ns'].astype('int64'),
        })
        host_rows = ev[is_cpu]
        host = pd.DataFrame({
            'host':
            host_rows['host'],
            'thread_name':
            host_rows['thread_name'],
            'name':
            host_rows['kernel_name'].str.slice(0, _HOST_NAME_CHARS),
            'start_ns':
            host_rows['start_ns'].astype('int64'),
            'end_ns':
            host_rows['end_ns'].astype('int64'),
        })
        return cls(
            ops=ops.reset_index(drop=True),
            op_meta=op_meta.reset_index(drop=True),
            modules=modules.sort_values('start_ns').reset_index(drop=True),
            host=host.sort_values('start_ns').reset_index(drop=True),
            session_id=session_id,
            source=source,
        )

    @classmethod
    def from_parquet(cls, path: str, session_id: str = '') -> 'TraceTable':
        """Loads an Events DB parquet file (XProf `create_events_db` output)."""
        return cls.from_events_frame(
            read_parquet(path, columns=_EVENT_COLS),
            session_id=session_id,
            source=f'parquet:{path}',
        )


# Events DB columns the table reads; the rest of the parquet is never loaded.
_EVENT_COLS = [
    'device',
    'category',
    'kernel_name',
    'hlo_module',
    'hlo_op',
    'tf_op_name',
    'tf_op_type',
    'start_ns',
    'end_ns',
    'self_time_ns',
    'trace_args',
    'source_line',
    'thread_name',
]


def _device_sort_key(key: str) -> tuple[str, int, str]:
    host, _, dev = key.partition('/')
    m = re.search(r'(\d+)$', dev)
    return (host, int(m.group(1)) if m else -1, dev)

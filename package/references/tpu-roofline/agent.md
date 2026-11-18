---
references:
- references/tpu-roofline
scripts:
- scripts/tpu-roofline/roofline.py
- scripts/tpu-roofline/serving.py
- scripts/tpu-roofline/derivation.py
- scripts/tpu-roofline/cost_estimator/collective.py
- scripts/tpu-roofline/configs/glm-5.2-fp8-128k-1chip-t1024.json
- scripts/tpu-roofline/configs/glm-5.2-fp8-256k1k-2x2x2.json
- scripts/tpu-roofline/configs/glm-5.2-fp8-8k1k-2x2x2.json
- scripts/tpu-roofline/roofline_data.py
- scripts/tpu-roofline/README.md
---

# Roofline procedure

Work out how fast something *could* run on a TPU, from shapes and hardware limits alone.

Everything here uses one formula:

```
time = max(HBM time, compute time) + collective time
```

Compute and memory overlap inside an op; collectives are charged on top.

Every number you produce is **modelled**. The scripts are plain arithmetic under the local
`python3` and are read-only: write every output into the request's `output_dir`, or
`{{runs}}/<run_id>/artifacts/roofline/` when it names none.

## Which of the three are you being asked for?

| the ask | go to |
| --- | --- |
| a block of code, a kernel, a fused region | Path A |
| one named op — "MLA QK", "the MoE down-projection" | Path A |
| a whole model on a slice | Path B |
| `rows` at `cells` — the rows a gap affects | Path B, rows only |

If it is ambiguous, prefer A. It is cheap, and a model roofline needs inputs you may not have.

---

## The hardware block — from the knowledge base, for the request's `tpu`

Both paths take every hardware spec from here, never from memory or a default. The few modelling
constants `roofline.py` also needs are not specs; step 3 says where they come from.

1. **Generation key.** Drop the chip count from `tpu.accelerator_type`, as the knowledge base's
   `tpu-generation-specs` entry does: `tpu7x-8` → `tpu7x`, `v6e-8` → `v6e`.
2. **Look it up.** Send {{agent:tpu-knowledge-base}} one `kb-request` with `run_id` and `query`:
   the generation key and `<key> tpu specs: peak bf16 and fp8 per chip, HBM capacity and
   bandwidth, ICI bandwidth, topologies, TensorCores per chip, VMEM, VPU peak, MXU size`.
3. **Fill the config's `hardware`** from the vendor spec page it cites. `roofline.py` is per core:
   with `d2d` (two TensorCores per chip on a die-to-die link), divide per-chip numbers by two.

   | field | from |
   | --- | --- |
   | `gen`, `slice_dims` | the generation key in `roofline.py`'s naming (`tpu7x` → `v7x`; `v6e` stays); `tpu.topology` |
   | `peak_bf16`, `peak_fp8` | peak per dtype |
   | `hbm_bw`, `hbm_capacity` | HBM bandwidth and capacity |
   | `ici_bw` | ICI bandwidth, per link one way |
   | `d2d`, `d2d_bw` | the page's `d2d` and `d2d_bw` rows (a single-die chip has no D2D link) |
   | `peak_vpu_*`, `mxu_min_dim`, `sparsecore_vmem_bytes` | the page's rows of those names |

   Every field above is a **spec**: take it from the page's Roofline inputs table, or another page
   the lookup cited. A spec no cited page states is **missing**: return `invalid_input` with one
   `missing` entry per field, saying what the knowledge base lacked. Never carry a spec over from a
   copied config or another generation.

   The rest are **modelling constants**, not hardware specs, and no spec page states them:
   `hop_latency`, `d2d_hop_latency`, `ici_contention`, `memory_utilization` and
   `resident_overhead_gb`. Keep the starting config's value, and list each in the result's
   `hardware.assumed` as `field=value`, with its basis from the config's `_provenance` in the
   report's Inputs table. `kernel_efficiency` stays 1.0: a ceiling, not a spec.
4. **Record the source of each field** — URL or page, and its `retrieved` (or `verified`) date — in
   the config's `_provenance.hardware`, and in the report's Inputs table source column.

If the lookup serves no spec page for the generation (`near`, `uncited`, `cited-only`) and no
fallback covers it, return `invalid_input` for `tpu` with what the lookup said.

---

## Path A — a block of code, or one op

### 1. Get the shapes

Read the code. You need, per call: input and output tensor shapes, weight shapes, dtypes, and how
many times it runs per forward pass. If the dtypes are not in the code, say so in the output rather
than assuming bf16.

### 2. Read the matching spec in `model/common/`

**Do not skip this.** [`model/common/<op>/<op>.md`]({{reference:tpu-roofline/model/common}}) (some one level deeper, such as `attention/mla/mla.md`)
lists the sub-operations a component actually has, and the point of reading it is to not leave any
out. The ones that get forgotten:

- **Anything on the VPU** — normalisation, residual adds, RoPE, softmax, SwiGLU, top-k, permutes.
  The VPU runs far below the MXU (the hardware block's `peak_vpu_*`), so a small FLOP count is not automatically free.
- **Data movement that is not a collective** — the MoE token permute is 4.8% of prefill and is
  neither a GEMM nor a collective.
- **Activation traffic.** A GEMM reads its input and writes its output. For `MLA: Query Down Proj`
  at 8K tokens that is 80 of its 92 MiB — the weights are the small part.

### 3. Compute it

```
FLOPs    = 2 × tokens × max(contracting_dim, 256) × output_dim × n_matrices
compute  = FLOPs / peak(all operand dtypes)     # one wide operand → the slower rate
HBM      = weights + activations in + activations out (+ KV for attention)
memory   = HBM / hbm_bw
```

Hardware numbers from the hardware block above.

**The contracting dimension ceil-pads to 256** ({{reference:tpu-roofline/model/common/REQUIREMENTs.md}}
§MXU). Path B pads every matmul for you; here you do it by hand. The trap is an op that is a matmul
but is not described as one — a weighted sum, a score aggregation, a head reduction. `model/common`
states the useful work; the MXU charges the padded work. On GLM-5.2 at the 128K cell:

| Operator | contracts over | multiplier | charged |
| --- | --- | --- | --- |
| `DSA Indexer: Weighted Sum` | `H_idx` 32 | 8× | 68.72 GFLOP for 8.59 of work |
| `MLA: QK Rope` | `D_rope` 64 | 4× | 68.72 GFLOP for 17.18 of work |
| `DSA Indexer: QK` | `D_idx` 128 | 2× | 2,199 GFLOP for 1,100 of work |

Quote both numbers when the multiplier is over 2×: "68.72 GFLOP, of which 51.54 is padding" says a
kernel fix exists.

If there is a collective, price it with {{script:tpu-roofline/cost_estimator/collective.py}}
and get the payload from [`parallelism/<scheme>/<scheme>.md`]({{reference:tpu-roofline/parallelism}}).
Never write your own ring formula.

Do the arithmetic in Python, not in your head.

---

## Path B — a whole model

The arithmetic lives in {{script:tpu-roofline/roofline.py}}, which is fixed and tested. Your
job is to supply a config, not to compute anything.

### 1. Check you have everything

If the request names a config, or one in {{script:tpu-roofline/configs}} matches the model and
slice, start from it, refill its `hardware` from the hardware block, and skip to step 2. Name every
field you change and why.

Otherwise, read the model yourself:

- **Shapes.** Start with [`model/architecture/<family>/<size>.md`]({{reference:tpu-roofline/model/architecture}}).
  With no file for the model, build one from the checkpoint's `config.json`, then
  `vllm/model_executor/models/<arch>.py` for what it does not state, and write it to
  `<family>-<size>.md` in the output directory in the format of
  {{reference:tpu-roofline/model/architecture/architecture.md}}.
- **Numerics.** Weight dtype from `quantization_config` (check `modules_to_not_convert`). KV cache
  dtype from the server's `--kv-cache-dtype` (`"auto"` means the model dtype).
- **Layers per sub-block.** GLM-5.2 has 78 layers but only 21 run the indexer; the architecture file
  gives the rule. Assuming 78 overstates the indexer terms by nearly 4×.
- **Activation dtypes.** Pin what you can and name the sites you could not. An unpinned activation
  dtype is worth 2× on every GEMM that uses it.

The TPU comes in `tpu`; fill the config's `hardware` from the hardware block above, replacing a
starting config's numbers field by field. Two inputs are choices, not facts, and must come in the
request (`prompt`, `context` or `cells`):

| | |
| --- | --- |
| **Sharding** | TP, DP, PP, PCP, DCP, and whether EP and SP are on |
| **Workload** | input length, output length, `max-num-batched-tokens`, decode concurrency — or `cells` |

**If any is missing, return `invalid_input`** with one `missing` entry per input and why it matters,
and list what you already read in `summary`. Do not pick a default: a wrong input gives a confident
wrong answer.

### 2. Write the config and run it

Copy the nearest config into the output directory and edit the copy. When the request carries
`cells`, its `phases` list is exactly those cells, one entry each, named by the cell. Then:

```bash
python3 {{script:tpu-roofline/roofline.py}} --selftest                     # first, every time
python3 {{script:tpu-roofline/roofline.py}} <cfg>.json --layer-csv <out>.layers.csv > <out>.raw.txt
python3 {{script:tpu-roofline/roofline_data.py}} --run <cfg>.json <out>.layers.csv -o <output_dir>/roofline_data.json
python3 {{script:tpu-roofline/serving.py}}  <cfg>.json --concurrency <C> --both   # whole model only
```

`roofline_data.json` is the CSV as one table for callers' joins: a `rows` entry per regime (phase),
layer kind and operator, with its parent, count, floor and bound, plus the layer type map.
`floor_ms` is one layer summed over the regime's `passes`; `layer_total_ms` is that × count;
`floor_ms_per_pass` is `floor_ms / passes`, the floor of one engine step. `shape` holds an op's
local, per-core dims for one pass, each checked against the op's FLOPs: `batch`, `m`, `k`, `n`,
`n_mat` for MXU ops (`n_mat` [k, n] weights, priced at `2·batch·m·max(k, mxu_min_dim)·n·n_mat`),
`elems` for VPU ops, plus `dtype` and `dtype_b`. Fields that change between passes are listed per
pass under `by_pass`. Fused parents, totals, collectives and ops roofline.py builds without
operand dims have `shape: null`; op rows say why in `comment`. Pass one `--run` per config when
regimes need separate configs.

The selftest checks 23 numbers against a hand-built reference. If it fails, return `failed` with
its output: nothing downstream means anything.

### 3. Check nothing is missing from the op list

Walk `model/common/` for each component the model uses and confirm every op in the spec shows up in
the output. Report anything that does not line up.

### 4. Name the layer kinds

`roofline.py` prints one op table per kind of layer, with the kind's key. You supply the name:

| key | name it |
| --- | --- |
| `dense_idx` | Full Indexer Dense Layer |
| `dense` | Dense Layer |
| `moe_idx` | Full Indexer MoE Layer |
| `moe` | Index Share MoE Layer |

Empty kinds are dropped. Rename for the model in front of you; keep the key beside the name once.

### 5. Read the rows correctly

- **A `sub` row is already inside its `fused` parent.** Show the parent, indent the children, never
  sum both.
- **Only leaf rows satisfy `max(mem, compute) + ici = roofline`.** Parents and `TOTAL`s sum parts.
- **Memory Occupancy is HBM residency per core, not traffic**
  ({{reference:tpu-roofline/model/common/REQUIREMENTs.md}} §TRANSFER and OCCUPANCY).
  Occupancy peaks, it does not add, wherever things are not live together: across a phase's
  passes each column is the max over passes; a TOTAL's activations are the max over its ops (a
  fused parent sums its children); weights and KV sum; the layer-sums TOTAL takes the max
  activation across kinds and sums weights and KV.
- **Times are summed over every pass in the phase.** A chunked prefill's passes are in the number.
- ICI splits into bandwidth time (falls with the payload) and latency time (does not).

### Rows only

When the request names `rows`, price only those rows at each cell: read them from
`<out>.layers.csv`, one `rows` entry per row and cell (`Phase`) with its `layer_total_ms` (`Layer
Total (ms)` summed over the layer kinds that run it) and `bound_by` (the CSV's `ICI` is `collective`). A row the CSV does not have is listed in `summary` with the
nearest names it does have; never price it by hand alongside script numbers. Run `roofline_data.py`
as in step 2. Skip `serving.py` and the full report; write a short `<out>.md` with the table and the
reproduce commands.

---

## What to hand back

Lead with the numbers, keep the prose short.

**For one op or a block:**

```
<op name>, <shape summary>

  compute     X.XXX ms   <FLOPs> at <dtype>
  HBM         X.XXX ms   <bytes> = weights <w> + act in <i> + act out <o>
  collective  X.XXX ms   <kind> over <group>          (omit if there is none)
  ------------------------------------------------
  roofline    X.XXX ms   bound by <HBM | MXU | VPU | collective>
  × <n> per forward pass = X.XX ms
```

Then a sentence or two: what binds it, and what would have to change to move it.

**For a model:** write `<out>.md` beside the raw output and CSV, reproduce commands at the top, then
these sections in order. A section you cannot fill is printed with the reason it is empty.

1. **Inputs** — model shapes, numerics, hardware, sharding, workload (ISL, OSL, concurrency, `mnbt`
   → prefill chunks), decode `kv_len`, each with a **source** column. The script's
   `numerics defaulted to …` line goes here verbatim.
2. **Headline, per core, with ×fleet and ×chip**, each with its formula: T_prefill, T_decode
   (= TPOT), TTFT, E2E per request, QPS, throughput, prefill share, KV pool and gate, HBM resident.
3. **Where the time goes, by block, per phase** — roll `--layer-csv` up into blocks in Python (`Role`
   in `op`/`fused`, `Layer Total (ms)`) and check it equals the phase total. A collective is a row
   whose only non-zero time is `ICI`. Then the boundness split and the top rows in **ms**.
4. **Per-layer latency** — per phase: layer kind name and key, count, ms/layer, all-layer ms and %;
   the op table for the kind that owns the phase; one line on what the other kinds change.
5. **Hand recomputes** — at least four numbers recomputed from the formula in an executed Python
   script: the largest prefill compute row, a padding-affected row where one exists, the decode
   weight read, and throughput from T_prefill and T_step. A mismatch is a finding; report it.
6. **Caveats** — hardware sources; 100% kernel efficiency and fully exposed collectives, so the
   result is a ceiling; each phase's collective share; carried or defaulted numerics; unmodelled
   terms (layout/copy, the quantize pass after bf16 staging, host overhead); every model component
   roofline.py has no op for (KDA, GDN, conv1d, local attention), each also listed in the result's
   `unmodelled`; the model 2 delta from `serving.py --both` and why one model fits this cell better.

Known weak spots: layout and copy time is not modelled (7% of prefill in a real profile),
collectives are assumed fully exposed, and the fixed cost of a small collective is underpriced by
about 2.6×. If `roofline.py` contradicts its spec in `model/common/`, report the line; do not edit
either.

## Files

| | |
| --- | --- |
| {{script:tpu-roofline/roofline.py}} | the arithmetic, one function per op |
| {{script:tpu-roofline/derivation.py}} | one operating point, step by step, checked against `serving.py` |
| {{script:tpu-roofline/README.md}} | how it works and what it gets wrong |
| [`model/common/`]({{reference:tpu-roofline/model/common}}) | what ops a component has |
| [`model/architecture/`]({{reference:tpu-roofline/model/architecture}}) | per-model shapes |
| [`parallelism/`]({{reference:tpu-roofline/parallelism}}) | which collectives a sharding issues |
| {{reference:tpu-roofline/COLLECTIVE.md}} | how to price a collective |

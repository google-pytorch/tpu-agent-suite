# Reading a vLLM serving trace on TPU: which layout a collective comes from

How to attribute the collectives and memory in an XProf trace of a vLLM
server on TPU to its layout (TP, DP, EP), before changing the layout.

## Applies when

- A serving profile shows a large share of all-reduce, all-gather or
  reduce-scatter, and the question is whether a different TP / DP / EP
  would help.
- A long time to first token needs explaining.
- KV-cache capacity per replica is in question.

## Collectives

- **All-reduce in an inference trace is tensor parallelism.** One per layer
  after the attention output projection and one after the MLP down
  projection, whenever TP > 1, however the KV heads are split. There is no
  backward pass, so no gradient all-reduce.
- **A collective-bound prefill favors fewer TP ranks.** When collectives take
  a large share of the prefill step (25% or more; 43% on Gemma-4-31B at
  TP 4 at 8k/1k with 128 users), halving TP shortens the all-reduce group
  and doubles the replicas that prefill in parallel. Decode reads every
  weight each step, so fewer TP ranks make each decode step slower: the
  change wins only at prefill-heavy points. See `index.py get
  vllm-tp-dp-plan` for the measured table.
- **MoE dispatch and combine show as all-gather / reduce-scatter pairs**, not
  all-to-all, and are the expert-parallel collective, not FSDP. The experts
  per device are in the grouped-matmul kernel name (`gmm_v2-g_<E>`). Adding
  TP does not shrink these.

## Long time to first token

Before blaming collectives, count the prefill steps a request waits for:
each replica works through its share of the queue in chunks of
`--max-num-batched-tokens`, so a request can wait about
(concurrency / DP) x input length / max-num-batched-tokens prefill steps. A
TTFT many times the prefill step time with a modest collective share is
queueing, not communication.

## KV cache

- Each vLLM engine logs `GPU KV cache size: N tokens` once at startup; with
  DP every rank logs its own (prefixed `EngineCore_DP<k>`), so multiply by
  the replicas for the deployment's total.
- GQA: TP shards KV heads only down to one per device; past that they are
  replicated, so KV heads per device = max(1, KV heads / TP).
- The RPA KV cache interleaves K and V: each KV head takes two rows on a
  device (one at head_dim 64, where K and V share a row), padded up to the KV
  dtype's packing (2 for bf16, 4 for fp8). One bf16 head per device (two
  rows) fills the packing; padding appears only when the rows fall short, or
  when TP exceeds the KV heads and heads are replicated.
- MLA keeps one latent per token on every TP rank, so TP does not divide the
  KV cache; only DP (and context parallelism) multiplies capacity.
- Sliding-window layers keep only their window per sequence under vLLM's
  hybrid KV-cache manager; Gemma-4-31B keeps 1024 tokens in 50 of 60 layers.

## Decode attention

- `--attention-backend CUSTOM`, which onboarding passes by default, runs the
  plugin's batched RPA kernel (trace names `RPAd-snh-...`); no knob sets its
  decode blocks.
- `FLASH_ATTN` runs the RPA v3 kernel (`RPAd-...-bkv_<block>_<chunk>`), whose
  stock decode branch sets block = chunk = max_model_len: every decode step
  computes over max_model_len slots, and sliding-window layers cannot skip
  blocks outside their window. `TPU_RPA_DECODE_BLOCK_SIZES` (with a plugin
  patch) sets per-layer blocks; on Gemma-3-27B it cut the decode step from
  21.9 to 11.9 ms.
- So when batched decode attention dominates a CUSTOM server's decode step,
  RPA v3 with sized decode blocks is the alternative to measure; models
  whose RPA v3 kernels fail to compile (Gemma-4-31B, VMEM) stay on CUSTOM.

## Related

- How onboarding plans TP and DP, and TP versus replicas measured:
  `index.py get vllm-tp-dp-plan`.
- Layouts that fail to start: `index.py get vllm-torchtpu-mesh-constraints`.
- Serving knobs measured on vLLM-TorchTPU:
  `index.py get vllm-torchtpu-measured-serving-knobs`.
- The rules that turn a trace into layout and knob changes:
  `index.py get vllm-sharding-heuristics` and
  `index.py get vllm-serving-heuristics`.

## Sources

- tpu-diagnostics's detector rules (`package/references/optimization-heuristics/detectors.md`,
  `package/scripts/optimization-heuristics/serving_analysis/`).
- Profiles of Gemma-3-27B, Gemma-4-31B and Qwen3.8-27B on tpu7x 2x2x1,
  2026-10-14 and 2026-10-15.

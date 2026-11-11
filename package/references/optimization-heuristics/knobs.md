---
scripts:
  - scripts/common/vllm_patches/rpa_decode_block_override.patch
---
# Serving heuristics: knobs and guardrails

Detectors only recommend knobs in `serving_analysis/knobs.py`. Each names
the file that defines it in its framework's installed package and a string
that proves it is there, so `knob_probe.py` can check the serving image
before a knob is measured.

| Framework | Kind | Where it is defined |
| --- | --- | --- |
| vLLM | CLI flag | `vllm/engine/arg_utils.py` |
| vLLM-TorchTPU | env var | `vllm_torchtpu/envs.py` |
| vLLM-TPU (tpu-inference) | env var, `--additional-config` key | `tpu_inference/envs.py`, `tpu_inference/layers/common/sharding.py` |
| MaxText | yml key | `maxtext/configs/base.yml` (MaxText traces only) |

A knob that exists only with a plugin patch says so (`requires_patch`):
`TPU_RPA_DECODE_BLOCK_SIZES` comes from
{{script:common/vllm_patches/rpa_decode_block_override.patch}}, which reads
the decode block and compute chunk per layer kind from the env var and
holds no sizes itself; it changes only the non-batched RPA v3 kernel (no
`--attention-backend CUSTOM`). When the image lacks such a knob and
`knob_probe.py --patch` finds the patch applies, the candidate's manifest
lists it in `vllm_patches`, and the launch recipe applies it with
`apply_vllm_patch.sh` before `serve_vllm.sh`.

## Guardrails (`serving_analysis/guardrails.py`)

- **Raising `--max-num-batched-tokens`** is dropped while time to first
  token matters, unless prefill stalls decode by at least 25% of the step:
  on three vLLM-TorchTPU models it raised median TTFT by 13% to 124% with no
  throughput gain.
- **`--async-scheduling`** is proposed only when D29 fires (the device idles
  between decode steps). On Qwen3.8-27B it measured no gain twice and raised
  TTFT by 18% at 128 users, so watch TTFT when it is measured.
- **Mesh candidates** change only TP and DP; knob candidates keep the
  baseline mesh, so each candidate is measured alone.
- **MLA and decode-context parallelism (`D18`, `D20`)**: `D20` never proposes
  `--tensor-parallel-size=1` when `--decode-context-parallel-size` or
  `--prefill-context-parallel-size` is above 1, or when the MLA KV cache each
  rank would hold at `--max-model-len` exceeds `--kv-cache-memory-bytes`
  (36 GB assumed when unset). `D18` proposes `--decode-context-parallel-size=8`
  only for the decode engine of a disaggregated pair (`--kv-transfer-config`
  with `"kv_role": "kv_consumer"`), on vLLM-TorchTPU when the sparse-attention
  indexer takes >= 8% of decode step time, DCP >= 16, TP x PCP is a multiple
  of 8, the decode batch is small (<= 16 sequences or a mean decode bucket
  <= 64 tokens) and the KV cache still fits at DCP 8. Keep prefill DCP >= 16
  on the prefill engine so prefix-cache KV capacity is not halved.
- **TP-sharded MoE decode (`D52`)**: `moe_tp_decode` takes at most 64 token
  rows per step (sequences x (1 + speculative tokens)). `D52` proposes
  `--enable-expert-parallel` when MoE grouped matmuls take >= 20% of decode
  step time and at least 50% of decode step time is in steps above 64 rows;
  `D31` stops advising against EP once decode steps exceed 64 rows.
  Prefill-heavy traffic (inputs >= 2x outputs) suppresses `D52` except on the
  decode engine of a disaggregated pair (`--kv-transfer-config` with
  `"kv_role": "kv_consumer"`).
- **Token buckets (`D2`)**: on vLLM-TorchTPU, when the smallest padded token
  bucket a phase runs is above 256 rows, `D2` also proposes
  `TPU_TOKEN_BUCKET_STEP=128` (linear buckets every 128 rows on top of the
  power-of-two ladder), unless a step below 128 is already set.
- **Few-token MoE (`D5`)**: on vLLM-TorchTPU decode steps of at most 64
  tokens, `D5` proposes `TPU_MOE_LOWTOK_MAX_TOKENS` set to the largest decode
  bucket: the few-token MoE kernel skips sort, permute, pad and unpermute for
  expert-parallel MoE calls up to that many tokens.
- **RoPE cache layout (`D9`)**: on vLLM-TorchTPU, when the per-step rotary
  cache layout copy remains with `TPU_ROPE_CACHE_TRUNCATE=1` already set, `D9`
  proposes `TPU_ROPE_CACHE_ROW_MAJOR=1` (row-major cos/sin caches built at
  load, which can use more HBM for padding).

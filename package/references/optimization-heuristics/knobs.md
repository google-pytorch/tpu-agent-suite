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

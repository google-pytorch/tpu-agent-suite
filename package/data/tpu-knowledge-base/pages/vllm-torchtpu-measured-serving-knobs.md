<!--
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
---
entry: vllm-torchtpu-measured-serving-knobs
keyword: vllm-torchtpu-measured-serving-knobs
url: not applicable -- see Sources
authority: first-party-verified
retrieved: 2026-10-16
source_updated: 2026-10-16
generator: authored
---

# vLLM-TorchTPU serving knobs, measured

Serving knobs tried on vLLM-TorchTPU servers and measured against a
baseline running at the same time on the same cluster: tpu7x 2x2x1 (8
devices), `vllm bench serve` with random prompts, 5 x concurrency requests,
two runs per point (mean shown). Each was proposed from an XProf trace by a
fixed detector rule (tpu-diagnostics' `D` ids).

## Applies when

- Choosing env vars or flags for a vLLM server on `vllm_torchtpu`.
- Deciding whether `--async-scheduling` or a larger
  `--max-num-batched-tokens` is worth a run.

## `--async-scheduling` (D29): no gain

| Model, layout | Point | Throughput | TTFT |
| --- | --- | --- | --- |
| Gemma-4-31B, TP 8 | 1k/1k, 1 | +1.0% | -3.0% |
| Gemma-4-31B, TP 8 | 8k/1k, 1 | -0.2% | +2.1% |
| Qwen3.8-27B, TP 4 x DP 2 | 1k/1k, 1 | +0.2% | +0.2% |
| Qwen3.8-27B, TP 4 x DP 2 | 1k/1k, 128 | +2.3% (within run spread) | +17.8% |
| Mistral-Small-24B, TP 4 x DP 2 (vLLM 0.32.0) | 1k/1k, 1 to 256 (9 points) | -1.5% to +0.1%, mean -0.4% (tie at every point) | not measured |

Recent vLLM turns async scheduling on by itself; the server log's "Async
scheduling" line says whether it is.

## A larger `--max-num-batched-tokens`: TTFT regression

Raising the prefill token budget raised median time to first token on both
models tried, with no throughput gain:

| Model | Change | Throughput | Median TTFT |
| --- | --- | --- | --- |
| Gemma-4-31B | default -> 8192 | -0.9% to -5.5% | +42% to +124% |
| Qwen3.8-27B | default -> 4096 | -2.6% to +0.3% | +24% to +89% |

Raise it only when time to first token does not matter, or when a profile
shows prefill chunks stalling decode for a large share of the step.

Lowering it has the opposite limit: a budget below the benchmark's prompt
length splits every prefill across steps and raises time to first token
(Gemma-3-27B at 8k inputs: +67%). The prefill-interference rule (D3)
therefore never proposes a budget below the prompt length.

## Knobs an image did not have

- `TPU_MOE_SKIP_PADDED_TOKENS` (skip padded MoE token slots; 94% of decode
  slots were padding on gpt-oss-20b, TP 8 + expert parallelism): not in
  the vLLM 0.22.1 build.

Check an image's installed plugin source for a knob before spending a run on
it (tpu-diagnostics' `knob_probe.py` does).

## Related

- The rules that propose these knobs: `index.py get vllm-serving-heuristics`.
- TP versus replicas, measured on Gemma-4-31B: `index.py get vllm-tp-dp-plan`.
- Layouts that fail to start: `index.py get vllm-torchtpu-mesh-constraints`.
- Reading a serving trace: `index.py get serving-trace-attribution`.

## Sources

- A/B runs of 2026-10-14 and 2026-10-15 on tpu7x 2x2x1, each candidate
  beside a same-session baseline; profiles in a separate run per point.
- `package/references/optimization-heuristics/detectors.md`.

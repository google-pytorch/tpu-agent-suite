Hill-climb google/gemma-4-31B-it, served with vLLM on TPU, to improve the output throughput (tokens/s) of the 1k/1k benchmark cell. Use the `/close-gaps` skill if it is available; otherwise, optimize it on your own.

The baseline is the vLLM server job `job-20261021054117-48973b`. Three benchmark client jobs ran against that server, one per cell:

- `job-20261021054837-82d4ad`: 1k/1k at concurrency 512
- `job-20261021060211-c6b620`: 8k/1k at concurrency 64
- `job-20261021060724-8618a7`: 1k/8k at concurrency 256

The onboarding that produced these jobs (serving recipe, benchmark harness and results) is in `/workspace/onboarded/`.

Early in the run, while your baseline measurements run, compute a whole-model roofline for the 1k/1k cell (ISL 1024, OSL 1024, concurrency 512) on this deployment. Use the roofline-analysis skill if it is available; otherwise, compute it on your own.

When you are done, write `/logs/artifacts/benchmark_results.json` with the 1k/1k results of the baseline and the optimized server, both measured in this run. Each row points to the raw benchmark output it came from, saved under `/logs/artifacts/`. Record the roofline's output throughput and the report it came from. Then write your final 1k/1k speedup (optimized output throughput divided by baseline):

```json
{
  "baseline": [
    {"regime": "1k/1k", "concurrency": 512, "output_tps": <measured>, "raw_output_path": "/logs/artifacts/<baseline raw output>"}
  ],
  "optimized": [
    {"regime": "1k/1k", "concurrency": 512, "output_tps": <measured>, "raw_output_path": "/logs/artifacts/<optimized raw output>"}
  ],
  "roofline": {"regime": "1k/1k", "concurrency": 512, "output_tps": <computed>, "source": "/logs/artifacts/<roofline report>"},
  "speedup": <final speedup>
}
```

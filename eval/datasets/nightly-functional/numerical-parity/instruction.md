Run a numerical parity check for the model `HuggingFaceTB/SmolLM2-135M-Instruct` on the GKE TPU cluster configured for this package, comparing a teacher-forced CPU reference forward pass against a Cloud TPU candidate forward pass on identical inputs.

Hook and compare intermediate outputs for `model.layers.0.self_attn` and `model.layers.29.self_attn` in addition to the final `logits`.

Important execution instructions:
- Proceed without checking in; accept all suggested defaults and do not stop to confirm a plan.
- Run only the numerical parity verification: do not deploy an HTTP inference server or run throughput/latency benchmarks.

## Final Deliverable & Reporting

Write the measured results to `/workspace/parity_results.json` with this schema:

```json
{
  "model_id": "HuggingFaceTB/SmolLM2-135M-Instruct",
  "parity_passed": true,
  "top1_token_agreement": "<agree>/<total>",
  "logits": {
    "max_rel": 0.0,
    "max_abs": 0.0,
    "rms": 0.0
  },
  "modules": {
    "model.layers.0.self_attn": {
      "max_rel": 0.0,
      "max_abs": 0.0
    },
    "model.layers.29.self_attn": {
      "max_rel": 0.0,
      "max_abs": 0.0
    }
  }
}
```

In your final response, report the overall parity verdict, the top-1 token agreement, and the measured `max_rel`, `max_abs`, and `rms` differences for the logits and both hooked modules.

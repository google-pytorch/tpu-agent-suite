You are acting as `tpu-agent`. Load the `tpu-agent` skill into your session.

`inputs/trace` contains an XProf trace of a transformer training step on TPU v6e.

Route the request by delegating the trace inspection to `tpu-diagnostics`:
1. Analyze the roofline model for `inputs/trace`.
2. Determine whether the workload is bound by HBM or Compute.
3. Extract the operational intensity, peak FLOP rate, peak HBM bandwidth, and published ridge point.
4. Recompute the ridge point from the peak constants and determine if the workload sits above the ridge.
5. Write your findings to `answer.json` matching `answer.schema.json`.

Important execution instructions:
- Delegate to `tpu-diagnostics` to inspect and analyze the trace.
- Forward the delegation request directly and concisely without adding unrequested tasks.
- Do not run shell `sleep` commands while waiting on subagents.

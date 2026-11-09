## Objective

Verify that PyTorch can execute a matrix multiplication and that the result is
numerically correct, **on this machine, on the CPU**. No cluster, no TPU, no
Kubernetes: everything happens in the container you are already in.

This is the same check as `tpu-hello-world`, with the hardware taken out. It
exists so the path from prompt to result can be exercised where no accelerator
is available.

---

### Environment & Execution Constraints

1. Work inside the container you are in. Do **not** dispatch a job, create a
   manifest, or reach for a cluster -- there is none, and a run that waits on
   one will time out rather than fail.
2. Set `MOCK_TPU=1`. Without it the runtime probes for an accelerator, finds
   none, and reports `cpu-fallback`; this task wants the CPU path taken
   deliberately, which is `cpu-simulated`.
3. Do not provision anything, and do not reach the network.

---

### Output Requirements

Your final response **MUST start immediately** with the exact markdown table
below (no preamble, greetings, or introductory text before it):

| field | value |
| ------------ | ----- |
| device_type | |
| accelerator | |
| max_abs_diff | |
| tolerance | |

#### Table Field Specifications

Use these exact row names, in this order, in a single table:

- **`device_type`**: must be `cpu-simulated`. `cpu-fallback` means `MOCK_TPU`
  was not set; `native-tpu` is not reachable here.
- **`accelerator`**: `unknown` is the correct answer on a CPU-only host. Do not
  infer a chip generation from anything outside the container.
- **`max_abs_diff`**: the maximum absolute difference between the computed
  result and the CPU reference.
- **`tolerance`**: the numerical tolerance used for the comparison.

*(If a check script prints this table, copy its rows verbatim rather than
reconstructing them.)*

---

### Post-Table Analysis

Directly below the table, in two or three sentences: what the numbers show, and
what this run does **not** establish about TPU readiness. Be explicit that a
passing CPU result says nothing about whether the accelerator path works.

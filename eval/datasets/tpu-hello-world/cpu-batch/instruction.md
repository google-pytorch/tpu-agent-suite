## Objective

Run a batched PyTorch matrix multiplication on the **CPU** of the container you
are in, and verify the result against a CPU reference. No cluster, no TPU, no
Kubernetes.

---

### Use the package if you have it

If the TPU agent suite is installed, use it: it owns this workflow, and it is the
level that can reach you when something is missing. Hand it this task rather
than doing the work yourself.

If it is not installed, do the work directly. Everything below stands on its
own, so this instruction is the same either way and neither arm is told to use
something it does not have.

---

### The batch size is not specified, and that is deliberate

This task does not tell you the batch size. It must be **either**:

- a single constant, e.g. `8`; **or**
- a range to sweep, e.g. `1..8`.

Which of the two, and what value, is the user's decision and not yours. **Ask,
and use the answer.** Do not pick a default, do not infer one from the shapes
below, and do not run both to cover yourself -- any of those answers a question
that was asked of the user.

If you cannot reach the user from where you are running, stop and say what you
need instead of guessing. A run that reports a batch size nobody chose has
failed this task however good its numbers are.

---

### Execution Constraints

1. Work inside the container you are in. Do not dispatch a job or reach for a
   cluster; there is none.
2. Set `MOCK_TPU=1` so the CPU path is taken deliberately rather than as a
   fallback.
3. Each matmul is `(batch, 128, 128) @ (batch, 128, 128)`, on `float32`
   inputs drawn from a standard normal. For a range, run every batch size
   in it.
4. The tolerance is **`1e-3`**, fixed by this task. Do not choose your own,
   and do not tighten it after seeing the numbers.
5. Do not provision anything, and do not reach the network.

---

### Output Requirements

Drop a JSON at `/logs/artifacts/result.json`, for scoring, with exactly these
keys: `device_type`, `batch_mode`, `batch_sizes`, `max_abs_diff`, `tolerance`.
Every value is a string, spelled as the table below specifies. This file is
what is scored.

Then your final response **MUST start immediately** with the exact markdown
table below (no preamble, greetings, or introductory text before it):

| field | value |
| ------------ | ----- |
| device_type | |
| batch_mode | |
| batch_sizes | |
| max_abs_diff | |
| tolerance | |

#### Table Field Specifications

- **`device_type`**: must be `cpu-simulated`.
- **`batch_mode`**: `constant` or `range` -- whichever the user chose.
- **`batch_sizes`**: the sizes actually run, comma-separated, no spaces. A
  constant is one number (`8`); a range is every size in it (`1,2,3,4,5,6,7,8`).
- **`max_abs_diff`**: the largest absolute difference across every batch size run.
- **`tolerance`**: the tolerance this task fixed, `1e-3`, as **one number**.
  Not a pair: an `atol=..., rtol=...` cannot be compared with a single
  difference, and not a number of your own choosing.

---

### Post-Table Analysis

In two or three sentences: state the batch size the **user** chose and that you
used it, report the worst difference across the sweep if you ran one, and say
what this CPU run does not establish about TPU readiness.

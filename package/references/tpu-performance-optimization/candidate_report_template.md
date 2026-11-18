# Candidate `<id>`: `<slug>`

<!--
File path: candidates/<id>-<slug>.md
The report must be self-contained. An engineer or subagent must be able to execute it without reading decomposition.md.
Write `unknown` for any missing value. Do not guess values. Do not delete fields.
-->

## 1. Summary

- **Target op or mechanism**: `<op or mechanism>`. Measured cost `<ms>` ms, `<pct>`% of the step.
- **Phase**: `<prefill | decode | both>`
- **Problem class**: `<kernel | redundant-relayout>` (Step 5.7)
  - `kernel`: the op does required work, and its implementation is slow.
  - `redundant-relayout`: the op is fully redundant. It exists only because of a shape or layout mismatch between producer and consumer.
- **Class evidence**: `<kernel: measured ms, floor ms, and the cause found. redundant-relayout: the producer op and its output shape/layout, the consumer op and its expected shape/layout, and the code location that introduces the mismatch.>`
- **Report type**: `<full | estimated-weight | short>` (Step 4 item 7)
- **Weight to enter the top five**: `<estimated-weight reports only: the regime weight at which E2E Ceiling equals the fifth candidate's; otherwise n/a>`

---

## 2. Rationale

<!--
Explain why this triage identifies this op or mechanism as a problem.
Fill in both fields below. Do not add or remove fields.
-->

- **Starting point**: `<The op that first drew attention. State its xprof cost (ms), its roofline floor (ms), and the resulting measured-to-floor ratio.>`
- **Analysis**: `<The results of the cross-layer, cross-rank, and cross-phase checks, and how they identify this op or mechanism as the cause. If the candidate differs from the starting-point op, explain how the analysis led from one to the other. If the candidate is the starting-point op, explain how the other possible causes were excluded.>`

---

## 3. Scope and Boundary

- **Why not smaller (closure)**: `<List each helper or adjacent op inside this boundary that the profile alone would not flag, and the op that requires its inclusion. Otherwise write: "No extra ops included. The block is self-contained because <reason>.">`
- **Why not larger (split)**: `<Name the adjacent expensive op that is excluded, with its measured ms. State the reason for the boundary: cannot be stated as a single change | requires a different tool | crosses a repository or owner boundary | different hardware bottleneck | exceeds ~25% of step time.>`
- **Considered and excluded**: `<Name each other op evaluated for inclusion, its measured ms, and the reason it was excluded.>`

---

## 4. Measured Cost and Roofline Floor

One row per regime in the decomposition's Target Metric & Regime Weights table, including padded regimes and regimes where the op is absent (write 0). Weights are copied from the decomposition.

| Regime | Weight | Device step time (ms) | Measured (ms) | Floor (ms) | Gap (ms) | Step share (%) | Ceiling (%) | Weight × Ceiling (%) | Binding resource |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`<regime>`** | `<w_p>` | `<val>` | `<val>` | `<val>` | `<max(measured − floor, 0)>` | `<measured / step>` | `<gap / step>` | `<w_p × ceiling>` | `<MXU / VPU / HBM / ICI>` |
| **E2E Ceiling** | | | | | | | | `<Σ weight × ceiling>` | |

- **Floor source and assumptions**: `<State whether the floor comes from roofline-analysis, a hand derivation, or floor.md, and list the assumptions.>`
- **Single-index outlier test (5.3.0)**: `<per regime: worst index, its time against the median of the other indices, excess ms per step. State whether an outlier changes the diagnosis.>`
- **Ceiling arithmetic**:
  ```text
  <Step-by-step arithmetic for Floor (ms), Gap (ms), Ceiling (%) per regime, and the E2E Ceiling sum>
  ```

---

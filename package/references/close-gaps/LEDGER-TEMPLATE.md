# close-gaps ledger: `<job-id>` — `<MODEL_DIR>`

<!-- The only state. Read at the top of every loop iteration; write after every artifact.
     A number goes in only with the job id behind it. Keep under ~150 lines. -->

## Now

- **Step:** `<0 baseline | 1 rank | 2 gap <rank> (2.x) | 3 combine | 4 e2e | blocked | done>`
- **Next action:** `<one line>`
- **In flight:** `<job ids + what each is, or none>` — poll: `<background task id>`
- **Jobs spent:** `<n>` / `<--max-jobs>`
- **Blocked:** `<none | reason — resume when …>`

## Setup

| field | value |
| --- | --- |
| source job | `<job-id>` (recipe read back: `{{runs}}/<run_id>/artifacts/<job-id>/recipe.yaml`) |
| model / workload | `<MODEL_DIR>`, `<ISL/OSL or trace>`, `<concurrency>` |
| microbench plan | `<path>.steps.json` — mode `<scaled/mirror>`, top `<K>`, coverage `<x>%` of busy |
| baseline image | `<repo:tag@sha256:…>` — vllm-torchtpu main `<sha>` |
| baseline T_mix (served, modelled) | `<s>` ± `<spread %>` — jobs `<j-…>`, `<j-…>` |
| decomposition | `<path>/decomposition.md` |

## Stack

| field | value |
| --- | --- |
| branch | `close-gaps/<job-id>/stack` @ `<sha>` |
| image | `<repo:tag@sha256:…>` |
| cumulative T_mix (served, modelled) | `<s>` (`<speedup>×` vs baseline) — job `<j-…>` |

## The list

| rank | gap | op / block | gap ms (served, modelled) | e2e ceiling % | route | status | closed % | gain % | class + artifact | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `<slug>` | `<op>` | `<before − floor>` | `<x>` | `<flag/env/code/kernel>` | `<open/digging/ab/done>` | | | | `gaps/1-<slug>.md` |
| 1.1 | `<neighbour slug>` | `<op>` | | | | | | | | |

**Parked (examined, not selected):** `<op — reason>`, …

**Outcomes** (one line per A/B, `needs-e2e` arm or crashed job, here or in its gap file):
`- outcome: <win | not a win | crash>` then each changed `KEY=VALUE` in backticks

## Spend

| gap | A/B rounds | jobs | notes |
| --- | --- | --- | --- |

## Combined / E2E

| | value | jobs |
| --- | --- | --- |
| product of individual speedups | | |
| combined T_mix speedup (modelled) | | |
| projected served gain (modelled) | | |
| E2E measured gain | | |
| accuracy (mmlu_pro, limit 100) | | |

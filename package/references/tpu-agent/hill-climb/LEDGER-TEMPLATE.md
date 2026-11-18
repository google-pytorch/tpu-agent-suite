# hill-climb ledger: `<job-id>` — `<MODEL_DIR>`

<!-- The only state. Read at the top of every loop iteration; write after every artifact.
     A number goes in only with the job id behind it. Keep under ~150 lines. -->

## Now

- **Round:** `<1, 2, …>`
- **Step:** `<0 baseline | 1 triage | 2 list | 3 review | 4 gap <rank> (4.x) | 5 combine | 6 e2e | 7 accuracy | 8 image | blocked>`
- **Re-read next:** `<reference> § <section>`
- **Next action:** `<one line>`
- **Waiting on delegate:** `<none | agent — what was asked, request path>`
- **run_id:** `<run_id the delegations below belong to>`
- **Blocked:** `<none | reason — resume when …>`
- **Entry baseline:** `<shape: summary.json path (source | measured, job id)>` — fixed, report only
- **Last verified stack:** `<round, stack sha, step-6 result path, accuracy record | base + entry baseline>`

**Microbenchmarks** (one row per tpu-forward-bench request; its jobs and slots are the delegate's):

| label | request | result | status | arms → verdict | job ids | ML Diagnostics |
| --- | --- | --- | --- | --- | --- | --- |
| `<base / <rank>-<slug> / combine>` | `bench/<label>.request.json` | `bench/<label>.result.json` | `<measured / …>` | `<arm: win>` | `<j-…>` | `<arm: its ml_diagnostics_urls … / none>` |

## Setup

| field | value |
| --- | --- |
| source job | `<job-id>` (recipe read back: `{{runs}}/<run_id>/artifacts/<job-id>/recipe.yaml`) |
| engine | `<vllm>` — read from `<deployments.json / bench_request / recipe>`; passed on every bench request |
| TPU slots | `<N>` — the `gke_job` profile's declared `slots`, from `tpu_slots.py show` after step 0 |
| model / workload | `<MODEL_DIR>`, `<ISL/OSL or trace>`, `<concurrency>` |
| microbench plan | `<path>.steps.json` — mode `<scaled/mirror>`, top `<K>`, coverage `<x>%` of busy; replay `<base replay path>` |
| baseline image | `<repo:tag@sha256:…>` (`compute.params.image`) |
| baseline T_mix (served, modelled) | `<s>` ± `<spread %>` — jobs `<j-…>`, `<j-…>` |
| triage | `triage.request.json`, `triage.result.json` |
| floors | `<roofline result path>` — `<cells>` |
| decomposition | `<path>/decomposition.md` |

## Stack

| field | value |
| --- | --- |
| image digest | `<repo:tag@sha256:…>` (the baseline image; never rebuilt) |
| campaign repo | `plugin/` — `base` `<sha>` (image `<digest>`), `stack` `<sha>`; wins in order: `<hyp/1-<slug> @ <sha>>`, … or none |
| last apply check | `<exit code>` — job `<j-…>` on `stack` `<sha>` |
| slot hardware key | `<hardware>` — the key the baseline ran on; every later request uses it |
| cumulative T_mix (served, modelled) | `<s>` (`<speedup>×` vs baseline) — job `<j-…>` |

## The list

| rank | gap | op / block | gap ms (served, modelled) | e2e ceiling % | route | review | status | closed % | gain % | class + artifact | file |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `<slug>` | `<op>` | `<before − floor>` | `<x>` | `<flag-env/kernel/code>` | `<pass / refuted — review path>` | `<open/digging/ab/done>` | | | | `gaps/1-<slug>.md` |
| 1.1 | `<neighbour slug>` | `<op>` | | | | | | | | | |

**Parked (no floor, or examined and not selected):** `<op — reason>`, …

## Spend

| gap | launches | relaunches | notes |
| --- | --- | --- | --- |

## Combined / E2E

| | value | jobs |
| --- | --- | --- |
| product of individual speedups | | |
| combined T_mix speedup (modelled) | | |
| projected served gain (modelled) | | |
| E2E measured gain | | |
| E2E bench | `bench/e2e.request.json`, `bench/e2e.result.json`, replay `<path>` | `<server job ids per arm>`; ML Diagnostics `<arm: link>` |
| accuracy (mmlu_pro, limit 100) | `bench/accuracy.request.json`, `bench/accuracy.result.json`, gate, delta_pp | `<eval job id per arm>` |

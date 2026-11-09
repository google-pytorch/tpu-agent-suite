---
name: tpu-kernel-development
description: 'Only when the user asked for it by name: drives the kernel optimization self-refinement loop, converting JAX or PyTorch into a Pallas TPU kernel, initializing and re-reading its state.json, dispatching one isolated iteration at a time, enforcing the stop condition, and reporting the final results table. Accepts an optional reference kernel (e.g. a CUDA implementation of the same computation) that the planner mines for design ideas; CUDA is accepted in that reference slot only, never as the source being converted. Not for generic kernel or speed requests that do not name it.'
tools:
- read
- write
- edit
accepts: schemas/kernel-request.json
produces: schemas/kernel-result.json
delegates:
- kernel-worker
- tpu-diagnostics
- tpu-knowledge-base
host_dependencies:
  - docstring-parser
  - google-cloud-aiplatform
  - vertexai
tpu_dependencies:
  - jax
  - libtpu
  - ml_dtypes
  - torch
  - tpu-info
scripts:
- scripts/tpu-kernel-development/run_autotune.py
---

# tpu-kernel-development

You are strictly an orchestrator. You manage the workflow of kernel optimization
by dispatching kernel-worker agents. Under no circumstances should you act as an
engineer directly. You should never read or write any code or debug any error
yourself. The one exception is Setup step 0, which copies the caller's pasted
text to disk unchanged: transcription, not engineering.

--------------------------------------------------------------------------------

## Artifact root

Your task request carries the session `run_id`, assigned at session start. It
belongs to the whole session, not to tpu-kernel-development — never invent one, and never
allocate a second run mid-session. (Only when tpu-kernel-development is started directly,
so that no `run_id` reaches you, do you allocate exactly one yourself, via the
`start-run` call the installed package contract describes.)

Every file this session produces lives under that run's artifact root, referred
to throughout as `<artifacts>`, in a slot of tpu-kernel-development's own subdirectory —
one slot per kernel, referred to throughout as `<work>`:

```
<artifacts> = {{runs}}/<run_id>/artifacts
<kernels>   = <artifacts>/tpu-kernel-development/kernels.json
<work>      = <artifacts>/tpu-kernel-development/<slot>
```

`<artifacts>` already exists — `start-run` created it. `<work>` is created
implicitly the first time you write `<work>/state.json`.

### One slot per kernel

The `run_id` belongs to the whole session, and one session can call you more
than once — once per kernel a profile turned up. A `<work>` derived from the
`run_id` alone would be shared by every one of those calls: the second call
would resume the first kernel's `state.json` and report its results as the
second kernel's, or, told to reset, write its `base.py` and `iter<n>/` over the
first kernel's and leave the first result's `best_kernel_artifact` pointing at
someone else's code. So each kernel gets a slot of its own, and `<kernels>`
says which slot is whose:

```json
{
  "kernels": [
    {"slot": "k01", "base_kernel_path": "/abs/path/attention.py", "target_op": "ragged paged attention", "real_path": "/abs/real/attention.py"},
    {"slot": "k02", "base_kernel_path": "/abs/path/model.py", "target_op": "rms norm", "real_path": null}
  ]
}
```

A kernel is the pair (`base_kernel_path`, `target_op`), each normalized before
it is stored or compared, so that a follow-up which writes the same kernel a
little differently resumes it instead of silently starting over:

-   `base_kernel_path` is made absolute — a relative path is resolved against
    your working directory, the same resolution you use to read the file in
    Setup step 1 — with `.` and `..` segments and repeated `/` collapsed.
-   `target_op` is lowercased, trimmed, and its runs of whitespace collapsed to
    one space; `null` when the request had none.

The path alone is not enough: two hot kernels can live in one source file, and
keyed on the path the second would resume the first one's slot.

Normalizing catches spelling, not renaming: "ragged paged attention" and "RPA"
are still two keys. That is what the request's `slot` is for. Every
`kernel-result` you return after resolving a slot carries it in `slot`, and a
caller continuing that kernel passes it back. Every such result also carries,
in `same_file_slots`, the other `<kernels>` entries that are the same file —
the same `base_kernel_path`, or the same non-null `real_path` — read from
`<kernels>` when you return, omitted when there is none.

Normalizing also cannot resolve a symlink, because you have no shell:
`/workspace/torchtpu-vllm/…` and `/root/torchtpu-vllm/…` can be one file and
two keys. Two things cover that. The router passes each kernel's path in one
spelling for the whole session, as the result that named it spelled it. And
when it slips, `real_path` catches it: the kernel-worker resolves the file's
real path in its first iteration and records it in `state.kernel.real_path`,
and the **same-file check** below compares it across slots.

**Same-file check.** Run it at the top of every control-loop iteration and
before you return a result from Finish:

1.  If `state.kernel.real_path` is set and your `<kernels>` entry has no
    `real_path`, copy it into the entry, write `<kernels>` back, and read it
    back as step 4 below requires.
2.  If an earlier slot — a lower number than yours — has the same non-null
    `real_path` and the same `target_op`, your slot is a second copy of that
    kernel, reached through another spelling of its path. Stop dispatching
    and return `invalid_input` with your `slot`, that slot in
    `same_file_slots`, and an `error_message` saying the kernel is already
    being optimized in that slot and to delegate again with its `slot`. Do
    not delete your entry: a later call that spells the path your way
    resumes your slot and is redirected again by this same check. At most
    the one iteration that resolved `real_path` is spent before this fires.
3.  A same-file slot with a different `target_op` is a second kernel in the
    same file. Carry on; `same_file_slots` reports it.

Resolve `<work>` right after Setup step 1 has read the input, and before
step 1b, the first step that needs it — so an input that cannot be read never
claims a slot:

1.  Read `<kernels>`. If it does not exist, treat it as `{"kernels": []}`.
2.  **The request names a `slot`.** Find that entry. If there is none, or its
    `base_kernel_path` differs from the request's normalized
    `base_kernel_path`, stop and return `invalid_input` with an
    `error_message` naming the slot and both paths — the slot's path is the
    spelling to delegate again with, if the request reached the same file
    through a symlink. Otherwise that slot is your `<work>`, whatever
    `target_op` the request spelled — the caller has told you which kernel
    this is. Setup then resumes it, or resets it if the user asked.
3.  **No `slot`.** Find the entry whose `base_kernel_path` and `target_op`
    both equal the request's, normalized.
    -   **Found** — that entry's `slot` is your `<work>`. Setup then resumes it,
        or resets it if the user asked.
    -   **Not found** — this is a new kernel. Its slot is `k` followed by
        `len(kernels) + 1`, zero-padded to two digits (`k01`, `k02`, …).
        Append the entry, write `<kernels>` back, and use the new slot as
        `<work>`. It has no `state.json`, so Setup starts it fresh: nothing a
        previous kernel left behind is read, resumed or overwritten.

        If other entries are already the same file, the caller may
        have renamed the op of a kernel it meant to continue, or this may be a
        second kernel in the same file. Only the caller knows which, so do not
        guess and do not stop: the result's `same_file_slots` lists them, so
        that opening a new slot is never silent and the caller can come back
        with `slot` if it meant one of them.
4.  **After any write, read `<kernels>` back** and confirm that your entry is
    in it and that no other entry has your slot. If either fails, another
    call wrote the index between your read and your write; go back to step 1
    and resolve again. Do not dispatch anything until this check passes.
5.  Never reorder, rename, or delete an entry. A slot names the same kernel
    for the rest of the session, which is what keeps every path an earlier
    result returned pointing at that kernel.

You are the only writer of `<kernels>`. You have no shell, so the write cannot
be made atomic with a temporary file and a rename; the index instead relies on
the caller running one tpu-kernel-development call at a time in a session, which the
router's contract states. If that rule is broken anyway, two checks catch the
lost append that follows: step 4, when the other call's write lands before your
read-back, and the ownership check at the top of every control-loop iteration,
when it lands after.

Never write session output anywhere else — in particular, never into a
`workspace/` directory relative to your current working directory, and never
into the installed package tree (`{{package}}`), which is read-only shipped
content. Package-relative paths such as `{{script:tpu-kernel-development/run_autotune.py}}` name
shipped files to read or execute, never output locations.

## External state, isolated iterations

1.  **The state of the kernel optimization task, including the iteration count
    and artifact paths, lives in a file (`<work>/state.json`), not in your
    memory.** You re-read that file at every decision point, and you branch on
    what you just read — never on what you remember reading earlier.
2.  **Each iteration's heavy work runs in an isolated agent**. Your own context
    only ever holds one state read + one kernel-worker dispatch + one profiling
    dispatch + two short return summaries per iteration — not five iterations'
    worth of plan/implement/compile/test/autotune/profile transcripts.

## How to dispatch kernel-worker agent

Every time you dispatch the kernel-worker agent, it gets its own context, has no memory
of your conversation, and exits after doing one iteration's work.

The dispatch instruction is always the same every iteration, except that you
must pass the session `run_id` verbatim, and `<work>` fully resolved as
`work_dir`, in every dispatch. The kernel-worker cannot derive the slot from the
`run_id` — only you know which kernel this is — so `work_dir` is how it works in
the same `<work>` you do. The dispatched agent is {{agent:kernel-worker}}, which
derives its own iteration number, input file, and previous latency by reading
`<work>/state.json` itself.

Send it as a JSON object, not as prose. `kernel-worker-request` requires
`prompt`, `run_id` and `work_dir`, so a prose-only dispatch is denied before the
kernel-worker starts:

```json
{"prompt": "Run one iteration of the self-refinement loop.", "run_id": "<run_id>", "work_dir": "{{runs}}/<run_id>/artifacts/tpu-kernel-development/<slot>"}
```

## What the loop converts

The loop no longer assumes the caller hands it JAX. The **primary** input is
the thing being converted and it defines the semantics; zero or more
**references** are independent implementations of roughly the same computation,
supplied so the planner can mine them for design ideas. A reference is never
executed, never measured, and never compared against.

```
        kernel-request.base_kernel_path / base_kernel_source
                            │
                  the primary input, whatever it is
                            │
        ┌───────────────────┼─────────────────────┐
     jax │              pytorch │      specification │
        ▼                      ▼                     ▼
  base.py is the      oracle captured       the file is prose;
  caller's file       from the module,      a baseline is
  unchanged           then ported and       synthesized, then
                      GATED against it      STOPS for approval
        └───────────────────┴─────────────────────┘
                            ▼
              base.py  — PURE JAX, the denominator of every
                         number this run reports
                            ▼
              plan -> implement -> compile -> test -> autotune -> profile
                            ▼
                    state.best_code_path

  kernel-request.reference_paths  (optional, typically a CUDA kernel)
                            │
              briefed, triaged into the ideas ledger, read by
              the planner. Never executed, never measured,
              never base.py. This is the only slot CUDA fills.
```

Two things about this shape matter when you read the final numbers.

**"Pure JAX" and "Pallas" are both JAX.** `pl.pallas_call` is a JAX primitive,
so no step leaves JAX. What the front end produces is *idiomatic* JAX — what a
competent engineer writes without a custom kernel — and that is the thing the
kernel has to beat.

**Every arrow that changes representation has a gate under it, and a failed
gate stops the run rather than degrading it.** Five iterations optimizing
against a wrong `base.py` end in a confident, wrong speedup, which is worse
than no run at all.

## Setup

Your request is a `kernel-request`, so the loop's inputs are typed fields rather
than things to infer from prose. Read them from the request first; fall back to
the `prompt` only for what the fields do not carry.

0.  **If the request carries `base_kernel_source` instead of
    `base_kernel_path`, put it on disk first**, then proceed as for a path.
    This is the one write the rule at the top allows you: a copy, not code.
    The file is named by `input_language`, which the schema requires with
    inline source: `base_kernel.py` for `jax`, `pallas` and `pytorch`;
    `base_kernel.md` for `specification`.

    A request carrying both `base_kernel_source` and `slot` is refused:
    pasted text arrives once, as a new kernel, and the caller continues it by
    the `base_kernel_path` every result returns, as it would any file. Stop
    and return `invalid_input` with an `error_message` naming that path from
    the slot's `<kernels>` entry, if the slot has one, to delegate again with.

    Otherwise write the text to
    `<artifacts>/tpu-kernel-development/inputs/k<NN>/<file>`, `<file>` being
    the name above. For `NN`, read `<kernels>` and count up from
    `len(kernels) + 1`, zero-padded to two digits, to the first directory
    that does not exist yet — a `Read` of its `<file>` misses. The number
    only keeps pastes apart: it usually equals the slot the kernel gets, but
    the `<kernels>` entry, not the directory name, records which path is
    whose. That absolute path is the request's `base_kernel_path` from here
    on.

    Write the text byte-for-byte. Do not edit it on the way in — not
    whitespace, not a missing `#`, not an import: it is the user's file and
    the denominator of every number this run reports, and if it does not run
    the baseline gate says so.

1.  **Read the input from the request's `base_kernel_path`.** The contract
    requires it, or `base_kernel_source` which step 0 has just turned into
    one, so there is always one to read and no default to fall back on.
    If the file does not exist or cannot be read, stop and return a
    `kernel-result` with `invalid_input` and an `error_message` naming the path
    — do not substitute a file you found nearby, and do not start the loop
    against an input the caller did not name. Optimizing the wrong file
    silently is worse than refusing, because the speedup you report would be
    real and about something no one asked for.

    What that file *is* depends on `input_language`, and the requirement does
    not bend for any of them:

    -   `jax` — a JAX module. It becomes `base.py` unchanged. This is the path
        the loop has always taken.
    -   `pallas` — an existing Pallas kernel the caller wants made faster. A
        Pallas kernel is already JAX, so this is the `jax` path under the name
        callers reach for first; Setup step 3b records it as `jax`.
    -   `pytorch` — the source to convert. The front end ports it and gates
        that port against values captured from the caller's own module.
    -   `specification` — a written description of the computation rather than
        code. The kernel-worker synthesizes a baseline from it and **stops for the
        user to approve it** before anything is optimized. Say so when you
        report the plan: in that mode an agent writes the file the run is
        scored against, so the approval step is the only external check the
        baseline will get. The description is still a file by the time you
        read it — the caller's, or the one Setup step 0 wrote from
        `base_kernel_source` — which is what keeps this mode inside the same
        rule as every other, and gives the approval gate something durable to
        point the user back at.

    **CUDA is not on that list, and a CUDA `base_kernel_path` is refused.**
    Converting CUDA to JAX and then to Pallas measured worse than planning
    natively with the CUDA source as context, so the port was withdrawn. The
    route that measured well is still here: put the `.cu` in `reference_paths`
    alongside a JAX or PyTorch primary and the planner mines it for design
    ideas every iteration. If a request arrives with a CUDA primary anyway —
    `input_language: "cuda"`, which the schema does not admit, or a `.cu` the
    classifier flags — return a `kernel-result` with `invalid_input` and an
    `error_message` saying which file to name as the primary instead. Do not
    port it yourself and do not fall back to `specification`.

1b. **Check whether Tokamax already has this kernel.**

    Before you dispatch any kernel-worker, check whether the Tokamax library
    already ships a TPU kernel for this op. You are the only agent in this
    loop that talks to tpu-knowledge-base. What to do depends on the request.

    **The request has no `library_kernel` field**

    1.  If `<work>/state.json` already exists, skip this step: the run is
        resuming and the check was already done.
    2.  Find the op name in `target_op`, or else in the `prompt`. If neither
        names it, skip the check and mention that in your final report.
    3.  Ask {{agent:tpu-knowledge-base}}:

        ```json
        {"query": "Is there a public tokamax function for <op> (tokamax-public-api), and does its implementation= include a TPU kernel (a mosaic* value) according to its tokamax-<op>-api entry? Give the call signature and the implementation value to pass. If it is not public, does tokamax-experimental or tokamax-experimental-tpu list a TPU kernel for it? Name the module and the URL.", "run_id": "<run_id>"}
        ```

    4.  Act on the answer:
        -   **Tokamax has a TPU kernel:** directly use it. Record `"use"` in
            `library_check`, with the call and its `implementation=` value as
            `module` (e.g.
            `tokamax.dot_product_attention(..., implementation="mosaic")`;
            for an experimental op, its module path) and the URL. Then go on
            to step 2.
        -   **No TPU kernel, GPU only, or the lookup failed:** go on to
            step 2. If the lookup failed, say so in your final report.

    **The request carries `library_kernel` with `decision: "use"`**

    Copy `module` and `url` from `library_kernel`, and go on to step 2.

    **The request carries `library_kernel` with `decision: "build"`**

    Copy `module` and `url` from `library_kernel`, and ask tpu-knowledge-base how
    the Tokamax kernel works, so the planner can start from it:

    ```json
    {"query": "How is <library_kernel.module> implemented for TPU: which kernel files under <library_kernel.url> make it up, and what tiling and blocking do they use?", "run_id": "<run_id>"}
    ```

    In every case, record the result in `library_check` (see **Initial
    state**).

2.  Determine `atol` and `rtol` from the request's `atol` and `rtol` fields,
    defaulting both to `1e-2` when absent.
3.  Determine the iteration budget from the request's `max_iterations`,
    defaulting to `5` when absent. This is the loop's only stop condition —
    there is no time budget and no target-speedup early exit, which is why the
    request declares neither. When `library_check.result` is `"use"`, the
    budget is 1 iteration.
3b. Take `input_language` and `reference_paths` from the
    request when it carries them, and
    leave them `null` otherwise. Record an `input_language` of `pallas` as
    `jax`: every downstream phase branches on `jax`, and the two take the same
    path. **Do not guess any of them.** The kernel-worker holds
    `shell` and settles what it cannot read here in its Phase 0.1, using
    `classify_inputs.py` — which refuses rather than guessing when several
    files could each be the thing being converted. A wrong guess silently
    benchmarks the wrong computation for five iterations.

    When step 0 wrote the input as `base_kernel.py`, leave `primary.language`
    `null` whatever `input_language` the request carried: the label only
    chose the extension, and Phase 0.1 classifies the content. Keep a
    `specification` label: prose has nothing to classify.
4.  If `<work>/state.json` does not exist yet, or the user said
    "reset"/"restart", write it with the contents under **Initial state**
    below.
5.  If `<work>/state.json` already exists and the user didn't ask to reset,
    first check that it is this kernel's: its `kernel.base_kernel_path` and
    `kernel.target_op` must equal the `base_kernel_path` and `target_op` of
    the slot's `<kernels>` entry (`null` matching `null`). If either differs,
    stop and return a `kernel-result` with `invalid_input` and an
    `error_message` naming `<work>`, both kernels, and which field differs —
    do not resume it and do not reset it. The slot lookup should make this
    impossible, so reaching it means `<kernels>` and the slot disagree, and
    resuming would report one kernel's results as another's. Otherwise resume from it as-is — then
    bring it up to schema before dispatching anything. A state file written
    before the front end existed has no `primary` object; build one with
    `language: "jax"`, `source_path` set to its `base_kernel_path`, everything
    else `null`, and `references: []`. A resumed run never acquires a
    reference it did not start with.

### Initial state

```json
{
  "kernel": {
    "base_kernel_path": null,
    "target_op": null,
    "real_path": null
  },
  "iteration": 0,
  "max_iterations": 5,
  "atol": 1e-2,
  "rtol": 1e-2,
  "seed": 1024,
  "base_kernel_path": "<work>/base.py",
  "baseline_time": null,
  "best_code_path": "<work>/base.py",
  "best_optimized_time": "Infinity",

  "tpu_version": null,
  "tpu_spec": null,
  "primary": {
    "language": null,
    "source_path": null,
    "context_path": null,
    "golden_path": null,
    "golden_meta_path": null,
    "port_verified": null,
    "baseline_approved": null
  },
  "references": [],
  "reference_trust": null,
  "reference_alignment_path": null,
  "ideas_ledger_path": null,
  "fusion_context": null,
  "served_calls": null,
  "serving_config": null,
  "library_check": null,

  "history": []
}
```

Set `"atol"`, `"rtol"` and `"max_iterations"` to the values you resolved in
Setup steps 2 and 3, and `"base_kernel_path"` to the path the request carried.

The new fields, and who fills each one:

-   `kernel` is which kernel this slot belongs to: the slot's `<kernels>`
    entry's `base_kernel_path` and `target_op`, normalized as that entry
    is, plus `real_path`, which starts `null` and which the kernel-worker
    fills once in its first iteration. You write the first two here and
    nothing ever rewrites them, which is why it is
    not `primary.source_path` or `base_kernel_path`: the kernel-worker
    repoints both at its own copies in `<work>`. Setup step 5 checks a resumed
    slot against it.
-   `primary.language` is `"jax"`, `"pytorch"` or `"specification"`.
    Never `"pallas"`: a request saying `pallas` is recorded as `"jax"`.
    Write it here only when the request stated it; otherwise leave it `null`
    and the kernel-worker's Phase 0.1 settles it. Once set it is **never revised
    mid-run** — every downstream agent branches on it.
-   `primary.source_path` is the fully resolved path to the untouched caller
    input. For a `jax` primary that is the same file as `base_kernel_path`.
-   `seed` is internal and always `1024` unless a resumed run already has one.
    It is not a request field, because the *value* is arbitrary — what matters
    is that it is fixed for the run, so Phase 0.3 builds the module the same
    way every time it is entered. Nothing upstream knows or cares what it
    should be.
-   `primary.context_path`, `golden_path`, `golden_meta_path`, `port_verified`
    and `baseline_approved` all stay `null` here; the kernel-worker fills them in
    Phases 0.2, 0.3, 0.5 and 0.9 as each artifact appears on disk.
-   `references` is `[]` when the caller supplied no reference kernel — the
    common case, and the whole advisory spine is then skipped.
-   `reference_trust`, `reference_alignment_path` and `ideas_ledger_path` stay
    `null`; the kernel-worker sets them in Phase 0.7.
-   `fusion_context` is the request's `fusion_context`, copied verbatim, or
    `null` when it carried none. It says what XLA fuses the target with in the
    served graph, and the planner reads it from here: a kernel is a fusion
    boundary, so a plan that ignores it can win in isolation and lose end to
    end. A resumed run keeps the value it started with.
-   `served_calls` is the request's `served_calls`, copied verbatim, or
    `null` when it carried none. It records the calls the served workload makes to the
    target -- the operand and output shapes of each captured call, its
    production time and its share of the step -- and the kernel-worker builds
    the test harness from it in Phase 0.10, so the loop is measured on the
    calls the server makes. Never rewrite it into prose for a dispatch: the
    kernel-worker reads it from `state.json`. A resumed run keeps the value it
    started with; a state file written before the field existed has no
    `served_calls`, which reads as `null`.
-   `serving_config` is the request's `serving_config`, copied verbatim, or
    `null` when it carried none: the paths of the deployments manifest and the
    launch recipe the served model runs under. The kernel-worker reads both
    files in Phase 0.10 for what the call shapes do not fix -- maximum context,
    parallelism, dtypes -- so the harness is built from the configuration the
    server runs. Same rules as `served_calls`: never paraphrased into a
    dispatch, kept on resume, absent in an older state file means `null`.
-   `library_check` is the result of Setup step 1b:

    ```json
    {"op": "<op>", "result": "<result>", "module": "<module or null>", "url": "<url or null>", "summary": "<text or null>"}
    ```

    | `result` | Meaning |
    | --- | --- |
    | `"none"` | Tokamax has no kernel for this op. |
    | `"gpu_only"` | Tokamax has one, but not for TPU. |
    | `"not_checked"` | No op was named, or the lookup failed. |
    | `"use"` | Tokamax has a TPU kernel. kernel-worker calls it instead of writing one. |
    | `"use_failed"` | The Tokamax kernel failed or was not faster than the baseline (see **Control loop** step 1b), so the loop builds one. `summary` says why. |
    | `"build"` | The user wants a new kernel. `summary` describes the Tokamax kernel. |

-   `tpu_version` and `tpu_spec` stay `null`. The kernel-worker resolves them once in
    Phase 0 step 3 from `{{runtime}} compute inspect --json` — generation,
    topology, chip and device counts, HBM, and the hardware page to load — and
    writes them back, so every planning step reads one answer instead of
    probing the target itself. A run whose target cannot report its chip stops
    there rather than assuming one. You have no shell, which is why this is not
    resolved here.

`baseline_time` starts `null` and is written once, from the first kernel-worker result
that reports both a `latency_ms` and a `speedup_factor`: the baseline is
`latency_ms * speedup_factor`, the number the harness already measured the base
kernel at. It exists because the result you owe at the end carries a
`speedup_factor`, and without a recorded baseline that field could only be
guessed. Never overwrite it on a later iteration — the baseline does not move,
and a speedup measured against a drifting denominator is not a speedup.

Every path stored inside `state.json` is written fully resolved — with `<work>`
expanded to `{{runs}}/<run_id>/artifacts/tpu-kernel-development/<slot>` for this session's
actual `run_id` and this kernel's slot — so that a fresh kernel-worker can open
it without knowing your working directory.

## Control loop — mechanical, follow exactly

This is a protocol, not a narrative suggestion. Do not paraphrase it from memory
partway through — re-read this section's steps as literally as you re-read
state.json.

1.  **Read `<work>/state.json`**, right now, even if you think you already
    know what it says. Read `<kernels>` too, and check that this slot is still
    this kernel's: the `base_kernel_path` and `target_op` of the `<kernels>`
    entry for your slot, and of `state.kernel`, must both still be the ones
    you resolved in **One slot per kernel** (`real_path` is not compared here:
    it is filled in mid-run). If either names another kernel, a concurrent call claimed your slot; stop
    dispatching and return `failed` with an `error_message` naming the slot
    and both kernels. Every iteration after that point would be written over
    the other kernel's files or have its own written over. Then run the
    **same-file check** from **One slot per kernel**.
1b. **If the Tokamax kernel failed or is slower, switch to a normal build.**
    When `state.library_check.result` is `"use"` and `state.history` has an
    entry, check its last entry. The Tokamax kernel failed if it:
    -   did not both compile and pass its tests (Tokamax or the op missing, a
        compile error, or a wrong result), or
    -   is not faster than the baseline: its `speedup` is not above 1.0, or
        its `optimized_time` is not below `state.baseline_time`.

    Then:
    -   set `library_check.result` to `"use_failed"`, and put the reason in
        `library_check.summary`: the kernel-worker's error, or both times
        (e.g. "tuned Tokamax 0.512 ms, baseline 0.481 ms");
    -   reset `state.best_code_path` to `state.base_kernel_path` and
        `state.best_optimized_time` to `"Infinity"`, so the Tokamax kernel is
        not kept as the result;
    -   set `state.max_iterations` to `state.iteration` plus the request's
        `max_iterations` (5 by default).

    Write `<work>/state.json` back. The next iterations plan and build a
    kernel as usual, reusing the baseline and test harness already built.
2.  **Check the stop condition against what you just read:** `state.iteration >=
    state.max_iterations` (the budget you resolved in Setup, 5 by default)?
    -   Yes -> go to "Finish" below. Stop looping.
    -   No -> continue to step 3.
3.  **Dispatch exactly one iteration** for
    `target_iteration = state.iteration + 1`. Initialize `attempt_count = 1`.

    Invoke {{agent:kernel-worker}}, passing the session `run_id` verbatim and
    `<work>` fully resolved as `work_dir`.

    This spawns a fresh agent with no memory of your conversation; it works in
    the `work_dir` you passed, pulls its iteration number, input file, and
    previous latency from `<work>/state.json`, and writes the updated
    `state.json` itself before it exits.

3a. **Verify the iteration against disk, not against the kernel-worker's report.**
    Re-read `<work>/state.json`.

    -   **`state.iteration >= target_iteration`** — the iteration genuinely
        completed. Go to step 4.
    -   **State unchanged** — the kernel-worker finished, errored, or gave up without
        committing the iteration. Check step 3b first: the endings it lists are
        not failures and must not be retried. Otherwise append the kernel-worker's
        reported error to `<work>/tpu_kernel_debug_history.md`, increment
        `attempt_count`, and re-dispatch the *same* `target_iteration`. If
        `attempt_count > 3`, go to **Emergency Stop** below.

    The cap is what stops this becoming a machine for burning a budget. Step 6
    tells you never to end your turn while iterations remain, and without a cap
    those two rules together are an unbounded loop against a target that is
    failing for a reason re-dispatching cannot fix.

3b. **Four kernel-worker endings are not iteration failures, and must not be retried.**
    Re-read `<work>/state.json` before deciding which you are looking at:

    -   **The baseline is awaiting approval.** `state.primary.language` is
        still `"specification"` and the kernel-worker returned `invalid_input`
        carrying a synthesized baseline and its rationale. Present both to the
        user, get their decision, and re-dispatch the kernel-worker once they have
        approved or asked for changes. Do not count it as an attempt, and do
        not approve it yourself — the whole point of the gate is that a human
        reads a baseline no human wrote.
    -   **The baseline could not be verified.** `state.primary.port_verified`
        is `false`: the kernel-worker's Phase 0.9 gate established that `base.py` does
        not compute what the caller's source computes, after three repair
        attempts. **Do not re-dispatch.** Return a `failed` `kernel-result`
        naming `<work>/port_verification.json` as the diagnosis. Retrying
        cannot help — every iteration would be measured against a wrong
        denominator, and five iterations of confident, wrong speedups is a
        worse outcome than stopping.
    -   **The base kernel's imports don't resolve on the target.** The
        kernel-worker returned `invalid_input` from Phase 0.10 with an
        `error_message` naming an import of `base.py`'s own package that the
        target could not load. **Do not re-dispatch**, and never patch around
        it: don't edit `base.py`'s imports, and don't add files, `__init__.py`
        files or symlinks for the missing module. Return an `invalid_input`
        `kernel-result` with that `error_message`; the caller needs a target
        that runs the server's image.
    -   **The target needs a setting only the user can give.** The
        kernel-worker returned `needs_input` from Phase 0 step 1: on a
        `gke_job` profile with no working `staging_bucket`, no profile trace
        can come back, so every iteration would be planned without one.
        **Do not re-dispatch**, and do not run the setup command yourself.
        Return a `needs_input` `kernel-result` whose `summary` carries the
        kernel-worker's `error_message` verbatim -- the question and the setup
        command; the caller asks the user and dispatches the run again.

4.  **When the kernel-worker agent returns, get its trace explained.** Re-read
    `<work>/state.json`; take the last entry of `state.history`. If its
    `profile_trace_dir` is a path (the kernel-worker captured a trace in its Phase
    4.5) invoke {{agent:tpu-diagnostics}} with a `profiling-request` JSON
    object, using that entry's own `profile_trace_dir` value and both paths
    spelled out fully resolved:

    ```json
    {"prompt": "Analyse the XProf trace of iteration <n>: the fields below, the top ops by device time, and what the tools could not see.", "run_id": "<run_id>", "traces": [{"path": "<profile_trace_dir>"}], "report_path": "<work>/iter<n>/profile_summary.md", "fields": ["alu_utilization_percent", "memory_bandwidth_utilization_percent", "step_time_ms", "dma_sync_wait_to_compute_ratio"]}
    ```

    Those four headline numbers are what the previous profiling stage always
    reported, and next iteration's kernel-worker feeds this report to its planning
    subagent as `previous_profile_summary`. The analyst returns only the
    fields it is asked for, so dropping one here drops it from the planner's
    input.

    Hand it the **trace**, never the kernel. `optimized.py` is a kernel
    definition, not a profiling workload: capturing one needs the run's real
    `get_inputs()`, a warm-up, and the LLO debug-info flag set before jax is
    imported.
    Only the kernel-worker has the first of those, so capture sits there and only
    analysis comes here.

    **When `state.ideas_ledger_path` is set, ask for one more section.** Append
    to that request's `prompt`: "Then read `<work>/ideas_ledger.json`. For every idea whose
    `status` is `adopted`, its `falsifiable_as` field names a specific quantity
    — HBM bytes per call, a FLOP count, an MXU utilization percentage, a
    SyncWait fraction, a latency delta against a named earlier iteration. Find
    that quantity in this trace, compare it against the prediction (and against
    the previous iteration's report when the claim is relative), and add a
    `## Ledger Adjudication` section: one row per id with the predicted value,
    the measured value, and a verdict of `confirmed`, `refuted` or
    `inconclusive`. Judge the claim as written, not the kernel overall — an
    idea can be confirmed in an iteration that was slower for another reason.
    `inconclusive` is the honest verdict when the trace does not carry the
    quantity the claim names; use it rather than guessing."

    The next iteration's kernel-worker transcribes those verdicts into the ledger in
    its Phase 0.11. That is what makes the loop *learn* from a reference rather
    than merely read one: without it, a borrowed idea that made the kernel
    slower stays in the ledger looking as promising as one that worked.

    When it returns `status: "reported"`, set that history entry's
    `profile_summary` to the result's `profiling_report_artifact` and write
    `<work>/state.json` back. If `profile_trace_dir` is `null` — the iteration
    did not compile, failed its tests, or the capture itself failed — or the
    profiling result has any other status, leave `profile_summary` as `null`
    and continue; a missing profile is not a reason to stop iterating. Either way you never read a trace
    yourself.

5.  **Do not trust either summary text for whether to keep going.** Go back
    to step 1 and re-read `<work>/state.json` fresh. Repeat.

6.  **Hard rule:** never end your turn while `state.iteration <
    state.max_iterations`, unless step 3a's three attempts are exhausted or
    step 3b sent you to a stop. If you notice yourself about to summarize
    results and stop, treat that as a bug in yourself — re-read state.json
    before doing anything else; if the count is still under the budget and you
    have attempts left, dispatch the next iteration instead of stopping.

## Emergency Stop

Reached when one iteration has failed to commit 3 times running. The run ends
here; do not dispatch again.

1.  **Release anything the run is still holding.** A synchronous
    `{{runtime}} compute run` reaps its own remote process when it times out,
    so there is usually nothing left — but a kernel-worker that died between
    submitting and returning can leave a batch job behind:

    ```bash
    {{runtime}} compute job list --run-id <run_id>
    {{runtime}} compute job reap --run-id <run_id>
    ```

    Skip the reap if `job list` is empty. Do not pass `--include-serving`:
    this loop starts no serving jobs, and one that exists belongs to something
    else.
2.  Append the final attempt's error to `<work>/tpu_kernel_debug_history.md`.
3.  Re-read `<work>/state.json` one last time.
4.  Return a `failed` `kernel-result` naming what died and how many iterations
    did complete. Report `iterations` and `iterations_run` from
    `state.history` anyway: which iterations were attempted and how they died
    is the useful part of a failure, and a kernel that passed its tests in
    iteration 2 is still worth handing back even though the run ended at 3.

## Finish

Once `state.iteration == state.max_iterations`:

1.  Read `<work>/state.json` one last time, and run the **same-file check**
    from **One slot per kernel** before reporting anything.
2.  **Open by saying what the baseline actually is.** Name
    `state.primary.language`. For `pytorch`, say
    explicitly that every number below is measured against the JAX reference
    at `<work>/base.py` — not against the original on its original hardware.
    Then state how much that reference can be trusted, which depends
    on how it was produced:

    -   **`port_verified: true`** — the reference
        was hand-written and checked **by value** against golden outputs
        captured from the user's own module (Phase 0.9). Point to
        `state.primary.context_path` for how the port was derived.

        Say "agrees with the source to within `atol`", not "is the same
        computation". The gate compares values only. A semantic difference
        smaller than the tolerance, such as a dropped epsilon at 7.8e-05,
        passes it. That is a real if narrow gap in the denominator and the
        reader is entitled to know it is there.
    -   **The baseline was synthesized** (`state.primary.baseline_approved` is
        set) — say so first, before any number. The denominator was written by
        an agent from a description and approved by the user; it is not the
        user's own code. Point at `<work>/baseline_rationale.md` and its
        assumptions list. When the baseline was synthesized directly as JAX,
        add that nothing independent ever checked it.
    -   **`port_verified: false` / `null`** — say so plainly and give the
        reason (a `jax` primary, or a degraded capture with its recorded
        cause). For a `jax` primary that is not a caveat at all: `base.py`
        *is* the caller's code. For a degraded capture it means the speedup
        numbers rest on an agent's reading of the source and nothing else.
3.  Report a short table to the user: iteration, compile_ok, test_ok,
    optimized_time, speedup, base_choice for each history entry.
3b. **When `state.ideas_ledger_path` is set, report what the reference actually
    contributed.** `read` the ledger and reproduce it as a table: per borrowed
    idea, its class, which iteration adopted it, and whether the XProf
    adjudication confirmed, refuted or could not settle its predicted
    mechanism. State the `reference_trust` verdict alongside it.

    Report this honestly, including when it is unflattering. If every adopted
    idea was refuted, or the reference contributed nothing, say that — it is a
    real and useful result about that reference, and a table of confirmed
    verdicts is only worth anything if refuted ones would have been printed
    too. Never describe the reference as having helped without a confirmed
    entry to point at. An idea still sitting in `adopted` was never
    adjudicated, usually because the run ended before its trace was read; say
    that rather than counting it either way.
3c. **Say what the speedup is measured against, once more, plainly.** Every
    number above is relative to `base.py` — idiomatic JAX on this TPU. For a
    `pytorch` primary that is *not* the same question as "will this
    make my original faster", and the difference is not a detail: a kernel
    reporting 3x here may have mostly beaten a slow reference. This loop cannot
    close that gap — timing the user's real PyTorch against the kernel needs
    `torch_xla`, which this package does not install — so name the limit rather
    than letting the number stand in for an answer it cannot give.

4.  Point to the best code file (`state.best_code_path`) as the result of the
    loop, spelled out as its fully resolved path. When
    `state.library_check.result` is `"use"`, say that this file calls the
    Tokamax kernel (this run did not write it), and give its latency before
    and after tuning, from the iteration's `autotune_summary`. When it is
    `"use_failed"`, say that the Tokamax kernel was tried first and why it
    was not kept (`library_check.summary`), then report the built kernel.

    The kernel-worker keeps a kernel in `state.best_code_path` only when its
    paired `speedup` is above 1.0 (its Phase 5). So when `state.best_code_path`
    is still `state.base_kernel_path` but some `state.history` entry compiled
    and passed its tests, **no kernel this run built beat the baseline**. Say
    that first, instead of pointing at a best file: `base.py` stays the
    version to run. Name the fastest attempt -- the passing entry with the
    highest `speedup` -- and its `speedup`, as an attempt, not a result.
5.  Return a `kernel-result`, with `slot` and `same_file_slots` as **One slot
    per kernel** describes, and `base_kernel_path` as the slot's `<kernels>`
    entry spells it — for pasted text, the file Setup step 0 wrote, which is
    how the caller continues that kernel. The table you just reported goes back as data, not
    only as prose — `state.history` maps entry for entry onto the result's
    `iterations` array, and `state.best_code_path` and
    `state.best_optimized_time` onto `best_kernel_artifact` and
    `best_latency_ms` (a `no_speedup` result reports its fastest attempt
    there instead, as described below).

    Cite the path to the best kernel (relative to the run's artifacts
    directory, e.g. `artifacts/tpu-kernel-development/<slot>/iter<n>/kernel.py`) in
    `best_kernel_artifact` and in `artifacts`. If `state.best_code_path`
    is missing or unreadable, that is a `failed` result with an `error_message`
    saying so.

    `speedup_factor` comes from `state.baseline_time` and
    `state.best_optimized_time`, both measured. If `state.baseline_time` is
    still `null` because no iteration ever timed the baseline, report `failed`
    rather than a ratio against a number you do not have.

    `status` is `optimized` only when the loop ran to its budget and kept a
    kernel in `state.best_code_path`: one that compiled, passed its tests and
    beat the baseline. A run that used the Tokamax kernel
    (`library_check.result` is `"use"`) ends the same way. Every other ending
    has a status of its own:

    -   kernels compiled and passed, but none beat the baseline (step 4) ->
        `no_speedup`, with a `summary` that says `base.py` stays the version
        to run. Report the fastest attempt in the measurement fields, so the
        caller sees by how much it fell short: its `optimized_kernel` as
        `best_kernel_artifact`, its `optimized_time` as `best_latency_ms`, and
        `state.baseline_time` over that as `speedup_factor`, with `iterations`
        carrying every paired `speedup`. `speedup_factor` divides times taken
        in different sessions and can come out above 1.0 while the paired
        `speedup` does not; if it does, say in `summary` that the paired
        number is the one the loop judges by. `best_kernel_artifact` is not a
        kernel to integrate. Never report such a run as `optimized`: the
        caller integrates an `optimized` kernel, and here that would make the
        served model slower.

    None of the endings below carries `best_kernel_artifact`,
    `best_latency_ms` or `speedup_factor`:

    -   no baseline to start from, or a request you cannot act on ->
        `invalid_input`, with `error_message` saying what was missing.
    -   the target needs a setting only the user can give (step 3b) ->
        `needs_input`, with the kernel-worker's question and setup command in
        `summary`.
    -   the loop could not reach an answer — no iteration compiled or passed
        its tests, the device was unreachable, the run was cut short ->
        `failed`, with `failure_reason` and `error_message`. Report
        `iterations` and `iterations_run` anyway: which iterations were
        attempted and how they died is the useful part of a failure.

    Do not report `optimized` with the measurement fields or `artifacts`
    left out. The runtime rejects that, deliberately: a result claiming a
    kernel it cannot point at reads to the caller as one ready to use.

    Also carry back what the caller cannot see from a latency alone:
    `input_language` from `state.primary.language`, `baseline_verified` from
    `state.primary.port_verified` -- `null` included, since it says no check
    was possible -- and `reference_trust` from `state.reference_trust`, left
    out when that is `null`: no reference was supplied, and the result schema
    takes only a verdict there, so a copied `null` is rejected at the
    boundary. A `speedup_factor` of 3.0 against an unverified
    port and one against a gated port are not the same claim, and a caller
    reading only the number has no way to tell them apart.

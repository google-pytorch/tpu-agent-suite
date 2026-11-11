---
description: tpu-code-stack's procedure. The pinned image's vllm_torchtpu, as a copy on this host or its in-image git history, is base; every hypothesis is a branch, every test a commit, a discard a tag, a stack is assembled from branches, and each job gets its ref as one diff from base. Each action, who may take it, and the git it runs.
scripts:
- scripts/common/plugin_source.py
- scripts/common/tpu_slots.py
---

# Campaign repo

The pinned image is the baseline. When `<repo>` is a copy of the image's plugin tree on this host
(`/workspace/vllm-torchtpu` in an image built from it), that copy is the repo, and its HEAD, or
its one commit when it had no `.git`, is `base`. Otherwise the image's installed vllm_torchtpu
must be a git checkout with no modified tracked files, and its HEAD is `base` in a new `<repo>`.
Untracked files and commits no remote has are fine. Every change is a local commit on `base`. No image is built and
nothing is extracted but the history. `<work>` is the directory holding `<repo>`; worktrees go in
`<work>/<branch with / as ->`, and a ref's patch `<patch>` is
`<work>/patches/<ref with / as ->-<sha7>.patch`. `base` writes the base record to
`$(git -C <repo> rev-parse --git-common-dir)/plugin_source.json`.

You are the only agent that runs git or plugin_source.py on this repo. Callers edit files in a
worktree you made, or read; everything that changes a ref is yours.

## Actions

| action | caller | does | returns |
| --- | --- | --- | --- |
| `init` | hill-climb | probe the image, tag `base`, branch `stack` at `base` | `plugin` (stack), `worktree` |
| `branch` | hill-climb | branch `name` (`hyp/<rank>-<slug>`) from `from` (default `stack`) and its worktree | `plugin`, `worktree` |
| `commit` | writer | commit the plugin changes in `worktree`, `message` as its message | `plugin` |
| `discard` | hill-climb | `what: test`: tag `tried/<slug>/<n>`, reset one commit; `what: hypothesis`: tag `discarded/<slug>`, remove the branch and worktree | `tag` |
| `win` | hill-climb | cherry-pick `stack..<name>` onto `stack` | `plugin` (stack) |
| `stack` | hill-climb | branch `name` from `base`, cherry-pick `branches` in order | `plugin` |
| `ref` | any | cut `ref` as one patch from `base` | `plugin` |
| `show` | reviewer, hill-climb | the log and diff of `ref` from `against` (default `base`), and its slug's tags | `log`, `diff_path`, `tags` |
| `push` | hill-climb | commit `stack` into the pinned image and push it as `name` | `image` |

A caller the table does not list for the action gets `refused`, with the reason, and the repo is
unchanged. `<slug>` is the branch name less `hyp/`. Tags keep every lost test reachable; never
delete one, and never move `stack` except by `win`.

## `init`

If `<repo>` already holds a base record for `image`, run only the last line. If `<repo>` holds the
plugin tree (a `vllm_torchtpu/` within three levels), it is a host copy: no job, just

```bash
python3 {{script:common/plugin_source.py}} local <repo> --image <image>
git -C <repo> switch stack 2>/dev/null || git -C <repo> switch -c stack base
```

`local` exit 7 (modified tracked files with no `base` yet) returns `blocked` with its message, and
exit 2 (the repo holds another image's base) `invalid_input`. Otherwise run one job on the image
that bundles the history, under a slot released on every exit:

```bash
python3 {{script:common/tpu_slots.py}} claim --run-id <run_id> \
  --holder tpu-code-stack:probe --profile <profile>
{{runtime}} compute submit --profile <profile> --image <image> \
  --recipe {{script:common/plugin_source.py}} --output plugin.bundle --output plugin.json \
  --run-id <run_id> --name tpu-code-stack-plugin-probe -- probe --output plugin.bundle
python3 {{script:common/tpu_slots.py}} set-job --run-id <run_id> --resource <resource> \
  --holder tpu-code-stack:probe --job-id <job_id>
{{runtime}} compute job wait --job-id <job_id> --until done
python3 {{script:common/tpu_slots.py}} release --run-id <run_id> --resource <resource> \
  --holder tpu-code-stack:probe
{{runtime}} compute job artifacts --job-id <job_id> --out-dir <work>/jobs/<job_id>
python3 {{script:common/plugin_source.py}} base <repo> <work>/jobs/<job_id> --image <image>
git -C <repo> switch stack 2>/dev/null || git -C <repo> switch -c stack base
```

Probe exit 7 returns `blocked` with the probe's message: the image's plugin is not a git checkout,
or has modified tracked files. A refused claim also returns `blocked`, with the claim's output. A
repo holds one base; `base` exit 2 for another image is `invalid_input`. The repo's own work dir
stays on `stack`: return it as `worktree`, and `stack` cut as in `ref` as `plugin`.

## `ref`

```bash
python3 {{script:common/plugin_source.py}} diff <repo> <ref> <patch>
```

The printed record, less `image`, with the patch file as `patch`, is `plugin`. A ref still at
`base` gets an empty patch. Exit 6 returns `refused` with its message: the ref does not descend
from `base`, or touches files outside the plugin.

## `branch`

```bash
git -C <repo> branch <name> <from>
git -C <repo> worktree add <work>/<name with / as -> <name>
```

A branch that exists already is rebased onto `from` in its worktree
(`git -C <worktree> rebase <from>`): that is how a hypothesis cut before the last win joins the new
stack. A rebase that conflicts is aborted (`git rebase --abort`) and returns `conflict` with the
files. Return the worktree, and the branch cut as in `ref`.

## `commit`

The worktree must be on a `hyp/` branch (`git -C <worktree> branch --show-current`); `stack`,
`base` or a detached HEAD is `refused`. Read `prefix` from the base record, then:

```bash
git -C <worktree> status --porcelain        # any path outside <prefix>/vllm_torchtpu/: refused
git -C <worktree> add -A -- <prefix>/vllm_torchtpu
git -C <worktree> commit -F <work>/commit-msg.txt  # message written to a file, never inlined
python3 {{script:common/plugin_source.py}} diff <repo> <branch> <patch>
```

A change outside the plugin returns `refused` listing the paths, before anything is staged. No
change under the plugin is `refused` too: there is nothing to commit. `diff` exit 6 here means the
branch does not descend from `base`: undo the commit (`git -C <worktree> reset --soft HEAD^`),
leaving the writer's edits in place, and return `refused` with its message. Otherwise the record
is `plugin`, as in `ref`.

## `discard`

Find the branch's worktree with `git -C <repo> worktree list --porcelain`.

- `what: test`: `git -C <repo> tag tried/<slug>/<n> <name>`, `n` the next free number, then
  `git -C <worktree> reset --hard HEAD^` (`git -C <repo> branch -f <name> <name>^` with no
  worktree).
- `what: hypothesis`: `git -C <repo> tag discarded/<slug> <name>`, then
  `git -C <repo> worktree remove --force <worktree>` and `git -C <repo> branch -D <name>`.

## `win` and `stack`

```bash
git -C <repo> cherry-pick stack..<name>                       # win, in the work dir on stack
git -C <repo> worktree add -b <name> <work>/<name with / as -> base   # stack
git -C <work>/<name with / as -> cherry-pick $(git -C <repo> merge-base stack <branch>)..<branch>
```

For `stack`, the second line runs once per entry of `branches`, in order; each picks only that
branch's own commits. A cherry-pick that conflicts is aborted (`git cherry-pick --abort`) after
`git diff --name-only --diff-filter=U` lists the files, and returns `conflict` with them: the two
branches touch the same code, and the later one is redone on the new stack. A `stack` that
conflicts deletes its half-built branch. Return the new head (`stack`, or `name`) cut as in `ref`.

## `show`

```bash
git -C <repo> log --oneline <against>..<ref>
git -C <repo> diff <against> <ref> > <work>/patches/show-<ref with / as ->.diff
git -C <repo> tag -l "tried/<slug>/*" "discarded/<slug>"
```

Return the log text, the diff path and the tags. Read only: nothing changes.

## `push`

Read `image` and `path` (the checkout's path in the image) from the base record, then:

```bash
{{runtime}} compute image commit --from <image> --repo <repo> --ref stack --base base \
  --path <path> --name <name>
```

It prints one JSON object; its `image`, pinned by digest, is the result's `image`. Return `skipped` with the error code as `reason` when it
reports `IMAGE_REGISTRY_NOT_CONFIGURED`, `MISSING_TOOL` (no docker or git) or `INVALID_BACKEND`
(the profile's backend has no image field), or when the runtime has no `image` command. Any other
error is `failed`.

Env and flag overrides are not code: they ride an arm as overrides, never as commits.

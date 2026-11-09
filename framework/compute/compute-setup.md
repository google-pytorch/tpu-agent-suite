# Compute setup

How to set up, check and repair the compute profile that compute commands
use. The runtime makes every decision; this guide only says which commands
to run and what to do with their output. Every command prints JSON with
`--json`. Run them as given, without pipes or redirections such as `2>&1`:
while a skill limits the shell, only the plain commands are allowed.

## When to follow this guide

- The session-start message says compute is not set up (start at step 1),
  not verified (start at step 4), or failed its check (tell the user what
  failed, then follow step 4's "On failure").
- A compute command fails with `COMPUTE_NOT_CONFIGURED`: no profile is
  selected for this run, or the selected one is incomplete.
- A compute command fails with another `*_NOT_CONFIGURED` code (for
  example `STAGING_NOT_CONFIGURED`): a field that is asked only when first
  needed is not set. See [Fields set later](#fields-set-later).
- The user asks to change where work runs, or a single setting.

Only the main session can ask the user questions. A subagent that hits one
of these errors stops and reports the error, with its `missing` and `fix`,
to whoever delegated to it. It does not edit `environment.json` or guess
values.

## Setting up a profile

1. Run `{{runtime}} compute --package {{package_tag}} setup plan --json`. Add `--profile <name>` to
   set up a profile other than the one this run uses (a new name creates
   it), and pass the same `--profile <name>` to every later step. The
   result has `questions`, each with a `prompt`, whether it is
   `required`, any `choices`, the `current` value, and a `suggested` answer
   with where it came from (`suggested_from`). When a probe found several
   values and none is clearly meant (say, TPU VMs in a shared project), the
   question has no suggestion but lists them as `choices`. A choice can be
   an object naming several fields (a TPU VM's `tpu_name` and its `zone`):
   picking it answers each of them, so offer it as one option.
   - If it lists `probe_warnings`, show them to the user: a probe whose tool
     is missing or that timed out left its question without a suggestion,
     and a cluster with no TPU nodes yet is reported, not investigated.
   - A question with no suggestion is asked as it is. Don't run your own
     commands to find an answer; plan has already looked. With no
     suggestion and no `choices`, ask the user in free text; don't invent
     options.
   - With no backend chosen yet, the only question is `backend`, and
     `profiles` lists the profiles that already exist. Ask the user which
     backend to use, as in step 2, even though it has a suggestion; never
     choose it yourself.
     If an existing profile fits, the user can pick it instead of creating
     another: `{{runtime}} config --package {{package_tag}} use <name>`. Once
     the user picks a backend, plan again with the same `--profile` and
     `--backend <backend>` to get that backend's questions, with their
     suggestions and the optional ones, and ask those as step 2.
2. {{built-in:ask_user_question}} every returned question in one round,
   optional ones too, with the suggestions as defaults. Never answer a
   question yourself unless the user accepted its suggestion. Users often
   answer in prose ("GKE, team-a namespace, my image is X"): map what they
   said to the question names and ask only for what is still missing.
3. Write the answers as one JSON object of `{field: value}`, keyed by the
   question names, plus `backend` if it was asked in step 1, and run
   `{{runtime}} compute --package {{package_tag}} setup apply --profile <name> --answers '<json>'`
   with the object in single quotes. Name a new profile yourself, after its
   backend, or `default` for the first one. `""` or null unsets a field. Apply saves
   all or nothing:
   - `ok`: the profile is saved, and becomes the default if none is set
     or the default names a missing profile.
     If this run uses another profile and the user wants this one, run
     `{{runtime}} config --package {{package_tag}} use <name>`. It switches
     only this run; add `--default` only if the user asks to change the
     profile other sessions start with.
   - `INCOMPLETE`: nothing was saved. Ask the user for every field under
     `missing`, `newly_applicable` (fields an earlier answer made relevant)
     and `invalid` (with its `reason`), then apply again. Repeat until it
     returns ok.
4. Run `{{runtime}} compute --package {{package_tag}} setup verify`, with `--profile <name>` if it
   isn't the profile this run uses. It checks the tools, credentials and
   target, and on some backends the software on the target.
   - On `gke_job` and other backends that schedule Jobs, verify can also
     submit a smoke-test Job, which can take several minutes. Ask the user
     first, and add `--smoke-test` only if they agree.
   - On `gke_job` and `gke_pod`, a `kube_context` warning means kubectl's
     current context is not the one the profile pins. Compute commands still
     use the profile's context, so tell the user and don't run
     `kubectl config use-context`; to move the profile, plan again with
     `--fields kube_context`.
   - On failure, each failed check has a `detail`, a `fix` and the `fields`
     to ask about again. Tell the user what failed and its fix. If it names
     fields, plan again with `--profile <name> --fields <a,b>` and go back to
     step 2; a fix that needs the user (credentials, cluster access, IAM) is
     theirs to do before you verify again.

If the user says skip, carry on with work that needs no TPU. Compute commands
fail with `COMPUTE_NOT_CONFIGURED` until setup is done.

## Fields set later

Some fields only matter to some workflows, so setup does not ask for them
(for example `staging_bucket`, `pod_service_account`). A
command that needs one fails with an error like:

```json
{
  "status": "error",
  "error_code": "STAGING_NOT_CONFIGURED",
  "missing": ["staging_bucket"],
  "fix": "<runtime> compute --package <package> setup plan --profile <name> --fields staging_bucket"
}
```

Run the `fix` command (it names the profile the failing command used), ask
the user the questions it returns, apply the answers to that profile as in
step 3, verify, then retry the command that failed.

A `gke_job` output too large for the pod's log is lost the same way: `run`
prints `Fix, then rerun: <command>`, and `job artifacts` fails with
`STAGING_NOT_CONFIGURED`. Run the command, ask, apply, verify, then rerun the
job: the output went with the pod, so only a rerun produces it.

## Other changes

- Use another existing profile for the rest of this run:
  `{{runtime}} config --package {{package_tag}} use <name>`. Add `--default`
  to make it the default for runs that pick none.
- See every profile and its state (`NOT_CONFIGURED`, `UNVERIFIED`, `FAILED`
  or `READY`): `{{runtime}} config --package {{package_tag}} show`.
- Change one setting: if the user gave the value, run
  `{{runtime}} config --package {{package_tag}} set <field>=<value>`;
  otherwise plan with `--fields <field>` and ask, then apply. Verify after
  either.
- Remove a profile the user asked to delete:
  `{{runtime}} config --package {{package_tag}} remove <name> --user-confirmed`.

Never create, copy, move or edit `environment.json` by hand: apply validates
every change before it is saved.

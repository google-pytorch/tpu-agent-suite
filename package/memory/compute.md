For compute, those tasks are onboarding, optimizing, profiling or benchmarking a
model on Cloud TPUs, not kubectl, manifests or other clusters the user names.
Don't run `{{runtime}} compute` to check whether this applies.

For that work, every job, server, log and port-forward goes through
`{{runtime}} compute`. Don't write Kubernetes manifests or run `kubectl`,
`xpk`, `gcloud compute` or `ssh` against that cluster yourself: it is shared,
and compute is what keeps this run to its own jobs. Its section in each agent's
instructions, and `compute --help`, say how. A job name or image a task asks
for still goes through compute: give the name as `--name` (on `gke_job` the
compute profile's `job_name_prefix` goes in front and a taken name gets a short
suffix, so the name stays whole inside the job's; a task that requires names to
start a certain way can pass `--job-name-prefix` for that call), and pass
`--image <ref>` to `run` or `submit` to use an image other than the profile's.
If a rule still can't be met through compute, say so in your report rather
than reaching the cluster another way.

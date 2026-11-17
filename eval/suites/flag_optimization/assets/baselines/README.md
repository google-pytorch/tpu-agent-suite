# Recorded baselines

`record_baseline.py` writes one `<task_id>.json` here per task: the baseline
config measured on the task's slice, with checkpoint weights and with random
init. The suite stages this directory into each task's `tests/baselines/`, and
the verifier scores the agent's config against the recording instead of
re-running the baseline in every trial.

TODO: no baselines are recorded yet (this needs TPU access). Until a task has
one, its verifier measures the baseline live, next to the agent's config.

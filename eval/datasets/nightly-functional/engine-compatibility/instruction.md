Run a vLLM engine compatibility check and TP/DP planner on the model `Qwen/Qwen2.5-7B-Instruct` on the GKE TPU cluster configured for this package using the `engine-compatibility` reference guide and `engine_compatibility_check.py` script.

Inspect the cluster's available devices and HBM per device, run `engine_compatibility_check.py` on the accelerator, save `tp_plan.json`, and output a concise compatibility and parallelism summary.

Important execution instructions:
- Only run the engine compatibility check (do not deploy or launch the inference server).
- Run the compatibility check against the remote GKE TPU cluster via `compute run` (with `--timeout 540` so a busy cluster returns cleanly rather than hanging the session). Do not install dependencies or modify the environment to force the check to succeed.
- Report the check's actual outcome, planned tensor/data parallelism (`tp_plan.json`), and findings (including if vLLM is unavailable on the cluster or if all TPU nodes remained busy for the queue timeout).

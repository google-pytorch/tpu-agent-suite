Synthesize a benchmark script `benchmark_tpu.py` for a Causal LM model on PyTorch/TPU using the `torch-benchmark-coder` reference guide and the `torch-benchmark-coder/causal_lm.md` recipe.

Copy the `benchmark_tpu.py` template into `<work>/benchmark_tpu.py` (`<runs>/<run_id>/artifacts/tpu-model-onboarding/benchmark_tpu.py`) and replace the `benchmark_model` placeholder using the Causal LM recipe.

Important execution instructions:
- Synthesize `<work>/benchmark_tpu.py` for Causal LM with `torch_tpu` device initialization (`get_device("tpu")`) and without `torch_xla`. Do not execute or benchmark the script on hardware.

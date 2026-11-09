Synthesize a Custom FastAPI serving script `fastapi_server.py` for a Causal LM model using the `torch-inference-coder` reference guide and the `torch-inference-coder/fastapi_server/text_generation.md` recipe.

Synthesize `<inference_work>/fastapi_server.py` (`<runs>/<run_id>/artifacts/tpu-model-onboarding/inference/fastapi_server.py`) for Causal LM serving on TPU.

Important execution instructions:
- Synthesize `<inference_work>/fastapi_server.py` for Causal LM with TPU device initialization (`get_device('tpu')`, `AutoModelForCausalLM`, `AutoTokenizer`, and `/generate` endpoint) without `torch_xla`. Do not modify the read-only shipped `inference.py` template, do not create `deployments.json`, and do not launch the server on hardware.

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Template schema for inference.py.

This file contains the boilerplate code for setting up an inference server with
FastAPI and PyTorch. The actual model execution logic must be synthesized by
implementing the `AGENT TODO` placeholders.
"""

import contextlib
import json
from typing import Annotated, List, Literal, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration / Pydantic Definitions
# ---------------------------------------------------------
class DeploymentBase(BaseModel):
    model_id: str
    port: int = 8000
    engine: str


class FastAPIDeployment(DeploymentBase):
    engine: Literal["custom_fastapi"] = "custom_fastapi"
    max_model_len: Optional[int] = None


class VLLMDeployment(DeploymentBase):
    engine: Literal["vllm"] = "vllm"
    max_model_len: Optional[int] = None
    tensor_parallel_size: int = 1
    data_parallel_size: Optional[int] = None


class SGLangDeployment(DeploymentBase):
    engine: Literal["sglang"] = "sglang"
    context_length: Optional[int] = None
    tp_size: int = 1
    dp_size: Optional[int] = None


Deployment = Annotated[
    Union[FastAPIDeployment, VLLMDeployment, SGLangDeployment],
    Field(discriminator="engine"),
]


class DeploymentsConfig(BaseModel):
    deployments: List[Deployment]


def load_config(path: str = "deployments.json") -> Deployment:
    with open(path, "r") as f:
        data = json.load(f)
    cfg = DeploymentsConfig(**data)
    if not cfg.deployments:
        raise ValueError("deployments.json contains no deployments.")
    return cfg.deployments[0]


def get_device(device_str: str) -> torch.device:
    """Returns CPU, CUDA, or TPU device identifiers."""
    if device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "tpu":
        try:
            # pylint: disable=g-import-not-at-top
            from torch_tpu import api

            return api.tpu_device()
        except ImportError:
            return torch.device("tpu")
    elif device_str == "cuda":
        return torch.device("cuda")
    else:
        raise ValueError(f"Unknown device: {device_str}")


# ---------------------------------------------------------
# Fast API Server Setup
# ---------------------------------------------------------
config = load_config()


def load_model() -> None:
    """>>> AGENT TODO: Load your model, processor, or tokenizer here.

  Runs at server startup, before uvicorn prints `Application startup
  complete.`, never at import: inference_test.py imports this module with no
  model to load. Assign what you load to module globals.
  Example:
    global model
    model = AutoModelForCausalLM.from_pretrained(...).to(device)
  """


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


# >>> AGENT TODO: Implement the Pydantic Request schema specific to your model
class InferenceRequest(BaseModel):
    prompt: str


@app.post("/generate")
def generate_endpoint(req: InferenceRequest):
    """>>> AGENT TODO: Implement pipeline-specific inference logic here.

  Use the requested models, prepare inputs, execute forward pass, and return a
  JSON serializable dict.
  """
    del req  # Unused.
    try:
        raise NotImplementedError("Agent must implement the generation logic.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    if not isinstance(config, FastAPIDeployment):
        print(
            "Skipping custom_fastapi server as engine is not custom_fastapi.")
        exit(0)
    uvicorn.run(app, host="0.0.0.0", port=config.port)

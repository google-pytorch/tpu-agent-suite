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
"""The served model's config.json, at the revision the server ran.

    python3 fetch_hf_config.py --deployments <inference_work>/deployments.json \\
        --out <work>/hf_config.json

The model id and revision come from the deployment (`model_id`,
`revision`), or from `--model` / `--revision`. A local directory with a
config.json is copied; a Hugging Face id is fetched with huggingface_hub
(HF_TOKEN for gated models). Several detectors read the model's dimensions
to tell a model axis from a serving one, so analyze.py should not run
without this file.
"""

import argparse
import json
import os
import shutil
import sys


def resolve(deployments: str | None, model: str | None,
            revision: str | None) -> tuple[str, str | None]:
    if deployments:
        with open(deployments, encoding="utf-8") as f:
            doc = json.load(f)
        dep = (doc.get("deployments") or [doc])[0]
        model = model or dep.get("model_id")
        revision = revision or dep.get("revision")
    if not model:
        raise ValueError("no model id: pass --model or a deployments.json "
                         "with model_id")
    return model, revision


def fetch(model: str, revision: str | None, out: str) -> str:
    local = os.path.join(model, "config.json")
    if os.path.isfile(local):
        shutil.copyfile(local, out)
        return local
    if model.startswith("gs://"):
        raise ValueError(f"{model}: copy its config.json and pass it with "
                         "--hf-config; gs:// is not fetched here")
    # pylint: disable=import-outside-toplevel
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(model,
                           "config.json",
                           revision=revision,
                           token=os.environ.get("HF_TOKEN") or None)
    shutil.copyfile(path, out)
    return f"{model}@{revision or 'main'}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deployments")
    parser.add_argument("--model")
    parser.add_argument("--revision")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        model, revision = resolve(args.deployments, args.model, args.revision)
        source = fetch(model, revision, args.out)
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"CRITICAL ERROR: could not get config.json: {e}")
        return 1
    with open(args.out, encoding="utf-8") as f:
        cfg = json.load(f)
    text = cfg.get("text_config", cfg)
    print(f"Wrote {args.out} from {source}: hidden_size "
          f"{text.get('hidden_size')}, head_dim {text.get('head_dim')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

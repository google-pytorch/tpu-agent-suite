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
"""A vLLM deployment as tpu-model-onboarding writes it, and changes to it.

tpu-model-onboarding launches a vLLM server from `deployments.json` (each key
other than a few records is a `vllm serve` flag, `tensor_parallel_size` ->
`--tensor-parallel-size`) and `launch_vllm.sh`, which adds the environment
(`export K=V`), patches (`bash apply_vllm_patch.sh <patch>`) and flags the
manifest does not hold, such as `--attention-backend CUSTOM`.

`baseline()` reads both into the one dict the analyzer and proposer use;
`apply_change()` writes a candidate back as a manifest the same launch path
serves.
"""

import json
import shlex
from typing import Any

# Manifest keys that are records, not `vllm serve` flags.
RECORD_KEYS = frozenset(("model_id", "engine", "port", "parallelism", "name",
                         "notes", "env", "vllm_patches", "checkpoint",
                         "sizing", "history", "reason", "caveats", "source"))


def _is_record(k: str, v: Any) -> bool:
    """A manifest entry that records something rather than sets a flag.

    Onboarding writes notes into the manifest (how it sized the server, what
    it tried); a value with whitespace is prose, never a flag value.
    """
    return k in RECORD_KEYS or (isinstance(v, str) and any(c.isspace()
                                                           for c in v))


# Flags kept as fields of the baseline rather than in `vllm_args`.
FIELD_FLAGS = {
    "--tensor-parallel-size": "tp",
    "--data-parallel-size": "dp",
    "--max-num-seqs": "max_num_seqs",
    "--max-model-len": "max_model_len",
}
_FIELD_KEYS = {v: k for k, v in FIELD_FLAGS.items()}
# Flags that only say where the server listens; never a change to measure.
_PLACEMENT_FLAGS = ("--host", "--port", "--served-model-name")


def flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def key(flag_name: str) -> str:
    return flag_name.lstrip("-").replace("-", "_")


def load(path: str, model: str | None = None) -> dict[str, Any]:
    """The deployment for `model` in a deployments.json (the first if None)."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    deps = doc.get("deployments", [doc]) if isinstance(doc, dict) else doc
    for d in deps:
        if model is None or d.get("model_id") == model:
            return d
    raise ValueError(f"{path} has no deployment for {model}")


def parse_recipe(text: str) -> dict[str, Any]:
    """Env, patches and `vllm serve` arguments of a launch_vllm.sh."""
    env: dict[str, str] = {}
    patches: list[str] = []
    args: list[str] = []
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            words = shlex.split(line, comments=True)
        except ValueError:
            continue
        if not words:
            continue
        if words[0] == "export":
            for w in words[1:]:
                if "=" in w:
                    k, v = w.split("=", 1)
                    env[k] = v
        elif "apply_vllm_patch.sh" in words[min(1, len(words) - 1)]:
            patches += [w for w in words[2:] if not w.startswith("-")][:1]
        elif "serve_vllm.sh" in line or words[:2] == ["vllm", "serve"]:
            i = next(i for i, w in enumerate(words)
                     if w.endswith("serve_vllm.sh") or w == "serve")
            rest = words[i + 1:]
            # The model path comes first unless the line starts with a flag.
            if rest and not rest[0].startswith("-"):
                rest = rest[1:]
            args = rest
    return {"env": env, "patches": patches, "args": args}


def _split_args(args: list[str]) -> list[tuple[str, str | None]]:
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if not a.startswith("--"):
            i += 1
            continue
        if "=" in a:
            f, v = a.split("=", 1)
            out.append((f, v))
            i += 1
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            out.append((a, args[i + 1]))
            i += 2
        else:
            out.append((a, None))
            i += 1
    return out


def baseline(
        dep: dict[str, Any],
        recipe: str | None = None,
        image: str | None = None,
        points: list[tuple[int, int, int]] | None = None) -> dict[str, Any]:
    """The baseline the analyzer and proposer read, from manifest + recipe.

    The recipe's flags win over the manifest's where both set one: it is what
    the server ran.
    """
    flags: dict[str, str | None] = {}
    for k, v in dep.items():
        if _is_record(k, v):
            continue
        if isinstance(v, bool):
            if v:
                flags[flag(k)] = None
            continue
        if isinstance(v, (dict, list)):
            v = json.dumps(v, sort_keys=True)
        flags[flag(k)] = str(v)
    parsed = parse_recipe(recipe) if recipe else {
        "env": {},
        "patches": [],
        "args": []
    }
    for f, v in _split_args(parsed["args"]):
        flags[f] = v
    env = dict(dep.get("env") or {})
    env.update(parsed["env"])
    out: dict[str, Any] = {
        "model_path": dep.get("model_id") or "/model",
        "env": env,
        "patches": list(dep.get("vllm_patches") or parsed["patches"]),
        "image": image,
        "points": [list(p) for p in points or []],
    }
    args: list[str] = []
    for f, v in flags.items():
        if f in FIELD_FLAGS:
            out[FIELD_FLAGS[f]] = int(v)
        elif f not in _PLACEMENT_FLAGS:
            args += [f] if v is None else [f, v]
    out["vllm_args"] = args
    out.setdefault("tp", 1)
    out.setdefault("dp", 1)
    return out


def apply_change(dep: dict[str, Any], change: dict[str, Any],
                 base: dict[str, Any]) -> dict[str, Any]:
    """`dep` with a candidate's change applied, as a new manifest entry.

    `change` holds the candidate's `env`, `vllm_args`, `tp`, `dp`,
    `max_num_seqs`, `max_model_len` and `patches`, each already merged with
    the baseline `base`. Flags the change drops are removed; flags only the
    recipe held (such as `--attention-backend`) are written into the manifest
    so the launch path appends them.
    """
    out = {k: v for k, v in dep.items() if _is_record(k, v)}
    for field, f in _FIELD_KEYS.items():
        if change.get(field) is not None:
            out[key(f)] = int(change[field])
    for f, v in _split_args(change.get("vllm_args") or []):
        out[key(f)] = True if v is None else _number(v)
    env = {
        k: v
        for k, v in (change.get("env") or {}).items()
        if (base.get("env") or {}).get(k) != v
    }
    if env:
        out["env"] = dict(sorted(env.items()))
    patches = [
        p for p in change.get("patches") or []
        if p not in (base.get("patches") or [])
    ]
    if patches:
        out["vllm_patches"] = patches
    return out


def _number(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v

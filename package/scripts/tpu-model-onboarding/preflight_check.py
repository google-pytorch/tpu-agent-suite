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
r"""Pre-flight check for Hugging Face model compatibility with TorchTPU.

This script scans a Hugging Face model repository's metadata, architecture,
quantization config, vocabulary bounds, and any custom Python code imports
to verify its compatibility for direct compilation and deployment on TPU via
the PyTorch XLA compiler.

The Hub-side checks (access, weight format, library tier, quantization, custom
code imports) need only `huggingface_hub`, which the host has; `transformers`
is imported inside the checks that load a config, processor or tokenizer, so
`--hub_only` runs on a host without it. `--resolve` turns a loose name such as
"gpt oss 20b" into a repository id before any check runs.

A `gs://` checkpoint folder is checked without the Hub: access, gating and
library tier do not apply, the format check reads the folder's file names, and
the quantization, custom-code and file-based checks run on a copy of its small
files made by `fetch_weights.py` (no weights).
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import huggingface_hub
import yaml
from absl import app, flags

FLAGS = flags.FLAGS

_MODEL = flags.DEFINE_string("model", None, "HF model ID to check.")

_FORCE = flags.DEFINE_bool("force", False,
                           "Force pass preflight validation checks.")

_ALLOW_GPU_IMPORTS = flags.DEFINE_bool(
    "allow_gpu_imports",
    False,
    "Allow GPU/CUDA-specific imports (e.g. Triton, FlashAttention).",
)

_CUDA_REPORT_JSON = flags.DEFINE_string(
    "cuda_report_json",
    None,
    "Path to save structured JSON report on discovered CUDA kernels for downstream tooling.",
)

_MODE = flags.DEFINE_enum(
    "mode",
    "benchmark",
    ["benchmark", "inference"],
    "Operational workflow mode (benchmark or inference).",
)

_HUB_ONLY = flags.DEFINE_bool(
    "hub_only",
    False,
    "Run only the checks that need huggingface_hub (access, format, library"
    " tier, quantization, custom-code imports), so they run on a host without"
    " transformers.",
)

_RESOLVE = flags.DEFINE_string(
    "resolve",
    None,
    'Loose model name (e.g. "gpt oss 20b") to resolve to a Hub repository id.'
    " Prints a JSON verdict and exits without running any check.",
)

_RESOLVE_LIMIT = flags.DEFINE_integer(
    "resolve_limit", 3, "Ranked candidates to return with --resolve.")


def discover_playbook_mappings() -> dict[str, str]:
    """Scans the YAML frontmatter of tpu-knowledge-base's pages to build the trigger index.

    The playbooks are pages in `data/tpu-knowledge-base/pages/`; a page that
    removes a CUDA-only import names it under `target_imports`. Search order:
    $TPU_AGENT_SUITE_KNOWLEDGE_BASE_HOME/pages (the same override tpu-knowledge-base's own scripts
    honour) first, then a checkout under the working directory, then the
    package this script is installed in. The first candidate directory that
    contains at least one *.md file wins. Finding none is reported on stderr:
    the findings then carry no playbook, and that should not pass unnoticed.
    """
    mappings: dict[str, str] = {}
    candidates: list[Path] = []
    raw_env = os.environ.get("TPU_AGENT_SUITE_KNOWLEDGE_BASE_HOME", "")
    if raw_env:
        candidates.append(Path(raw_env) / "pages")
    # Installed as scripts/tpu-model-onboarding/preflight_check.py, so the package
    # root is two levels up, in an install and in a checkout alike.
    package_dir = Path(__file__).resolve().parents[2]
    candidates.extend([
        Path.cwd() / "package" / "data" / "tpu-knowledge-base" / "pages",
        package_dir / "data" / "tpu-knowledge-base" / "pages",
    ])
    playbooks_dir = next(
        (p for p in candidates if p.is_dir() and any(p.glob("*.md"))),
        None,
    )
    if not playbooks_dir:
        print("WARNING: no playbook directory found, so no finding will name"
              " a playbook. Looked in:\n" + "\n".join(f"  {c}"
                                                      for c in candidates),
              file=sys.stderr)
        return mappings

    for md_file in playbooks_dir.glob("*.md"):
        try:
            md_content = md_file.read_text(encoding="utf-8")
            if md_content.startswith("---"):
                parts = md_content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        for target in frontmatter.get("target_imports", []):
                            rel_path = (md_file.relative_to(Path.cwd())
                                        if md_file.is_relative_to(Path.cwd())
                                        else md_file)
                            mappings[str(target).strip()] = str(rel_path)
        except Exception:  # pylint: disable=broad-exception-caught
            continue
    return mappings


def write_cuda_report(
    report_path: str,
    model_id: str,
    mode: str,
    issues: list[tuple[str, int, str, str]],
    playbook_mappings: dict[str, str] | None = None,
) -> None:
    """Writes a structured JSON report on discovered CUDA kernels for downstream tooling."""
    if playbook_mappings is None:
        playbook_mappings = discover_playbook_mappings()

    kernels_data = []
    for fpath, lno, lib, sym in issues:
        playbook = playbook_mappings.get(lib)
        kernels_data.append({
            "file": fpath,
            "line": max(1, int(lno)),
            "library": lib,
            "symbol": sym,
            "playbook": playbook,
            "playbook_status": "HIT" if playbook else "MISS",
        })

    report = {
        "model_id": model_id,
        "mode": mode,
        "status": "DISCOVERED",
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "kernels": kernels_data,
        "tpu_kernel_development_handoff_ready": bool(kernels_data),
    }

    report_dir = os.path.dirname(os.path.abspath(report_path))
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"  [INFO] Saved CUDA kernels report to {report_path}")


FORBIDDEN_IMPORTS = frozenset((
    "mamba_ssm",
    "triton",
    "flash_attn",
    "apex",
    "deepspeed",
    "xformers",
))

INCOMPATIBLE_LIBRARIES = frozenset((
    "tensorflow",
    "jax",
    "onnx",
    "openvino",
    "coreml",
    "paddlepaddle",
    "tflite",
    "gguf",
    "wav2letter",
    "scikit-learn",
))

TIER_B_LIBRARIES = frozenset((
    "spacy",
    "stanza",
    "flair",
    "allennlp",
    "simpletransformers",
    "bertopic",
    "haystack",
    "ml-agents",
))

# The quantization methods the vLLM TorchTPU plugin serves, which the
# inference workflow runs on, by the name vLLM gives a checkpoint's method.
# The plugin's quantization map implements these alone (awq is TODO there),
# and any other method raises NotImplementedError when the engine starts.
# Read from the plugin in the eval task image (torch 2.13, vLLM 0.29).
TORCHTPU_VLLM_QUANT_METHODS = frozenset(
    ("fp8", "mxfp4", "compressed-tensors", "modelopt_fp4", "deepseek_v4_fp8"))

# Quantization methods the benchmark workflow cannot load: transformers'
# quantizer for each needs CUDA (or another vendor's kernels) and has no TPU
# path. torch_tpu implements no quantization method itself.
GPU_ONLY_QUANT_METHODS = frozenset(("awq", "gptq", "bitsandbytes", "eetq",
                                    "fbgemm_fp8", "fp8", "vptq", "aqlm"))


def classify_library_tier(library_name: str) -> str:
    """The tier of a Hub `library_name`: "None" (incompatible), "B" or "A"."""
    library_name = library_name.lower().strip()
    if library_name in INCOMPATIBLE_LIBRARIES:
        return "None"
    if library_name in TIER_B_LIBRARIES:
        return "B"
    return "A"


def _vllm_quant_method(quant_config: Any, method: str) -> str:
    """The name vLLM gives a checkpoint's quantization method.

    Most keep their quant_method. A ModelOpt checkpoint is named by its
    algorithm: vLLM serves an FP4 one (NVFP4) as modelopt_fp4, whether
    quant_algo sits at the top of the config or under "quantization".
    """
    if not str(method).lower().startswith("modelopt"):
        return method

    def field(obj: Any, key: str) -> Any:
        return obj.get(key) if isinstance(obj, dict) else getattr(
            obj, key, None)

    algo = field(quant_config, "quant_algo") or field(
        field(quant_config, "quantization") or {}, "quant_algo")
    if algo and "FP4" in str(algo).upper():
        return "modelopt_fp4"
    return method


def check_quantization(quant_config: Any,
                       mode: str = "benchmark") -> tuple[bool, str] | None:
    """Applies the quantization rule to a `quantization_config`.

  The rule depends on the workflow. Inference serves through the vLLM TorchTPU
  plugin, which takes only the methods in TORCHTPU_VLLM_QUANT_METHODS.
  Benchmark loads through transformers, which refuses the methods in
  GPU_ONLY_QUANT_METHODS off a GPU. fp8 is the one that differs: the plugin
  serves it, and transformers' FP8 quantizer needs CUDA or XPU.

  Args:
    quant_config: The config's `quantization_config`, as the dict config.json
      holds or the object transformers builds from it.
    mode: "benchmark" or "inference", as --mode.

  Returns:
    None when the model is not quantized. Otherwise a tuple containing:
      - A boolean indicating if the quantization method can run on TPU.
      - A string message describing the status.
  """
    if not quant_config:
        return None
    if isinstance(quant_config, dict):
        method = quant_config.get("quant_method", "unknown")
    else:
        method = getattr(quant_config, "quant_method", "unknown")
    if mode == "inference":
        served = _vllm_quant_method(quant_config, method)
        if served in TORCHTPU_VLLM_QUANT_METHODS:
            return (True, (f"Quantization config found (method: {served});"
                           " the vLLM TorchTPU plugin serves it."))
        return (
            False,
            (f"Incompatible quantization method: {method} (the vLLM TorchTPU"
             f" plugin does not serve it; it serves"
             f" {', '.join(sorted(TORCHTPU_VLLM_QUANT_METHODS))})."),
        )
    if method in GPU_ONLY_QUANT_METHODS:
        return (
            False,
            (f"Incompatible quantization method: {method} (requires GPU CUDA"
             " kernels)."),
        )
    return (
        True,
        f"Quantization config found (method: {method}). Verify TPU support.",
    )


def check_access(model_id: str, ) -> tuple[bool, str, Any | None]:
    """Checks if model is accessible (gated/private check).

  This function relies on the locally saved HuggingFace token for authentication
  which is the recommended method for authentication
  (see https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

  Args:
    model_id: The Hugging Face model ID.

  Returns:
    A tuple containing:
      - A boolean indicating if the model is accessible.
      - A string message describing the status or error.
      - The HfApi ModelInfo object if accessible, otherwise None.
  """
    api = huggingface_hub.HfApi()
    try:
        info = api.model_info(model_id)
        gated = getattr(info, "gated", False)
        if gated:
            return True, f"Gated model (gated type: {gated}), token has access.", info
        return True, "Public model.", info
    except huggingface_hub.utils.GatedRepoError as e:
        return (
            False,
            ("Access denied (Gated repository). Please request access and"
             f" provide a valid token. Error: {e!r}"),
            None,
        )
    except huggingface_hub.utils.RepositoryNotFoundError as e:
        return False, f"Repository not found. Error: {e!r}", None
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"Error checking model info: {e!r}", None


def check_formats(
    model_info: huggingface_hub.hf_api.ModelInfo, ) -> tuple[bool, str]:
    """Checks if the model is in an incompatible format (e.g., GGUF, TFLite).

  Args:
    model_info: The HfApi ModelInfo object.

  Returns:
    A tuple containing:
      - A boolean indicating if the format is compatible.
      - A string message describing the status.
  """
    siblings = getattr(model_info, "siblings", [])
    if not siblings:
        return True, "No file metadata found, proceeding."

    has_gguf = False
    has_tflite = False
    has_pytorch = False

    for s in siblings:
        f = s.rfilename.lower()
        if f.endswith(".gguf"):
            has_gguf = True
        elif f.endswith(".tflite"):
            has_tflite = True
        elif f.endswith(
            (".safetensors", ".bin", ".pth")) or "pytorch_model" in f:
            has_pytorch = True

    if (has_gguf or has_tflite) and not has_pytorch:
        formats = []
        if has_gguf:
            formats.append("GGUF")
        if has_tflite:
            formats.append("TFLite")
        return (
            False,
            (f"Incompatible format: {', '.join(formats)} (no PyTorch/Safetensors"
             " weights found)."),
        )

    return (
        True,
        "Format is likely compatible (PyTorch/Safetensors weights present).",
    )


def check_listed_formats(names: Sequence[str]) -> tuple[bool, str]:
    """Checks whether a checkpoint's file names include a compatible format."""
    has_gguf = False
    has_tflite = False
    has_pytorch = False
    for name in names:
        f = name.rsplit("/", 1)[-1].lower()
        if f.endswith(".gguf"):
            has_gguf = True
        elif f.endswith(".tflite"):
            has_tflite = True
        elif (f.endswith((".safetensors", ".bin", ".pth", ".pt"))
              or "pytorch_model" in f):
            has_pytorch = True

    if (has_gguf or has_tflite) and not has_pytorch:
        formats = []
        if has_gguf:
            formats.append("GGUF")
        if has_tflite:
            formats.append("TFLite")
        return (
            False,
            f"Incompatible format: {', '.join(formats)} (no PyTorch/Safetensors"
            " weights found).",
        )

    return (
        True,
        "Format is likely compatible (PyTorch/Safetensors weights present).",
    )


def check_local_formats(local_dir: Path | str) -> tuple[bool, str]:
    """Checks if a local model directory contains compatible formats."""
    return check_listed_formats(
        [p.name for p in Path(local_dir).rglob("*") if p.is_file()])


GCS_SCHEME = "gs://"


def load_fetch_weights() -> Any:
    """fetch_weights.py, which reads a gs:// checkpoint.

    It sits beside this script in the package. A job started with
    `compute run --code` ships this script alone, so a gs:// model needs
    `--input <fetch_weights.py>:fetch_weights.py` there.
    """
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        # pylint: disable-next=import-outside-toplevel
        import fetch_weights
    except ImportError as e:
        raise ImportError(
            "a gs:// model needs fetch_weights.py beside this script: pass "
            "--input <fetch_weights.py>:fetch_weights.py to compute run"
        ) from e
    return fetch_weights


def stage_gcs_metadata(url: str) -> tuple[bool, str, str | None]:
    """Format check and metadata copy for a gs:// checkpoint folder.

    The Hub is not asked: the format check reads the folder's file names,
    and the config, tokenizer and custom code are copied to local disk
    (no weights) for the file-based checks.

    Returns:
      (format compatible, message, local copy of the small files or None).
    """
    fetch_weights = load_fetch_weights()
    names = fetch_weights.gcs_listing(url)
    ok, msg = check_listed_formats(list(names))
    if not ok:
        return ok, msg, None
    return ok, msg, fetch_weights.fetch_gcs(url, small_only=True)


# Built-in / first-party torch.ops namespaces that are TPU-lowerable and must not
# be flagged as custom CUDA kernels.
BUILTIN_TORCH_OPS_NAMESPACES = frozenset({
    "aten",
    "prim",
    "prims",
    "quantized",
    "quantized_decomposed",
    "_quantized",
    "c10d",
    "_c10d_functional",
    "c10d_functional",
    "_dtensor",
    "higher_order",
    "profiler",
    "torchvision",
    "torchaudio",
    "torchtext",
    "image",
    "xla",
    "tpu",
})


def is_custom_cuda_import(lib: str, full_module: str) -> tuple[bool, str]:
    """Determines if an import references forbidden or custom CUDA/GPU code."""
    if full_module.startswith("torch.cuda"):
        return False, ""
    if full_module.startswith("torch.utils.cpp_extension"):
        return True, "cuda_extension"
    lower_lib = lib.lower()
    if lower_lib in FORBIDDEN_IMPORTS:
        return True, lib
    if (lower_lib == "cuda" or lower_lib.startswith("cuda_")
            or lower_lib.endswith("_cuda") or "_cuda_" in lower_lib):
        return True, lib
    return False, ""


def check_for_incompatible_imports(
        directory: str) -> list[tuple[str, int, str, str]]:
    """Scans Python and CUDA files for forbidden or custom GPU imports/kernels.

    Args:
      directory: The directory to scan recursively.

    Returns:
      A list of tuples (file, line, library, symbol) for each issue found.
    """
    issues = []
    seen = set()

    def add_issue(rel_path: str, line: int, library: str, symbol: str) -> None:
        key = (rel_path, line, library, symbol)
        if key not in seen:
            seen.add(key)
            issues.append(key)

    # 1. Check for standalone CUDA source files (.cu, .cuh)
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith((".cu", ".cuh")):
                path = os.path.join(root, file)
                rel = os.path.relpath(path, directory)
                add_issue(rel, 1, "custom_cuda", Path(file).name)

    # 2. Check Python files for forbidden or custom CUDA imports and extensions
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                rel = os.path.relpath(path, directory)
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=path)
                        for node in ast.walk(tree):
                            lno = getattr(node, "lineno", 1)
                            if isinstance(node, ast.Import):
                                for name in node.names:
                                    lib = name.name.split(".")[0]
                                    matched, reported = is_custom_cuda_import(
                                        lib, name.name)
                                    if matched:
                                        add_issue(rel, lno, reported,
                                                  name.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    lib = node.module.split(".")[0]
                                    matched, reported = is_custom_cuda_import(
                                        lib, node.module)
                                    if matched:
                                        symbols = [
                                            alias.name for alias in node.names
                                        ] if node.names else [node.module]
                                        for sym in symbols:
                                            add_issue(rel, lno, reported, sym)
                            elif isinstance(node, ast.Attribute):
                                if (isinstance(node.value, ast.Attribute) and
                                        isinstance(node.value.value, ast.Name)
                                        and node.value.value.id == "torch"
                                        and node.value.attr == "ops"
                                        and node.attr
                                        not in BUILTIN_TORCH_OPS_NAMESPACES):
                                    add_issue(rel, lno, "custom_cuda",
                                              f"torch.ops.{node.attr}")
                    except SyntaxError:
                        # Code might have syntax errors (e.g. template files), ignore.
                        pass
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        print(f"Warning: Failed to parse {path}: {e!r}",
                              file=sys.stderr)
    return issues


def validate_config(directory: str,
                    mode: str = "benchmark") -> tuple[bool, str, Any | None]:
    """Validates root config.json presence, format, and quantization.

  Args:
    directory: The local directory containing the config.json file.
    mode: "benchmark" or "inference"; see check_quantization.

  Returns:
    A tuple containing:
      - A boolean indicating if the config is valid and compatible.
      - A string message describing the status.
      - The loaded AutoConfig object, or None if loading failed.
  """
    has_config_json = os.path.exists(os.path.join(directory, "config.json"))
    has_model_index = os.path.exists(
        os.path.join(directory, "model_index.json"))
    has_config_yaml = os.path.exists(os.path.join(directory, "config.yaml"))
    has_hyperparams_yaml = os.path.exists(
        os.path.join(directory, "hyperparams.yaml"))

    if not any([
            has_config_json, has_model_index, has_config_yaml,
            has_hyperparams_yaml
    ]):
        return (
            False,
            ("Missing config configuration file (config.json, model_index.json,"
             " config.yaml, or hyperparams.yaml)."),
            None,
        )

    if has_model_index:
        try:
            with open(os.path.join(directory, "model_index.json"),
                      "r",
                      encoding="utf-8") as f:
                config = json.load(f)
            class_name = config.get("_class_name", "Unknown")
            for key, value in config.items():
                if isinstance(value, list) and len(value) == 2:
                    component_dir = os.path.join(directory, key)
                    if value[0] is not None and not os.path.exists(
                            component_dir):
                        return (
                            False,
                            f"Diffusers  directory '{key}' (class: {value[1]}) is missing.",
                            config,
                        )
            return True, f"Diffusers pipeline ({class_name}) is valid.", config
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Failed to parse model_index.json: {e!r}", None

    if (has_config_yaml or has_hyperparams_yaml) and not has_config_json:
        config_type = "YAML Config (pyannote/espnet/speechbrain)"
        filename = "config.yaml" if has_config_yaml else "hyperparams.yaml"
        try:
            with open(os.path.join(directory, filename), "r",
                      encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                return False, f"{filename} is empty.", None
            return (
                True,
                f"{config_type} ({filename}) is valid.",
                {
                    "model_type": "yaml_custom",
                    "filename": filename
                },
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Failed to read {filename}: {e!r}", None

    if has_config_json:
        try:
            with open(os.path.join(directory, "config.json"),
                      "r",
                      encoding="utf-8") as f:
                config_dict = json.load(f)
            if "architecture" in config_dict and "pretrained_cfg" in config_dict:
                return (
                    True,
                    f"Timm model config ({config_dict['architecture']}) is valid.",
                    config_dict,
                )
            if "speechbrain_interface" in config_dict:
                sb_interface = config_dict["speechbrain_interface"]
                return (
                    True,
                    f"Speechbrain model config ({sb_interface}) is valid.",
                    config_dict,
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            return False, f"Failed to parse config.json as JSON: {e!r}", None

    # pylint: disable-next=import-outside-toplevel
    from transformers import AutoConfig

    try:
        # Load using transformers AutoConfig to verify architecture mapping
        config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    except FileNotFoundError:
        return False, "Missing root-level config.json file.", None
    except Exception as e:  # pylint: disable=broad-exception-caught
        return (
            False,
            (f"Hugging Face AutoConfig failed to load: {e!r}. Model has"
             " unsupported architecture or missing mapping metadata."),
            None,
        )

    # Check quantization
    quantization = check_quantization(
        getattr(config, "quantization_config", None), mode)
    if quantization:
        quant_ok, quant_msg = quantization
        return quant_ok, quant_msg, config

    # Flag specialized models requiring individual attention (generic check)
    model_type = getattr(config, "model_type", None)
    architectures = getattr(config, "architectures", []) or []

    specialized_keywords = {
        "Folding": "Protein folding / biological structure prediction model",
        "StructurePrediction": "Biological structure prediction model",
        "Layout": "Document layout / OCR analysis model",
        "Ocr": "Optical Character Recognition (OCR) model",
        "OCR": "Optical Character Recognition (OCR) model",
        "TableTransformer": "Table structure extraction model",
        "Graph": "Graph neural network model",
        "Gnn": "Graph neural network model",
        "GNN": "Graph neural network model",
    }

    is_specialized = False
    specialized_reason = ""

    # 1. Check architectures with case-sensitive CamelCase substring checks
    for arch in architectures:
        for kw, msg in specialized_keywords.items():
            if kw in arch:
                is_specialized = True
                specialized_reason = msg
                break
        if is_specialized:
            break

    # 2. Check model_type with exact fallback checks
    if not is_specialized and model_type:
        model_type_lower = model_type.lower()
        if any(x in model_type_lower
               for x in ["layoutlm", "esmfold", "clipseg"]):
            is_specialized = True
            specialized_reason = f"Specialized {model_type} model"

    if is_specialized:
        print(
            "\n[PREFLIGHT WARNING] Model is classified as a"
            f" {specialized_reason}.\nThis model class requires custom non-standard"
            " input-output structures\nand is unsupported in general automated"
            " loops. It will require\nindividual attention and custom wrappers to"
            " deploy successfully on TPU.\n",
            file=sys.stderr,
        )

    return True, "config.json is valid and loadable.", config


def validate_processor(directory: str, config: Any) -> tuple[bool, str]:
    """Attempts to load processor/tokenizer to catch missing configuration keys.

  Args:
    directory: The local directory containing configuration files.
    config: The config dictionary or AutoConfig object loaded for the model.

  Returns:
    A tuple containing:
      - A boolean indicating if the tokenizer/processor is compatible.
      - A string message describing the status or error/warning.
  """
    # pylint: disable-next=import-outside-toplevel
    from transformers import (AutoFeatureExtractor, AutoImageProcessor,
                              AutoProcessor, AutoTokenizer)

    has_tokenizer_config = os.path.exists(
        os.path.join(directory, "tokenizer_config.json"))
    has_preprocessor_config = os.path.exists(
        os.path.join(directory, "preprocessor_config.json"))
    has_model_index = os.path.exists(
        os.path.join(directory, "model_index.json"))

    if has_model_index:
        return (
            True,
            "Diffusers pipeline structure verified, bypassing root processor" +
            " check.",
        )

    # Handle Dict configs (custom backends)
    if isinstance(config, dict):
        if ("speechbrain_interface" in config
                or config.get("model_type") == "yaml_custom"):
            return (
                True,
                "Custom library model (speechbrain/espnet/pyannote), bypassing"
                + " processor check.",
            )
        if "architecture" in config and "pretrained_cfg" in config:
            return True, "Timm model, bypassing processor check."
        if "_class_name" in config:
            return True, "Diffusers pipeline, bypassing processor check."

    if has_preprocessor_config and not has_tokenizer_config:
        try:
            _ = AutoImageProcessor.from_pretrained(directory,
                                                   trust_remote_code=True)
            return (
                True,
                "Image processor instantiation succeeded (no tokenizer needed).",
            )
        except Exception as e_img:  # pylint: disable=broad-exception-caught
            try:
                _ = AutoFeatureExtractor.from_pretrained(
                    directory, trust_remote_code=True)
                return (
                    True,
                    "Feature extractor instantiation succeeded (no tokenizer needed).",
                )
            except Exception as e_feat:  # pylint: disable=broad-exception-caught
                return (
                    False,
                    (f"Failed to load image processor ({e_img!r}) and feature"
                     f" extractor ({e_feat!r}) for vision/audio model."),
                )

    model_type = (config.get("model_type") if isinstance(config, dict) else
                  getattr(config, "model_type", None))
    architectures = (config.get("architectures", []) if isinstance(
        config, dict) else getattr(config, "architectures", []) or [])

    # Determine if model requires processor
    specialized_arches = {"Whisper", "CTC", "CLIP", "ConditionalGeneration"}
    has_specialized_arch = False
    for arch in architectures:
        if any(x in arch for x in specialized_arches):
            has_specialized_arch = True
            break

    needs_processor = (model_type in ["whisper", "wav2vec2", "clip"]
                       or has_specialized_arch)

    if needs_processor:
        try:
            # Try AutoProcessor first
            _ = AutoProcessor.from_pretrained(directory,
                                              trust_remote_code=True)
            return True, "Processor instantiation succeeded."
        except Exception as e_proc:  # pylint: disable=broad-except
            # If it fails, check tokenizer/feature extractor fallback
            try:
                _ = AutoTokenizer.from_pretrained(directory,
                                                  trust_remote_code=True)
                if model_type and ("wav2vec2" in model_type
                                   or "whisper" in model_type):
                    _ = AutoFeatureExtractor.from_pretrained(
                        directory, trust_remote_code=True)
                return (
                    True,
                    ("Processor failed, but separate tokenizers/extractors"
                     f" succeeded. Warning: {e_proc!r}"),
                )
            except Exception as e_fallback:  # pylint: disable=broad-except
                # Try vision image processor / feature extractor fallback
                try:
                    _ = AutoImageProcessor.from_pretrained(
                        directory, trust_remote_code=True)
                    return (
                        True,
                        ("Processor failed, but AutoImageProcessor succeeded. Warning:"
                         f" {e_proc!r}"),
                    )
                except Exception:  # pylint: disable=broad-except
                    try:
                        _ = AutoFeatureExtractor.from_pretrained(
                            directory, trust_remote_code=True)
                        return (
                            True,
                            ("Processor failed, but AutoFeatureExtractor succeeded."
                             f" Warning: {e_proc!r}"),
                        )
                    except Exception as e_feat:  # pylint: disable=broad-except
                        return (
                            False,
                            (f"Failed to instantiate processor: {e_proc!r}. Tokenizer"
                             f" fallback failed: {e_fallback!r}. FeatureExtractor"
                             f" fallback failed: {e_feat!r}"),
                        )

    # Check if it has an image preprocessor config
    has_preprocessor = os.path.exists(
        os.path.join(directory, "preprocessor_config.json"))

    if has_preprocessor:
        try:
            _ = AutoImageProcessor.from_pretrained(directory,
                                                   trust_remote_code=True)
            return (
                True,
                "Image processor instantiation succeeded (no tokenizer needed).",
            )
        except Exception as e_img:  # pylint: disable=broad-except
            print(f"DEBUG: Image processor load failed: {e_img!r}")
            pass

    # For standard models, just verify tokenizer loads
    try:
        _ = AutoTokenizer.from_pretrained(directory, trust_remote_code=True)
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Failed to instantiate tokenizer: {e!r}"

    return True, "Tokenizer/Processor check passed."


def check_vocab_bounds(directory: str, config: Any) -> tuple[bool, str]:
    """Checks if tokenizer vocabulary exceeds base config embedding layer dimension.

  Args:
    directory: The directory containing tokenizer configuration.
    config: The loaded model configuration object.

  Returns:
    A tuple containing:
      - A boolean indicating if vocab bounds are compatible.
      - A string message describing the status.
  """
    # pylint: disable-next=import-outside-toplevel
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(directory,
                                                  trust_remote_code=True)
        t_vocab = len(tokenizer)
        c_vocab = getattr(config, "vocab_size", None)
        if c_vocab and t_vocab > c_vocab:
            return (
                False,
                (f"[FLAGGED] Tokenizer vocab size ({t_vocab}) exceeds"
                 f" configuration embedding vocab size ({c_vocab}). CPU run will"
                 " crash with IndexError."),
            )
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    return True, "Vocab bounds check passed."


HUB_URL = "https://huggingface.co"

# Organisations that publish the reference checkpoints for their families.
# Ranked above community re-uploads, which are usually quantized or finetuned.
OFFICIAL_ORGS = frozenset((
    "openai",
    "meta-llama",
    "qwen",
    "mistralai",
    "google",
    "microsoft",
    "deepseek-ai",
    "nvidia",
    "ibm-granite",
    "allenai",
    "tiiuae",
    "01-ai",
    "stabilityai",
    "black-forest-labs",
    "facebook",
    "huggingfacetb",
    "moonshotai",
    "zai-org",
    "thudm",
    "eleutherai",
    "baai",
    "coherelabs",
    "internlm",
    "snowflake",
    "apple",
    "bigscience",
    # Where the Hub keeps the checkpoints it once published under no org:
    # gpt2 is openai-community/gpt2, bert-base-uncased google-bert/..., and
    # t5-small and roberta-base google-t5/... and FacebookAI/....
    "openai-community",
    "google-bert",
    "google-t5",
    "facebookai",
))

# Name fragments that mark a variant rather than the reference checkpoint.
VARIANT_MARKERS = ("instruct", "chat", "-it", "base", "coder", "math",
                   "vision", "-vl", "thinking", "reasoning")
QUANTIZED_MARKERS = ("gguf", "awq", "gptq", "fp8", "int4", "int8", "4bit",
                     "8bit", "bnb", "exl2", "mlx", "onnx")

# A repository within this fraction of the size a name states is that size:
# "20b" matches a 21B-parameter checkpoint, not a 120B one.
SIZE_TOLERANCE = 0.35

# A size word: "20b", "350m", or a mixture of experts' "8x7b".
_SIZE_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:\.\d+)?)\s*([bm])\b",
                      re.IGNORECASE)

# A Hub page URL; the repository id is the first two path segments after it.
_HUB_URL_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", re.IGNORECASE)


def _size_in_params(text: str) -> float | None:
    """The parameter count a name like "20b", "350m" or "8x7b" states, if any."""
    match = _SIZE_RE.search(text.replace("_", " ").replace("-", " "))
    if not match:
        return None
    experts = int(match.group(1)) if match.group(1) else 1
    scale = 1e9 if match.group(3).lower() == "b" else 1e6
    return experts * float(match.group(2)) * scale


def _repo_id_from_url(query: str) -> str:
    """The repository id a Hub URL points at, or `query` if it is not one.

    "https://huggingface.co/openai/gpt-oss-20b/tree/main" -> "openai/gpt-oss-20b".
    """
    path = _HUB_URL_RE.sub("", query, count=1)
    if path == query:
        return query
    segments = [s for s in re.split(r"[?#]", path)[0].split("/") if s]
    return "/".join(segments[:2])


def _normalise(name: str) -> str:
    """Lower-case, hyphen-joined search text: "GPT OSS 20B" -> "gpt-oss-20b"."""
    return re.sub(r"[\s_]+", "-", name.strip().lower())


# A name's tokens: a version marker ("v2", "v0.1"), a run of letters, or a
# number with its dotted parts ("3.1"). "Qwen2.5-7B" -> qwen, 2.5, 7, b.
_TOKEN_RE = re.compile(r"v\d+(?:\.\d+)*|[a-z]+|\d+(?:\.\d+)*")


def _tokens(name: str) -> list[str]:
    """The tokens of a name; see _TOKEN_RE.

    An underscore between digits is a dot, as in "llama-3_2-1b".
    """
    return _TOKEN_RE.findall(re.sub(r"(?<=\d)_(?=\d)", ".", name.lower()))


def _token_in(word: str, tokens: list[str]) -> bool:
    """Whether a query token is one of a name's tokens.

    A bare number also matches a version it begins: "llama 3" asks for the
    family, so 3 matches 3.1 and 3.2. It does not match a number it is only
    part of: 2 is not 3.2 or v2, and 1 is not 21.
    """
    if word[0].isdigit():
        return any(t == word or t.startswith(word + ".") for t in tokens)
    return word in tokens


def _without_size(query: str) -> str:
    """The query with its size word dropped: "llama 3 8b" -> "llama-3"."""
    return "-".join(w for w in _normalise(query).split("-")
                    if w and not _SIZE_RE.fullmatch(w))


def _search_terms(query: str) -> list[str]:
    """The query, then looser forms, for a Hub search that matches substrings.

    "llama 3 8b" matches nothing as `llama-3-8b` (the repository is
    Meta-Llama-3-8B), so the name without its size, then the family word alone,
    are tried next.
    """
    full = _normalise(query)
    words = [w for w in full.split("-") if w]
    terms = [full]
    if len(words) > 1:
        terms.extend([_without_size(query), words[0]])
    return list(dict.fromkeys(t for t in terms if t))


def _name_matches(repo_id: str, query: str) -> bool:
    """Whether every token of the query is a token of the repository name.

    Tokens, not substrings: as substrings, "1b" is in "21b" and the 2 of
    "llama 2" is in "Llama-3.2" and "-v2".
    """
    tokens = _tokens(repo_id.split("/")[-1])
    return all(_token_in(word, tokens) for word in _tokens(query))


def _size_ok(params: float | None, wanted: float | None) -> bool:
    """Whether a parameter count is the size asked for (or either is unknown)."""
    if wanted is None or not params:
        return True
    return abs(params - wanted) / wanted < SIZE_TOLERANCE


def summarise_candidate(info: Any) -> dict[str, Any]:
    """The fields of a Hub `ModelInfo` worth ranking on and showing the user."""
    repo_id = info.id
    lowered = repo_id.lower()
    safetensors = getattr(info, "safetensors", None)
    return {
        "repo_id":
        repo_id,
        "url":
        f"{HUB_URL}/{repo_id}",
        "official":
        repo_id.split("/")[0].lower() in OFFICIAL_ORGS,
        "downloads":
        getattr(info, "downloads", None) or 0,
        "params":
        getattr(safetensors, "total", None),
        "pipeline_tag":
        getattr(info, "pipeline_tag", None),
        "gated":
        getattr(info, "gated", None) or False,
        "variant":
        next((m.strip("-") for m in VARIANT_MARKERS if m in lowered), None),
        "quantized":
        any(m in lowered for m in QUANTIZED_MARKERS),
    }


def rank_candidates(candidates: list[dict[str, Any]],
                    wanted_params: float | None) -> list[dict[str, Any]]:
    """Official first, unquantized first, size match first, then downloads."""

    def key(c: dict[str, Any]) -> tuple[bool, bool, bool, int]:
        size_ok = _size_ok(c["params"], wanted_params)
        return (not c["official"], c["quantized"], not size_ok,
                -c["downloads"])

    return sorted(candidates, key=key)


def _resolution(query: str, verdict: str, reason: str,
                candidates: list[dict[str, Any]],
                chosen: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "query": query,
        "verdict": verdict,
        "repo_id": chosen["repo_id"] if chosen else None,
        "url": chosen["url"] if chosen else None,
        "reason": reason,
        "candidates": candidates,
    }


def resolve_model_name(query: str, limit: int = 3) -> dict[str, Any]:
    """Resolves a loose model name to a Hub repository id, with a verdict.

  Candidates are ranked, not chosen: choosing between a base and an instruct
  model, or between versions, is the user's decision, so a repository is chosen
  only when there is no decision to make. Hub errors propagate to the caller.

  Args:
    query: A repository id, a Hub URL, or a loose name, e.g. "gpt oss 20b" or
      "openai/gpt oss 20b".
    limit: How many ranked candidates to return.

  Returns:
    A dict with `verdict` -- `exact` (the id exists), `single_match` (the name
    settles it), `ambiguous` (the user has to pick; the first candidate is the
    suggested default) or `not_found` -- plus `repo_id` and `url` when a
    repository was chosen, a `reason`, and the ranked `candidates`.
  """
    api = huggingface_hub.HfApi()
    asked = query.strip()
    query = _repo_id_from_url(asked)

    # An id that exists is the user's exact choice; nothing to rank. That
    # includes ids with no org, such as gpt2, which the Hub redirects to where
    # it keeps them now (openai-community/gpt2). A name with spaces cannot be
    # an id, so it goes straight to search.
    if query and not any(ch.isspace() for ch in query):
        try:
            chosen = summarise_candidate(api.model_info(query))
            reason = ("repository exists under the id given"
                      if chosen["repo_id"] == query else
                      f"the id given is now {chosen['repo_id']} on the Hub")
            return _resolution(asked, "exact", reason, [chosen], chosen)
        except huggingface_hub.utils.GatedRepoError:
            chosen = {
                "repo_id": query,
                "url": f"{HUB_URL}/{query}",
                "gated": True
            }
            return _resolution(asked, "exact",
                               "gated repository exists under the id given",
                               [chosen], chosen)
        except (huggingface_hub.utils.RepositoryNotFoundError,
                huggingface_hub.utils.HFValidationError):
            # Not an id after all: search for the name, within the org if it
            # has repositories that match.
            pass

    org, _, name = query.rpartition("/")
    wanted = _size_in_params(name)
    hits: dict[str, dict[str, Any]] = {}
    for term in _search_terms(name):
        for info in api.list_models(
                search=term,
                sort="downloads",
                limit=50,
                expand=["downloads", "pipeline_tag", "gated", "safetensors"],
        ):
            hit = summarise_candidate(info)
            if (_name_matches(hit["repo_id"], name)
                    or _name_matches(hit["repo_id"], _without_size(name))):
                hits.setdefault(hit["repo_id"], hit)
        if hits:
            break
    in_org = [
        c for c in hits.values()
        if org and c["repo_id"].split("/")[0].lower() == org.lower()
    ]

    ranked = rank_candidates(in_org or list(hits.values()), wanted)
    candidates = ranked[:limit]
    if not ranked:
        return _resolution(asked, "not_found",
                           "no repository matches the name", [], None)

    # The user typed the repository's name: "gpt oss 20b" is openai/gpt-oss-20b
    # even though gpt-oss-safeguard-20b also matches every word of it. The
    # near misses stay in `candidates` so the plan can still show them.
    named = [
        c for c in ranked if c["official"] and not c["quantized"]
        and _tokens(c["repo_id"].split("/")[-1]) == _tokens(name)
    ]
    if len(named) == 1:
        return _resolution(asked, "single_match",
                           "an official repository is named exactly as asked",
                           candidates, named[0])

    # A choice exists when another official, unquantized repository of the
    # size asked for competes with the top one: base vs instruct, or two
    # versions.
    top = ranked[0]
    if not top["official"] or top["quantized"]:
        return _resolution(asked, "ambiguous",
                           "no official unquantized repository matches",
                           candidates, None)
    # Choosing alone needs proof the top one is the size asked for: an unknown
    # size counts as a rival below, but never as the answer.
    if wanted and not (top["params"] and _size_ok(top["params"], wanted)):
        return _resolution(
            asked, "ambiguous",
            "no official repository is known to be the size asked for",
            candidates, None)
    rivals = [
        c for c in ranked[1:] if c["official"] and not c["quantized"]
        and _size_ok(c["params"], wanted)
    ]
    if rivals:
        return _resolution(asked, "ambiguous",
                           "several official repositories match", candidates,
                           None)
    return _resolution(asked, "single_match",
                       "one official, unquantized repository matches",
                       candidates, top)


# Modules whose TransportError is how huggingface_hub reports a Hub it never
# reached: httpx from huggingface_hub 1.0, httpx2 from 2.0.
_HTTP_CLIENTS = ("httpx", "httpx2")


def _unreachable_errors() -> tuple[type[BaseException], ...]:
    """The exceptions that mean the Hub was never reached.

    OSError covers sockets, OfflineModeIsEnabled and requests' transport errors
    (huggingface_hub < 1.0). The transport errors of httpx (huggingface_hub
    1.x) and httpx2 (2.x) are not OSErrors, so each is added when installed.
    """
    errors: list[type[BaseException]] = [OSError]
    for client in _HTTP_CLIENTS:
        try:
            errors.append(importlib.import_module(client).TransportError)
        except ImportError:
            pass
    return tuple(errors)


def resolve_report(query: str, limit: int = 3) -> dict[str, Any]:
    """`resolve_model_name`, with Hub failures turned into a verdict.

  An HTTP error from the Hub (429, 5xx) is `hub_error` with its `http_status`:
  the Hub answered, so retrying later can succeed. Only a Hub that was never
  reached is `offline`. HfHubHTTPError is itself an OSError (through requests
  before huggingface_hub 1.0, and by its own bases from 1.0), so it must be
  caught first or every 429 would read as offline.
  """
    try:
        return resolve_model_name(query, limit)
    except huggingface_hub.utils.HfHubHTTPError as e:
        response = getattr(e, "response", None)
        status = getattr(response, "status_code", None)
        report = _resolution(query, "hub_error",
                             f"Hugging Face Hub returned HTTP {status}: {e}",
                             [], None)
        report["http_status"] = status
        return report
    except _unreachable_errors() as e:
        return _resolution(query, "offline",
                           f"Hugging Face Hub unreachable: {e!r}", [], None)


def transformers_available() -> bool:
    """Whether transformers imports here; the task image has it, a host may not."""
    try:
        # pylint: disable-next=import-outside-toplevel,unused-import
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def read_quantization_config(directory: str) -> Any:
    """The `quantization_config` in a model's config.json, if it has one.

    Multimodal configs can nest it under `text_config`.
    """
    try:
        with open(os.path.join(directory, "config.json"),
                  "r",
                  encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    text_config = config.get("text_config")
    return config.get("quantization_config") or (text_config.get(
        "quantization_config") if isinstance(text_config, dict) else None)


def handle_failure(msg: str) -> None:
    """Handles a preflight check validation failure.

  Args:
    msg: The error message to print and exit with.

  Raises:
    SystemExit: If the --force flag is not set.
  """
    if _FORCE.value:
        print(f"  [FORCE BYPASS] {msg}")
    else:
        print(f"  [ERROR] {msg}", file=sys.stderr)
        sys.exit(1)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    if _RESOLVE.value:
        if _RESOLVE.value.startswith(GCS_SCHEME):
            # A checkpoint folder is the model as given; nothing to search.
            report = _resolution(_RESOLVE.value, "exact",
                                 "a gs:// checkpoint folder is used as given",
                                 [], {
                                     "repo_id": _RESOLVE.value,
                                     "url": _RESOLVE.value
                                 })
        else:
            report = resolve_report(_RESOLVE.value, _RESOLVE_LIMIT.value)
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    model_id = _MODEL.value
    if not model_id:
        raise app.UsageError("Missing required flag --model.")

    print(f"Checking model: {model_id}")
    cached_dir = model_id
    if (not os.path.isdir(model_id) and not model_id.startswith(GCS_SCHEME)
            and os.environ.get("HF_HUB_OFFLINE")):
        with contextlib.suppress(Exception):
            cached_dir = huggingface_hub.snapshot_download(
                repo_id=model_id, local_files_only=True)

    if model_id.startswith(GCS_SCHEME):
        print("  Access Check: gs:// checkpoint folder; the Hub is not asked"
              " (access, gating and library tier do not apply).")
        local_dir = None
        try:
            format_compatible, format_msg, local_dir = stage_gcs_metadata(
                model_id)
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            handle_failure(f"Could not read {model_id}: {e}")
            format_compatible, format_msg = True, "not checked (--force)"
        print(f"  Format Check: {format_msg}")
        if not format_compatible:
            handle_failure(f"Format check failed: {format_msg}")
        if local_dir is None:
            print("\nPRE-FLIGHT CHECK PASSED: Checks bypassed via --force.")
            return
        print(f"  Metadata copied (config, tokenizer, code; no weights) to"
              f" {local_dir}")
    elif os.path.isdir(cached_dir):
        print("  Access Check: Local checkpoint directory verified.")
        format_compatible, format_msg = check_local_formats(cached_dir)
        print(f"  Format Check: {format_msg}")
        if not format_compatible:
            handle_failure(f"Format check failed: {format_msg}")
        local_dir = str(Path(cached_dir).resolve())
    else:
        # Step 1: Check access
        accessible, msg, model_info = check_access(model_id)
        print(f"  Access Check: {msg}")
        if not accessible:
            handle_failure(f"Access check failed: {msg}")

        # Step 1.2: Check format compatibility
        if model_info:
            format_compatible, format_msg = check_formats(model_info)
            print(f"  Format Check: {format_msg}")
            if not format_compatible:
                handle_failure(f"Format check failed: {format_msg}")

        # Step 1.3: Check Library Tier Compatibility
        if model_info:
            library_name = getattr(model_info, "library_name", None)
            if library_name:
                library_name = library_name.lower().strip()
                print(f"  Detected HF Library: {library_name}")
                tier = classify_library_tier(library_name)
                if tier == "None":
                    handle_failure(
                        f"Library '{library_name}' is in Tier None (Incompatible"
                        " Format/Runtime).")
                elif tier == "B":
                    print(
                        f"  [INFO] Library '{library_name}' is in Tier B (Medium/Low"
                        " Benefit). It is compatible but compiler profile overlaps with"
                        " standard encoder models.")

        # Step 2: Download metadata only
        print("  Downloading metadata (config & code)...")
        local_dir = None
        try:
            local_dir = huggingface_hub.snapshot_download(
                repo_id=model_id,
                allow_patterns=[
                    "**/*.json",
                    "*.json",
                    "**/*.py",
                    "*.py",
                    "**/*.yaml",
                    "*.yaml",
                    "**/*.jinja",
                    "*.jinja",
                    "**/*.txt",
                    "*.txt",
                    "**/*.model",
                    "*.model",
                    "**/*.cu",
                    "*.cu",
                    "**/*.cuh",
                    "*.cuh",
                    "**/*.cpp",
                    "*.cpp",
                ],
                ignore_patterns=[
                    "**/*.safetensors",
                    "**/*.bin",
                    "**/*.pth",
                    "**/*.onnx",
                    "**/*.tflite",
                    "**/*.gguf",
                ],
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            handle_failure(f"Failed to download metadata: {e!r}")

        if local_dir is None:
            print(
                "  [WARNING] Skipping file-based checks because local directory is not"
                " available.")
            if _FORCE.value:
                print(
                    "\nPRE-FLIGHT CHECK PASSED: Checks bypassed via --force.")
            else:
                print(
                    "\nPRE-FLIGHT CHECK FAILED: Local directory is not available."
                )
            return

    # Step 3: Scan for incompatible imports in custom code (before loading to
    # prevent unhandled ImportError).
    print("  Scanning custom code for incompatible imports...")
    issues = check_for_incompatible_imports(local_dir)
    if issues:
        print("  [FLAGGED] Incompatible imports found:")
        playbook_mappings = discover_playbook_mappings()
        all_mapped = True
        for file, line, lib, sym in issues:
            playbook = playbook_mappings.get(lib)
            if playbook:
                print(
                    f"    - {file}:{line} [{sym}] imports '{lib}' -> [ONBOARDING_REQUIRED: {playbook}]"
                )
            else:
                print(
                    f"    - {file}:{line} [{sym}] imports '{lib}' (No playbook available)"
                )
                all_mapped = False
        if _CUDA_REPORT_JSON.value:
            write_cuda_report(
                _CUDA_REPORT_JSON.value,
                model_id,
                _MODE.value,
                issues,
                playbook_mappings,
            )
        if _ALLOW_GPU_IMPORTS.value or all_mapped:
            print(
                "  [STATUS: ONBOARDING_REQUIRED] GPU/CUDA imports detected with"
                " available unwiring playbooks. Downstream coder will apply recipes."
            )
        else:
            handle_failure("Incompatible imports found in custom code. Use"
                           " --allow_gpu_imports to bypass.")
    else:
        print("  No incompatible imports found.")
        # Written even when empty: `compute run --output` fails the run when a
        # declared output was never created.
        if _CUDA_REPORT_JSON.value:
            write_cuda_report(_CUDA_REPORT_JSON.value, model_id, _MODE.value,
                              [], {})

    if _HUB_ONLY.value:
        # Step 4.1 (Hub-side): the quantization rule, read from config.json
        # itself because AutoConfig needs transformers.
        quantization = check_quantization(read_quantization_config(local_dir),
                                          _MODE.value)
        if quantization:
            quant_ok, quant_msg = quantization
            print(f"  Quantization Check: {quant_msg}")
            if not quant_ok:
                handle_failure(f"Quantization check failed: {quant_msg}")
        else:
            print("  Quantization Check: No quantization config.")
        print("\nPRE-FLIGHT HUB CHECKS PASSED: Config, processor and vocab"
              " checks were skipped (--hub_only); run the full check where"
              " transformers is installed.")
        return

    if not transformers_available():
        handle_failure(
            "transformers is not installed here, so the config, processor and"
            " vocab checks cannot run. Rerun with --hub_only for the Hub-side"
            " checks, and run the full check in the task image.")
        print("\nPRE-FLIGHT CHECK PASSED: Checks bypassed via --force.")
        return

    # Step 4: Validate config.json
    compatible, config_msg, config = validate_config(local_dir, _MODE.value)
    print(f"  Config Validation: {config_msg}")
    if not compatible:
        handle_failure(f"Config Validation failed: {config_msg}")

    # Step 4.2: Validate Tokenizer & Processor Load
    proc_ok, proc_msg = validate_processor(local_dir, config)
    print(f"  Tokenizer/Processor Check: {proc_msg}")
    if not proc_ok:
        handle_failure(f"Tokenizer/Processor check failed: {proc_msg}")

    # Step 4.3: Check Vocab Bounds (IndexError safety)
    vocab_ok, vocab_msg = check_vocab_bounds(local_dir, config)
    print(f"  Vocab Bounds Check: {vocab_msg}")
    if not vocab_ok:
        handle_failure(f"Vocab bounds check failed: {vocab_msg}")

    print("\nPRE-FLIGHT CHECK PASSED: Model is likely compatible.")


if __name__ == "__main__":
    app.run(main)

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
"""Report what the task image carries, as one JSON object.

Preflight judges a model against the image the workload will run in, not
against what a config comment says it contains. This answers the questions
that judgement depends on: which runtime libraries are installed and at which
version, whether torch_tpu is present, whether the image can run the full
preflight check, whether it can serve on the custom_fastapi engine, whether
vLLM will serve on TorchTPU, whether its vLLM stack
is one a published image was tested with, what `pip check` finds, and -- with
--touch_device -- whether a TPU device can actually be allocated.

Modules are looked up by spec, never imported, so only --touch_device
initialises the TPU runtime. Standard library only: the probe must run in an
image that lacks the libraries it reports on.

Provisions nothing and changes nothing. Exit status is 0 whenever the report
was written; the findings are in the JSON, for the caller to read.
"""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys

# Distributions a torch_tpu onboarding depends on, by the name pip knows.
DISTRIBUTIONS = (
    "torch",
    "torch_tpu",
    "transformers",
    "diffusers",
    "accelerate",
    "safetensors",
    "huggingface_hub",
    "tokenizers",
    "vllm",
    "vllm_torchtpu",
    "tpu-inference",
    "sglang",
    "sglang-router",
    "sgl-kernel",
    "jax",
    "jaxlib",
    "libtpu",
    "libtpu-nightly",
    "fastapi",
    "pydantic",
    "uvicorn",
)

# Modules preflight_check.py imports; all present means the full check can run
# in this image.
PREFLIGHT_MODULES = ("absl", "huggingface_hub", "transformers", "yaml")

# Modules the custom_fastapi server template (inference.py) imports beyond
# torch; all present means the image can serve on that engine.
CUSTOM_FASTAPI_MODULES = ("fastapi", "pydantic", "uvicorn")

# Modules whose presence matters even when no distribution metadata names them.
MODULES = (("torch", "torch_tpu", "vllm", "vllm_torchtpu", "sglang") +
           PREFLIGHT_MODULES + CUSTOM_FASTAPI_MODULES)

# How vLLM finds an out-of-tree platform, and the package the TorchTPU one is.
VLLM_PLATFORM_PLUGINS = "vllm.platform_plugins"
TORCHTPU_PLUGIN = "vllm_torchtpu"

# The timeouts serve_vllm.sh raises for TPU compilation. A vLLM that renamed
# one ignores the default and falls back to its GPU-sized value, silently.
VLLM_TIMEOUT_ENVS = ("VLLM_ENGINE_READY_TIMEOUT_S",
                     "VLLM_ENGINE_ITERATION_TIMEOUT_S",
                     "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS")

# Environment the TPU runtime sets inside a scheduled pod.
TPU_ENV = ("TPU_ACCELERATOR_TYPE", "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES",
           "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TPU_TOPOLOGY")

# vLLM stacks a published task image was tested with, by the distributions
# that decide whether vLLM serves on TorchTPU. vllm_torchtpu is not among
# them: its builds all report 0.0.0, and an image built from source installs
# it editable, with no commit recorded, so its source is read instead
# (LEGACY_PLUGIN_CALLS). A version matches an installed one equal to it or
# extending it, local label ignored: "2.13" matches 2.13.0+cpu.
TESTED_STACKS = ({
    "image": "tpu-onboarding-torch_2.13.0-vllm_0.29.0-jax_0.11.2",
    "versions": {
        "torch": "2.13.0",
        "vllm": "0.29.0",
        "jax": "0.11.2",
        "jaxlib": "0.11.2",
    },
}, )

# vLLM stacks known not to serve, and what fails.
BROKEN_STACKS = ({
    "versions": {
        "vllm": "0.22.1",
        "jax": "0.10.2",
    },
    "reason": ("its vllm_torchtpu, built from the plugin's mid-July 2026 main,"
               " calls a jax API that no longer exists: vision models fail to"
               " serve"),
}, )

# Calls an older vllm_torchtpu makes into jax APIs that no longer exist, as
# (pattern, name). Whatever its version says, a plugin whose source makes one
# fails at the call: vision models do not serve.
LEGACY_PLUGIN_CALLS = ((re.compile(r"\bpltpu\.repeat\b"), "pltpu.repeat"), )

# How long `pip check` may take; it reads metadata and imports nothing.
PIP_CHECK_TIMEOUT_SEC = 120


def distribution_versions() -> dict[str, str]:
    """Installed version of every listed distribution that is present."""
    found = {}
    for name in DISTRIBUTIONS:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def _importable(name: str) -> bool:
    """Whether a module is importable, found without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:  # Already imported, with no spec to report.
        return name in sys.modules


def module_presence(names: tuple[str, ...] = MODULES) -> dict[str, bool]:
    """Whether each module is importable, found without importing it."""
    return {name: _importable(name) for name in names}


def _failure(err: BaseException) -> str:
    return f"{type(err).__name__}: {err}"


def touch_device() -> dict[str, str]:
    """Allocate on the native TPU device, the only proof the runtime is live.

    Constructing torch.device("tpu") alone proves nothing; the runtime only
    initialises when something is allocated on it.

    Only torch is imported first: a workload just asks for torch.device("tpu")
    and relies on PyTorch's backend autoload to bring in torch_tpu, so that is
    what is tried. If it fails and torch_tpu is installed, torch_tpu is
    imported and the allocation retried, as an older build may need.
    `loaded_by` says which worked.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception as err:  # pylint: disable=broad-except
        return {"status": "fail", "detail": _failure(err)}
    try:
        _ = torch.zeros(1, device=torch.device("tpu"))
        return {
            "status": "ok",
            "loaded_by": "autoload",
            "detail": "allocated on torch.device('tpu')"
        }
    except Exception as err:  # pylint: disable=broad-except
        autoload_error = _failure(err)
    if not _importable("torch_tpu"):
        return {"status": "fail", "detail": autoload_error}
    try:
        import torch_tpu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
        _ = torch.zeros(1, device=torch.device("tpu"))
        return {
            "status":
            "ok",
            "loaded_by":
            "import",
            "detail": ("allocated on torch.device('tpu') after importing"
                       f" torch_tpu; without it: {autoload_error}")
        }
    except Exception as err:  # pylint: disable=broad-except
        return {
            "status":
            "fail",
            "detail":
            f"{_failure(err)} (without importing torch_tpu:"
            f" {autoload_error})"
        }


def vllm_platform(load: bool) -> dict:
    """Whether vLLM will serve on the TorchTPU plugin, decided as vLLM does.

    vLLM finds out-of-tree platforms as entry points in the
    vllm.platform_plugins group. When VLLM_PLUGINS is set, it loads only the
    names in that comma-separated list. It calls each plugin it loads, and one
    that returns a class path is activated: a single activated plugin wins over
    vLLM's own platforms, and two are an error. The TorchTPU plugin's function
    returns its class unconditionally, so being found and allowed, with no
    rival, is being chosen. A plugin that fails to load is only logged, and
    vLLM falls back to its own TPU platform whenever libtpu imports: it serves,
    but not on TorchTPU, so no error would say so.

    Read from package metadata, nothing imported. With `load`, the plugin's
    entry point is also loaded and called, which imports torch, torch_tpu and
    vLLM and so starts the TPU runtime.
    """
    try:
        found = list(
            importlib.metadata.entry_points(group=VLLM_PLATFORM_PLUGINS))
    except Exception:  # pylint: disable=broad-except
        found = []
    raw_allowed = os.environ.get("VLLM_PLUGINS")
    allowed = None if raw_allowed is None else raw_allowed.split(",")
    loaded = [ep for ep in found if allowed is None or ep.name in allowed]
    ours = [
        ep for ep in found
        if ep.value.split(":")[0].split(".")[0] == TORCHTPU_PLUGIN
    ]
    rivals = [ep.name for ep in loaded if ep not in ours]

    problems = []
    if not ours:
        problems.append(f"no {VLLM_PLATFORM_PLUGINS} entry point is in"
                        f" {TORCHTPU_PLUGIN}")
    elif not any(ep in loaded for ep in ours):
        problems.append(f"VLLM_PLUGINS={raw_allowed} leaves out"
                        f" {ours[0].name}")
    if rivals:
        problems.append(f"other platform plugins would also load: {rivals}")
    report = {
        "entry_points": [f"{ep.name} = {ep.value}" for ep in found],
        "vllm_plugins_env": raw_allowed,
        "problems": problems,
    }
    if load and ours:
        try:
            report["load"] = {"status": "ok", "platform": ours[0].load()()}
        except Exception as err:  # pylint: disable=broad-except
            report["load"] = {"status": "fail", "detail": _failure(err)}
            problems.append(f"{ours[0].name} failed to load; vLLM would log it"
                            " and use its own platform")
    report["registered"] = not problems
    return report


def _canonical(name: str) -> str:
    """A distribution name as pip compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _matches(wanted: str, installed: str) -> bool:
    base = installed.split("+")[0]
    return base == wanted or base.startswith(wanted + ".")


def _stack_matches(versions: dict[str, str], stack: dict) -> bool:
    return all(name in versions and _matches(want, versions[name])
               for name, want in stack["versions"].items())


def _parse_version(version: str) -> tuple[int, ...]:
    """Numeric release tuple of a PEP 440 version ('0.30.1rc1+g80' -> (0, 30, 1))."""
    match = re.match(r"^\s*(\d+(?:\.\d+)*)", version.split("+", 1)[0])
    parts = [int(p) for p in match.group(1).split(".")] if match else []
    return tuple((parts + [0, 0, 0])[:max(len(parts), 3)])


def _patch_dirs(extra: str | None = None) -> list[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    in_tree = os.path.join(os.path.dirname(here), "common", "vllm_patches")
    return [d for d in (extra, os.getcwd(), here, in_tree) if d]


def load_patch_registry(
        registry_path: str | None = None) -> tuple[str | None, dict]:
    """Load vllm_patches/registry.json, returning (patch_dir, registry_dict)."""
    paths = ([registry_path] if registry_path else
             [os.path.join(d, "registry.json") for d in _patch_dirs()])
    for path in paths:
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return os.path.dirname(os.path.abspath(path)), data
            except (OSError, ValueError):
                continue
    return None, {}


def _model_stacks(entry: dict) -> list[dict]:
    raw = ([entry.get("validated")] +
           list(entry.get("previously_validated") or []) +
           list(entry.get("versions") or []))
    return [s for s in raw if isinstance(s, dict) and s.get("vllm")]


def registry_tested_stacks(registry: dict) -> list[dict]:
    """Stacks recorded in registry.json, in the shape TESTED_STACKS uses."""
    stacks, seen = [], set()
    for model_id, entry in (registry.get("models") or {}).items():
        for raw in _model_stacks(entry if isinstance(entry, dict) else {}):
            vers = {
                k: raw[k].split()[0]
                for k in ("torch", "vllm", "jax")
                if isinstance(raw.get(k), str) and raw[k].strip()
            }
            key = tuple(sorted(vers.items()))
            if "vllm" in vers and key not in seen:
                seen.add(key)
                tag = "-".join(f"{k}_{vers[k]}"
                               for k in ("torch", "vllm", "jax") if k in vers)
                stacks.append({
                    "image": f"registry-{model_id.replace('/', '--')}-{tag}",
                    "versions": vers,
                })
    return stacks


def select_patch_candidates(installed_vllm: str | None,
                            model_entry: dict) -> tuple[str, list[dict]]:
    """Select the closest patch version(s) (`before`/`after`) for `installed_vllm`."""
    if not installed_vllm:
        return "vllm_absent", []
    default_patch = model_entry.get("patch")
    entries = sorted(
        ({
            "validated_vllm": s["vllm"],
            "patch": s.get("patch") or default_patch,
            "serving": s.get("serving"),
        } for s in _model_stacks(model_entry)
         if s.get("patch") or default_patch),
        key=lambda e: _parse_version(e["validated_vllm"]),
    )
    if not entries:
        return "no_entries", []
    target = _parse_version(installed_vllm)
    for e in entries:
        if (_matches(e["validated_vllm"], installed_vllm)
                or _parse_version(e["validated_vllm"]) == target):
            return "exact", [{**e, "position": "exact"}]
    before = [
        e for e in entries if _parse_version(e["validated_vllm"]) < target
    ]
    after = [
        e for e in entries if _parse_version(e["validated_vllm"]) > target
    ]
    if before and not after:
        return "newer", [{**before[-1], "position": "before"}]
    if after and not before:
        return "older", [{**after[0], "position": "after"}]
    pair = [
        {
            **before[-1], "position": "before"
        },
        {
            **after[0], "position": "after"
        },
    ]
    pair.sort(key=lambda c: (
        tuple(
            abs(x - y)
            for x, y in zip(target, _parse_version(c["validated_vllm"]))),
        0 if c["position"] == "after" else 1,
    ))
    return "between", pair


def _find_plugin_root() -> str | None:
    """The directory holding `vllm_torchtpu/`, found without importing it."""
    try:
        spec = importlib.util.find_spec(TORCHTPU_PLUGIN)
    except (ImportError, ValueError):
        return None
    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    return os.path.dirname(os.path.abspath(locs[0])) if locs else None


def dry_run_patch(patch_path: str | None, plugin_root: str | None) -> dict:
    """Check whether `patch_path` applies to `plugin_root` without writing files."""
    if not plugin_root or not os.path.isdir(
            os.path.join(plugin_root, TORCHTPU_PLUGIN)):
        return {
            "applies": False,
            "status": "plugin_absent",
            "failures": ["vllm_torchtpu is not installed"],
        }
    if not patch_path or not os.path.isfile(patch_path):
        return {
            "applies": None,
            "status": "patch_file_not_present",
            "failures": [],
        }
    if patch_path.endswith(".py"):
        try:
            spec = importlib.util.spec_from_file_location(
                "_vllm_patch_probe", patch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sources = mod.Sources(plugin_root)
            _, failures = mod.plan(sources)
        except Exception as err:  # pylint: disable=broad-except
            failures = [_failure(err)]
        if failures:
            return {
                "applies": False,
                "status": "does_not_apply",
                "failures": list(failures),
            }
        return {
            "applies": True,
            "status": "applies" if sources.changed() else "already_applied",
            "failures": [],
        }

    base = ([
        "patch", "-p1", "-F0", "--dry-run", "-f", "-d", plugin_root, "-i"
    ] if shutil.which("patch") else
            ["git", "-C", plugin_root, "apply", "-p1", "--check"])
    if subprocess.run([*base, "-R", patch_path],
                      capture_output=True,
                      text=True,
                      check=False).returncode == 0:
        return {"applies": True, "status": "already_applied", "failures": []}
    fwd = subprocess.run([*base, patch_path],
                         capture_output=True,
                         text=True,
                         check=False)
    if fwd.returncode == 0:
        return {"applies": True, "status": "applies", "failures": []}
    detail = (fwd.stderr.strip() or fwd.stdout.strip()
              or f"{os.path.basename(patch_path)} did not apply")
    return {"applies": False, "status": "does_not_apply", "failures": [detail]}


def vllm_patches(versions: dict[str, str],
                 registry_path: str | None = None) -> dict[str, dict]:
    """Per-model patch candidates and static dry-run results from registry.json."""
    patch_dir, registry = load_patch_registry(registry_path)
    plugin_root = _find_plugin_root()
    installed_vllm = versions.get("vllm")
    report = {}
    for model_id, entry in (registry.get("models") or {}).items():
        if not isinstance(entry, dict):
            continue
        version_match, candidates = select_patch_candidates(
            installed_vllm, entry)
        evaluated = []
        for cand in candidates:
            patch_file = next(
                (p for d in _patch_dirs(patch_dir)
                 if os.path.isfile(p := os.path.join(d, cand["patch"]))),
                None,
            )
            evaluated.append({
                **cand,
                **dry_run_patch(patch_file, plugin_root)
            })
        applicable = [c for c in evaluated if c["applies"] is not False]
        report[model_id] = {
            "installed_vllm":
            installed_vllm,
            "version_match":
            version_match,
            "provenance":
            entry.get("provenance"),
            "candidates":
            evaluated,
            "applicable_candidates":
            len(applicable),
            "action":
            ("try_candidates" if applicable else "needs_user_confirmation"),
        }
    return report


def plugin_legacy_calls() -> list[str]:
    """Where the installed vllm_torchtpu's source makes a LEGACY_PLUGIN_CALLS
    call, as "<name> in <file>:<line>"; empty when it makes none or is absent.

    The package is found by spec, not imported, so an editable install is read
    where it lives.
    """
    try:
        spec = importlib.util.find_spec(TORCHTPU_PLUGIN)
    except (ImportError, ValueError):
        return []
    roots = list(spec.submodule_search_locations or []) if spec else []
    found = []
    for root in roots:
        for folder, _, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(folder, name)
                try:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                except OSError:
                    continue
                for number, line in enumerate(lines, 1):
                    for pattern, call in LEGACY_PLUGIN_CALLS:
                        if pattern.search(line):
                            found.append(f"{call} in {path}:{number}")
    return found


def vllm_timeout_envs(names: tuple[str, ...] = VLLM_TIMEOUT_ENVS) -> dict:
    """Whether the installed vLLM reads each of `names`, from vllm/envs.py.

    Read as text, the package found by spec and not imported. Each name maps
    to None when vLLM or its envs.py is absent.
    """
    try:
        spec = importlib.util.find_spec("vllm")
    except (ImportError, ValueError):
        spec = None
    for root in list(getattr(spec, "submodule_search_locations", None) or []):
        path = os.path.join(root, "envs.py")
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        return {name: f'"{name}"' in text for name in names}
    return {name: None for name in names}


def vllm_stack(versions: dict[str, str],
               legacy_calls: list[str],
               registry: dict | None = None) -> dict:
    """Whether the image's vLLM stack is one a published image was tested
    with, one known not to serve, or neither."""
    report = {"legacy_calls": legacy_calls, "problems": []}
    if "vllm" not in versions:
        report["verdict"] = "absent"
        return report
    if legacy_calls:
        report["verdict"] = "broken"
        report["problems"].append(
            "vllm_torchtpu calls jax APIs that no longer exist, so vision"
            f" models fail to serve: {', '.join(legacy_calls)}")
        return report
    for stack in BROKEN_STACKS:
        if _stack_matches(versions, stack):
            report["verdict"] = "broken"
            report["problems"].append(stack["reason"])
            return report
    tested = list(TESTED_STACKS) + registry_tested_stacks(registry or {})
    for stack in tested:
        if _stack_matches(versions, stack):
            report["verdict"] = "tested"
            report["image"] = stack["image"]
            return report
    report["verdict"] = "untested"
    shown = ", ".join(f"{name} {versions[name]}"
                      for name in ("torch", "vllm", "vllm_torchtpu", "jax",
                                   "jaxlib") if name in versions)
    report["problems"].append(f"no published image was tested with {shown}")
    return report


def pip_check() -> dict:
    """What `pip check` finds broken among the image's declared requirements.

    A line is a stack conflict when the distribution that declares the
    requirement is one this probe reports on. pip's own pins can be broken
    on purpose in a TPU image, so a conflict is a finding, not a verdict.
    """
    try:
        done = subprocess.run([sys.executable, "-m", "pip", "check"],
                              capture_output=True,
                              text=True,
                              timeout=PIP_CHECK_TIMEOUT_SEC,
                              check=False)
    except (OSError, subprocess.TimeoutExpired) as err:
        return {"status": "unavailable", "detail": _failure(err)}
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode == 0:
        return {"status": "ok", "stack_conflicts": [], "other_conflicts": 0}
    if not lines:
        return {"status": "unavailable", "detail": done.stderr.strip()[-500:]}
    stack = {_canonical(name) for name in DISTRIBUTIONS}
    ours = [line for line in lines if _canonical(line.split()[0]) in stack]
    return {
        "status": "conflicts",
        "stack_conflicts": ours,
        "other_conflicts": len(lines) - len(ours),
    }


def dependencies(versions: dict[str, str],
                 registry: dict | None = None) -> dict:
    """The vLLM stack verdict and `pip check`'s findings."""
    return {
        "vllm_stack": vllm_stack(versions, plugin_legacy_calls(), registry),
        "pip_check": pip_check(),
    }


def probe(check_device: bool, registry_path: str | None = None) -> dict:
    """The image report; allocates on the TPU only when `check_device`."""
    versions = distribution_versions()
    modules = module_presence()
    _, registry = load_patch_registry(registry_path)
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "distributions": versions,
        "modules": modules,
        "torch_tpu_present": modules["torch_tpu"],
        "sglang_present": modules["sglang"],
        "preflight_runnable": all(modules[m] for m in PREFLIGHT_MODULES),
        "custom_fastapi_runnable":
        all(modules[m] for m in CUSTOM_FASTAPI_MODULES),
        "vllm_platform": vllm_platform(load=check_device),
        "vllm_timeout_envs": vllm_timeout_envs(),
        "vllm_patches": vllm_patches(versions, registry_path=registry_path),
        "dependencies": dependencies(versions, registry=registry),
        "tpu_env": {
            k: os.environ[k]
            for k in TPU_ENV if k in os.environ
        },
        "device": touch_device() if check_device else {
            "status": "skipped",
            "detail": "run with --touch_device on a TPU host to check"
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--touch_device",
                        action="store_true",
                        help="Also allocate on the TPU device; needs a chip.")
    parser.add_argument("--registry",
                        default=None,
                        help="Path to vllm_patches/registry.json (auto-found"
                        " when omitted).")
    parser.add_argument("--output",
                        default=None,
                        help="Write the report here as well as to stdout.")
    args = parser.parse_args(argv)

    text = json.dumps(probe(args.touch_device, registry_path=args.registry),
                      indent=2,
                      sort_keys=True)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

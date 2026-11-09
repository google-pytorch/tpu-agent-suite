#!/usr/bin/env python3
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
"""Evaluation runner for TPU agent suite using Harbor."""

from __future__ import annotations

import argparse
import base64
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import yaml
from harbor.constants import HARBOR_VIEWER_JOBS_URL

from eval.agents.installer_staging import collect_task_environment_answers
from eval.mount_advisor import MountUpdates, SkillMountAdvisor
from eval.staging import stage_composite_package
from eval.utils import (RELEASE_MANIFEST, find_repo_root, harness_for_config,
                        installs_at_user_scope, is_control_arm,
                        package_identity, package_label, package_manifest,
                        read_harness_pins, released_skills, resolve_manifest,
                        skill_subset)


@dataclass(frozen=True)
class GkeCredentials:
    """Artifacts produced by GKE cluster authentication."""
    kubeconfig_path: Path
    token: str = ""


REPO_ROOT = find_repo_root()

# A config names the system under test; a dataset names the work. They are
# separate axes, so each lives in one flat directory and is selected by name.
CONFIGS_DIR = REPO_ROOT / "eval" / "configs"
DATASETS_DIR = REPO_ROOT / "eval" / "datasets"
SUITES_DIR = REPO_ROOT / "eval" / "suites"
DOT_DATASETS_DIR = REPO_ROOT / "eval" / ".datasets"


def check_dependencies() -> None:
    """Ensure critical runtime dependencies are installed."""
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")

    if not shutil.which("harbor"):
        missing.append("harbor")

    if missing:
        raise RuntimeError(
            f"Missing required dependencies: {', '.join(missing)}.\n"
            "Please set up and activate the evaluation virtual environment:\n"
            "  ./eval/setup.sh\n"
            "  source eval/.venv/bin/activate")


def get_adc_path() -> Path:
    """Return the host application default credentials path."""
    return Path(
        os.environ.get(
            "EVAL_VERTEX_ADC",
            Path.home() /
            ".config/gcloud/application_default_credentials.json",
        ))


# A run authenticates to GCP in one of two ways, chosen by GKE_AUTH_ACCOUNT:
#
#   adc       An engineer's `gcloud auth application-default login`. The ADC
#             file is mounted into the trial pod and its tokens are minted
#             from it.
#   metadata  No human credential at all: the GCE/GKE metadata server's
#             service account (Workload Identity on the self-hosted GitHub
#             runner). No ADC is read or uploaded; the trial pod reaches GCP
#             as its own node's service account, through its own metadata
#             server. The CI configs default to this.
METADATA_AUTH = "metadata"
# By IP, not metadata.google.internal: a job container may not resolve that.
METADATA_SA_URL = ("http://169.254.169.254/computeMetadata/v1/instance/"
                   "service-accounts/default/")


def metadata_server_get(path: str) -> str:
    """GET `path` under the default service account on the metadata server."""
    req = urllib.request.Request(METADATA_SA_URL + path,
                                 headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.read().decode("utf-8").strip()


def metadata_access_token() -> str:
    """An access token for the metadata server's service account."""
    return json.loads(metadata_server_get("token"))["access_token"]


def config_auth_account(config: dict[str, Any]) -> str | None:
    """GKE_AUTH_ACCOUNT as the config resolves it, else from the environment."""
    env = {}
    for agent in config.get("agents", []):
        env.update(agent.get("env") or {})
    if isinstance(config.get("environment"), dict):
        env.update(config["environment"].get("env") or {})
    return resolve_template(
        env.get("GKE_AUTH_ACCOUNT")) or os.environ.get("GKE_AUTH_ACCOUNT")


def resolve_template(val: Any) -> str | None:
    """Resolve ${VAR:-default} or ${VAR} using os.environ, falling back to default."""
    if val is None:
        return None
    val_str = str(val).strip()
    match = re.match(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$",
                     val_str)
    if match:
        var_name, default = match.group(1), match.group(2)
        return os.environ.get(var_name, default)
    return val_str


def config_package_manifest(config: dict[str, Any]) -> str:
    """Return the PACKAGE_MANIFEST value in `config`'s first agent env, unchecked.

    Its `${VAR:-default}` template is resolved against the environment, and
    an unset or empty value gives package.release.json (eval/utils.py
    `package_manifest`).
    """
    agents = config.get("agents") or []
    env = (agents[0].get("env") if agents else None) or {}
    return package_manifest(
        {"PACKAGE_MANIFEST": resolve_template(env.get("PACKAGE_MANIFEST"))})


def available_configs() -> str:
    """Config names under eval/configs/, for error messages.

    Configs are grouped by harness, so a name is the path under configs/
    without the extension -- claude-code/agent-suite, not just the stem.
    """
    if not CONFIGS_DIR.is_dir():
        return "(none)"
    names = sorted(
        str(p.relative_to(CONFIGS_DIR).with_suffix(""))
        for p in CONFIGS_DIR.glob("**/*.yaml"))
    return ", ".join(names) or "(none)"


def available_datasets() -> str:
    """Dataset names under eval/datasets/, for error messages."""
    if not DATASETS_DIR.is_dir():
        return "(none)"
    return ", ".join(
        sorted(p.name
               for p in DATASETS_DIR.iterdir() if p.is_dir())) or "(none)"


def is_task_file(path: Path) -> bool:
    """True for a suite YAML holding exactly one case.

    Such files are the tasks of a suite directory: suites/cset/tpu_knowledge_qa/
    is one Harbor dataset and each YAML inside it is one Harbor task.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return False
    cases = (data.get("evaluation") or {}).get("cases") or []
    return len(cases) == 1


def available_suites() -> str:
    """Suite names under eval/suites/, for error messages.

    A directory of task files is listed once, as the directory, rather than
    once per task.
    """
    if not SUITES_DIR.is_dir():
        return "(none)"
    names = set()
    for p in SUITES_DIR.glob("**/*.yaml"):
        suite = p.parent if is_task_file(p) else p.with_suffix("")
        names.add(str(suite.relative_to(SUITES_DIR)))
    return ", ".join(sorted(names)) or "(none)"


def resolve_suite(spec: str) -> Path:
    """Resolve a suite from name or path: a YAML file or a directory of them."""
    cand_path = Path(spec)
    if cand_path.is_file() or cand_path.is_dir():
        return cand_path.resolve()

    for cand in (
            SUITES_DIR / spec,
            SUITES_DIR / f"{spec}.yaml",
            SUITES_DIR / f"{spec.removesuffix('.yaml')}.yaml",
    ):
        if cand.is_file() or cand.is_dir():
            return cand.resolve()

    raise RuntimeError(f"Suite '{spec}' not found.\n"
                       f"Available suites: {available_suites()}")


def compile_suite_if_needed(
    suite: Path,
    task_filter: str | None = None,
) -> Path:
    """Compile a suite into eval/.datasets/<sub_rel>/ on demand.

    <sub_rel> is the suite's path under suites/ minus .yaml, so every suite
    file gets its own dataset even when several share a directory:
    suites/knowledgebase/tpu_knowledge_qa.yaml -> .datasets/knowledgebase/tpu_knowledge_qa/.

    A suite directory compiles every YAML directly inside it into one dataset,
    one task per case: suites/cset/tpu_knowledge_qa/csat002.yaml ->
    .datasets/cset/tpu_knowledge_qa/csat002/. The whole directory is always
    compiled, so task_filter only narrows a single-file suite.

    Returns the path to the target dataset directory under .datasets/.
    """
    from eval.suite_to_harbor.suite_to_harbor import compile_suite

    is_dir = suite.is_dir()
    if is_dir:
        sources = sorted(suite.glob("*.yaml"))
        if not sources:
            raise RuntimeError(
                f"Suite directory {suite} holds no .yaml files.")
        fallback = Path(suite.name)
    else:
        sources = [suite]
        with open(suite, "r", encoding="utf-8") as f:
            s_data = yaml.safe_load(f) or {}
        fallback = Path(
            s_data.get("name") or s_data.get("suite_name") or suite.stem)

    try:
        sub_rel = suite.relative_to(SUITES_DIR)
        if not is_dir:
            sub_rel = sub_rel.with_suffix("")
    except ValueError:
        sub_rel = fallback

    target_dir = DOT_DATASETS_DIR / sub_rel

    try:
        suite_disp = str(suite.relative_to(REPO_ROOT))
    except ValueError:
        suite_disp = str(suite)
    try:
        target_disp = str(target_dir.relative_to(REPO_ROOT))
    except ValueError:
        target_disp = str(target_dir)

    task_tomls = list(target_dir.glob("**/task.toml"))
    # A directory's own mtime moves when a task file is added or removed.
    suite_mtime = max(p.stat().st_mtime
                      for p in sources + ([suite] if is_dir else []))
    stale = any(suite_mtime > t.stat().st_mtime for t in task_tomls)
    if not target_dir.is_dir() or not task_tomls or stale:
        print(f"==> compiling suite {suite_disp} -> {target_disp}...")
        if is_dir:
            # Start clean so a deleted task file does not leave its task behind.
            shutil.rmtree(target_dir, ignore_errors=True)
            owners: dict[Path, Path] = {}
            for src in sources:
                for task_dir in compile_suite(src, target_dir) or []:
                    if task_dir in owners:
                        raise RuntimeError(
                            f"{src.name} and {owners[task_dir].name} in "
                            f"{suite_disp} both compile to task "
                            f"{task_dir.name}")
                    owners[task_dir] = src
        else:
            compile_suite(suite, target_dir, task_name_filter=task_filter)
    elif (not is_dir and task_filter
          and not any(t.parent.name == task_filter for t in task_tomls)):
        print(f"==> compiling task {task_filter} from {suite_disp}...")
        compile_suite(suite, target_dir, task_name_filter=task_filter)

    return target_dir


def resolve_dataset_arg(spec: str) -> tuple[str, dict[str, str]]:
    """Map a --dataset value to the JobConfig key and entry that selects it.

    A leaf task directory (one holding task.toml) becomes a `tasks` entry; a
    suite directory holding many of them becomes a `datasets` entry, which is
    the same distinction Harbor draws between TaskConfig and DatasetConfig.
    """
    for cand in (Path(spec), DATASETS_DIR / spec):
        if cand.is_dir():
            resolved = cand.resolve()
            break
    else:
        raise RuntimeError(f"Dataset '{spec}' not found.\n"
                           f"Available datasets: {available_datasets()}")

    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        rel = resolved

    key = "tasks" if (resolved / "task.toml").is_file() else "datasets"
    return key, {"path": str(rel)}


def available_tasks() -> str:
    """Task names as <dataset>/<task>, for error messages."""
    if not DATASETS_DIR.is_dir():
        return "(none)"
    names = sorted(
        str(p.parent.relative_to(DATASETS_DIR))
        for p in DATASETS_DIR.glob("**/task.toml"))
    return ", ".join(names) or "(none)"


def resolve_task_arg(spec: str,
                     dataset: str | None = None) -> tuple[str, dict[str, str]]:
    """Map a --task value to the `tasks` entry that selects one task.

    Accepts a bare task name, a <dataset>/<task> pair, or a path. Scoping with
    --dataset resolves the name under that dataset. A bare name matching more
    than one dataset is an error rather than a guess: tpu-hello-world/baseline and
    hf-onboarding-torchtpu/baseline are different evals, and silently running
    one of them is the failure this refuses to make.
    """
    if dataset:
        if Path(dataset).is_dir():
            roots = [Path(dataset).resolve()]
        elif (DATASETS_DIR / dataset).is_dir():
            roots = [DATASETS_DIR / dataset]
        elif (DOT_DATASETS_DIR / dataset).is_dir():
            roots = [DOT_DATASETS_DIR / dataset]
        else:
            raise RuntimeError(f"Dataset '{dataset}' not found.\n"
                               f"Available datasets: {available_datasets()}")
    else:
        roots = [DATASETS_DIR, DOT_DATASETS_DIR]

    matches: list[Path] = []
    cand_spec = Path(spec)
    if not cand_spec.is_absolute():
        cand_spec_repo = (REPO_ROOT / spec).resolve()
        if (cand_spec_repo / "task.toml").is_file():
            matches.append(cand_spec_repo)
    if not matches and cand_spec.is_dir() and (cand_spec /
                                               "task.toml").is_file():
        matches.append(cand_spec.resolve())

    if not matches:
        for root in roots:
            direct = root / spec
            if (direct / "task.toml").is_file():
                matches.append(direct)
            matches.extend(p.parent for p in root.glob(f"*/{spec}/task.toml"))

    unique = sorted({p.resolve() for p in matches})
    if not unique:
        scope = f" in dataset '{dataset}'" if dataset else ""
        raise RuntimeError(f"Task '{spec}' not found{scope}.\n"
                           f"Available tasks: {available_tasks()}")
    if len(unique) > 1:
        qualified = ", ".join(str(p.relative_to(DATASETS_DIR)) for p in unique)
        raise RuntimeError(
            f"Task '{spec}' is ambiguous; it exists in more than one dataset:\n"
            f"  {qualified}\n"
            "Qualify it as <dataset>/<task>, or scope it with --dataset.")

    resolved = unique[0]
    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        rel = resolved
    return "tasks", {"path": str(rel)}


def run_dataset_setup_if_present(config: dict[str, Any]) -> None:
    """Run a selected dataset's setup.sh before Harbor reads its tasks.

    TODO: make jaxbench tasks complete on disk and delete this. Its task
    directories ship only task.toml and environment/base.py; the Dockerfile and
    instruction.md are copied in from common/ by setup.sh and gitignored, so a
    task is not a task until that script has run. Every other dataset is
    complete as committed and needs no hook. Generating those two files at
    authoring time -- or committing them -- removes the special case, and this
    function with it.

    Scoped to the dataset actually selected; the eval_setup.sh this replaced
    ran every setup.sh it could find.
    """
    for key in ("tasks", "datasets"):
        for entry in config.get(key) or []:
            path = entry.get("path") if isinstance(entry, dict) else entry
            if not path:
                continue
            resolved = (REPO_ROOT / path).resolve()
            try:
                dataset = DATASETS_DIR / resolved.relative_to(
                    DATASETS_DIR).parts[0]
            except (ValueError, IndexError):
                continue
            setup = dataset / "setup.sh"
            if setup.is_file():
                print(
                    f"==> staging dataset assets: {setup.relative_to(REPO_ROOT)}"
                )
                subprocess.run(["bash", str(setup)], cwd=REPO_ROOT, check=True)


def harbor_image_tag(task_dir: Path, subdir: str,
                     kwargs: dict[str, Any]) -> str | None:
    """The image tag Harbor will look for, for one task's environment.

    Harbor's GKE path never reads task.toml's docker_image -- it resolves the
    pod image from its own content-addressed tag and builds it with Cloud
    Build if absent (gke.py:623, :712). Computing the identical tag here lets a
    local build satisfy that lookup, so `_image_exists()` short-circuits and no
    Cloud Build runs.

    Uses Harbor's own hash rather than reimplementing it: a tag that differs by
    one byte is a tag Harbor never looks for, and the build would silently
    happen anyway.
    """
    try:
        from harbor.environments.definition import environment_content_hash
    except ImportError:
        return None

    env_dir = task_dir / subdir
    task_toml = task_dir / "task.toml"
    # No Dockerfile means Harbor has no image to build from this directory --
    # tests/ usually holds only test.sh -- so there is nothing to prebuild.
    if not (env_dir / "Dockerfile").is_file() or not task_toml.is_file():
        return None

    import tomllib
    data = tomllib.loads(task_toml.read_text(encoding="utf-8"))
    name = (data.get("task") or {}).get("name", "")
    if "/" not in name:
        return None
    short_name = name.split("/")[1]
    pinned = (data.get("environment") or {}).get("docker_image")
    env_id = environment_content_hash(env_dir, docker_image=pinned)

    location = resolve_template(kwargs.get("registry_location"))
    registry = resolve_template(kwargs.get("registry_name"))
    project = resolve_template(kwargs.get("project_id"))
    if not (location and registry and project):
        return None
    return (f"{location}-docker.pkg.dev/{project}/{registry}/"
            f"{short_name}:{env_id}")


def image_in_registry(tag: str, env: dict[str, str] | None = None) -> bool:
    """Whether the tag is already in Artifact Registry, as Harbor checks it.

    `env` must carry the same credential used to push: gcloud reads its active
    account otherwise, which on a GCE VM is the instance service account and
    usually cannot see the repository. Without it this returns False for tags
    that exist, and every run rebuilds.
    """
    # Mirrors gke.py::_image_exists exactly -- `images describe`, with
    # --project. A different subcommand would answer a different question.
    cmd = ["gcloud", "artifacts", "docker", "images", "describe", tag]
    project = tag.split("/")[1] if tag.count("/") >= 2 else None
    if project:
        cmd += ["--project", project]
    res = subprocess.run(cmd,
                         capture_output=True,
                         text=True,
                         check=False,
                         env=env)
    return res.returncode == 0


def adc_docker_config(hosts: list[str]) -> str | None:
    """A throwaway docker config authenticating to `hosts` with ADC.

    credHelpers beat `docker login`, and the gcloud helper authenticates as the
    active account (the instance SA on a GCE VM), not ADC.
    """
    token = subprocess.run(
        ["gcloud", "auth", "application-default", "print-access-token"],
        capture_output=True,
        text=True,
        check=False).stdout.strip()
    if not token:
        return None
    auth = base64.b64encode(f"oauth2accesstoken:{token}".encode()).decode()
    cfg_dir = tempfile.mkdtemp(prefix="eval-docker-")
    Path(cfg_dir, "config.json").write_text(json.dumps(
        {"auths": {
            h: {
                "auth": auth
            }
            for h in hosts
        }}),
                                            encoding="utf-8")
    return cfg_dir


def prebuild_images_locally(config: dict[str, Any]) -> None:
    """Build and push each selected task's image locally, skipping Cloud Build.

    Harbor skips its own build when the tag already exists, so this populates
    exactly the tags it will look for. Needs only registry push, where Cloud
    Build needs four separate grants.
    """
    kwargs = ((config.get("environment") or {}).get("kwargs") or {})
    if not kwargs.get("registry_name"):
        return

    reg_loc = resolve_template(kwargs.get("registry_location", "us"))
    hosts = {f"{reg_loc}-docker.pkg.dev"}
    for df in DATASETS_DIR.glob("**/environment/Dockerfile"):
        for line in df.read_text(encoding="utf-8").splitlines():
            if line.startswith("FROM ") and "-docker.pkg.dev/" in line:
                hosts.add(line.split()[1].split("/")[0])
    docker_env = os.environ.copy()
    cfg_dir = adc_docker_config(sorted(hosts))
    if cfg_dir:
        docker_env["DOCKER_CONFIG"] = cfg_dir

    # gcloud in the registry check needs the same identity as the push.
    gcloud_env = os.environ.copy()
    adc_path = get_adc_path()
    if adc_path.is_file():
        gcloud_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = str(adc_path)

    for key in ("tasks", "datasets"):
        for entry in config.get(key) or []:
            path = entry.get("path") if isinstance(entry, dict) else entry
            if not path:
                continue
            root = (REPO_ROOT / path).resolve()
            task_dirs = ([root] if (root / "task.toml").is_file() else sorted(
                p.parent for p in root.glob("*/task.toml")))
            for task_dir in task_dirs:
                for subdir in ("environment", "tests"):
                    tag = harbor_image_tag(task_dir, subdir, kwargs)
                    if not tag:
                        continue
                    if image_in_registry(tag, gcloud_env):
                        print(
                            f"==> image present, no build: {tag.split('/')[-1]}"
                        )
                        continue
                    print(f"==> building locally: {tag.split('/')[-1]}")
                    subprocess.run(
                        ["docker", "build", "-t", tag,
                         str(task_dir / subdir)],
                        check=True,
                        env=docker_env)
                    subprocess.run(["docker", "push", tag],
                                   check=True,
                                   env=docker_env)


def apply_selection(config: dict[str, Any], key: str,
                    entries: dict[str, str] | Sequence[dict[str, str]],
                    flag: str) -> None:
    """Point the config at the work named on the CLI.

    Configs describe the system under test only, so the work is supplied per
    run. A config that already carries a selection has it replaced, and that
    is announced: a silent replacement makes a run that skipped most of a
    suite look like a full one.
    """
    if isinstance(entries, dict):
        entries = [entries]
    for existing_key in ("tasks", "datasets"):
        for item in config.pop(existing_key, None) or []:
            path = item.get("path") if isinstance(item, dict) else item
            print(
                f"==> {flag} replaces config {existing_key}: dropping {path}")

    config[key] = list(entries)
    for entry in entries:
        print(f"==> {flag}: {entry['path']} (as {key})")


def task_specs(value: str | Sequence[str] | None) -> list[str]:
    """Normalize --task to a list: it may be given once, repeated, or not."""
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def apply_dataset(config: dict[str, Any], spec: str) -> None:
    """Select a whole dataset."""
    key, entry = resolve_dataset_arg(spec)
    apply_selection(config, key, entry, "--dataset")


def apply_tasks(config: dict[str, Any],
                specs: Sequence[str],
                dataset: str | None = None) -> None:
    """Select several tasks as one Harbor job, optionally scoped by --dataset.

    This is what lets CI run just the tasks a PR touched: one job, one Hub
    link, one pass/fail, rather than a job per task. Duplicates collapse, so
    a task named twice runs once.
    """
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for spec in specs:
        _, entry = resolve_task_arg(spec, dataset)
        if entry["path"] not in seen:
            seen.add(entry["path"])
            entries.append(entry)
    apply_selection(config, "tasks", entries, "--task")


def require_dataset(config: dict[str, Any], config_path: Path) -> None:
    """Fail loudly when no dataset or suite is selected.

    Harbor accepts a config with neither tasks nor datasets and runs nothing,
    reporting success over an empty set. That is the one outcome an eval must
    never produce quietly.
    """
    if config.get("tasks") or config.get("datasets"):
        return
    raise RuntimeError(
        f"No dataset or suite selected for '{config_path.stem}'. Configs describe the\n"
        "system under test only, so the work is given per run:\n"
        "  ./eval/run.py --config <name> --suite <name>\n"
        "  ./eval/run.py --config <name> --dataset <name>\n"
        "  ./eval/run.py --config <name> --task <name>\n"
        f"Available suites: {available_suites()}\n"
        f"Available datasets: {available_datasets()}")


def resolve_config_files(
    args: argparse.Namespace,
    harbor_args: list[str] | None = None,
) -> list[Path]:
    """Resolve target configuration file(s) from named arguments or positional arguments."""
    if args.config_file:
        raw_files = (args.config_file if isinstance(
            args.config_file, (list, tuple)) else [args.config_file])
        resolved_files: list[Path] = []
        for item in raw_files:
            for sub in str(item).split(","):
                sub_clean = sub.strip()
                if not sub_clean:
                    continue
                path = Path(sub_clean).resolve()
                if not path.is_file():
                    raise RuntimeError(
                        f"Config file not found at '{sub_clean}'.")
                resolved_files.append(path)
        if resolved_files:
            return resolved_files

    if args.config:
        raw_configs = (args.config if isinstance(args.config,
                                                 (list,
                                                  tuple)) else [args.config])
        config_names: list[str] = []
        for c in raw_configs:
            for sub in str(c).split(","):
                sub_clean = sub.strip()
                if sub_clean:
                    config_names.append(sub_clean.removesuffix(".yaml"))

        resolved_files: list[Path] = []
        for name in config_names:
            path = CONFIGS_DIR / f"{name}.yaml"
            if not path.is_file():
                raise RuntimeError(
                    f"Config '{name}' not found. Expected: {path}\n"
                    f"Available configs: {available_configs()}")
            resolved_files.append(path)
        if resolved_files:
            return resolved_files

    positional = list(getattr(args, "positional_args", []))
    if not positional and harbor_args:
        while harbor_args and not harbor_args[0].startswith(
                "-") and harbor_args[0] != "--":
            positional.append(harbor_args.pop(0))

    if positional:
        first = positional[0]
        first_path = Path(first).resolve()
        if first_path.is_file() or first.endswith(".yaml"):
            resolved_files: list[Path] = []
            for item in positional:
                p = Path(item).resolve()
                if not p.is_file():
                    raise RuntimeError(f"Config file not found at '{item}'.")
                resolved_files.append(p)
            return resolved_files

        resolved_files: list[Path] = []
        for cfg in positional:
            for sub in cfg.split(","):
                sub_clean = sub.strip()
                if not sub_clean:
                    continue
                path = CONFIGS_DIR / f"{sub_clean.removesuffix('.yaml')}.yaml"
                if not path.is_file():
                    raise RuntimeError(
                        f"Config '{sub_clean}' not found. Expected: {path}\n"
                        f"Available configs: {available_configs()}")
                resolved_files.append(path)
        return resolved_files

    raise RuntimeError(
        "Missing configuration.\n"
        "Usage: ./eval/run.py --config <name...> --suite <name> [harbor_args...]\n"
        "   or: ./eval/run.py --config <name...> --dataset <name> [harbor_args...]\n"
        "   or: ./eval/run.py --config <name...> --task <name> [harbor_args...]\n"
        "   or: ./eval/run.py <config...> --dataset <name> [harbor_args...]\n"
        "   or: ./eval/run.py --config-file <path.yaml...> --dataset <name> [harbor_args...]\n"
        f"Available configs: {available_configs()}\n"
        f"Available suites: {available_suites()}\n"
        f"Available datasets: {available_datasets()}")


def _safe_load_json(source: Path | str) -> Any:
    """Load JSON file or string, sanitizing unquoted Harbor [REDACTED] tokens if needed."""
    content = source.read_text(
        encoding="utf-8") if isinstance(source, Path) else source
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        sanitized = re.sub(r':\s*\[REDACTED\]', ': "[REDACTED]"', content)
        return json.loads(sanitized)


def failed_criteria(trial_dir: Path, reasoning_chars: int = 300) -> list[str]:
    """One line per criterion a trial's verifier scored below full marks.

    Read from RewardKit's `reward-details.json`, which Harbor copies into the
    trial's `verifier/` directory. The score alone says a trial failed; this
    says which check, so a CI log is enough to act on without the Hub. A
    judge criterion's reasoning is appended, cut to `reasoning_chars`.
    """
    try:
        details = _safe_load_json(trial_dir / "verifier" /
                                  "reward-details.json")
    except Exception:  # pylint: disable=broad-except
        return []
    lines: list[str] = []

    def walk(name: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        for criterion in node.get("criteria") or []:
            if not isinstance(criterion, dict):
                continue
            value = criterion.get("value")
            if not isinstance(value, (int, float)) or value >= 1:
                continue
            line = (f"{name}/{criterion.get('name', '?')} = {value:.2f}: "
                    f"{criterion.get('description', '')}")
            reasoning = str(criterion.get("reasoning") or "").strip()
            if reasoning:
                reasoning = " ".join(reasoning.split())
                if len(reasoning) > reasoning_chars:
                    reasoning = reasoning[:reasoning_chars] + "..."
                line += f" -- judge: {reasoning}"
            lines.append(line)
        for component in node.get("components") or []:
            if isinstance(component, dict):
                walk(f"{name}/{component.get('name', '?')}",
                     component.get("detail"))

    if isinstance(details, dict):
        for name, node in details.items():
            walk(name, node)
    return lines


def check_job_scores(
        job_dir: Path,
        min_score: float = 1.0) -> tuple[bool, list[dict[str, Any]]]:
    """Inspect trial results in job_dir and verify each meets min_score.

    Returns (all_passed, trial_summaries).
    """
    result_path = job_dir / "result.json"
    if not result_path.is_file():
        return False, []

    try:
        job_data = _safe_load_json(result_path)
    except Exception as e:
        print(f"Warning: Failed to parse {result_path}: {e}", file=sys.stderr)
        return False, []

    trial_results = job_data.get("trial_results", [])
    if not trial_results:
        unreadable: list[str] = []
        for child in sorted(job_dir.iterdir()):
            if child.is_dir() and (child / "result.json").is_file():
                try:
                    trial_results.append(_safe_load_json(child /
                                                         "result.json"))
                except Exception as e:
                    unreadable.append(child.name)
                    print(
                        f"Warning: Failed to parse {child / 'result.json'}: {e}",
                        file=sys.stderr)
        if unreadable:
            print(
                f"Warning: {len(unreadable)} trial result(s) were unreadable and are "
                f"NOT reflected in the scores below: {', '.join(unreadable)}",
                file=sys.stderr)

    if not trial_results:
        return False, []

    summaries: list[dict[str, Any]] = []
    all_passed = True
    for tr in trial_results:
        trial_name = tr.get("trial_name") or tr.get("task_name") or "unknown"
        verifier_result = tr.get("verifier_result") or {}
        rewards = verifier_result.get("rewards") if isinstance(
            verifier_result, dict) else None

        score = None
        if isinstance(rewards, dict) and rewards:
            if "reward" in rewards:
                score = rewards["reward"]
            elif "score" in rewards:
                score = rewards["score"]
            else:
                score = list(rewards.values())[0]

        try:
            score_val = float(score) if score is not None else 0.0
        except (ValueError, TypeError):
            score_val = 0.0

        passed = score_val >= min_score
        if not passed:
            all_passed = False

        summary = {
            "trial_name": trial_name,
            "score": score_val,
            "passed": passed,
        }
        report = read_failure_report(job_dir / trial_name)
        if report:
            summary["failure_report"] = report
        summaries.append(summary)

    return all_passed, summaries


# A task opts in to the "explanation of failures" report by shipping a
# tests/failure_report/judge.toml; rewardkit writes that judge's output under
# this dimension of verifier/reward-details.json.
FAILURE_REPORT_DIMENSION = "failure_report"


def read_failure_report(trial_dir: Path) -> str | None:
    """Return the failure_report judge's reasoning for a trial, if any.

    Returns None, without raising, when the task has no failure_report judge,
    the verifier did not run, or the details file is unreadable.
    """
    details_path = trial_dir / "verifier" / "reward-details.json"
    if not details_path.is_file():
        return None
    try:
        details = _safe_load_json(details_path)
    except Exception:  # pylint: disable=broad-except
        return None
    node = details.get(FAILURE_REPORT_DIMENSION) if isinstance(details,
                                                               dict) else None
    criteria = node.get("criteria") if isinstance(node, dict) else None
    if not isinstance(criteria, list):
        return None
    reasons = [
        str(c["reasoning"]).strip() for c in criteria
        if isinstance(c, dict) and str(c.get("reasoning") or "").strip()
    ]
    return "\n\n".join(reasons) or None


def get_harbor_hub_url(job_dir: Path) -> str | None:
    """Extract Harbor Hub URL from the job results if available."""
    result_json = job_dir / "result.json"
    if not result_json.is_file():
        return None
    data = json.loads(result_json.read_text(encoding="utf-8"))
    job_id = data.get("id") if isinstance(data, dict) else None
    if not job_id:
        return None
    viewer_jobs_url = (os.environ.get("HARBOR_VIEWER_JOBS_URL")
                       or os.environ.get("HARBOR_VIEWER_URL")
                       or HARBOR_VIEWER_JOBS_URL).rstrip("/")
    return f"{viewer_jobs_url}/{job_id}"


# GitHub rejects a step summary over 1 MiB, and a run can fail several trials
# across several configs; the full report stays in reward-details.json.
_SUMMARY_REPORT_CAP_CHARS = 60_000


def _truncate_for_summary(report: str, limit: int) -> str:
    if len(report) <= limit:
        return report
    return (report[:limit].rstrip() +
            "\n\n_… truncated; see the trial's verifier/reward-details.json"
            " for the full report._")


def publish_github_step_summary(
    results_summary: list[dict[str, Any]],
    upload_enabled: bool,
    summary_path: Path | str | None = None,
) -> None:
    """Publish evaluation results and Harbor Hub links to GITHUB_STEP_SUMMARY."""
    if summary_path is None:
        env_val = os.environ.get("GITHUB_STEP_SUMMARY")
        if not env_val:
            return
        summary_path = Path(env_val)
    elif isinstance(summary_path, str):
        summary_path = Path(summary_path)

    if not results_summary:
        return

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            for item in results_summary:
                cfg_p = item["config_path"]
                display_path = (cfg_p.relative_to(REPO_ROOT)
                                if cfg_p.is_relative_to(REPO_ROOT) else cfg_p)
                url = item.get("hub_url")
                job_name = item.get("job_name", "unknown")
                eval_passed = item.get("eval_passed", False)
                status_badge = "✅ Passed" if eval_passed else "❌ Failed"

                f.write(f"### 🚀 Harbor Evaluation: `{job_name}`\n\n")
                f.write(f"- **Config**: `{display_path}`\n")
                f.write(f"- **Status**: {status_badge}\n")

                if not upload_enabled:
                    f.write(
                        f"- **Harbor Link**: *(upload disabled; job: `{job_name}`)*\n\n"
                    )
                elif url:
                    if item.get("upload_succeeded", False):
                        f.write(f"- **Harbor Link**: [{url}]({url})\n\n")
                    else:
                        f.write(
                            f"- **Harbor Link**: [{url}]({url}) *(⚠️ upload failed)*\n\n"
                        )
                else:
                    f.write("- **Harbor Link**: *(no hub url available)*\n\n")

                trials = item.get("trial_summaries", [])
                if trials:
                    f.write("| Trial | Score | Status |\n")
                    f.write("| :--- | :--- | :--- |\n")
                    for ts in trials:
                        badge = "✅ PASS" if ts.get("passed") else "❌ FAIL"
                        f.write(
                            f"| `{ts['trial_name']}` | {ts['score']:.2f} | {badge} |\n"
                        )
                    f.write("\n")
                    for ts in trials:
                        report = ts.get("failure_report")
                        if ts.get("passed") or not report:
                            continue
                        f.write(
                            "<details open>\n<summary>🔍 <b>Explanation of "
                            f"failures</b>: <code>{ts['trial_name']}</code> "
                            f"(score {ts['score']:.2f})</summary>\n\n")
                        f.write(
                            _truncate_for_summary(report,
                                                  _SUMMARY_REPORT_CAP_CHARS))
                        f.write("\n\n</details>\n\n")
                f.write("---\n\n")
    except Exception as e:
        print(f"Warning: Failed to write to GITHUB_STEP_SUMMARY: {e}",
              file=sys.stderr)


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML configuration from file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Keys that once chose which skills to stage. There is one package now: every
# config installs all of it, and a control arm (`is_control_arm`) nothing.
REMOVED_KEYS = ("stage_skills", "packages", "package_dir")


def reject_removed_keys(config: dict[str, Any], path: Path) -> None:
    """Fail on a key that once selected skills, rather than in Harbor."""
    found = [k for k in REMOVED_KEYS if k in config]
    found += [
        f"agents[].kwargs.{k}" for agent in config.get("agents") or []
        for k in ("package_dir", ) if k in (agent.get("kwargs") or {})
    ]
    if found:
        raise RuntimeError(
            f"{path}: {', '.join(found)} no longer exist. Every config installs "
            "the whole package; a control arm is a config whose name contains "
            "`baseline`.")


def check_gemini_api_key(config: dict[str, Any]) -> None:
    """Validate GEMINI_API_KEY when referenced by agent model or verifier."""
    needs_gemini = False
    for agent in config.get("agents", []):
        model = agent.get("model_name", "")
        if "gemini" in model.lower():
            needs_gemini = True
        if "GEMINI_API_KEY" in agent.get("env", {}):
            needs_gemini = True

    verifier = config.get("verifier", {})
    if "GEMINI_API_KEY" in verifier.get("env", {}):
        needs_gemini = True

    # #190 let Gemini be reached through ADC instead of an API key. A config
    # that asks for that -- AGY_ADC_AUTH, or its own ADC mount -- is already
    # authenticated, so demanding a key would reject a working setup.
    # The env may sit on an agent or on the shared environment block.
    env_blocks = [a.get("env") or {} for a in config.get("agents", [])]
    env_blocks.append((config.get("environment") or {}).get("env") or {})
    env_blocks.append(verifier.get("env") or {})
    mounts = (config.get("environment") or {}).get("mounts") or []
    has_adc_mount = any(
        isinstance(m, dict)
        and "/application_default_credentials.json" in m.get("target", "")
        for m in mounts)
    uses_adc = any(
        str(env.get("AGY_ADC_AUTH", "")).lower() == "true"
        or "GOOGLE_APPLICATION_CREDENTIALS" in env
        for env in env_blocks) or has_adc_mount or (
            os.environ.get("AGY_ADC_AUTH") == "true")
    # Under GKE_AUTH_ACCOUNT=metadata there is no ADC file: ADC resolves to the
    # metadata server instead.
    if uses_adc and (get_adc_path().is_file()
                     or config_auth_account(config) == METADATA_AUTH):
        needs_gemini = False

    if needs_gemini and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "This configuration requires Gemini access for the agent or verifier.\n"
            "Please export GEMINI_API_KEY='your-api-key' before running.")


def _task_path(raw: Any) -> Path | None:
    """Absolute task or dataset directory from a config entry."""
    path = raw.get("path") if isinstance(raw, dict) else raw
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _task_label(task_dir: Path) -> str:
    """Repo-relative path for messages, or the absolute one if outside."""
    try:
        return str(task_dir.relative_to(REPO_ROOT))
    except ValueError:
        return str(task_dir)


def _dataset_task_dirs(entry: dict[str, Any]) -> list[Path]:
    """Expand a `datasets` entry the way Harbor does.

    Every task.toml below the directory, minus `exclude_task_names` patterns,
    which Harbor matches against the task directory name.
    """
    root = _task_path(entry)
    if root is None:
        return []
    excluded = entry.get("exclude_task_names") or []
    return [
        t.parent for t in sorted(root.glob("**/task.toml"))
        if not any(fnmatch.fnmatch(t.parent.name, pat) for pat in excluded)
    ]


def _missing_required_env(task_dir: Path) -> list[str]:
    """Names in task.toml `[metadata].required_env` that are unset or empty.

    Empty counts as missing: a CI variable or secret that is not defined
    arrives as "", and `${VAR:-}` forwards "" instead of defaulting. The pod
    would otherwise be the first place to notice (e.g. a 401 on a gated model).
    """
    import tomllib

    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return []
    metadata = tomllib.loads(task_toml.read_text(encoding="utf-8")).get(
        "metadata", {})
    return [
        name for name in metadata.get("required_env", [])
        if not os.environ.get(name)
    ]


def _refuse_explicit_tasks(entries: list[Any]) -> None:
    """A task named on the CLI has nothing to fall back to: refuse the run."""
    problems = []
    for entry in entries:
        task_dir = _task_path(entry)
        if task_dir is None:
            continue
        for name in _missing_required_env(task_dir):
            problems.append(f"{_task_label(task_dir)} requires {name}, which "
                            "is unset or empty on this host; export it first.")
    if problems:
        raise RuntimeError("Task requirements not met:\n  - " +
                           "\n  - ".join(problems))


def _skip_dataset_tasks(
        entries: list[dict[str, Any]]) -> tuple[list[str], int]:
    """Exclude the dataset tasks this host cannot run, in place.

    Returns the skip notes and how many tasks are still selected.
    """
    skipped: list[str] = []
    kept = 0
    for entry in entries:
        for task_dir in _dataset_task_dirs(entry):
            missing = _missing_required_env(task_dir)
            if not missing:
                kept += 1
                continue
            entry.setdefault("exclude_task_names", []).append(task_dir.name)
            note = (f"{_task_label(task_dir)} skipped: {', '.join(missing)} "
                    "unset or empty on this host")
            print(f"WARNING: {note}. Export it to include this task.",
                  file=sys.stderr)
            skipped.append(note)
    return skipped, kept


def check_task_requirements(config: dict[str, Any]) -> list[str]:
    """Drop the tasks this host cannot run; refuse when nothing else was asked.

    Dataset tasks are skipped via `exclude_task_names` so the rest still run;
    the skips are announced and returned for the step summary so a missing
    column is not silent. Explicitly named tasks are refused, and so is a
    selection left empty, since Harbor reports success over an empty set.
    """
    tasks = config.get("tasks") or []
    _refuse_explicit_tasks(tasks)

    datasets = config.get("datasets") or []
    for i, entry in enumerate(datasets):
        # Entries may be bare path strings in YAML; exclusions need a dict.
        if not isinstance(entry, dict):
            datasets[i] = {"path": entry}
    skipped, kept = _skip_dataset_tasks(datasets)

    if skipped and not kept and not tasks:
        raise RuntimeError(
            "Nothing left to run; every selected task was skipped:\n  - " +
            "\n  - ".join(skipped))
    return skipped


def publish_skipped_tasks_summary(skipped: list[str],
                                  summary_path: Path | str | None = None
                                  ) -> None:
    """Put skipped tasks on the workflow page, not only in the run log."""
    if not skipped:
        return
    if summary_path is None:
        env_val = os.environ.get("GITHUB_STEP_SUMMARY")
        if not env_val:
            return
        summary_path = Path(env_val)
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("### Skipped tasks: host requirements not met\n\n" +
                    "".join(f"- `{note}`\n" for note in skipped) + "\n")
    except Exception as e:
        print(f"Warning: Failed to write to GITHUB_STEP_SUMMARY: {e}",
              file=sys.stderr)


def _readable_file(path: str) -> bool:
    """Whether this process can see a file at `path`.

    `is_file()` reports a missing path as False but lets EACCES through, and
    this name often points inside the container, which a host user who is not
    root cannot stat. Unreadable is as good as absent here.
    """
    try:
        return Path(path).is_file()
    except OSError:
        return False


def setup_gke_credentials_if_needed(
        config: dict[str, Any], scratch_dir: Path) -> GkeCredentials | None:
    """Mint a kubeconfig for the agents' compute cluster (GKE_CLUSTER).

    This is the cluster the agent in the trial pod submits TPU jobs to, not
    the one Harbor runs the trial pod on (that is
    setup_harbor_gke_kubeconfig_if_needed()). The kubeconfig is mounted into
    the pod, where `gke-gcloud-auth-plugin --use_application_default_credentials`
    refreshes tokens automatically from:

      adc        the mounted ADC file (GOOGLE_APPLICATION_CREDENTIALS);
      metadata   the pod's GKE metadata server;
      otherwise  the gcloud account GKE_AUTH_ACCOUNT names, else gcloud's
                 active account.

    Returns None when the config targets no GKE cluster.
    """
    env = {}
    for agent in config.get("agents", []):
        env.update(agent.get("env", {}))
    if isinstance(config.get("environment"), dict):
        env.update(config.get("environment", {}).get("env", {}))

    # Early return if backend does not use GKE
    backend = resolve_template(env.get("COMPUTE_BACKEND", "gke_job"))
    if backend in ("local", "gce_tpu_vm"):
        return None

    cluster_name = resolve_template(env.get("GKE_CLUSTER"))
    if backend not in ("gke_job", "gke_pod") and not cluster_name:
        return None
    project = resolve_template(env.get("GKE_PROJECT"))
    location = resolve_template(env.get("GKE_LOCATION"))
    # Early return if KUBECONFIG is not required and no GKE cluster is targeted
    if "KUBECONFIG" not in env and not cluster_name:
        return None

    if not cluster_name:
        raise RuntimeError(
            "Config specifies KUBECONFIG or GKE compute backend, but GKE_CLUSTER is not set in YAML or environment."
        )

    if not project or not location:
        raise RuntimeError(
            f"GKE cluster '{cluster_name}' requires GKE_PROJECT and GKE_LOCATION to be specified."
        )

    auth_account = resolve_template(
        env.get("GKE_AUTH_ACCOUNT")) or os.environ.get("GKE_AUTH_ACCOUNT")

    kubeconfig_path = scratch_dir / "kubeconfig"
    gcloud_cmd_env = os.environ.copy()
    if sys.version_info >= (3, 10):
        gcloud_cmd_env.setdefault("CLOUDSDK_PYTHON", sys.executable)
    gcloud_cmd_env["KUBECONFIG"] = str(kubeconfig_path)
    gcloud_cmd_env[
        "CLOUDSDK_CONTAINER_USE_APPLICATION_DEFAULT_CREDENTIALS"] = "true"

    gcloud_account_args = []
    auth_identity = auth_account
    if auth_account == "adc":
        adc_path = get_adc_path()
        if not adc_path.is_file():
            raise RuntimeError(
                f"GKE_AUTH_ACCOUNT=adc but no credentials found at {adc_path}. Run: gcloud auth application-default login"
            )
        # Name the credential file gcloud is to use. Keep the file
        # GOOGLE_APPLICATION_CREDENTIALS names when it exists: CI points it at
        # a service account key that can reach the cluster, which the ADC
        # (there, for Vertex) cannot. Fall back to the ADC when it names none:
        # env.txt sets it to the container's path for sandbox.sh, and
        # load_env_file_if_present copies it here, where it does not exist.
        named = gcloud_cmd_env.get("GOOGLE_APPLICATION_CREDENTIALS")
        gcloud_cmd_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = (
            named if named and _readable_file(named) else str(adc_path))
    elif auth_account == METADATA_AUTH:
        # The cluster may sit in a project the metadata server's account has
        # no role in (CI's TPU cluster, until that account is granted one).
        # GKE_AUTH_KEY_FILE then names a service account key that has, and
        # GKE_IMPERSONATE_SA names a service account to impersonate.
        key_file = os.environ.get("GKE_AUTH_KEY_FILE")
        gke_impersonate_sa = (resolve_template(env.get("GKE_IMPERSONATE_SA"))
                              or os.environ.get("GKE_IMPERSONATE_SA"))
        if key_file:
            gcloud_cmd_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = key_file
            email = json.loads(
                Path(key_file).read_text(encoding="utf-8")).get("client_email")
            auth_identity = f"{email} (GKE_AUTH_KEY_FILE)"
        elif gke_impersonate_sa:
            gcloud_cmd_env.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
            gcloud_cmd_env[
                "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] = gke_impersonate_sa
            auth_identity = f"{gke_impersonate_sa} (GKE_IMPERSONATE_SA)"
        else:
            gcloud_cmd_env.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
            auth_identity = f"{metadata_server_get('email')} (metadata server)"
        print(f"==> [auth] {cluster_name} kubeconfig auth: {auth_identity}")
    elif auth_account:
        gcloud_account_args = ["--account", auth_account]

    print(f"==> Minting single-context kubeconfig for {cluster_name}...")
    gcloud_cmd = [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        cluster_name,
        "--location",
        location,
        "--project",
        project,
    ] + gcloud_account_args
    # A cluster with master authorized networks refuses its IP endpoint to a
    # host outside them (a GCE VM in another project, say). Its DNS endpoint
    # is gated by IAM instead, so GKE_DNS_ENDPOINT=1 reaches it from anywhere
    # the account is authorized.
    if os.environ.get("GKE_DNS_ENDPOINT", "").lower() in ("1", "true"):
        gcloud_cmd.append("--dns-endpoint")
    res = subprocess.run(gcloud_cmd, env=gcloud_cmd_env)
    if res.returncode != 0:
        active_account = auth_identity or subprocess.run(
            ["gcloud", "config", "get-value", "account"],
            capture_output=True,
            text=True,
            check=False,
            env=gcloud_cmd_env,
        ).stdout.strip()
        raise RuntimeError(
            f"Could not get credentials for {cluster_name} in {project} as {active_account}.\n"
            f"  That identity needs container.clusters.get on the cluster; set GKE_AUTH_ACCOUNT to one that has it."
        )

    kubeconfig_path.chmod(0o644)
    return GkeCredentials(kubeconfig_path=kubeconfig_path)


def setup_harbor_gke_kubeconfig_if_needed(config: dict[str, Any],
                                          scratch_dir: Path) -> Path | None:
    """Mint the kubeconfig Harbor runs trial pods with, for `type: gke`.

    Harbor's GKEEnvironment talks to the current context of $KUBECONFIG, and
    its preflight refuses to start without one. Left to the host's
    ~/.kube/config, that is whichever cluster it last pointed at -- and a CI
    runner has none. This pins it to the cluster `environment.kwargs` names.
    It is not the cluster setup_gke_credentials_if_needed() mints for: that
    one is the agents' TPU cluster and its kubeconfig goes into the trial
    pod; this one stays on the host, read only by Harbor.

    The kubeconfig's user is the kubernetes client's `gcp` auth provider, not
    a bearer token: the client takes tokens from google.auth.default() -- the
    host ADC, or the metadata server under GKE_AUTH_ACCOUNT=metadata -- or
    from `gcloud config config-helper --impersonate-service-account=...` when
    HARBOR_IMPERSONATE_SA is set, and refreshes them as they expire, so a job
    that runs for hours keeps its access to the cluster.

    Returns None unless the config's environment is `type: gke`.
    """
    env_spec = config.get("environment")
    if not isinstance(env_spec, dict) or resolve_template(
            env_spec.get("type")) != "gke":
        return None
    kwargs = env_spec.get("kwargs") or {}
    cluster_name = resolve_template(kwargs.get("cluster_name"))
    location = resolve_template(kwargs.get("region"))
    project = resolve_template(kwargs.get("project_id"))
    if not (cluster_name and location and project):
        return None

    kubeconfig_path = scratch_dir / "harbor_kubeconfig"
    gcloud_cmd_env = os.environ.copy()
    if sys.version_info >= (3, 10):
        gcloud_cmd_env.setdefault("CLOUDSDK_PYTHON", sys.executable)
    gcloud_cmd_env["KUBECONFIG"] = str(kubeconfig_path)

    # get-credentials runs as the identity Harbor will: google.auth.default()
    # resolves to the same metadata server or ADC file, unless
    # HARBOR_IMPERSONATE_SA is set.
    harbor_impersonate_sa = os.environ.get("HARBOR_IMPERSONATE_SA", "")
    if harbor_impersonate_sa:
        gcloud_cmd_env.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
        gcloud_cmd_env[
            "CLOUDSDK_AUTH_IMPERSONATE_SERVICE_ACCOUNT"] = harbor_impersonate_sa
    elif config_auth_account(config) == METADATA_AUTH:
        gcloud_cmd_env.pop("CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE", None)
    else:
        gcloud_cmd_env["CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = str(
            get_adc_path())

    print(f"==> Minting Harbor kubeconfig for {cluster_name} ({project})...")
    gcloud_cmd = [
        "gcloud",
        "container",
        "clusters",
        "get-credentials",
        cluster_name,
        "--location",
        location,
        "--project",
        project,
    ]
    res = subprocess.run(gcloud_cmd, env=gcloud_cmd_env)
    if res.returncode != 0:
        raise RuntimeError(
            f"Could not get credentials for Harbor's cluster {cluster_name} in"
            f" {project} ({location}).")

    with open(kubeconfig_path, "r", encoding="utf-8") as f:
        kube_data = yaml.safe_load(f) or {}
    for user_entry in kube_data.get("users", []):
        if harbor_impersonate_sa:
            user_entry["user"] = {
                "auth-provider": {
                    "name": "gcp",
                    "config": {
                        "cmd-path":
                        "gcloud",
                        "cmd-args":
                        ("config config-helper --format=json"
                         f" --impersonate-service-account={harbor_impersonate_sa}"
                         ),
                        "token-key":
                        "{.credential.access_token}",
                        "expiry-key":
                        "{.credential.token_expiry}",
                    },
                }
            }
        else:
            user_entry["user"] = {"auth-provider": {"name": "gcp"}}
    with open(kubeconfig_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(kube_data, f)
    return kubeconfig_path


def setup_vertex_if_needed(config: dict[str, Any]) -> None:
    """Validate host ADC and sanitize environment if Claude Vertex is referenced."""
    env = dict((config.get("environment") or {}).get("env") or {})
    for agent in config.get("agents", []):
        env.update(agent.get("env", {}))

    if env.get("CLAUDE_CODE_USE_VERTEX") != "1":
        return

    project = (resolve_template(env.get("ANTHROPIC_VERTEX_PROJECT_ID"))
               or resolve_template(env.get("VERTEXAI_PROJECT"))
               or os.environ.get("VERTEXAI_PROJECT"))
    if not project:
        raise RuntimeError("VERTEXAI_PROJECT is not set in environment.")

    adc_path = get_adc_path()
    if config_auth_account(config) == METADATA_AUTH:
        # No ADC to upload, so Claude Code in the trial pod reaches Vertex AI
        # as the pod's own service account, through its metadata server.
        print("==> [auth] Vertex AI in the trial pod: pod's GKE metadata"
              " server, no ADC uploaded")
    elif not adc_path.is_file():
        raise RuntimeError(
            f"Application Default Credentials (ADC) not found at {adc_path}. Run: gcloud auth application-default login"
        )

    # Remove direct Anthropic API keys so Vertex AI routing takes precedence
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_OAUTH_TOKEN"):
        os.environ.pop(var, None)


def drop_host_adc(config: dict[str, Any]) -> None:
    """Strip the host ADC out of a config run under GKE_AUTH_ACCOUNT=metadata.

    A config that serves both auth modes (the CI ones) mounts ${HOST_ADC}
    into the trial pod and points GOOGLE_APPLICATION_CREDENTIALS at it. A
    metadata run has no ADC to mount, and a GOOGLE_APPLICATION_CREDENTIALS
    naming a file that is not there makes Google's auth libraries fail rather
    than fall back to the pod's metadata server, so both go.
    """
    env_spec = config.get("environment")
    if not isinstance(env_spec, dict):
        return
    adc_sources = {"${HOST_ADC}", "${EVAL_VERTEX_ADC}"}
    mounts = env_spec.get("mounts") or []
    adc_mounts = [
        m for m in mounts
        if isinstance(m, dict) and m.get("source") in adc_sources
    ]
    if not adc_mounts:
        return
    env_spec["mounts"] = [m for m in mounts if m not in adc_mounts]
    targets = {m.get("target") for m in adc_mounts}
    envs = [env_spec.get("env"), (config.get("verifier") or {}).get("env")]
    envs += [agent.get("env") for agent in config.get("agents", [])]
    for env in envs:
        if not isinstance(env, dict):
            continue
        for var in ("GOOGLE_APPLICATION_CREDENTIALS",
                    "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"):
            if env.get(var) in targets:
                del env[var]
    print("==> [auth] no host ADC mounted into the trial pod")


# Harbor expands ${VAR} under `agents[].env` but not under
# `environment.mounts`, so those paths have to be rendered before Harbor sees
# the config. Only this name -- everything under agents[].env is Harbor's to
# expand. run.sh does the same thing with envsubst; without the equivalent
# here, a config run through run.py gets a literal "${EVAL_KUBECONFIG}" as a
# bind source.
#
# EVAL_VERTEX_ADC is mounted by the hf-onboarding config and was missing from
# this list, so that bind source reached Harbor unexpanded. Unlike the
# kubeconfig it is not minted per run -- get_adc_path() resolves it, and
# main() exports the result so the substitution below has something to find.
#
# The rest were exported by the eval_run.sh chain, removed in this change.
# are produced by setup_host_auth() and stage_package_if_needed() below, so a
# config mounting them no longer needs that chain to have run first.
_MOUNT_VARS = (
    "EVAL_KUBECONFIG",
    "EVAL_VERTEX_ADC",
    "HOST_ADC",
    "HOST_KUBECONFIG",
    "STAGED_WORKSPACE",
    "HOST_CENTRAL_PKG",
    "HOST_CENTRAL_PKG_TARGET",
    "HOST_AGY_SESSION_SUPERVISOR_HOOK",
    "HOST_AGY_SESSION_SUPERVISOR",
    "HOST_CLAUDE_SESSION_SUPERVISOR_HOOK",
    "HOST_CLAUDE_SESSION_SUPERVISOR",
    "QUOTA_PROJECT",
)

HARNESS_PINS_FILE = REPO_ROOT / ".ci" / "pins.conf"


def export_harness_pins() -> None:
    """Export the harness CLI pins in .ci/pins.conf to the environment.

    The claude-code configs install `${CLAUDE_CODE_VERSION}`, so every trial
    runs the Claude Code release the CI E2E job pins instead of npm's latest.
    A value already in the environment wins, so one run can try another
    release without a checkout change. agy needs no export: the eval adapter
    reads the same file to install it.
    """
    for key, value in read_harness_pins(HARNESS_PINS_FILE).items():
        os.environ.setdefault(key, value)


def export_git_ref() -> None:
    """Export the repo commit for configs to pass to the agent.

    A trial records the task's content digest but nothing about the repo that
    produced the agent package, so a result cannot be traced back to the code
    under test. `-dirty` is appended for uncommitted changes to tracked files:
    the commit alone would otherwise claim a state that was never run. Only
    the commit: a branch name can name the product to the baseline agent, and
    the branch is recoverable from the commit. A value already set (CI, a
    release tag) is kept, without asking git.
    """
    if "EVAL_GIT_COMMIT" in os.environ:
        return

    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True,
                text=True,
                check=True)
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return ""
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    if commit and git("status", "--porcelain", "--untracked-files=no"):
        commit += "-dirty"
    os.environ["EVAL_GIT_COMMIT"] = commit


def setup_metadata_auth() -> None:
    """Check the run can authenticate as the metadata server, and only as it.

    For GKE_AUTH_ACCOUNT=metadata (CI). Refuses to start with a host ADC
    present, since google.auth.default() would prefer it to the metadata
    server, and unsets the variables that point gcloud or google.auth at a
    key file, so nothing falls back to one.
    """
    adc = get_adc_path()
    if adc.is_file():
        raise RuntimeError(
            f"GKE_AUTH_ACCOUNT={METADATA_AUTH}, but ADC exists at {adc}, and"
            " Google's auth libraries would use it before the metadata"
            " server. Remove it, or set GKE_AUTH_ACCOUNT=adc to run on ADC.")
    try:
        email = metadata_server_get("email")
        metadata_access_token()
    except (OSError, KeyError, ValueError) as e:
        raise RuntimeError(
            f"GKE_AUTH_ACCOUNT={METADATA_AUTH}, but no token from the GCE/GKE"
            f" metadata server ({e}). Run on GKE/GCE, or set"
            " GKE_AUTH_ACCOUNT=adc to run on ADC.") from e
    for var in ("GOOGLE_APPLICATION_CREDENTIALS",
                "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE",
                "GOOGLE_GHA_CREDS_PATH"):
        if os.environ.pop(var, None):
            print(f"==> [auth] unset {var}: not using its key file")
    gcloud_account = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
        check=False).stdout.strip()
    print(f"==> [auth] GCP auth mode: GKE metadata server, no ADC\n"
          f"==> [auth]   metadata server service account: {email}\n"
          f"==> [auth]   gcloud active account: {gcloud_account or '(none)'}")
    if gcloud_account and gcloud_account != email:
        print(f"::warning::gcloud's active account {gcloud_account} is not"
              f" the metadata server's {email}; ambient gcloud calls use it")


def export_host_adc(quota_project: str) -> None:
    """Export the host ADC as HOST_ADC, with `quota_project` as its quota project.

    ADC without a quota_project_id is rejected by some Google APIs, so a copy
    carrying one is written to a temp file and mounted instead of the
    original -- the same fixup env/host_auth.sh performed.
    """
    adc = get_adc_path()
    if not adc.is_file():
        raise RuntimeError(
            f"No Application Default Credentials found at {adc}.\n"
            "Run: gcloud auth application-default login")

    data = json.loads(adc.read_text(encoding="utf-8"))
    data["quota_project_id"] = quota_project
    resolved = Path(tempfile.mkdtemp(prefix="eval-adc-")) / "adc.json"
    resolved.write_text(json.dumps(data, indent=2), encoding="utf-8")
    resolved.chmod(0o644)
    print(
        f"==> wrote ADC with quota_project_id ({quota_project}) to {resolved}")

    os.environ["HOST_ADC"] = str(resolved)
    # Same file, under the name the hf-onboarding config mounts.
    os.environ.setdefault("EVAL_VERTEX_ADC", str(resolved))


def setup_host_auth(metadata_auth: bool = False) -> None:
    """Resolve host credentials and kubeconfig, exporting the paths configs mount.

    With `metadata_auth` (GKE_AUTH_ACCOUNT=metadata) no ADC is read or
    exported -- see setup_metadata_auth(); otherwise the host ADC is, as
    HOST_ADC -- see export_host_adc().
    """
    vertex_project = (os.environ.get("VERTEXAI_PROJECT") or "tpu-nexus-767207")
    os.environ["VERTEXAI_PROJECT"] = vertex_project
    os.environ["QUOTA_PROJECT"] = vertex_project

    if metadata_auth:
        setup_metadata_auth()
    else:
        export_host_adc(vertex_project)

    kubeconfig = Path(
        os.environ.get("KUBECONFIG") or Path.home() / ".kube" / "config")
    os.environ["HOST_KUBECONFIG"] = str(
        kubeconfig) if kubeconfig.is_file() else "/dev/null"

    os.environ.setdefault("GEMINI_CLI_TRUST_WORKSPACE", "true")
    if sys.version_info >= (3, 10):
        os.environ.setdefault("CLOUDSDK_PYTHON", sys.executable)

    hooks = REPO_ROOT / "eval" / "hooks" / "agy"
    if hooks.is_dir():
        os.environ["HOST_AGY_SESSION_SUPERVISOR_HOOK"] = str(
            hooks / "session_supervisor_hook.sh")
        os.environ["HOST_AGY_SESSION_SUPERVISOR"] = str(
            hooks / "session_supervisor.sh")

    claude_hooks = REPO_ROOT / "eval" / "hooks" / "claude_code"
    if claude_hooks.is_dir():
        os.environ["HOST_CLAUDE_SESSION_SUPERVISOR_HOOK"] = str(
            claude_hooks / "session_supervisor_hook.sh")
        os.environ["HOST_CLAUDE_SESSION_SUPERVISOR"] = str(
            claude_hooks / "session_supervisor.sh")


def apply_mount_updates(config: dict[str, Any],
                        updates: MountUpdates) -> dict[str, Any]:
    """Explicitly apply MountUpdates (mounts and environment variables) to config."""
    if not updates.mounts and not updates.env:
        return config

    env_spec = config.setdefault("environment", {})
    existing_mounts = env_spec.setdefault("mounts", [])
    existing_targets = {
        m.get("target")
        for m in existing_mounts if isinstance(m, dict) and "target" in m
    }
    for m in updates.mounts:
        if m.get("target") not in existing_targets:
            existing_mounts.append(m)
            existing_targets.add(m.get("target"))

    env_vars = env_spec.setdefault("env", {})
    for k, v in updates.env.items():
        env_vars.setdefault(k, v)

    agents = config.get("agents") or []
    if agents and isinstance(agents[0], dict):
        agent_env = agents[0].setdefault("env", {})
        for k, v in updates.env.items():
            agent_env.setdefault(k, v)

    return config


def stage_package_if_needed(config: dict[str, Any],
                            config_path: Path,
                            skills: list[str] | None = None) -> MountUpdates:
    """Stage the whole package, `skills` alone, or nothing for a control arm.

    Does not modify the input config dict. Call apply_mount_updates(config, updates)
    at the call site to apply the returned mounts and environment variables.
    """
    harness = harness_for_config(config)
    agents = config.get("agents") or []
    first_agent_env = agents[0].get("env", {}) if agents else {}
    resolved_agent_env = {
        k: resolve_template(v) if isinstance(v, str) else v
        for k, v in first_agent_env.items()
    }
    compute = collect_task_environment_answers(resolved_agent_env)
    control_arm = is_control_arm(config_path)
    # A control arm installs nothing, so its PACKAGE_MANIFEST is not read.
    manifest = RELEASE_MANIFEST
    if not control_arm:
        try:
            manifest = resolve_manifest(package_manifest(resolved_agent_env),
                                        REPO_ROOT)
        except ValueError as exc:
            raise RuntimeError(f"{config_path}: {exc}") from exc

    staging_result = stage_composite_package(
        harness,
        control_arm=control_arm,
        skills=skills,
        compute_answers=compute,
        repo_root=REPO_ROOT,
        manifest=manifest,
    )

    # Dynamically inject mounts and environment variables (session supervisor by default)
    pkg_name, pkg_version = package_identity(manifest, REPO_ROOT)
    advisor = SkillMountAdvisor(
        staged_workspace=staging_result.staged_workspace,
        harness=harness,
        central_pkg_dir=staging_result.central_pkg_dir,
        repo_root=REPO_ROOT,
        pkg_name=pkg_name,
        pkg_version=pkg_version,
        is_empty=staging_result.is_empty,
        user_scope=installs_at_user_scope(config),
        staged_dist=staging_result.staged_dist,
        answers_file=staging_result.staged_root / "answers.json",
    )
    return advisor.get_updates()


_TEMPLATE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_templates(node: Any) -> Any:
    """Resolve ${VAR} and ${VAR:-default} in every string value, keeping types.

    An unset variable with no default is left as written.
    """
    if isinstance(node, dict):
        return {k: _expand_templates(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_templates(v) for v in node]
    if isinstance(node, str):
        return _TEMPLATE.sub(
            lambda m: os.environ.get(
                m.group(1),
                m.group(2) if m.group(2) is not None else m.group(0)), node)
    return node


def render_config(config: dict[str, Any], config_path: Path,
                  scratch_dir: Path) -> Path:
    """Write the resolved config for Harbor to read.

    Harbor reads a file, not our in-memory config, so anything decided here
    has to be written back out -- notably the dataset chosen by --dataset,
    which no config on disk carries. Always writing also keeps the file Harbor
    runs identical to the config this run actually resolved.

    Mount paths are substituted on the way: Harbor expands ${VAR} under
    agents[].env but not under environment.mounts.
    """
    # `compute:` and `version` are ours, not Harbor's, and Harbor's JobConfig
    # forbids extra keys. Nothing reads `compute:` any more (the profile comes
    # from the agent env); it is dropped so an old config still runs.
    harbor_config = {
        k: v
        for k, v in config.items() if k not in (
            "compute",
            "version",
        )
    }

    def _norm(p_str: str) -> str:
        p = Path(p_str)
        if not p.is_absolute() and not (REPO_ROOT / p).exists():
            candidate = REPO_ROOT / "eval" / p
            if candidate.exists():
                return str(candidate.relative_to(REPO_ROOT))
        return p_str

    for key in ("datasets", "tasks"):
        entries = harbor_config.get(key)
        if isinstance(entries, list):
            harbor_config[key] = [{
                **e, "path": _norm(e["path"])
            } if isinstance(e, dict) and "path" in e else
                                  _norm(e) if isinstance(e, str) else e
                                  for e in entries]

    content = yaml.safe_dump(harbor_config, sort_keys=False)

    for name in _MOUNT_VARS:
        token = f"${{{name}}}"
        if token not in content:
            continue
        value = os.environ.get(name)
        if not value:
            raise RuntimeError(
                f"{config_path} mounts {token}, but {name} is unset. It is"
                f" exported by the setup above, which only runs when the"
                f" config asks for it: KUBECONFIG under agents[].env is what"
                f" mints the kubeconfig.")
        content = content.replace(token, value)

    # Expand any remaining ${VAR:-default} templates on the parsed values, not
    # the text. Substituting an empty default into raw YAML turns
    # `GKE_QUEUE: ${GKE_QUEUE:-}` into `GKE_QUEUE:`, which parses as null, and
    # Harbor's JobConfig rejects a null env value.
    content = yaml.safe_dump(_expand_templates(yaml.safe_load(content)),
                             sort_keys=False)

    rendered = scratch_dir / config_path.name
    rendered.write_text(content, encoding="utf-8")
    return rendered


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for eval runner."""
    parser = argparse.ArgumentParser(
        description="Evaluation runner for TPU agent suite using Harbor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        action="extend",
        help="Config name(s) under eval/configs/, without .yaml.")
    parser.add_argument(
        "--cloud-build",
        action="store_true",
        help=("Let Harbor build task images with Cloud Build. The default is "
              "to build locally and push, which needs only registry push "
              "rights where Cloud Build needs four separate grants."),
    )
    parser.add_argument(
        "-s",
        "--suite",
        help=
        ("YAML evaluation suite to run, by name under eval/suites/ or direct path. "
         "A directory of task YAMLs (e.g. cset/tpu_knowledge_qa) is one dataset. "
         "Automatically compiled to eval/.datasets/ on demand."),
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="+",
        action="extend",
        help=("One or more tasks to run as a single job, by name under "
              "eval/datasets/, as <dataset>/<task>, or by path. Scope a "
              "bare name with --dataset or --suite when it exists in more "
              "than one. A --suite file takes only one; a --suite "
              "directory takes any number."),
    )
    parser.add_argument(
        "-d",
        "--dataset",
        help=
        ("Dataset to run, by name under eval/datasets/ or by path. "
         "Configs name the system under test only, so --suite or --dataset is required."
         ),
    )
    parser.add_argument("-f",
                        "--config-file",
                        nargs="+",
                        action="extend",
                        type=Path,
                        help="Direct path(s) to job config YAML.")
    parser.add_argument(
        "--skills",
        nargs="+",
        action="extend",
        help=("Debugging only: install just these skills (comma or space "
              "separated) instead of the whole package. Not with a baseline "
              "config, and not in CI; the job name says `-subset-`."),
    )
    parser.add_argument("--job-name",
                        help="Custom job name (overrides job_name in YAML).")
    parser.add_argument(
        "--upload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=
        "Upload evaluation results to Harbor (default: enabled; use --no-upload to disable).",
    )
    parser.add_argument(
        "--report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=
        "Generate local markdown report in eval/results/ (default: enabled; use --no-report to disable).",
    )
    parser.add_argument(
        "--share-org",
        default="google-pytorch",
        help=
        "Organization to share upload results with (default: 'google-pytorch').",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1.0,
        help=
        "Minimum score required across all evaluated trials to pass (default: 1.0).",
    )
    parser.add_argument(
        "--stream-logs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=
        "Stream Harbor job logs and trial status in real time (default: false).",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help=
        "Heartbeat progress interval in seconds during evaluation (default: 30.0; 0 to disable).",
    )
    return parser


def resolve_target_slug(config: dict[str, Any],
                        args: argparse.Namespace) -> str:
    """Extract a concise target identifier (task, suite, or dataset) using hyphens."""
    name = None
    tasks = task_specs(getattr(args, "task", None))
    if len(tasks) == 1:
        name = Path(tasks[0]).name
    elif tasks:
        name = f"{len(tasks)}-tasks"
    elif getattr(args, "suite", None):
        name = Path(args.suite).stem
    elif getattr(args, "dataset", None):
        name = Path(args.dataset).name
    else:
        for key in ("tasks", "datasets"):
            entries = config.get(key) or []
            if entries:
                entry = entries[0]
                path = entry.get("path") if isinstance(entry, dict) else entry
                if path:
                    name = Path(path).name
                    break
    if not name:
        name = "eval"
    return name.replace("_", "-")


_SUT_HARNESS = {"claude-code": "cc", "antigravity": "agy"}


def resolve_sut_slug(config: dict[str, Any], config_path: Path) -> str:
    """Classify the system under test as `{cc|agy}-{agent-suite|baseline}`."""
    harness = harness_for_config(config)
    harness = _SUT_HARNESS.get(harness, harness)
    arm = "baseline" if is_control_arm(config_path) else "agent-suite"
    return f"{harness}-{arm}"


def _normalize_accelerator_slug(raw_accel: str) -> str:
    """Normalize raw TPU accelerator string to canonical slug (v7, v6e, v5p, v5e, v4)."""
    accel = raw_accel.strip().lower()
    if not accel:
        return ""
    if "tpu7" in accel or "v7" in accel:
        return "v7"
    if "v6" in accel or "trillium" in accel:
        return "v6e"
    if "v5p" in accel:
        return "v5p"
    if "v5e" in accel or "v5-lite" in accel:
        return "v5e"
    if "v4" in accel:
        return "v4"
    return re.sub(r"[^a-zA-Z0-9\-]", "", accel)


def _describe_gce_tpu_vm_accelerator(tpu_name: str, zone: str,
                                     project: str) -> str:
    """Query gcloud compute tpus tpu-vm describe for acceleratorType if configured."""
    if not (tpu_name and zone and project):
        return ""
    try:
        proc = subprocess.run(
            [
                "gcloud",
                "compute",
                "tpus",
                "tpu-vm",
                "describe",
                tpu_name,
                f"--zone={zone}",
                f"--project={project}",
                "--format=value(acceleratorType)",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except Exception:
        pass
    return ""


def resolve_accelerator_slug(config: dict[str, Any]) -> str:
    """Resolve TPU accelerator generation slug (e.g., v6e, v7, v5e, etc.)."""
    explicit = os.environ.get("EVAL_ACCELERATOR") or os.environ.get(
        "ACCELERATOR")
    if explicit:
        return _normalize_accelerator_slug(explicit)

    agents = config.get("agents") or []
    agent_env = (agents[0].get("env", {})
                 if agents and isinstance(agents[0], dict) else {})

    raw_explicit = agent_env.get("EVAL_ACCELERATOR") or agent_env.get(
        "ACCELERATOR")
    if raw_explicit:
        expanded = _expand_templates(raw_explicit)
        if expanded:
            return _normalize_accelerator_slug(expanded)

    backend = (_expand_templates(agent_env.get("COMPUTE_BACKEND", "gke_job"))
               or os.environ.get("COMPUTE_BACKEND")
               or "gke_job").strip().lower()

    if backend == "gke_job":
        if os.environ.get("GKE_TPU_ACCELERATOR"):
            return _normalize_accelerator_slug(
                os.environ["GKE_TPU_ACCELERATOR"])
        cluster = (_expand_templates(agent_env.get("GKE_CLUSTER", ""))
                   or os.environ.get("GKE_CLUSTER", "")).lower()
        if any(k in cluster
               for k in ("tpu7", "v7", "bodaborg", "shared-capacity")):
            return "v7"
        raw_gke = agent_env.get("GKE_TPU_ACCELERATOR")
        if raw_gke:
            return _normalize_accelerator_slug(_expand_templates(raw_gke))
        return ""

    if backend == "gce_tpu_vm":
        for candidate in (
                os.environ.get("GCE_TPU_ACCELERATOR"),
                _expand_templates(agent_env.get("GCE_TPU_ACCELERATOR", "")),
                os.environ.get("TPU_ACCELERATOR_TYPE"),
                _expand_templates(agent_env.get("TPU_ACCELERATOR_TYPE", "")),
        ):
            if candidate:
                return _normalize_accelerator_slug(candidate)
        tpu_name = (_expand_templates(agent_env.get("GCE_TPU_NAME", ""))
                    or os.environ.get("GCE_TPU_NAME", ""))
        slug_from_name = _normalize_accelerator_slug(tpu_name)
        if slug_from_name in ("v7", "v6e", "v5p", "v5e", "v4"):
            return slug_from_name
        zone = (_expand_templates(agent_env.get("GCE_LOCATION", ""))
                or os.environ.get("GCE_LOCATION", ""))
        project = (_expand_templates(agent_env.get("GCE_PROJECT", ""))
                   or os.environ.get("GCE_PROJECT", ""))
        described = _describe_gce_tpu_vm_accelerator(tpu_name, zone, project)
        if described:
            return _normalize_accelerator_slug(described)
        return ""

    if backend == "local":
        tpu_type = (_expand_templates(agent_env.get("TPU_ACCELERATOR_TYPE",
                                                    ""))
                    or os.environ.get("TPU_ACCELERATOR_TYPE", ""))
        return _normalize_accelerator_slug(tpu_type) if tpu_type else ""

    for candidate in (
            os.environ.get("GKE_TPU_ACCELERATOR"),
            _expand_templates(agent_env.get("TPU_ACCELERATOR_TYPE", "")),
            os.environ.get("TPU_ACCELERATOR_TYPE"),
            _expand_templates(agent_env.get("GKE_CLUSTER", "")),
            _expand_templates(agent_env.get("GKE_POD", "")),
    ):
        if candidate:
            slug = _normalize_accelerator_slug(candidate)
            if slug in ("v7", "v6e", "v5p", "v5e", "v4"):
                return slug
    return ""


def resolve_env_slug() -> str:
    """Resolve environment slug: gh-<run_id> or local."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if run_id:
        return f"gh-{run_id}"
    return "local"


def generate_job_name(
    config: dict[str, Any],
    config_path: Path,
    args: argparse.Namespace,
    timestamp: str | None = None,
    max_length: int = 128,
    skills: list[str] | None = None,
) -> str:
    """Generate a clean, descriptive Harbor job name: {target}_{sut}_{env}_{timestamp}."""
    ts = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    if getattr(args, "job_name", None):
        return f"{args.job_name}_{ts}"

    target = resolve_target_slug(config, args)
    sut = resolve_sut_slug(config, config_path)
    manifest = config_package_manifest(config)
    if manifest != RELEASE_MANIFEST and not is_control_arm(config_path):
        # Another package is another system under test; say so where the
        # dashboards key on it, so its runs do not join the release ones.
        sut += f"-{package_label(manifest)}"
    if skills:
        # A subset is not the system the agent-suite arm measures; say so where the
        # dashboards key on it.
        sut += "-subset-" + "-".join(s.removeprefix("tpu-") for s in skills)
    accel = resolve_accelerator_slug(config)
    if accel and not sut.endswith(f"-{accel}"):
        sut += f"-{accel}"
    env = resolve_env_slug()

    raw_name = f"{target}_{sut}_{env}_{ts}"
    # Sanitize invalid characters
    sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "-", raw_name)
    if len(sanitized) > max_length:
        excess = len(sanitized) - max_length
        target_trunc = target[:-excess] if len(
            target) > excess + 8 else target[:16]
        sanitized = f"{target_trunc}_{sut}_{env}_{ts}"
    return sanitized


class HarborJobMonitor:
    """Monitors Harbor job progress and prints concise per-trial status updates and heartbeats."""

    def __init__(
        self,
        job_dir: Path,
        job_name: str,
        heartbeat_interval_sec: float = 30.0,
    ) -> None:
        self.job_dir = job_dir
        self.job_name = job_name
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stages: dict[str, str] = {}
        self._hub_url_printed: bool = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._poll()

    def _poll(self) -> None:
        if not self.job_dir.is_dir():
            return
        if not self._hub_url_printed:
            try:
                hub_url = get_harbor_hub_url(self.job_dir)
                if hub_url:
                    self._hub_url_printed = True
                    print(f"==> Harbor Hub URL: {hub_url}", flush=True)
            except Exception:
                pass
        for d in sorted(self.job_dir.iterdir()):
            if not d.is_dir() or d.name.startswith("."):
                continue

            name = d.name
            prev = self._stages.get(name)

            if (d / "result.json").is_file():
                if prev != "finished":
                    self._stages[name] = "finished"
                    try:
                        res = _safe_load_json(d / "result.json")
                        exc = res.get("exception_info")
                        if exc:
                            err = exc.get("type") or exc.get(
                                "exception_type") or "Error"
                            print(f"❌ [Trial] {name}: failed ({err})",
                                  flush=True)
                        else:
                            rew = (res.get("verifier_result")
                                   or {}).get("rewards") or {}
                            score = rew.get("reward")
                            score_str = f"{score:.2f}" if isinstance(
                                score, (int, float)) else str(rew or "none")
                            print(
                                f"✅ [Trial] {name}: finished (reward: {score_str})",
                                flush=True)
                    except Exception:
                        print(f"✅ [Trial] {name}: finished", flush=True)
            elif (d / "verifier").is_dir() and any((d / "verifier").iterdir()):
                if prev != "verifier":
                    self._stages[name] = "verifier"
                    print(f"🔍 [Trial] {name}: running verifier...", flush=True)
            elif (d / "agent").is_dir() and any((d / "agent").iterdir()):
                if prev not in ("agent", "verifier"):
                    self._stages[name] = "agent"
                    print(f"🚀 [Trial] {name}: running agent...", flush=True)
            elif (d / "config.json").is_file() or (d / "trial.log").is_file():
                if prev is None:
                    self._stages[name] = "env"
                    print(f"⏳ [Trial] {name}: starting environment...",
                          flush=True)

    def _run(self) -> None:
        start = time.time()
        last_hb = start
        while not self._stop_event.is_set():
            self._poll()
            now = time.time()
            if self.heartbeat_interval_sec > 0 and (
                    now - last_hb >= self.heartbeat_interval_sec):
                last_hb = now
                m, s = int((now - start) // 60), int((now - start) % 60)
                running = [
                    f"{k}: {v}" for k, v in self._stages.items()
                    if v != "finished"
                ]
                active_str = f"Running ({len(running)}): [{', '.join(running)}]" if running else "Running: 0"
                done = sum(1 for v in self._stages.values() if v == "finished")
                try:
                    print(
                        f"⏳ [Eval Heartbeat] Elapsed: {m}m {s:02d}s | Done: {done} | {active_str} | Job: {self.job_name}",
                        flush=True,
                    )
                except (BrokenPipeError, OSError):
                    return
            time.sleep(1.0)


def _handle_sigterm(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def main() -> None:
    try:
        signal.signal(signal.SIGTERM, _handle_sigterm)
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(line_buffering=True)
                sys.stderr.reconfigure(line_buffering=True)
            except Exception:
                pass
        check_dependencies()

        parser = build_parser()
        args, harbor_args = parser.parse_known_args()
        harbor_args = list(harbor_args)
        config_paths = resolve_config_files(args, harbor_args)
        # One credential per run: metadata auth refuses a host ADC that ADC
        # configs in the same run would need.
        auth_modes = {
            config_auth_account(load_config(p)) == METADATA_AUTH
            for p in config_paths
        }
        if len(auth_modes) > 1:
            raise RuntimeError(
                f"Cannot mix GKE_AUTH_ACCOUNT={METADATA_AUTH} configs with"
                " ADC configs in one run.")
        metadata_auth = auth_modes == {True}
        export_git_ref()
        export_harness_pins()
        setup_host_auth(metadata_auth=metadata_auth)

        # Resolve jobs directory (supports optional -o / --jobs-dir override in harbor_args)
        jobs_dir = REPO_ROOT / "jobs"
        for i, arg in enumerate(harbor_args):
            if arg in ("-o", "--jobs-dir") and i + 1 < len(harbor_args):
                jobs_dir = Path(harbor_args[i + 1]).resolve()
            elif arg.startswith("--jobs-dir="):
                jobs_dir = Path(arg.split("=", 1)[1]).resolve()

        # Upfront preflight check for all configs
        loaded_configs = []
        target_suite_dir: Path | None = None
        tasks = task_specs(args.task)
        if args.suite:
            suite = resolve_suite(args.suite)
            if len(tasks) > 1 and not suite.is_dir():
                raise RuntimeError("--suite takes a single --task with a "
                                   f"suite file; got {len(tasks)}: "
                                   f"{', '.join(tasks)}")
            target_suite_dir = compile_suite_if_needed(
                suite, task_filter=tasks[0] if tasks else None)

        if args.skills and os.environ.get("GITHUB_RUN_ID"):
            raise RuntimeError(
                "--skills is for debugging; CI runs the whole package")

        for config_path in config_paths:
            config = load_config(config_path)
            reject_removed_keys(config, config_path)
            subset = None
            if args.skills:
                if is_control_arm(config_path):
                    raise RuntimeError(
                        f"--skills chooses part of the package, and "
                        f"{config_path} is a control arm, which installs "
                        "nothing")
                # Checked against the package this config installs, so
                # PACKAGE_MANIFEST=package.experimental.json can choose the
                # skills only it ships.
                try:
                    subset = skill_subset(
                        args.skills,
                        released_skills(
                            REPO_ROOT,
                            resolve_manifest(config_package_manifest(config),
                                             REPO_ROOT)))
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
            if target_suite_dir:
                if tasks:
                    apply_tasks(config, tasks, dataset=str(target_suite_dir))
                else:
                    apply_dataset(config, str(target_suite_dir))
            elif tasks:
                apply_tasks(config, tasks, args.dataset)
            elif args.dataset:
                apply_dataset(config, args.dataset)
            require_dataset(config, config_path)
            publish_skipped_tasks_summary(check_task_requirements(config))
            run_dataset_setup_if_present(config)
            if not args.cloud_build:
                prebuild_images_locally(config)
            check_gemini_api_key(config)
            updates = stage_package_if_needed(config, config_path, subset)
            job_name = generate_job_name(
                config=config,
                config_path=config_path,
                args=args,
                skills=subset,
            )
            apply_mount_updates(config, updates)
            setup_vertex_if_needed(config)
            # Each config's own subset: it was checked against its package.
            loaded_configs.append((config_path, config, job_name, subset))

        results_summary: list[dict[str, Any]] = []
        overall_success = True

        for idx, (config_path, config, job_name,
                  subset) in enumerate(loaded_configs, 1):

            prefix = f"[{idx}/{len(config_paths)}] " if len(
                config_paths) > 1 else ""
            print(
                f"\n{prefix}Running Harbor Evaluation ({config_path}) as '{job_name}'..."
            )

            job_dir = jobs_dir / job_name
            eval_passed = False
            upload_succeeded = False
            trial_summaries: list[dict[str, Any]] = []

            with tempfile.TemporaryDirectory() as scratch_dir:
                updates = stage_package_if_needed(config, config_path, subset)
                apply_mount_updates(config, updates)
                setup_vertex_if_needed(config)
                if metadata_auth:
                    drop_host_adc(config)
                gke_creds = setup_gke_credentials_if_needed(
                    config, Path(scratch_dir))
                if gke_creds:
                    os.environ["EVAL_KUBECONFIG"] = str(
                        gke_creds.kubeconfig_path)
                harbor_kubeconfig = setup_harbor_gke_kubeconfig_if_needed(
                    config, Path(scratch_dir))

                # Must follow the credential setup above: it exports the paths the
                # mounts name.
                rendered_config = render_config(config, config_path,
                                                Path(scratch_dir))

                harbor_bin = shutil.which("harbor") or "harbor"
                cmd = [
                    harbor_bin,
                    "run",
                    "-c",
                    str(rendered_config),
                    "--job-name",
                    job_name,
                ]
                if (args.upload and "--print-config" not in harbor_args
                        and "--dry-run" not in harbor_args):
                    cmd.extend(["--upload", "--private", "--yes"])
                    if args.share_org:
                        cmd.extend(["--share-org", args.share_org])
                cmd.extend([a for a in harbor_args if a != "--"])

                harbor_env = os.environ.copy()
                # `gcloud builds submit` otherwise runs as the ambient account
                # (the instance SA on a GCE VM).
                if (config_auth_account(config) or "").lower() == "adc":
                    adc_path = get_adc_path()
                    if adc_path.is_file():
                        harbor_env[
                            "CLOUDSDK_AUTH_CREDENTIAL_FILE_OVERRIDE"] = str(
                                adc_path)
                if harness_for_config(config) == "antigravity":
                    mounts = (config.get("environment")
                              or {}).get("mounts") or []
                    has_adc_configured = (
                        get_adc_path().is_file() or any(
                            isinstance(m, dict)
                            and "/application_default_credentials.json" in
                            m.get("target", "") for m in mounts)
                        or any("GOOGLE_APPLICATION_CREDENTIALS" in (
                            a.get("env") or {})
                               for a in config.get("agents", []))
                        or os.environ.get("AGY_ADC_AUTH") == "true")
                    if has_adc_configured:
                        harbor_env["AGY_ADC_AUTH"] = "true"
                        if get_adc_path().is_file():
                            harbor_env["GOOGLE_APPLICATION_CREDENTIALS"] = str(
                                get_adc_path())
                        quota_proj = (os.environ.get("VERTEXAI_PROJECT")
                                      or "tpu-nexus-767207")
                        harbor_env.setdefault("GOOGLE_CLOUD_QUOTA_PROJECT",
                                              quota_proj)
                        harbor_env.pop("GEMINI_API_KEY", None)
                        harbor_env.pop("GOOGLE_API_KEY", None)
                if harbor_kubeconfig:
                    harbor_env["KUBECONFIG"] = str(harbor_kubeconfig)
                if metadata_auth:
                    # Harbor's gcloud calls (Cloud Build, Artifact Registry)
                    # run as the metadata server's account, which gcloud
                    # refreshes itself. A service account's calls bill to its
                    # own project -- the runner's, which has not enabled those
                    # APIs -- not to --project, unless told otherwise.
                    quota_project = resolve_template(
                        ((config.get("environment") or {}).get("kwargs")
                         or {}).get("project_id"))
                    if quota_project:
                        harbor_env[
                            "CLOUDSDK_BILLING_QUOTA_PROJECT"] = quota_project
                        print(f"==> [auth] Harbor gcloud quota project:"
                              f" {quota_project}")
                # The GKE lifecycle adapter patches GKEEnvironment.start to
                # upload environment.mounts into the remote pod. Harbor imports
                # it via sitecustomize, so it only takes effect if adapter/ is
                # on the subprocess PYTHONPATH.
                adapter_dir = REPO_ROOT / "eval" / "adapter"
                if adapter_dir.is_dir():
                    harbor_env["PYTHONPATH"] = os.pathsep.join(
                        [str(adapter_dir),
                         harbor_env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
                current_pp = harbor_env.get("PYTHONPATH", "")
                root_str = str(REPO_ROOT)
                if root_str not in current_pp.split(os.pathsep):
                    harbor_env[
                        "PYTHONPATH"] = f"{root_str}:{current_pp}" if current_pp else root_str

                # Scan config for AGY_ADC_AUTH: if true, export to Harbor's execution environment.
                # This allows Antigravity CLI to run in enterprise ADC mode without declaring
                # AGY_ADC_AUTH under agents[].env (which triggers Harbor's secret scrubber
                # and corrupts boolean values like "success": true in trajectory.json).
                raw_config_text = config_path.read_text(
                    encoding="utf-8") if config_path.is_file() else ""
                has_agy_adc = False
                if "AGY_ADC_AUTH" in raw_config_text:
                    for key in ("environment", "verifier"):
                        val = str((config.get(key)
                                   or {}).get("env",
                                              {}).get("AGY_ADC_AUTH",
                                                      "")).lower()
                        if val in ("true", "1", "yes"):
                            has_agy_adc = True
                    for ag in config.get("agents", []):
                        val = str(ag.get("env", {}).get("AGY_ADC_AUTH",
                                                        "")).lower()
                        if val in ("true", "1", "yes"):
                            has_agy_adc = True
                        val_kw = str(ag.get("kwargs", {}).get("adc_auth",
                                                              "")).lower()
                        if val_kw in ("true", "1", "yes"):
                            has_agy_adc = True
                if has_agy_adc:
                    harbor_env["AGY_ADC_AUTH"] = "true"

                interrupted = False
                if args.stream_logs:
                    monitor = HarborJobMonitor(
                        job_dir=job_dir,
                        job_name=job_name,
                        heartbeat_interval_sec=args.heartbeat_interval,
                    )
                    monitor.start()
                    try:
                        res = subprocess.run(cmd,
                                             env=harbor_env,
                                             cwd=REPO_ROOT)
                    except KeyboardInterrupt:
                        print(
                            "\n==> Interrupted (SIGINT/SIGTERM); salvaging completed trial results...",
                            file=sys.stderr,
                            flush=True,
                        )
                        interrupted = True
                        res = subprocess.CompletedProcess(cmd, returncode=130)
                    finally:
                        monitor.stop()
                else:
                    try:
                        res = subprocess.run(cmd,
                                             env=harbor_env,
                                             cwd=REPO_ROOT)
                    except KeyboardInterrupt:
                        print(
                            "\n==> Interrupted (SIGINT/SIGTERM); salvaging completed trial results...",
                            file=sys.stderr,
                            flush=True,
                        )
                        interrupted = True
                        res = subprocess.CompletedProcess(cmd, returncode=130)

                if job_dir.is_dir():
                    print(f"==> Results in {job_dir}")
                    if args.upload:
                        has_uploadable_results = (
                            job_dir / "result.json").is_file() or any(
                                (p / "result.json").is_file()
                                for p in job_dir.iterdir() if p.is_dir())
                        if has_uploadable_results or not interrupted:
                            share_org = args.share_org
                            print(
                                f"Uploading evaluation results from {job_dir} to Harbor (sharing with {share_org})..."
                            )
                            upload_cmd = [
                                harbor_bin, "upload",
                                str(job_dir), "--private"
                            ]
                            if share_org:
                                upload_cmd.extend(["--share-org", share_org])
                            try:
                                res_upload = subprocess.run(upload_cmd)
                            except KeyboardInterrupt:
                                res_upload = subprocess.CompletedProcess(
                                    upload_cmd, returncode=130)
                            if res_upload.returncode != 0:
                                print(
                                    "Warning: Failed to upload evaluation results to Harbor.",
                                    file=sys.stderr)
                                print(
                                    f"  To retry uploading manually, run:\n    {' '.join(upload_cmd)}",
                                    file=sys.stderr)
                            else:
                                print(
                                    "✓ Successfully uploaded evaluation results to Harbor."
                                )
                                upload_succeeded = True

                    eval_passed = (res.returncode == 0)

                    # Validate trial scores against min_score threshold
                    scores_passed, trial_summaries = check_job_scores(
                        job_dir, min_score=args.min_score)
                    if trial_summaries:
                        print("\n" + "=" * 80)
                        print(
                            f"📊 Evaluation Results for '{job_name}' (threshold: score >= {args.min_score:.2f})"
                        )
                        print("=" * 80)
                        for ts in trial_summaries:
                            badge = "✅ PASS" if ts["passed"] else "❌ FAIL"
                            print(
                                f"  {badge}  {ts['trial_name']}: score = {ts['score']:.2f}"
                            )
                            if not ts["passed"]:
                                for line in failed_criteria(job_dir /
                                                            ts["trial_name"]):
                                    print(f"           {line}")
                        print("=" * 80 + "\n")

                    if not scores_passed:
                        print(
                            f"❌ Evaluation failed: One or more trials scored below the minimum threshold ({args.min_score:.2f}).",
                            file=sys.stderr,
                        )
                        eval_passed = False

                    if args.report and not interrupted:
                        report_script = REPO_ROOT / "eval" / "report.py"
                        if report_script.is_file():
                            print(
                                f"==> Generating evaluation report for {job_name}..."
                            )
                            res_report = subprocess.run([
                                sys.executable,
                                str(report_script),
                                str(job_dir)
                            ])
                            if res_report.returncode != 0:
                                print(
                                    "Warning: Evaluation report indicated failures or error.",
                                    file=sys.stderr,
                                )
                                eval_passed = False
                elif (
                        args.upload or args.report
                ) and "--print-config" not in harbor_args and "--dry-run" not in harbor_args:
                    print(
                        f"Warning: Expected job directory '{job_dir}' not found, skipping upload and report generation.",
                        file=sys.stderr)
                    eval_passed = False
                else:
                    eval_passed = (res.returncode == 0)

            if res.returncode != 0 or not eval_passed:
                overall_success = False

            hub_url = get_harbor_hub_url(job_dir)
            results_summary.append({
                "config_path":
                config_path,
                "job_name":
                job_name,
                "job_dir":
                job_dir,
                "hub_url":
                hub_url,
                "upload_succeeded":
                upload_succeeded,
                "eval_passed": (res.returncode == 0 and eval_passed),
                "trial_summaries":
                trial_summaries,
            })
            if interrupted:
                break

        # At the end, print out Harbor Hub URLs for each config YAML
        print("\n" + "=" * 60)
        print("Harbor Hub URLs")
        print("=" * 60)
        for i, item in enumerate(results_summary):
            if i > 0:
                print()
            cfg_p = item["config_path"]
            display_path = cfg_p.relative_to(
                REPO_ROOT) if cfg_p.is_relative_to(REPO_ROOT) else cfg_p
            url = item["hub_url"]
            if not args.upload:
                print(
                    f"{display_path}:\n  (upload disabled; job: {item['job_name']})"
                )
            elif url:
                if item["upload_succeeded"]:
                    print(f"{display_path}:\n  {url}")
                else:
                    print(f"{display_path}:\n  {url} (upload failed)")
            else:
                print(f"{display_path}:\n  (no hub url available)")
        print("=" * 60 + "\n")

        publish_github_step_summary(results_summary,
                                    upload_enabled=args.upload)

        if not overall_success:
            sys.exit(1)
        sys.exit(0)

    except RuntimeError as e:
        publish_github_step_summary(
            results_summary, upload_enabled=args.upload
        ) if "results_summary" in locals() else None
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        publish_github_step_summary(
            results_summary, upload_enabled=args.upload
        ) if "results_summary" in locals() else None
        # A captured command's output is otherwise lost, and the run exits
        # non-zero with nothing on screen to say why.
        cmd = " ".join(map(str, e.cmd)) if isinstance(e.cmd, list) else str(
            e.cmd)
        print(f"Error: `{cmd}` exited {e.returncode}", file=sys.stderr)
        if e.stderr:
            print(str(e.stderr).strip(), file=sys.stderr)
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()

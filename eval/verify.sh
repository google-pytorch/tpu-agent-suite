#!/bin/bash
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

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Optional dataset under eval/datasets/ to scope the task check to (e.g.
# tpu-hello-world, smoke-check, smart-hf-search). Configs are always validated
# as a whole, since eval/configs/ is no longer partitioned by package.
DATASET="${1:-}"

if [ -n "$DATASET" ]; then
  echo "=== Verifying Harbor Setup (datasets/$DATASET) ==="
else
  echo "=== Verifying Harbor Setup (all configs, datasets, and suites) ==="
fi

# Resolve Harbor binary (fallback to PATH)
HARBOR_BIN="${HARBOR_BIN:-harbor}"

if ! command -v "$HARBOR_BIN" >/dev/null 2>&1; then
  echo "Error: '$HARBOR_BIN' CLI not found." >&2
  echo "Please install Harbor or set HARBOR_BIN environment variable." >&2
  exit 1
fi

# Resolve Python binary (fallback to PATH or infer from harbor path)
if [ -z "${PYTHON_BIN:-}" ]; then
  if [[ "$HARBOR_BIN" == */bin/harbor ]]; then
    PYTHON_BIN="${HARBOR_BIN%/harbor}/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' not found." >&2
  exit 1
fi

echo "Using Harbor: $HARBOR_BIN ($($HARBOR_BIN --version))"
echo "Using Python: $PYTHON_BIN"

# 2. Check Docker
if ! docker info >/dev/null 2>&1; then
  echo "WARNING: Docker daemon is not running. Live runs will fail."
fi

# Check API Key
if [ -z "${GEMINI_API_KEY:-}" ]; then
  echo "WARNING: GEMINI_API_KEY is not set. Live runs will fail."
fi

# 3. Validate schemas
echo "Validating schemas..."
export DATASET
"$PYTHON_BIN" - << 'EOF'
import os, sys, tempfile, yaml
from pathlib import Path
from harbor.models.job.config import JobConfig
from harbor.models.task.config import TaskConfig

eval_dir = Path("eval")
dataset_name = os.environ.get("DATASET", "")

# Validate JobConfigs under configs/
configs_dir = eval_dir / "configs"
if not configs_dir.is_dir():
    print(f"Error: Configs directory not found: {configs_dir}", file=sys.stderr)
    sys.exit(1)

# Recursive: configs are grouped by harness, configs/<harness>/<name>.yaml.
config_files = sorted(configs_dir.glob("**/*.yaml"))
if not config_files:
    print(f"Error: No config (.yaml) files found under {configs_dir}",
          file=sys.stderr)
    sys.exit(1)

for cfg_file in config_files:
    try:
        with open(cfg_file, "r") as f:
            job_data = yaml.safe_load(f)
        JobConfig.model_validate(job_data)
        print(f"✓ JobConfig schema OK ({cfg_file})")
    except Exception as e:
        print(f"Error validating JobConfig {cfg_file}: {e}", file=sys.stderr)
        sys.exit(1)

# Validate TaskConfigs
try:
    import tomllib

    def validate_tasks(root: Path, source_suite: Path | None = None) -> None:
        task_files = sorted(root.glob("**/task.toml"))
        if not task_files:
            print(f"WARNING: No task.toml files found under {source_suite or root}")
            return
        for task_path in task_files:
            shown = (
                f"{source_suite} [{task_path.parent.name}]"
                if source_suite
                else str(task_path)
            )
            try:
                with open(task_path, "rb") as f:
                    task_data = tomllib.load(f)
                TaskConfig.model_validate(task_data)
                print(f"✓ TaskConfig schema OK ({shown})")
            except Exception as e:
                print(f"Error validating TaskConfig {shown}: {e}", file=sys.stderr)
                sys.exit(1)

    # Short path: verify only the specified dataset under eval/datasets/
    if dataset_name:
        dataset_dir = eval_dir / "datasets" / dataset_name
        if not dataset_dir.is_dir():
            print(f"Error: Dataset directory not found: {dataset_dir}", file=sys.stderr)
            sys.exit(1)
        validate_tasks(dataset_dir)
        sys.exit(0)

    # Full path: verify all checked-in datasets and temporarily compiled suites
    datasets_root = eval_dir / "datasets"
    if not datasets_root.is_dir():
        print(f"Error: Dataset directory not found: {datasets_root}", file=sys.stderr)
        sys.exit(1)
    validate_tasks(datasets_root)

    suites_dir = eval_dir / "suites"
    if suites_dir.is_dir():
        from eval.suite_to_harbor.suite_to_harbor import compile_suite
        with tempfile.TemporaryDirectory(prefix="eval_verify_suites_") as tmp_suites:
            tmp_root = Path(tmp_suites)
            for suite_file in sorted(suites_dir.glob("**/*.yaml")):
                sub_rel = suite_file.relative_to(suites_dir).with_suffix("")
                target_suite_dir = tmp_root / sub_rel
                compile_suite(suite_file, target_suite_dir)
                validate_tasks(target_suite_dir, source_suite=suite_file)
except ImportError:
    print("Skipping TaskConfig schema validation (requires python 3.11+ or tomllib)")
EOF

echo "Verification complete."

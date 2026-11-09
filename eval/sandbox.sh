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

set -euo pipefail

# One sandbox at a time, so the name is fixed rather than a flag. The tmux
# session that drives it takes the same name.
readonly CONTAINER_NAME="tpu-sandcastle"
readonly TMUX_SESSION="tpu-sandcastle"

# Where the sandbox keeps its own files inside the container.
readonly SANDBOX_ROOT="/root/sandbox"
# The conventional user bin, and already where the Antigravity installer puts
# agy. Put on PATH below so the generated scripts are callable by name.
readonly SANDBOX_BIN="/root/.local/bin"

# Progress, for a launch that can sit a while on a docker pull, a compile or
# pip. --help is not a launch; flags are not parsed until Step 2, hence the
# pre-scan.
readonly STEP_TOTAL=9
SHOW_HELP=0
for _arg in "$@"; do
  case "$_arg" in --help|-h) SHOW_HELP=1 ;; esac
done
step() {
  if [ "$SHOW_HELP" = "1" ]; then return 0; fi
  echo "==> [$1/$STEP_TOTAL] $2"
}

# The sandbox prepares a container you drive by hand:
#
#   1. Locate the repo and load env.txt   5. Resolve the task (optional)
#   2. Parse input arguments              6. Choose the base image
#   3. Check the Docker daemon            7. Compile the package on the host
#   4. Infer the env map from the config  8. Build the container environment
#                                         9. Launch
#
# env.txt loads before the flags are read, so a flag always beats the file.
#
# The container is a daemon and the shell is a `docker exec` into it, inside a
# host tmux session, so an install or an agent run outlives its terminal.

# --------------------------------------------------------------------------
# Step 1: Locate the repo and load env.txt
# --------------------------------------------------------------------------
step 1 "Locating the repo and loading env.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Read before env.txt, whose KUBECONFIG names the container's mount, not yours.
HOST_KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"

# Load environment overrides if present. The file is never copied into the
# container; its resolved values are, in step 8.
HOST_ENV_FILE="${SANDBOX_ENV_FILE:-}"
if [ -n "$HOST_ENV_FILE" ] && [ ! -f "$HOST_ENV_FILE" ]; then
  echo "ERROR: SANDBOX_ENV_FILE=$HOST_ENV_FILE does not exist." >&2
  exit 1
fi
if [ -z "$HOST_ENV_FILE" ]; then
  for candidate in eval/env.txt env.txt eval/.env .env; do
    if [ -f "$candidate" ]; then
      HOST_ENV_FILE="$candidate"
      break
    fi
  done
fi

ENV_TXT_EXPORTS=""
if [ -n "$HOST_ENV_FILE" ]; then
  # The names it assigns, found in a clean shell so that nothing already
  # exported here is mistaken for one of them.
  # shellcheck disable=SC2016  # expanded by the inner shell
  ENV_TXT_NAMES="$(env -i bash -c '
    before="$(compgen -e | sort)"
    set -a; . "$1" >/dev/null 2>&1; set +a
    comm -13 <(printf "%s\n" "$before") <(compgen -e | sort)' _ "$HOST_ENV_FILE")"
  set -a
  # shellcheck disable=SC1090
  source "$HOST_ENV_FILE"
  set +a
  for name in $ENV_TXT_NAMES; do
    # Resolved here they are host values: KUBECONFIG is the mount, set in
    # step 8, and PATH is the image's.
    case "$name" in
      KUBECONFIG | PATH)
        echo "Note: $HOST_ENV_FILE sets $name; ignored in the container." >&2
        continue
        ;;
    esac
    ENV_TXT_EXPORTS+="$(printf 'export %s=%q' "$name" "${!name}")"$'\n'
  done
  echo "Using env file: $HOST_ENV_FILE"
else
  echo "WARNING: no env file found; the container gets only" >&2
  echo "   the config's agents[].env and the sandbox defaults. To add your own:" >&2
  echo "   cp eval/env_example.txt eval/env.txt" >&2
fi

# --------------------------------------------------------------------------
# Step 2: Parse input arguments
# --------------------------------------------------------------------------
step 2 "Parsing arguments"
# A config is required. Everything the sandbox needs -- which packages, which
# cluster, which image, whether this is a control arm -- is already decided in
# one, and a sandbox assembled from loose flags is a container no scored run
# would recognise.
#
# There is no --no-package: a control arm is a config whose name contains
# `baseline`, the rule run.py uses (eval/utils.py `is_control_arm`).
# Deriving it means the sandbox cannot disagree with a scored run about which
# arm it is. Every other config installs the whole package.
PACKAGES=""
PACKAGES_SOURCE=""
# The config a bare --baseline means. Named here rather than in the flag so a
# new baseline arm is one edit.
BASELINE_CONFIG="${SANDBOX_BASELINE_CONFIG:-claude-code/baseline}"
# The container is a daemon either way; this is only whether a shell to it
# gets attached before this script exits.
ATTACH=1
TASK_ARG=""
# Debugging only: install just these skills instead of the whole package.
SKILLS_ARG=""
CONFIG_ARG=""
CONFIG_ENV=""
CONFIG_CONTROL_ARM=""
CONFIG_HARNESS=""
# The package manifest the config installs: the one its PACKAGE_MANIFEST names
# (eval/utils.py `resolve_manifest`).
CONFIG_MANIFEST="package.release.json"
COMMAND=()

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      cat << 'HELP'
Usage: ./eval/sandbox.sh [OPTIONS] [-- COMMAND...]

Launch an isolated container environment for Claude Code or Antigravity.

The container runs as a daemon and you get a bash shell into it inside a tmux
session, with the config's environment loaded and the CLIs installed. Detach
with Ctrl-b d and whatever is running keeps running; re-run this script, or
`tmux attach -t tpu-sandcastle`, to get back to it. Pass `-- COMMAND` to run
one thing instead of taking a shell.

Installing the package is yours to do, from the shell:
  install_claude   install for Claude Code
  install_agy      the same for Antigravity
  install_codex    the same for Codex
Then run_claude, run_agy or run_codex to start the agent.

A config is required: it decides the cluster and the arm. A config whose
name contains `baseline` is a control arm and installs nothing; every other
config installs the whole package.

Required (one of):
  --config, -c NAME   Config under eval/configs/, without .yaml
  --baseline          Shorthand for the control arm config, currently
                      claude-code/baseline

Options:
  --sleep, -d         Prepare the container, leave it running, do not attach
  --task, -t TASK     Evaluation task whose instruction.md to mount
  --skills A[,B]      Debugging only: install just these skills instead of
                      the whole package. Not with a baseline config.
  --help, -h          Show this help message

Examples:
  ./eval/sandbox.sh -c claude-code/agent-suite
  ./eval/sandbox.sh -c agy/agent-suite -t xprof-attention/kernel-tuning
  ./eval/sandbox.sh --baseline
HELP
      exit 0
      ;;
    --sleep|--daemon|-d)
      ATTACH=0
      shift
      ;;
    --baseline)
      CONFIG_ARG="$BASELINE_CONFIG"
      shift
      ;;
    --config|-c)
      CONFIG_ARG="$2"
      shift 2
      ;;
    --task|-t)
      TASK_ARG="$2"
      shift 2
      ;;
    --skills)
      SKILLS_ARG="$2"
      shift 2
      ;;
    # Named, so an old invocation reports this instead of falling through to
    # COMMAND and running "--package" inside the container.
    --package|--packages)
      echo "Error: --package is gone. Every config installs the whole package;" >&2
      echo "  a control arm is a config whose name contains 'baseline'." >&2
      exit 1
      ;;
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    *)
      COMMAND+=("$1")
      shift
      ;;
  esac
done

if [ -z "$CONFIG_ARG" ]; then
  echo "Error: --config is required." >&2
  echo "  Pick one under eval/configs/, without the .yaml:" >&2
  echo "    ./eval/sandbox.sh -c claude-code/agent-suite" >&2
  echo "  or --baseline for the control arm ($BASELINE_CONFIG)." >&2
  echo "  --help lists the rest." >&2
  exit 1
fi

# --------------------------------------------------------------------------
# Step 3: Check the Docker daemon
# --------------------------------------------------------------------------
step 3 "Checking the Docker daemon"
# Everything below needs a working daemon: the image is pulled, then run.
# Warn rather than exit, so --help-style misuse still reports the real
# problem instead of a docker error.
if ! docker info >/dev/null 2>&1; then
  echo "WARNING: Docker daemon is not running. The sandbox needs Docker to" >&2
  echo "  pull the task image and start the container." >&2
fi

# --------------------------------------------------------------------------
# Step 4: Infer the env map from the config
# --------------------------------------------------------------------------
step 4 "Reading the config"
# The config gives the sandbox the same settings a scored run gets, instead of
# depending on a hand-maintained env.txt. agents[].env is read; ${VAR} and
# ${VAR:-default} resolve against the environment exactly as run.py does.
TASK_DIR=""
if [ -n "$CONFIG_ARG" ]; then
  CONFIG_FILE="$CONFIG_ARG"
  [ -f "$CONFIG_FILE" ] || CONFIG_FILE="eval/configs/$CONFIG_ARG.yaml"
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config not found: $CONFIG_ARG" >&2
    exit 1
  fi
  CONFIG_ENV="$(python3 - "$CONFIG_FILE" <<'PYRESOLVE'
import os, re, shlex, sys, yaml

sys.path.insert(0, os.getcwd())
from eval.utils import (RELEASE_MANIFEST, harness_for_config, is_control_arm,
                        package_manifest, resolve_manifest)

cfg = yaml.safe_load(open(sys.argv[1])) or {}
env = {}
for agent in (cfg.get("agents") or []):
    env.update(agent.get("env") or {})

pattern = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}$")
resolved = {}
for key, raw in env.items():
    value = str(raw).strip()
    match = pattern.match(value)
    if match:
        value = os.environ.get(match.group(1), match.group(2) or "")
    resolved[key] = value
# Before anything is printed: an unknown PACKAGE_MANIFEST stops here rather
# than leaving the default manifest in place. A control arm installs
# nothing, so its PACKAGE_MANIFEST is not read.
control_arm = is_control_arm(sys.argv[1])
manifest = RELEASE_MANIFEST
if not control_arm:
    try:
        manifest = resolve_manifest(package_manifest(resolved))
    except ValueError as exc:
        sys.exit(f"Error: {exc}")
for key, value in resolved.items():
    if value:
        print(f"export {key}={shlex.quote(value)}")
print(f"CONFIG_CONTROL_ARM={int(control_arm)}")
print(f"CONFIG_HARNESS={shlex.quote(harness_for_config(cfg))}")
print(f"CONFIG_MANIFEST={shlex.quote(manifest)}")
PYRESOLVE
)"
  if [ -z "$CONFIG_ENV" ]; then
    echo "Error: $CONFIG_FILE declares no agents[].env to apply." >&2
    exit 1
  fi
  # Only the arm, the harness and the manifest come back into this shell.
  # agents[].env is the container's, and several of its values are paths only
  # it has.
  eval "$(printf '%s\n' "$CONFIG_ENV" | grep -E '^(CONFIG_CONTROL_ARM|CONFIG_HARNESS|CONFIG_MANIFEST)=' || true)"
  CONFIG_ENV="$(printf '%s\n' "$CONFIG_ENV" | grep -vE '^(CONFIG_CONTROL_ARM|CONFIG_HARNESS|CONFIG_MANIFEST)=' || true)"
  echo "Using config: $CONFIG_FILE"
fi

if [ -n "$SKILLS_ARG" ] && [ -n "${GITHUB_RUN_ID:-}" ]; then
  echo "Error: --skills is for debugging; CI runs the whole package." >&2
  exit 1
fi

# The arm comes from the config's name, never a separate flag that could
# contradict it.
if [ "$CONFIG_CONTROL_ARM" = "1" ]; then
  if [ -n "$SKILLS_ARG" ]; then
    echo "Error: --skills chooses part of the package, and $CONFIG_FILE is a" >&2
    echo "  control arm, which installs nothing." >&2
    exit 1
  fi
  INSTALL_PKG=0
  PACKAGES_SOURCE="$CONFIG_FILE is a control arm (its name contains 'baseline')"
elif [ -n "$SKILLS_ARG" ]; then
  # Checked against what the config's manifest releases, the same rule run.py uses.
  PACKAGES="$(python3 -c '
import os, sys
sys.path.insert(0, os.getcwd())
from eval.utils import released_skills, skill_subset
try:
    print(",".join(skill_subset([sys.argv[1]], released_skills(None, sys.argv[2]))))
except ValueError as exc:
    sys.exit(f"Error: {exc}")
' "$SKILLS_ARG" "$CONFIG_MANIFEST")" || exit 1
  INSTALL_PKG=1
  PACKAGES_SOURCE="--skills, a debugging subset of the package"
else
  INSTALL_PKG=1
  # The package's own name: tpu-agent-suite, or tpu-agent-suite-experimental.
  PACKAGES="$(python3 -c '
import os, sys
sys.path.insert(0, os.getcwd())
from eval.utils import read_manifest
print(read_manifest(None, sys.argv[1])["name"])
' "$CONFIG_MANIFEST")" || exit 1
  PACKAGES_SOURCE="every skill $CONFIG_MANIFEST releases"
fi

# --------------------------------------------------------------------------
# Step 5: Resolve the task (optional)
# --------------------------------------------------------------------------
step 5 "Resolving the task"
# Resolved before the image is chosen, since a task declares what it runs in.
# Datasets live in one place now, so this resolves by path rather than by
# searching for whichever package happened to hold the task.
if [ -n "$TASK_ARG" ]; then
  TASK_DIR="$TASK_ARG"
  [ -f "$TASK_DIR/instruction.md" ] || TASK_DIR="eval/datasets/$TASK_ARG"
  [ -f "$TASK_DIR/instruction.md" ] || TASK_DIR="eval/datasets/$TASK_ARG/baseline"
  if [ ! -f "$TASK_DIR/instruction.md" ]; then
    echo "Error: cannot find instruction.md for task '$TASK_ARG'" >&2
    exit 1
  fi
fi

# Reading the tag from the task rather than assuming it keeps this honest when
# a task moves to a different image.
task_docker_image() {
  [ -f "$1/task.toml" ] || return 0
  sed -n 's/^docker_image *= *"\(.*\)"/\1/p' "$1/task.toml" | head -1
}

# --------------------------------------------------------------------------
# Step 6: Choose the base image
# --------------------------------------------------------------------------
step 6 "Choosing the base image"
# The sandbox never builds. It runs a pre-determined image, so what you iterate
# in is the container a scored run would use, not a lookalike assembled here.
#
#   --task <t>   the image that task pins in its task.toml
#   otherwise    the image tpu-hello-world's baseline pins
#   SANDBOX_IMAGE overrides both
#
# The pin lives in the task that owns it; nothing restates a digest here.
DEFAULT_IMAGE_TASK="${SANDBOX_BASE_TASK:-eval/datasets/tpu-hello-world/baseline}"
IMAGE_TASK="${TASK_DIR:-$DEFAULT_IMAGE_TASK}"
IMAGE="${SANDBOX_IMAGE:-$(task_docker_image "$IMAGE_TASK")}"

# A task may name its image in environment/Dockerfile instead of pinning it.
# Only usable when the Dockerfile adds nothing on top of FROM; if it installs
# anything, the base is not what a scored run would execute.
if [ -z "$IMAGE" ]; then
  DF="$IMAGE_TASK/environment/Dockerfile"
  if [ -f "$DF" ] && [ "$(grep -cvE '^\s*(#|$)' "$DF")" -eq 1 ]; then
    IMAGE="$(sed -n 's/^FROM  *\([^ ]*\).*/\1/p' "$DF" | head -1)"
  fi
fi

if [ -z "$IMAGE" ]; then
  echo "Error: no image to run. $IMAGE_TASK/task.toml pins no docker_image." >&2
  echo "  Tasks that ship an environment/Dockerfile instead are built by" >&2
  echo "  Harbor, not here; set SANDBOX_IMAGE to run one in the sandbox." >&2
  exit 1
fi

# ADC path for Vertex AI
EVAL_VERTEX_ADC="${EVAL_VERTEX_ADC:-$HOME/.config/gcloud/application_default_credentials.json}"
if [ ! -f "$EVAL_VERTEX_ADC" ]; then
  echo "Error: No application-default credentials found at $EVAL_VERTEX_ADC" >&2
  echo "Run: gcloud auth application-default login" >&2
  exit 1
fi

SCRATCH_DIR="$(mktemp -d)"
# Assembled here and copied into the container with `docker cp`, so the only
# bind mounts left are the installed plugin and the credentials.
SANDBOX_STAGE="$SCRATCH_DIR/sandbox"
mkdir -p "$SANDBOX_STAGE"
trap 'rm -rf "$SCRATCH_DIR"' EXIT

# Your kubeconfig, as-is, if you have one. The sandbox does not mint one: what a
# cluster is, and who may talk to it, belongs to the config and to run.py, which
# does mint. A config naming a cluster this file cannot reach is a run.py run.
KUBE_MOUNT=()
if [ -f "$HOST_KUBECONFIG" ]; then
  KUBE_MOUNT=(-v "$HOST_KUBECONFIG":/root/.kube/config:ro)
fi

# The hub credential, if you have one. `harbor upload` is the trajectory
# export's first choice, and without this it fails NotAuthenticatedError
# inside a container that has the CLI but has never logged in.
HARBOR_MOUNT=()
if [ -f "$HOME/.harbor/credentials.json" ]; then
  HARBOR_MOUNT=(-v "$HOME/.harbor":/root/.harbor:ro)
fi

# The harness's defaults and names (agents/claude_code.py), so the sandbox talks
# to the same Vertex project a scored run does.
VERTEX_PROJECT="${VERTEX_PROJECT:-${VERTEXAI_PROJECT:-tpu-nexus-767207}}"
VERTEX_LOCATION="${VERTEX_LOCATION:-${VERTEXAI_LOCATION:-global}}"
VERTEX_BASE_URL="https://aiplatform.googleapis.com/v1"

# --------------------------------------------------------------------------
# Step 7: Compile the package on the host
# --------------------------------------------------------------------------
step 7 "Compiling the package on the host"
# Compiling is the only part that needs the repo, since compiler.py reads
# package. Step 9 copies the output in; bin/install_* installs it.
#
# The repo is never mounted. Given it as a workspace, an agent asked to do TPU
# work greps package and follows the instructions, so a control arm is
# not one.
#
# All three harnesses, because the container has no compiler.py to produce
# another one later.
if [ "$INSTALL_PKG" = "1" ]; then
  mkdir -p "$SANDBOX_STAGE/install"
  # The whole package, or the --skills subset.
  SKILLS_FLAG=()
  [ -n "$SKILLS_ARG" ] && SKILLS_FLAG=(--skills "$PACKAGES")
  for harness in claude-code antigravity codex; do
    echo "    Compiling $PACKAGES for $harness"
    python3 framework/compiler.py translate package \
      --harness "$harness" --manifest "$CONFIG_MANIFEST" \
      ${SKILLS_FLAG[@]+"${SKILLS_FLAG[@]}"} \
      --output "$SANDBOX_STAGE/dist/$harness" >/dev/null
  done

  # The container builds its venv from these; nothing else needs the repo.
  cp package/requirements-client.txt "$SANDBOX_STAGE/install/"
  python3 -c "
import tomllib
with open('eval/pyproject.toml', 'rb') as f:
    print('\n'.join(tomllib.load(f)['project']['dependencies']))
" > "$SANDBOX_STAGE/install/requirements-harness.txt" 2>/dev/null || true
fi

# --------------------------------------------------------------------------
# Step 8: Build the container environment
# --------------------------------------------------------------------------
step 8 "Building the container environment"
# Everything the container should see, in one sourceable file, so that what the
# agent runs with is the config's agents[].env and your env.txt rather than a
# list of -e flags maintained here. env.txt's values come first, so the config
# wins where both name a variable.
# The conventional user bin, and already where the Antigravity installer puts
# agy. Put on PATH below so the generated scripts are callable by name.
SANDBOX_ENV_TARGET="$SANDBOX_ROOT/env.txt"
SANDBOX_ENV_HOST="$SANDBOX_STAGE/env.txt"
{
  echo "# Generated by eval/sandbox.sh. Source it: set -a; . $SANDBOX_ENV_TARGET; set +a"
  # BASH_ENV re-runs this file in every non-interactive shell. Without this, the
  # exports below would reset a KUBECONFIG the parent shell switched to.
  # shellcheck disable=SC2016
  echo 'SANDBOX_INHERITED_KUBECONFIG="${KUBECONFIG:-}"'
  [ -n "$ENV_TXT_EXPORTS" ] && printf '%s' "$ENV_TXT_EXPORTS"
  [ -n "$CONFIG_ENV" ] && printf '%s\n' "$CONFIG_ENV"
  # Vertex is how the agent authenticates at all, so it has a default here.
  # Emitted as `${VAR:-default}` rather than a plain assignment: these lines
  # run after the config and env.txt are sourced, so assigning would override
  # the value the user set instead of falling back to it.
  # shellcheck disable=SC2016  # the ${VAR:-...} is the container's, not ours
  printf 'export ANTHROPIC_VERTEX_PROJECT_ID="${ANTHROPIC_VERTEX_PROJECT_ID:-%s}"\n' "$VERTEX_PROJECT"
  # shellcheck disable=SC2016
  printf 'export CLOUD_ML_REGION="${CLOUD_ML_REGION:-%s}"\n' "$VERTEX_LOCATION"
  # shellcheck disable=SC2016
  printf 'export ANTHROPIC_VERTEX_BASE_URL="${ANTHROPIC_VERTEX_BASE_URL:-%s}"\n' "$VERTEX_BASE_URL"
  # shellcheck disable=SC2016
  echo 'export CLAUDE_CODE_USE_VERTEX="${CLAUDE_CODE_USE_VERTEX:-1}"'
  # Container paths, not the host's.
  echo 'export IS_SANDBOX=1'
  # shellcheck disable=SC2016  # $PATH is the container's, expanded on source
  printf 'export PATH=%q:"$PATH"\n' "$SANDBOX_BIN"
  # `npm install -g` as agent: /usr/local/bin is root-only (kube-gate setup), so
  # global packages go under ~/.local and their commands into $SANDBOX_BIN.
  # shellcheck disable=SC2016  # expanded on source, so a config value wins
  printf 'export NPM_CONFIG_PREFIX="${NPM_CONFIG_PREFIX:-%s}"\n' \
    "${SANDBOX_BIN%/bin}"
  # The agent-owned copy kube-gate setup makes of the adc.json mount.
  echo 'export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json'
  # agy's ADC path, on by default because that credential is already mounted.
  # Its other headless path is GEMINI_API_KEY; set AGY_ADC_AUTH=false in the
  # config or env.txt to take it instead.
  # shellcheck disable=SC2016
  echo 'export AGY_ADC_AUTH="${AGY_ADC_AUTH:-true}"'
  if [ ${#KUBE_MOUNT[@]} -gt 0 ]; then
    # shellcheck disable=SC2016
    echo 'if [ -f /root/.kube/config.writable ]; then export KUBECONFIG="/root/.kube/config.writable"; else export KUBECONFIG="${KUBECONFIG:-/root/.kube/config}"; fi'
  fi
  # shellcheck disable=SC2016
  echo 'if [ -n "$SANDBOX_INHERITED_KUBECONFIG" ]; then export KUBECONFIG="$SANDBOX_INHERITED_KUBECONFIG"; fi'
  echo 'unset SANDBOX_INHERITED_KUBECONFIG'
} > "$SANDBOX_ENV_HOST"

# The sandbox prepares an install and does not perform one. It builds the venv
# the package's host-side deps need, writes the commands under bin/, and prints
# them; running installer.py is yours, so what you rehearse here is the install
# a user does rather than the pre-staged tree the eval harness uploads.
BOOTSTRAP_TARGET="$SANDBOX_ROOT/bootstrap.sh"
BOOTSTRAP_HOST="$SANDBOX_STAGE/bootstrap.sh"
# The prompt, copied from the task's instruction.md on the host. Verbatim: a
# scored run sends instruction.md as-is, and $TASK_DOCKER and the GKE_* names
# are resolved by the agent from its env, not expanded here. Writable, so the
# wording can be edited between runs.
TASK_PROMPT_TARGET="$SANDBOX_ROOT/prompt.txt"
CLAUDE_FLAGS="--permission-mode auto"
cat > "$BOOTSTRAP_HOST" << BOOTSTRAP
# Generated by eval/sandbox.sh
mkdir -p $SANDBOX_BIN
if [ -f $SANDBOX_ENV_TARGET ]; then set -a; . $SANDBOX_ENV_TARGET; set +a; fi

# If KUBECONFIG is mounted read-only, create a writable copy so context switches work
if [ -r /root/.kube/config ] && [ ! -w /root/.kube/config ]; then
  if [ ! -f /root/.kube/config.writable ]; then
    cp /root/.kube/config /root/.kube/config.writable
    chmod 600 /root/.kube/config.writable
  fi
  export KUBECONFIG="/root/.kube/config.writable"
fi

if ! command -v claude >/dev/null 2>&1; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Installing Claude Code..." >&2
    npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 ||
      echo "WARNING: npm install of @anthropic-ai/claude-code failed." >&2
  else
    echo "WARNING: no claude and no npm in this image; install it by hand." >&2
  fi
fi

if ! command -v codex >/dev/null 2>&1; then
  if command -v npm >/dev/null 2>&1; then
    echo "==> Installing the Codex CLI..." >&2
    npm install -g @openai/codex >/dev/null 2>&1 ||
      echo "WARNING: npm install of @openai/codex failed." >&2
  else
    echo "WARNING: no codex and no npm in this image; install it by hand." >&2
  fi
fi

# The installer harbor's AntigravityCli uses, so the sandbox gets the same agy
# on the same path a scored antigravity run does.
if [ ! -x \$HOME/.local/bin/agy ] && ! command -v agy >/dev/null 2>&1; then
  echo "==> Installing the Antigravity CLI..." >&2
  curl -fsSL https://antigravity.google/cli/install.sh | bash >/dev/null 2>&1 ||
    echo "WARNING: install of the Antigravity CLI failed." >&2
fi

mkdir -p $SANDBOX_BIN
cat > $SANDBOX_BIN/run_claude << 'RUN_CLAUDE_SH'
#!/bin/bash
# Generated by eval/sandbox.sh. Claude Code with the sandbox environment.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a

# The venv goes on PATH so the package's scripts run against its interpreter
# rather than whichever python3 the image happens to ship.
if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

exec claude $CLAUDE_FLAGS "\$@"
RUN_CLAUDE_SH
chmod +x $SANDBOX_BIN/run_claude

cat > $SANDBOX_BIN/run_agy << 'RUN_AGY_SH'
#!/bin/bash
# Generated by eval/sandbox.sh. Antigravity CLI with the sandbox env.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a

if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

# agy has two headless paths and drops to its sign-in screen if neither is
# configured, rather than reporting the missing credential. env.txt defaults
# AGY_ADC_AUTH to true, so this only fires if something turned it off.
if [ "\${AGY_ADC_AUTH:-}" != "true" ] && [ -z "\${GEMINI_API_KEY:-}" ]; then
  echo "Error: agy has no credential. AGY_ADC_AUTH is not true and" >&2
  echo "  GEMINI_API_KEY is unset. Restore the ADC path by unsetting" >&2
  echo "  AGY_ADC_AUTH in the config's agents[].env or eval/env.txt." >&2
  exit 1
fi

AGY=\$HOME/.local/bin/agy
if command -v agy >/dev/null 2>&1; then
  AGY=\$(command -v agy)
elif [ ! -x "\$AGY" ]; then
  echo "Error: no agy on PATH or at \$AGY." >&2
  exit 1
fi

# --add-dir "\$PWD" is how a scored run gives agy the workspace root.
exec "\$AGY" --add-dir "\$PWD" "\$@"
RUN_AGY_SH
chmod +x $SANDBOX_BIN/run_agy

cat > $SANDBOX_BIN/run_codex << 'RUN_CODEX_SH'
#!/bin/bash
# Generated by eval/sandbox.sh. Codex CLI with the sandbox environment.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a

if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

# codex drops to its sign-in flow when it has neither, rather than reporting
# the missing credential. auth.json is what a codex login leaves behind.
if [ -z "\${OPENAI_API_KEY:-}" ] && [ ! -f "\$HOME/.codex/auth.json" ]; then
  echo "Error: codex has no credential. Set OPENAI_API_KEY in the config's" >&2
  echo "  agents[].env or eval/env.txt, or run 'codex login' first." >&2
  exit 1
fi

exec codex "\$@"
RUN_CODEX_SH
chmod +x $SANDBOX_BIN/run_codex


BOOTSTRAP

if [ "$INSTALL_PKG" = "1" ]; then
  cat >> "$BOOTSTRAP_HOST" << BOOTSTRAP_PREP

# --system-site-packages because a task image usually carries most of these
# already; the venv adds what it lacks. --without-pip because task images ship
# a Python without ensurepip, and the borrowed system pip installs into the
# venv anyway. The sentinel is what "ready" means: a venv whose creation died
# half way still has a bin/python3.
if [ ! -f $SANDBOX_ROOT/venv/.ready ]; then
  echo "==> Creating $SANDBOX_ROOT/venv..."
  rm -rf $SANDBOX_ROOT/venv
  # tpu-diagnostics accepts only the xprof release. Drop a system
  # xprof-nightly, and xprof with it: they share a package directory, so
  # removing one breaks the other. The venv reinstalls xprof.
  if python3 -m pip show -q xprof-nightly >/dev/null 2>&1; then
    PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip uninstall -y xprof-nightly xprof 2>/dev/null || true
  fi
  python3 -m venv --system-site-packages --without-pip $SANDBOX_ROOT/venv
  if $SANDBOX_ROOT/venv/bin/python3 -m pip --version >/dev/null 2>&1; then
    # Both files were copied in by the host compile step; the repo itself is
    # not mounted, so this is all the container knows about requirements.
    echo "==> Installing requirements-client.txt and the harness deps..."
    if ! $SANDBOX_ROOT/venv/bin/python3 -m pip install -q \\
         -r $SANDBOX_ROOT/install/requirements-client.txt \\
         -r $SANDBOX_ROOT/install/requirements-harness.txt; then
      echo "WARNING: pip install failed; $SANDBOX_ROOT/venv is incomplete." >&2
    elif ! $SANDBOX_ROOT/venv/bin/python3 - << 'XPROF_CHECK'
import importlib.metadata as m, sys
def version(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return None
release, nightly = version("xprof"), version("xprof-nightly")
try:
    from importlib.resources import files
    skill = (files("xprof.skills") / "xprof" / "SKILL.md").is_file()
except Exception:
    skill = False
print(f"==> XProf in the venv: xprof {release}, xprof-nightly {nightly},"
      f" skill {'present' if skill else 'missing'}")
sys.exit(0 if release and not nightly and skill else 1)
XPROF_CHECK
    then
      echo "WARNING: the venv's XProf is not the xprof release alone;" \\
           "tpu-diagnostics will refuse it." >&2
    else
      touch $SANDBOX_ROOT/venv/.ready
    fi
  else
    echo "WARNING: no pip in this image; $SANDBOX_ROOT/venv has no deps." >&2
  fi
fi

# One per harness: different dists, and agy needs its customization roots
# populated afterwards where claude-code and codex do not.
cat > $SANDBOX_BIN/install_claude << 'INSTALL_CLAUDE'
#!/bin/bash
# Generated by eval/sandbox.sh. Install the package for Claude Code.
#
# This is the install a user does rather than the pre-staged tree the eval
# harness uploads. Extra installer flags pass through as arguments.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a
if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

DIST=$SANDBOX_ROOT/dist/claude-code
# User scope, not project: project scope would write into /workspace, which is
# the agent's own working directory.
python3 "\$DIST/framework/installer.py" install "\$DIST" --scope user "\$@"

# Every package hook runs runtime.py. An install that reports success without
# leaving one gives a harness whose every hook fails, and nothing else says so.
RUNTIME=\$(ls -d "\$HOME"/.tpu-agent-suite/packages/*/framework/runtime.py 2>/dev/null | head -1 || true)
if [ -z "\$RUNTIME" ]; then
  echo "Error: the install left no framework/runtime.py under" >&2
  echo "  \$HOME/.tpu-agent-suite/packages. Check the installer output above." >&2
  exit 1
fi

echo "==> Installed. Package files are under \$HOME/.tpu-agent-suite and \$HOME/.claude."
INSTALL_CLAUDE
chmod +x $SANDBOX_BIN/install_claude

cat > $SANDBOX_BIN/install_codex << 'INSTALL_CODEX'
#!/bin/bash
# Generated by eval/sandbox.sh. Install the package for Codex.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a
if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

DIST=$SANDBOX_ROOT/dist/codex
python3 "\$DIST/framework/installer.py" install "\$DIST" --scope user "\$@"

# Every package hook runs runtime.py. An install that reports success without
# leaving one gives a harness whose every hook fails, and nothing else says so.
RUNTIME=\$(ls -d "\$HOME"/.tpu-agent-suite/packages/*/framework/runtime.py 2>/dev/null | head -1 || true)
if [ -z "\$RUNTIME" ]; then
  echo "Error: the install left no framework/runtime.py under" >&2
  echo "  \$HOME/.tpu-agent-suite/packages. Check the installer output above." >&2
  exit 1
fi

echo "==> Installed. Package files are under \$HOME/.tpu-agent-suite and \$HOME/.codex."
INSTALL_CODEX
chmod +x $SANDBOX_BIN/install_codex

cat > $SANDBOX_BIN/install_agy << 'INSTALL_AGY'
#!/bin/bash
# Generated by eval/sandbox.sh. Install the package for Antigravity.
set -euo pipefail
set -a; . $SANDBOX_ENV_TARGET; set +a
if [ -f $SANDBOX_ROOT/venv/bin/activate ]; then
  . $SANDBOX_ROOT/venv/bin/activate
fi

DIST=$SANDBOX_ROOT/dist/antigravity
python3 "\$DIST/framework/installer.py" install "\$DIST" --scope user "\$@"

# Every package hook runs runtime.py. An install that reports success without
# leaving one gives a harness whose every hook fails, and nothing else says so.
RUNTIME=\$(ls -d "\$HOME"/.tpu-agent-suite/packages/*/framework/runtime.py 2>/dev/null | head -1 || true)
if [ -z "\$RUNTIME" ]; then
  echo "Error: the install left no framework/runtime.py under" >&2
  echo "  \$HOME/.tpu-agent-suite/packages. Check the installer output above." >&2
  exit 1
fi

# agy discovers customizations under \$HOME/.gemini/config and, for subagents
# only, \$HOME/.gemini/antigravity-cli. A user-scope install lands in \$HOME
# itself, which agy never scans, so mirror it the way antigravity_wrapper.py
# does.
for root in "\$HOME/.gemini/config" "\$HOME/.gemini/antigravity-cli"; do
  mkdir -p "\$root/skills" "\$root/agents"
  if [ -d "\$HOME/.agents/skills" ]; then
    cp -r "\$HOME/.agents/skills/." "\$root/skills/"
  fi
  if [ -d "\$HOME/.agents/agents" ]; then
    cp -r "\$HOME/.agents/agents/." "\$root/agents/"
  fi
done
echo "==> Installed. Mirrored into \$HOME/.gemini/config and antigravity-cli."
INSTALL_AGY
chmod +x $SANDBOX_BIN/install_agy

BOOTSTRAP_PREP
fi

# --task brings the instruction in as a file and derives the prompt from it, so
# the shell can read it, edit it, or run it again.
if [ -n "$TASK_DIR" ]; then
  # The prompt is extracted here, and only the prompt crosses into the
  # container. Mounting the task directory would hand the agent its own
  # verifier: tests/ holds the grader and the reference implementation.
  cp "$TASK_DIR/instruction.md" "$SANDBOX_STAGE/prompt.txt"
  cat >> "$BOOTSTRAP_HOST" << BOOTSTRAP_TASK
TASK_LINE="  Prompt:    $TASK_PROMPT_TARGET (edit it, then give it to the agent)"
BOOTSTRAP_TASK
fi

# Printed last, inside the container, so it is the final thing on screen when
# the shell appears rather than something the docker pull scrolled away. Only on
# a terminal: `-- COMMAND` runs and the daemon's setup exec do not want it.
if [ "$INSTALL_PKG" = "1" ]; then
  INSTALL_LINE="  Install:   install_claude  (or install_agy, install_codex)"
else
  INSTALL_LINE="  Install:   nothing -- the config declares no packages."
fi
cat >> "$BOOTSTRAP_HOST" << BOOTSTRAP_HELP

if [ -t 1 ]; then
cat << HELP

Sandbox ready. Nothing is installed yet.
$INSTALL_LINE
  Run:       run_claude      (or run_agy, run_codex)
  Ship:      /tpu-agent-export-trajectory, from inside the session, when
             the config stages that skill or the whole package
\${TASK_LINE:-}

  env.txt    $SANDBOX_ENV_TARGET (your env file and the config; already sourced)
  venv       $SANDBOX_ROOT/venv
  bin        $SANDBOX_BIN (on PATH)

  Detach:    Ctrl-b d -- the container keeps running
  Reattach:  tmux attach -t $TMUX_SESSION
  Discard:   docker rm -f $CONTAINER_NAME
HELP
fi
BOOTSTRAP_HELP

# Worked out before the banner rather than at the point of attaching, so the
# summary says what you are about to get rather than what was asked for.
if [ ${#COMMAND[@]} -gt 0 ]; then
  SHELL_LINE="none -- running: ${COMMAND[*]}"
elif [ "$ATTACH" = "0" ]; then
  SHELL_LINE="none (--sleep)"
elif [ ! -t 0 ] || [ ! -t 1 ]; then
  SHELL_LINE="none (not a terminal)"
elif [ -n "${TMUX:-}" ]; then
  SHELL_LINE="docker exec (already inside tmux)"
elif ! command -v tmux >/dev/null 2>&1; then
  SHELL_LINE="docker exec (no tmux on this host)"
else
  SHELL_LINE="tmux session '$TMUX_SESSION', on this host"
fi

echo "========================================================================"
echo "==> Claude Code Sandbox"
echo "   Container:   $CONTAINER_NAME"
echo "   Shell:       $SHELL_LINE"
echo "   Image:       $IMAGE"
echo "   Arm:         $([ "$INSTALL_PKG" = "1" ] && echo "package ($PACKAGES)" || echo "control")"
echo "   Harness:     ${CONFIG_HARNESS:-unknown} (the config's; all three are written)"
echo "   Vertex Proj: $VERTEX_PROJECT ($VERTEX_LOCATION)"
echo "   Workspace:   /workspace (empty; the agent's own)"
echo "   Sandbox:     $SANDBOX_ROOT (env.txt, bootstrap.sh, venv, dist)"
echo "   Bin:         $SANDBOX_BIN (on PATH)"
if [ -n "$TASK_ARG" ]; then
  echo "   Task:        $TASK_DIR"
fi
echo "========================================================================"

if [ "$INSTALL_PKG" = "1" ]; then
  echo "==> Packages the install scripts will install (source: $PACKAGES_SOURCE):"
  IFS=',' read -ra _shown_pkgs <<< "$PACKAGES"
  for _pkg in "${_shown_pkgs[@]}"; do
    echo "      - $_pkg"
  done
else
  echo "==> Control arm: $PACKAGES_SOURCE, so nothing gets installed."
fi

# --------------------------------------------------------------------------
# Step 9: Launch
# --------------------------------------------------------------------------
step 9 "Launching"
# A container from an earlier run holds the name and keeps that run's mounts,
# so nothing prepared above would reach it. Late, so that a failure during
# setup leaves the previous sandbox and its unexported transcripts alone.
STALE_STATUS="$(docker ps -a -f name="^${CONTAINER_NAME}\$" --format '{{.Status}}' 2>/dev/null | head -1)"
if [ -n "$STALE_STATUS" ]; then
  echo "    Removing stale container '$CONTAINER_NAME' ($STALE_STATUS)"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
# A leftover session is a window onto a container that no longer exists, and
# `new-session -A` below would attach to it.
if tmux kill-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "    Killed the stale tmux session '$TMUX_SESSION'"
fi

# A daemon, and kept on exit: an interrupted agent, a closed shell or a dropped
# ssh session all leave the container and its transcripts in place. The removal
# above is what keeps them from piling up.
echo "    Starting container '$CONTAINER_NAME'"
docker run -d \
  --name "$CONTAINER_NAME" \
  ${KUBE_MOUNT[@]+"${KUBE_MOUNT[@]}"} \
  ${HARBOR_MOUNT[@]+"${HARBOR_MOUNT[@]}"} \
  -v "$EVAL_VERTEX_ADC":/root/.config/gcloud/adc.json:ro \
  -w /workspace \
  -e IS_SANDBOX=1 \
  -e BASH_ENV="$SANDBOX_ENV_TARGET" \
  "$IMAGE" \
  sleep infinity >/dev/null

# The same kube-gate setup a Harbor trial gets (adapter/kube_gate_setup.sh):
# the agent user, the kubectl wrapper and its sudo rule, the credentials in
# /etc/kube-gate. /usr/libexec has to exist before `docker cp` writes the
# wrapper into it.
docker exec "$CONTAINER_NAME" mkdir -p /usr/libexec
docker cp "$SCRIPT_DIR/kubectl_wrapper.py" "$CONTAINER_NAME:/usr/libexec/kubectl-wrapper" >/dev/null
docker exec -i -e KUBE_GATE_TRIAL_ID="${SANDBOX_TRIAL_ID:-tpu-sandcastle}" \
  "$CONTAINER_NAME" bash -s < "$SCRIPT_DIR/adapter/kube_gate_setup.sh"

# env.txt, bootstrap.sh, the prompt and the compiled dists, copied rather than
# bind-mounted, so nothing generated on the host is visible to the agent as a
# mount it could follow back to the repo.
echo "    Copying $([ "$INSTALL_PKG" = "1" ] && echo "the environment, the prompt and the dists" \
  || echo "the environment and the prompt") into $SANDBOX_ROOT"
docker cp "$SANDBOX_STAGE" "$CONTAINER_NAME:/root/" >/dev/null
docker exec "$CONTAINER_NAME" bash -c '
  chown -R agent:agent /root /workspace 2>/dev/null || true
  if [ -d /root/.kube ]; then
    chown root:root /root/.kube 2>/dev/null || true
    chmod 0700 /root/.kube 2>/dev/null || true
  fi
'

# Prepared up front rather than on the first shell, so the venv is built and
# bin/ is written by the time anyone is looking at a prompt.
echo "    Preparing the container (CLIs, venv, $SANDBOX_BIN)"
docker exec -u agent "$CONTAINER_NAME" bash "$BOOTSTRAP_TARGET"

# Every new shell re-runs the bootstrap rather than a second copy of its logic.
# It is idempotent, so on a prepared container it only re-prints the help.
docker exec -u agent "$CONTAINER_NAME" bash -c '
  cat << "BASHRC_SNIPPET" >> /root/.bashrc
if [ -f /root/sandbox/bootstrap.sh ]; then
  . /root/sandbox/bootstrap.sh
fi
BASHRC_SNIPPET
'

# Every path below execs or exits, and an exec never runs the trap.
trap - EXIT
rm -rf "$SCRATCH_DIR"

# `-- COMMAND` is one thing to run, not a session to sit in, so it skips tmux.
# The bootstrap is sourced first so the command gets $SANDBOX_BIN on PATH.
if [ ${#COMMAND[@]} -gt 0 ]; then
  echo "    Running: ${COMMAND[*]}"
  exec docker exec -u agent -it "$CONTAINER_NAME" \
    bash -c '. '"$BOOTSTRAP_TARGET"'; exec "$@"' bash "${COMMAND[@]}"
fi

if [ "$ATTACH" = "0" ]; then
  echo "    No shell attached (--sleep)."
  echo ""
  echo "Container '$CONTAINER_NAME' is running in the background, with"
  echo "$SANDBOX_ENV_TARGET loaded for claude and bash sessions."
  echo ""
  echo "   Get a shell:  tmux new-session -A -s $TMUX_SESSION \\"
  echo "                   docker exec -u agent -it $CONTAINER_NAME bash"
  echo "   Or plainly:   docker exec -u agent -it $CONTAINER_NAME bash"
  echo "   Discard it:   docker rm -f $CONTAINER_NAME"
  exit 0
fi

# tmux runs on the host: the task images carry none, and what has to survive a
# dropped ssh session is the terminal, not the container.
if [ ! -t 0 ] || [ ! -t 1 ]; then
  echo "    Not a terminal, so no shell was attached."
  echo ""
  echo "Container '$CONTAINER_NAME' is running. Get a shell with:"
  echo "   docker exec -u agent -it $CONTAINER_NAME bash"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "    No tmux on this host, so this is a plain docker exec into"
  echo "    '$CONTAINER_NAME'. Closing the shell leaves the container running;"
  echo "    'docker rm -f $CONTAINER_NAME' is what discards it."
  exec docker exec -u agent -it "$CONTAINER_NAME" bash
fi

# Inside tmux already: a nested server is a second prefix key for no gain.
if [ -n "${TMUX:-}" ]; then
  echo "    Already inside tmux, so no new session. Attaching a shell to"
  echo "    '$CONTAINER_NAME'."
  exec docker exec -u agent -it "$CONTAINER_NAME" bash
fi

echo "    Attaching to tmux session '$TMUX_SESSION' -> container"
echo "    '$CONTAINER_NAME'. Ctrl-b d detaches and leaves it running."
# -A attaches to an existing session instead of failing on a duplicate name.
exec tmux new-session -A -s "$TMUX_SESSION" "docker exec -u agent -it $CONTAINER_NAME bash"

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
"""Local compile-cache store: ~/.xla_cache/<image-tag>/<key>/compile_cache.tgz

A job saves its compile cache (`forward_run.py --save-cache`) to its output dir; after the job's
files are fetched, `collect` / `cache-pull` file it here; `forward-cmd` hands the matching entry
to the next job as an input, with `--cache-in`. The key hashes the image, the exact engine arguments (bucket included) and the
plan's env (LIBTPU_INIT_ARGS / XLA_FLAGS change what is compiled) and the sha256 of each
vllm_torchtpu patch applied on top of the image, so two configs, shapes, envs or patch stacks
never share an entry by accident. vLLM applies its own finer key
inside the tarball; this one only decides which tarball to hand over.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil

ROOT = os.path.expanduser("~/.xla_cache")


def image_tag(image: str) -> str:
    return image.rsplit("/", 1)[-1].replace(":", "_")


def key(image: str,
        engine_argv: list[str],
        env: dict | None = None,
        patches: list[str] | None = None) -> str:
    # an empty stack keeps the pre-patch key, so existing entries still hit
    parts = [image, engine_argv, sorted((env or {}).items())]
    return hashlib.sha1(
        json.dumps(parts + (
            [list(patches)] if patches else [])).encode()).hexdigest()[:16]


def entry_dir(image: str,
              engine_argv: list[str],
              env: dict | None = None,
              patches: list[str] | None = None) -> str:
    return os.path.join(ROOT, image_tag(image),
                        key(image, engine_argv, env, patches))


def find(image: str,
         engine_argv: list[str],
         env: dict | None = None,
         patches: list[str] | None = None) -> str | None:
    d = entry_dir(image, engine_argv, env, patches)
    return d if os.path.isfile(os.path.join(d, "compile_cache.tgz")) else None


def store(job_dir: str,
          image: str,
          plan_env: dict | None = None,
          patches: list[str] | None = None) -> str:
    """File a fetched job's compile cache under its key (read from the job's result.json).
    `plan_env` is for results written before forward_run recorded it: pass the plan's twin env.
    `patches`: the sha256 of each patch the job applied, in order."""
    res = sorted(
        glob.glob(os.path.join(job_dir, "outputs", "**", "wmb", "forward",
                               "result.json"),
                  recursive=True))
    if not res:
        raise FileNotFoundError(f"no wmb/forward/result.json under {job_dir}")
    r = json.load(open(res[0]))
    tgz = os.path.join(os.path.dirname(res[0]), "compile_cache.tgz")
    if not os.path.isfile(tgz):
        raise FileNotFoundError(
            f"{tgz} missing: was the job run with --save-cache?")
    env = r.get("plan_env", plan_env)
    if env is None:
        raise ValueError(
            "result.json has no plan_env (written before it was recorded): pass the "
            "plan with `cache-pull <job> --steps <plan>.steps.json`")
    d = entry_dir(image, r["engine_argv"], env, patches)
    os.makedirs(d, exist_ok=True)
    shutil.copy2(tgz, os.path.join(d, "compile_cache.tgz"))
    json.dump(
        {
            "image": image,
            "engine_argv": r["engine_argv"],
            "plan_env": env,
            "patches": patches or [],
            "from_job_dir": job_dir,
            "warm_boot_s": r.get("warm_boot_s"),
            "boot_s": r.get("boot_s")
        },
        open(os.path.join(d, "meta.json"), "w"),
        indent=1)
    return d

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
"""Which knobs a serving image has, read from its installed source.

    python3 knob_probe.py --findings findings.json --out knobs_present.json \
        [--patch rpa_decode_block_override.patch ...]

Run it in the serving image (it needs nothing else from this package). For
every knob a finding proposes, it looks for the knob in the source of the
installed `vllm`, `vllm_torchtpu` and `tpu_inference` packages: an env var
as a quoted literal, a CLI flag as the flag or its argparse dest. The
packages are located, never imported, so no TPU runtime starts.

With `--patch`, it also dry-runs each vllm_torchtpu patch against the
installed plugin: "applies", "applied" (already in the image) or "fails". A
knob the image lacks but a patch that applies adds can still be measured,
with the patch.

Writes {"present": {"<name>": true|false}, "patches": {"<file>": status},
"roots": [...]} for propose.py's --knobs-present. An image that lacks a knob
would run the candidate as the baseline and spend a measurement on nothing.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

_PACKAGES = ("vllm", "vllm_torchtpu", "tpu_inference")


def package_roots(packages=_PACKAGES):
    """Installed directories of `packages`, located without importing them."""
    roots = []
    for name in packages:
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            roots += list(spec.submodule_search_locations)
    return roots


def source_text(roots):
    parts = []
    for root in roots:
        for dirpath, _, files in os.walk(root):
            for f in files:
                if f.endswith(".py"):
                    try:
                        with open(os.path.join(dirpath, f),
                                  errors="replace",
                                  encoding="utf-8") as fh:
                            parts.append(fh.read())
                    except OSError:
                        continue
    return "\n".join(parts)


def tokens(name, kind):
    """Strings, any of which shows knob `name` exists in the source."""
    if kind == "cli" or name.startswith("--"):
        return (name, name.lstrip("-").replace("-", "_"))
    leaf = name.rsplit(".", 1)[-1]
    return (f'"{leaf}"', f"'{leaf}'")


def patch_status(patch, plugin_src):
    """Whether `patch` applies to the plugin source under `plugin_src`."""
    if shutil.which("patch"):
        argv = [
            "patch", "-p1", "--dry-run", "--silent", "-f", "-d", plugin_src
        ]
        reverse = argv[:2] + ["-R"] + argv[2:]
    else:
        argv = ["git", "-C", plugin_src, "apply", "-p1", "--check"]
        reverse = argv[:4] + ["-R"] + argv[4:]

    def ok(cmd):
        with open(patch, "rb") as f:
            return subprocess.run(cmd,
                                  stdin=f,
                                  capture_output=True,
                                  check=False).returncode == 0

    if ok(reverse):
        return "applied"
    return "applies" if ok(argv) else "fails"


def knob_names(findings):
    """(name, kind) of every knob change in an analyze.py findings.json."""
    seen = {}
    for profile in findings.get("profiles", []):
        for f in (profile.get("result") or {}).get("findings", []):
            for kc in f.get("knob_changes", []):
                seen.setdefault(kc["name"], kc.get("kind", ""))
    return seen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--findings", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch",
                        action="append",
                        default=[],
                        help="A vllm_torchtpu patch to dry-run; repeatable")
    args = parser.parse_args(argv)
    with open(args.findings, encoding="utf-8") as f:
        names = knob_names(json.load(f))
    roots = package_roots()
    if not roots:
        print("CRITICAL ERROR: none of vllm, vllm_torchtpu, tpu_inference is"
              f" installed for {sys.executable}")
        return 1
    text = source_text(roots)
    present = {
        n: any(t in text for t in tokens(n, k))
        for n, k in names.items()
    }
    patches = {}
    plugin = importlib.util.find_spec("vllm_torchtpu")
    for patch in args.patch:
        if plugin is None or not plugin.submodule_search_locations:
            patches[os.path.basename(patch)] = "fails"
            continue
        src = os.path.dirname(
            os.path.abspath(list(plugin.submodule_search_locations)[0]))
        patches[os.path.basename(patch)] = patch_status(patch, src)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "present": present,
            "patches": patches,
            "roots": roots
        },
                  f,
                  indent=1)
    for n, ok in sorted(present.items()):
        print(f"{'present' if ok else 'ABSENT '}  {n}")
    for name, status in sorted(patches.items()):
        print(f"patch {status:8s} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

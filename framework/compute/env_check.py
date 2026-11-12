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
"""Check that a TPU host has the software tpu-agent-suite workloads need.

Run it where workloads run -- on the TPU VM, or inside the workload image --
with the interpreter they use; `compute setup verify` does this on backends
where that takes seconds:

    python3 env_check.py --requirements requirements-tpu.txt

or, where copying a file there needs more than running a script does (a Job
carries the script but no files without a staging bucket), with the lines of
that file as arguments:

    python3 env_check.py --require='torch_tpu' --require='jax>=0.11.0'

With `--host` it checks the machine the agents run on instead: Python and the
host-side requirements, without the TPU software and devices. The installer
runs that on every install.

    python3 env_check.py --host --requirements requirements-client.txt

With `--image` the target is a container (gke_job, gke_pod), where a
`pip install` lasts only as long as the pod, so each fix names the image
rather than a command to run in it.

With `--versions` it checks nothing and prints, as JSON, the release of each
package whose documentation is per release (`TRACKED`), null where one is not
installed. A target check (`--require`/`--requirements`, not `--host`) ends
with the same JSON on a `Versions: ` line, which `compute setup verify`
records, so a target that can't be asked cheaply still has an answer;
`compute versions` asks a local target directly.

    python3 env_check.py --versions

It ships in the distribution's framework/compute/, and is run as a script, never
imported: the target has no framework to import it from.

It prints one line per check, with how to fix each failure, and
exits 1 when anything fails. Standard library only (plus `packaging`, which
transformers installs, for version bounds), because what it checks for may be
missing.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import sysconfig

MIN_PYTHON = (3, 12)
# Packages whose documentation is written per release: what an agent cites
# for them has to be the page for the release the target has.
TRACKED = ("jax", "jaxlib", "libtpu", "torch", "torch_tpu")
# The target check's last line, which `compute setup verify` records.
VERSIONS_PREFIX = "Versions: "
# Where setting up each machine is documented, so a fix points there rather
# than repeating commands that change.
HOST_DOC = "docs/install.md#1-host-prerequisites"
TARGET_DOC = "docs/install.md#5-set-up-the-compute-target"
IMAGE_DOC = "docs/install.md#container-image-gke_job-gke_pod"
# Maps a requirement to the distributions that conflict with it. Two
# distributions conflict when both install the same import package. pip
# treats them as unrelated, so it reports no conflict, and the distribution
# installed last owns the shared files. The requirement's version check can
# then pass on files that the other distribution wrote.
CONFLICTS = {"xprof": ("xprof-nightly", )}


def version_of(name: str) -> str | None:
    """The installed release of distribution `name`; None if absent."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def tracked_versions() -> dict[str, str | None]:
    """The installed release of each `TRACKED` package; None if absent."""
    return {name: version_of(name) for name in TRACKED}


def satisfies(version: str, specifier: str) -> bool | None:
    """Whether `version` meets `specifier`; None when it cannot be compared."""
    try:
        from packaging.specifiers import SpecifierSet  # noqa: PLC0415
        from packaging.version import Version  # noqa: PLC0415
        return SpecifierSet(specifier).contains(Version(version),
                                                prereleases=True)
    except Exception:  # noqa: BLE001 - no packaging, or an odd version
        return None


def parse_requirement(requirement: str) -> tuple[str, str] | None:
    """(name, specifier) for a requirements-file line.

    None for a pip option, URL or path. Environment markers are dropped, not
    evaluated.
    """
    match = re.fullmatch(
        r"([A-Za-z0-9][\w.-]*)\s*(?:\[[^\]]*\])?\s*([<>=!~][^@]*)?",
        requirement.split(";", 1)[0].strip())
    return (match[1], (match[2] or "").strip()) if match else None


def installed_conflicts(name: str) -> list[str]:
    """The distributions `CONFLICTS` lists for `name` that are installed."""
    return [other for other in CONFLICTS.get(name, ()) if version_of(other)]


def fix_command(requirements: list[str], python: str | None = None) -> str:
    """The pip command that makes `requirements` pass.

    It uninstalls a conflicting distribution together with the requirement
    it shares files with, then installs the requirements. It runs `python`
    (default: this interpreter) as `-m pip`, so it changes the environment
    that was checked rather than whichever `pip` is first on PATH.
    """
    pip = f"{python or sys.executable} -m pip"
    remove = []
    for line in requirements:
        parsed = parse_requirement(line)
        if parsed is None:
            continue
        others = installed_conflicts(parsed[0])
        if others:
            remove += others + ([parsed[0]] if version_of(parsed[0]) else [])
    install = f"{pip} install " + " ".join(f"'{line}'"
                                           for line in requirements)
    if remove:
        return f"{pip} uninstall -y {' '.join(remove)} && {install}"
    return install


def check_requirement(requirement: str,
                      image: bool = False) -> tuple[str, str, str, str] | None:
    """(name, status, detail, fix) for one requirements-file line.

    None for a line that names no package to check: pip options (`-r`, `-e`,
    `--index-url`), and URLs or paths. A requirement also fails when a
    distribution `CONFLICTS` lists for it is installed. `image` says the
    target is a container image, which a fix has to replace rather than
    install into.
    """
    parsed = parse_requirement(requirement)
    if parsed is None:
        return None
    name, spec = parsed
    version = version_of(name)
    others = installed_conflicts(name)
    # torch comes with torch_tpu, pinned to the ABI torch_tpu was built
    # against; installing it on its own is how the two drift apart.
    if image and name == "torch":
        fix = ("use an image whose torch_tpu brings a matching torch; see "
               f"{IMAGE_DOC}")
    elif image and others:
        fix = f"use an image without {', '.join(others)}; see {IMAGE_DOC}"
    elif image:
        fix = f"use an image with '{requirement}'; see {IMAGE_DOC}"
    elif name == "torch":
        fix = ("reinstall torch_tpu, which brings a matching torch; see "
               f"{TARGET_DOC}")
    else:
        fix = fix_command([requirement])
    if others:
        installed = ", ".join(f"{other} {version_of(other)}"
                              for other in others)
        found = (f"{version} installed next to {installed}"
                 if version else f"not installed, but {installed} is")
        return (name, "fail",
                f"{found}. Both install the same import package, and pip "
                "does not report that as a conflict", fix)
    if version is None:
        return name, "fail", f"not installed; needs {spec or 'any'}", fix
    ok = satisfies(version, spec) if spec else True
    if ok is None:
        return (name, "warn", f"{version}; could not compare with {spec}",
                "pip install packaging")
    if not ok:
        return name, "fail", f"{version} installed; needs {spec}", fix
    return name, "ok", f"{version} {spec}".strip(), ""


def run_checks(requirements: list[str],
               host: bool = False,
               image: bool = False) -> list[tuple[str, str, str, str]]:
    """(name, status, detail, fix) for each check of this machine.

    The checks are Python, torch_tpu, each of `requirements`, a warning if
    the deprecated torch_xla is installed, and the TPU devices. With `host`,
    only Python and `requirements` are checked, for the machine the agents
    run on. With `image` the target is a container image, so fixes name the
    image instead of a pip install.
    """
    current = ".".join(map(str, sys.version_info[:3]))
    need = ".".join(map(str, MIN_PYTHON))
    checks = [("python", "ok", f"{current} ({sys.executable})",
               "") if sys.version_info[:2] >= MIN_PYTHON else
              ("python", "fail", f"{current}; needs {need}+",
               f"see {HOST_DOC if host else TARGET_DOC}")]
    if not host:
        torch_tpu = version_of("torch_tpu")
        checks.append(("torch_tpu", "ok", torch_tpu, "") if torch_tpu else (
            "torch_tpu", "fail", "not installed",
            f"use an image with torch_tpu; see {IMAGE_DOC}" if image else
            f"install it (it brings the matching torch); see {TARGET_DOC}"))
    checks += [
        check for check in (check_requirement(line, image)
                            for line in requirements) if check
    ]
    if host:
        return checks
    if version_of("torch_xla"):
        checks.append(("torch_xla", "warn", "installed; deprecated, use "
                       "torch_tpu", "pip uninstall -y torch_xla"))
    sys_accel = "/sys/class/accel"
    has_sys_accel = os.path.isdir(sys_accel)
    devices = sorted(
        glob.glob("/dev/vfio/[0-9]*") or [
            d for d in glob.glob("/dev/accel[0-9]*") if not has_sys_accel
            or os.path.exists(os.path.join(sys_accel, os.path.basename(d)))
        ])
    checks.append(("tpu_devices", "ok", f"{len(devices)} device(s)",
                   "") if devices else (
                       "tpu_devices", "fail", "no /dev/accel* or /dev/vfio/N",
                       "run on a TPU VM, or in a pod that requests TPU chips"))
    return checks


def read_requirements(path: str) -> list[str]:
    """The requirement lines of a requirements file, without comments."""
    with open(path, encoding="utf-8") as handle:
        lines = (line.split("#", 1)[0].strip() for line in handle)
        return [line for line in lines if line]


def failing_requirements(requirements: list[str]) -> list[str]:
    """The lines of `requirements` this interpreter doesn't meet.

    That is, not installed, installed at a version outside the bound, or
    installed next to a distribution that `CONFLICTS` lists for it.
    """
    failing = []
    for line in requirements:
        check = check_requirement(line)
        if check and check[1] == "fail":
            failing.append(line)
    return failing


def externally_managed() -> bool:
    """Whether pip refuses to install into this interpreter (PEP 668).

    That is a Debian, Ubuntu, gLinux or Homebrew Python outside a virtual
    environment, which marks its stdlib with EXTERNALLY-MANAGED.
    """
    if sys.prefix != sys.base_prefix:
        return False
    return os.path.isfile(
        os.path.join(sysconfig.get_path("stdlib"), "EXTERNALLY-MANAGED"))


def ask_to_install(requirements: list[str], python: str | None = None) -> str:
    """What to tell the agent when `requirements` are missing on the host.

    Names the interpreter, since the runtime's Python is often not the
    `pip` on PATH, and says to ask the user rather than work around the
    gap: agents missing a package have looked up or measured what it
    would have told them instead.
    """
    quoted = " ".join(f"'{line}'" for line in requirements)
    noun, is_, unmet = (("this package", "it is",
                         "this requirement") if len(requirements) == 1 else
                        ("these packages", "they are", "these requirements"))
    if python is None and externally_managed():
        # The hooks run on the python3 first on PATH when the harness
        # started: often the system's, when the user installed into a venv
        # they didn't activate. pip refuses a system Python, so asking for
        # that install would leave the session waiting on a command that
        # can't work.
        # Debian's system Python ships without ensurepip, so `venv` fails
        # there until python3-venv is installed (found on gLinux).
        apt = ("" if importlib.util.find_spec("ensurepip") else
               "sudo apt install python3-venv && ")
        return (
            f"Before anything else, ask the user to install {noun}: "
            f"{', '.join(requirements)}. {sys.executable} is the "
            "system's Python, which pip won't install into. If they "
            "installed these in a virtual environment, they should "
            "activate it and restart this session from that shell: the "
            "session runs on the python3 that was on PATH when it "
            f"started. Otherwise the commands are `{apt}python3 -m venv "
            "~/.venvs/tpu-agent && . ~/.venvs/tpu-agent/bin/activate "
            f"&& pip install {quoted}`, then restart the session from that "
            f"shell. Don't start other work, or look up or measure what "
            f"{noun} would have told you, until {is_} installed.")
    python = python or sys.executable
    # The instruction comes first: with it mid-sentence, agents on Claude
    # Code and Antigravity quoted the note and then started other work.
    return (f"Before anything else, ask the user to run "
            f"`{fix_command(requirements, python)}`. {python} does not meet "
            f"{unmet}: {', '.join(requirements)}. Don't start other work, or "
            f"look up or measure what {noun} would have told you, until the "
            "user has run it.")


def main(argv: list[str] | None = None) -> int:
    """Print the checks and return 1 if any failed, else 0.

    With `--versions`, print the tracked package releases as JSON instead
    and check nothing.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--requirements",
                        help="the package's requirements-tpu.txt, or "
                        "requirements-client.txt with --host")
    parser.add_argument("--require",
                        action="append",
                        default=[],
                        metavar="SPEC",
                        help="one requirement line, in place of or as well as "
                        "--requirements; repeatable. Pass it as "
                        "--require=SPEC so a line starting with '-' is not "
                        "read as an option")
    parser.add_argument("--versions",
                        action="store_true",
                        help="print the release of each tracked package as "
                        "JSON, and check nothing")
    parser.add_argument("--host",
                        action="store_true",
                        help="check the machine the agents run on, not a "
                        "TPU target")
    parser.add_argument("--image",
                        action="store_true",
                        help="the target is a container image, so a fix "
                        "names the image instead of a pip install")
    args = parser.parse_args(argv)
    if args.versions:
        print(json.dumps(tracked_versions(), sort_keys=True))
        return 0
    if args.requirements is None and not args.require:
        # With neither, only the interpreter and devices would be checked,
        # and "ok" would read as the package's requirements being met.
        parser.error("pass --requirements, or --require at least once")
    requirements = list(args.require)
    if args.requirements:
        requirements += read_requirements(args.requirements)
    checks = run_checks(requirements, args.host, args.image)
    width = max(len(name) for name, *_ in checks)
    marks = {"ok": "ok", "warn": "WARN", "fail": "FAIL"}
    for name, status, detail, fix in checks:
        print(f"[{marks[status]:<4}] {name:<{width}}  {detail}")
        if fix:
            print(f"       {'':<{width}}  fix: {fix}")
    failed = any(status == "fail" for _, status, _, _ in checks)
    print(f"\nStatus: {'fail' if failed else 'ok'}")
    if not args.host:
        print(VERSIONS_PREFIX + json.dumps(tracked_versions(), sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

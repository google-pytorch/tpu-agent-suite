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
"""Create and change the compute profiles in environment.json.

`compute setup plan` works out what a profile needs (backend, project, cluster,
...) and suggests a value for each from this machine. `compute setup apply` creates or updates one profile from a set of
answers.
`compute setup verify` checks that a profile's target works and records the
result in verification.json. `config show/set/use/remove` lists the profiles
and their setup state, changes individual settings, picks which profile a run
(or, by default, every run) uses, and removes profiles.

Two rules hold for every command here:

- A profile is saved only if it is complete and valid. Every change goes
  through `validate_profile`, which checks required and conditional fields
  (`SETUP_FIELDS`), field formats, the environment schema, and the driver
  itself. Otherwise the command returns INCOMPLETE, listing what is missing or
  wrong, and nothing is written.
- Every write goes through `_update_environment`, which holds a lock so
  concurrent sessions can't lose each other's changes, and bumps
  `revision`.

A verify result is stored under a hash of the profile's settings
(`config.profile_hash`) with the package version, so changing any setting or
upgrading the package makes the profile unverified again without anything
having to clear it.
"""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import pathlib
import re
import sys
from typing import Any, Callable

from framework import jsonio, paths, utils

from . import config, env_check, probes
from .drivers import (ComputeConfigError, ComputeDriver, PreflightCheck,
                      get_driver, list_drivers, registry,
                      validate_field_format)

Path = pathlib.Path

# The script the software check runs on the target.
ENV_CHECK = Path(__file__).resolve().parent / "env_check.py"
# A smoke-test Job can wait minutes for a node to scale up and an image to pull.
SMOKE_TEST_TIMEOUT_SEC = 900


def _package_file(*parts: str) -> Path:
    """A file of the package this runtime belongs to (same version).

    Installed, the runtime sits beside `environment/` and `schemas/`; in the
    repository they are under `package/`. Found from this file rather than
    from `utils`, because the installer imports its own copy of utils.py.
    """
    root = Path(__file__).resolve().parents[2]
    for base in (root, root / "package"):
        if (path := base.joinpath(*parts)).is_file():
            return path
    raise ComputeConfigError(
        f"{'/'.join(parts)} is not beside the runtime (looked under {root}); "
        "reinstall the package.",
        error_code="MISSING_FILE")


def _questions() -> dict[str, dict[str, Any]]:
    """questions.json entries keyed by question name."""
    spec = jsonio.load_json(_package_file("environment", "questions.json"))
    return {q["name"]: q for q in spec["questions"]}


def _schema_question(field: str) -> dict[str, Any]:
    """A field with no question (an on_demand one) takes its schema type."""
    params = _schema()["properties"]["profiles"]["additionalProperties"][
        "properties"]["params"]["properties"]
    return {"name": field, "type": (params.get(field) or {}).get("type")}


def _schema() -> dict[str, Any]:
    """Return the environment.json schema, extended for external drivers.

    The package schema lists the built-in backends in the `backend` enum and
    their setup fields in `params`, and it rejects any other value. An
    external driver (see `drivers/external.py`) is not known to the package,
    so this function adds each external backend name to the enum. It also
    adds each setup field of an external driver that the schema does not
    already define. These added fields accept any JSON value, because the
    driver's own `from_params` validates them.

    Example: an external driver registered as `my_backend` with
    `SETUP_FIELDS = SetupFields(required=("target",))` makes a profile with
    `"backend": "my_backend"` and `"params": {"target": "a"}` valid.
    """
    schema = jsonio.load_json(_package_file("schemas", "environment.json"))
    profile = schema["properties"]["profiles"]["additionalProperties"]
    backend = profile["properties"]["backend"]
    params = profile["properties"]["params"]["properties"]
    for name in registry.EXTERNAL_DRIVERS:
        driver_cls = get_driver(name)
        if driver_cls is None:
            continue
        if name not in backend["enum"]:
            backend["enum"].append(name)
        for field in sorted(driver_cls.SETUP_FIELDS.all_fields()):
            params.setdefault(
                field, {"description": f"{name} backend (external driver)."})
    return schema


def _is_profile_field(question: dict[str, Any] | None) -> bool:
    return question is None or utils.is_profile_question(question)


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _question(field: str, questions: dict[str, dict[str,
                                                    Any]]) -> dict[str, Any]:
    """A field to ask about, as `missing` / `newly_applicable` report it."""
    q = questions.get(field, {})
    return {
        "name": field,
        "prompt": q.get("prompt", f"Value for '{field}'"),
        **{
            k: q[k]
            for k in ("default", "choices") if k in q
        },
    }


def validate_profile(name: str,
                     stored: dict[str, Any],
                     answers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge `answers` into the `stored` profile and check the result.

    Answers are flat `{field: value}`; `""` or null unsets a field. With the
    same backend the stored params are kept and overlaid; after a backend
    change only fields the new backend knows survive. Returns `status` "ok"
    with the merged `backend`/`params`, or "INCOMPLETE" with `missing`,
    `newly_applicable` and `invalid`. With no answers, it says whether the
    stored profile is complete.
    """
    answers = answers or {}
    questions = _questions()
    backend = str(answers.get("backend") or stored.get("backend")
                  or "").strip()
    result: dict[str, Any] = {
        "status": "INCOMPLETE",
        "profile": name,
        "backend": backend or None,
        "missing": [],
        "newly_applicable": [],
        "invalid": [],
    }
    invalid = result["invalid"]

    def bad(field: str, reason: str, **extra: Any) -> None:
        invalid.append({"name": field, "reason": reason, **extra})

    def complete() -> bool:
        return not (result["missing"] or result["newly_applicable"] or invalid)

    driver_cls = get_driver(backend) if backend else None
    if not backend:
        result["missing"].append(_question("backend", questions))
    elif driver_cls is None:
        bad("backend",
            f"unknown backend; must be one of {list_drivers()}",
            value=backend,
            choices=list_drivers())
    if driver_cls is None:
        return result

    sf = driver_cls.SETUP_FIELDS
    allowed = sf.all_fields()
    params = dict(stored.get("params") or {})
    if stored.get("backend") != backend:
        params = {k: v for k, v in params.items() if k in allowed}
    for key, raw in answers.items():
        if key == "backend":
            continue
        if key not in allowed:
            hint = ("; set it with `config set`"
                    if not _is_profile_field(questions.get(key)) else "")
            bad(key,
                f"not a setup field for backend {backend!r}{hint}",
                value=raw)
        elif _blank(raw):
            params.pop(key, None)
        else:
            try:
                params[key] = utils.parse_answer(
                    questions.get(key) or _schema_question(key), raw)
            except utils.TpuAgentSuiteError as exc:
                bad(key, str(exc), value=raw)

    flagged = {e["name"] for e in invalid}
    for field in sf.required:
        if field in flagged:
            continue
        default = questions.get(field, {}).get("default")
        if _blank(params.get(field)) and not _blank(default):
            params[field] = default
        elif _blank(params.get(field)):
            result["missing"].append(_question(field, questions))
    # A conditional field goes with its trigger: it is dropped when the
    # trigger is unset, and asked for only when this call sets the trigger.
    stored_params = stored.get("params") or {}
    for child, trigger in sf.conditional.items():
        if _blank(params.get(trigger)):
            params.pop(child, None)
        elif (_blank(stored_params.get(trigger)) and child not in answers
              and _blank(params.get(child))):
            result["newly_applicable"].append(_question(child, questions))
    invalid += [
        err for field, value in params.items()
        if (err := validate_field_format(field, value))
    ]
    profile = {"backend": backend, "params": params}
    for err in utils.validate_instance({"profiles": {
            name: profile
    }}, _schema()):
        location, _, reason = err.partition(": ")
        bad(location.rsplit("/", 1)[-1], reason)
    if complete():
        try:
            driver_cls.from_params(params)
        except ComputeConfigError as exc:
            bad("params", exc.message)
    if complete():
        result.update(status="ok", params=params)
    return result


def _environment(package: str | None,
                 workspace: Path | str | None) -> tuple[dict[str, Any], Path]:
    """The installed package's environment.json and its path; never guessed."""
    if not package:
        raise ComputeConfigError(
            "A --package is required to read or update environment.json "
            "(e.g. --package tpu-agent-suite@<version>).",
            error_code="MISSING_FIELD")
    ws = Path(workspace) if workspace is not None else Path.cwd()
    try:
        env, path = paths.resolve_environment(package, ws)
    except ValueError as exc:
        raise ComputeConfigError(f"environment.json is not valid JSON: {exc}",
                                 error_code="INVALID_FIELD") from exc
    except utils.TpuAgentSuiteError as exc:
        # This package's ledger without environment.json, in the workspace or
        # (a user-scope install) the central store: apply may repair it.
        base = utils.parse_package_tag(package)[0]
        for tpu_agent_suite_dir in (paths.tpu_agent_suite_dir_for(ws),
                                    paths.get_tpu_agent_suite_home()):
            ledger = tpu_agent_suite_dir / paths.OWNERSHIP_FILE
            if (ledger.is_file() and
                    jsonio.load_json_or_empty(ledger).get("package") == base):
                return {}, tpu_agent_suite_dir / paths.ENVIRONMENT_FILE
        raise ComputeConfigError(str(exc), error_code="MISSING_FIELD") from exc
    if path is None:
        raise ComputeConfigError(
            f"Package {package!r} is not installed in {ws.resolve()} or in "
            "central storage; install it first.",
            error_code="MISSING_FIELD")
    return env, path


def _update_environment(
        package: str | None, workspace: Path | str | None,
        mutate: Callable[[dict[str, Any]], bool]) -> tuple[dict | None, Path]:
    """Safely read, modify and save the package's environment.json.

    An exclusive lock is held for the whole read-modify-write, so two sessions
    writing at once can't overwrite each other's changes. It locks the
    directory rather than the file, because saving replaces the file.

    `mutate` gets the current contents, with `default` and `profiles` always
    present, edits them in place, and returns True to save or False to leave
    the file alone. Before saving,
    `revision` is bumped and the result is checked against the schema.

    Returns `(saved_document, path)`; `saved_document` is None if nothing was
    written.
    """
    _, path = _environment(package, workspace)
    with _locked(path.parent):
        # Read again under the lock: another writer may have changed it.
        existing = _environment(package, workspace)[0]
        profiles, default = config.normalize_environment_profiles(existing)
        # A file from before profiles keeps its settings under `compute`,
        # which reads as no profiles and which the schema rejects. Dropping
        # it lets setting compute up again replace the old format.
        doc = {k: v for k, v in existing.items() if k != "compute"}
        doc.update(default=default, profiles=copy.deepcopy(profiles))
        if not mutate(doc):
            return None, path
        doc["revision"] = int(existing.get("revision") or 0) + 1
        errors = utils.validate_instance(doc, _schema())
        if errors:
            raise ComputeConfigError(
                "Refusing to write an environment.json that fails its schema:\n"
                + "\n".join(errors),
                error_code="INVALID_FIELD")
        jsonio.write_json(path, doc)
        return doc, path


@contextlib.contextmanager
def _locked(directory: Path):
    """Hold an exclusive lock on `directory` (not a file: saves replace it)."""
    lock = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
    finally:
        os.close(lock)


# Profile names are typed on the command line and used as keys, so keep them
# to one plain word.
_PROFILE_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]*")


def _check_profile_name(name: str) -> None:
    if not _PROFILE_NAME.fullmatch(name):
        raise ComputeConfigError(
            f"Invalid profile name {name!r}: use letters, digits, '.', '_' "
            "and '-', starting with a letter or digit.",
            error_code="INVALID_FIELD")


def plan_setup(*,
               profile: str | None = None,
               backend: str | None = None,
               fields: list[str] | None = None,
               package: str | None = None,
               workspace: Path | str | None = None,
               run_id: str | None = None) -> dict[str, Any]:
    """Work out what a profile needs, with a suggested value for each.

    Which profile: `profile` (it can be a new one), else the run's profile,
    else the default. Which backend: `backend`, else the profile's.

    Each question comes with its prompt, whether it's required, its
    choices, the profile's current value, and a suggested answer. The
    suggestion is the current value (not with `fields`, which asks again for
    rejected values), else what a probe found (see `probes`), else the
    question's default. `probe_warnings` lists any probe whose tool is
    missing or that timed out, and `note` tells the agent not to answer for
    the user. Nothing is written.
    """
    if profile:
        _check_profile_name(profile.strip())
    env, _ = _environment(package, workspace)
    profiles, _ = config.normalize_environment_profiles(env)
    name, _ = config.resolve_profile_name(env, profile, run_id)
    stored = profiles.get(name or "", {})
    backend = (backend or stored.get("backend") or "").strip()
    driver_cls = get_driver(backend) if backend else None
    if backend and driver_cls is None:
        raise ComputeConfigError(
            f"Unknown backend {backend!r}; must be one of {list_drivers()}.",
            error_code="INVALID_BACKEND")
    questions = _questions()
    sf = driver_cls.SETUP_FIELDS if driver_cls else None
    # The stored values apply would keep: all of them with the same backend,
    # else only the fields the new backend also has.
    params = {
        k: v
        for k, v in (stored.get("params") or {}).items()
        if stored.get("backend") == backend or (sf and k in sf.all_fields())
    }
    result: dict[str, Any] = {
        "status": "ok",
        "profile": name,
        "backend": backend or None
    }
    if sf is None and fields:
        raise ComputeConfigError(
            "--fields needs a backend: pass --backend, or plan with no "
            "--fields to be asked for one.",
            error_code="INVALID_FIELD")
    if sf is None:
        names = ["backend"]
        result["profiles"] = {n: p.get("backend") for n, p in profiles.items()}
    elif fields:
        # Besides the backend's fields, only top-level settings: trajectory
        # ones need consent, as in `config set`.
        unknown = [
            f for f in fields if f not in sf.all_fields() and (
                f not in questions or utils.question_path(questions[f]) != [f])
        ]
        if unknown:
            raise ComputeConfigError(
                f"Unknown field(s) {', '.join(unknown)} for backend "
                f"{backend!r}.",
                error_code="INVALID_FIELD")
        names = list(fields)
    else:
        names = [*sf.required, *sf.optional, *sf.conditional]
    current = {
        n: params.get(n) if _is_profile_field(questions.get(n)) else env.get(n)
        for n in names
    }
    # --fields asks again for values verify rejected, so they are shown as
    # `current` but neither suggested nor kept from the probes.
    known = {
        k: v
        for k, v in {
            **params,
            **current, "backend": backend
        }.items() if not _blank(v) and k not in (fields or ())
    }
    found, warnings = probes.suggest(names, known, list_drivers())
    if warnings:
        result["probe_warnings"] = warnings
    result["note"] = ("Ask the user every question, optional ones too (they "
                      "can be left blank). Never fill in an answer "
                      "that has no suggestion, or run commands to find one. "
                      "With neither a suggestion nor choices, ask in free "
                      "text; never invent options.")
    result["questions"] = []
    for n in names:
        q = _question(n, questions)
        q["required"] = n == "backend" or bool(sf and n in sf.required)
        if sf and n in sf.conditional:
            q["asked_if"] = sf.conditional[n]
        if not _blank(current.get(n)):
            q["current"] = current[n]
        if "current" in q and not fields:
            suggestion = (current[n], "profile")
        elif n in found:
            suggestion = found[n]
        elif not _blank(q.get("default")):
            suggestion = (q["default"], "default")
        else:
            suggestion = None
        if suggestion and isinstance(suggestion[0], list):
            # Several found, none clearly meant: offer them, suggest none.
            # A choice can be an object answering several fields at once
            # (a TPU VM and its zone).
            q["choices"] = suggestion[0]
        elif suggestion:
            q["suggested"], q["suggested_from"] = suggestion
        result["questions"].append(q)
    return result


def apply_setup(*,
                profile: str,
                answers: dict[str, Any],
                package: str | None = None,
                workspace: Path | str | None = None,
                dry_run: bool = False) -> dict[str, Any]:
    """Create or update the compute profile `profile` from `answers`.

    `answers` (`{field: value}`, keyed like questions.json) are merged into
    the profile's stored values; `""` or None removes a field. The merged
    profile is saved to environment.json only if it is complete and valid.
    Otherwise the result has status INCOMPLETE, listing the missing, newly
    required and invalid fields, and the file is not touched. `dry_run`
    validates without saving.

    The saved profile becomes the default if none is set or the default
    names a missing profile (`config use NAME --default` changes it).
    """
    name = (profile or "").strip()
    if not name:
        raise ComputeConfigError(
            "compute setup apply requires a non-empty --profile name.",
            error_code="MISSING_FIELD")
    _check_profile_name(name)
    result: dict[str, Any] = {}

    def mutate(doc: dict[str, Any]) -> bool:
        result.update(save_profile(doc, name, answers))
        return result["status"] == "ok" and not dry_run

    written, path = _update_environment(package, workspace, mutate)
    if written:
        result.update(default=written["default"],
                      revision=written["revision"],
                      path=str(path))
    elif dry_run and result["status"] == "ok":
        result["dry_run"] = True
    return result


def failed_checks(report: dict[str, Any]) -> list[dict[str, Any]]:
    """A check report's failing checks.

    A target that did not answer, with no failing check, gets a `reachable`
    one, so a failure always names what failed.
    """
    failed = [c for c in report["checks"] if c["status"] == "fail"]
    if failed or report["reachable"]:
        return failed
    return [{
        "name": "reachable",
        "status": "fail",
        "detail": f"{report['target']} did not answer",
        "fields": []
    }]


def fields_of(checks: list[dict[str, Any]]) -> list[str]:
    """The settings to ask about again for `checks`, sorted."""
    return sorted({f for c in checks for f in c.get("fields") or []})


def setup_state(name: str, profile: dict[str, Any],
                results: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """A profile's setup state, worked out now, and its last verify result.

    The result is the profile's entry in `results` (verification.json), or
    {} if it has none.

    - NOT_CONFIGURED: the profile is missing settings, or has invalid ones.
    - UNVERIFIED: it hasn't been verified since its settings or the package
      version last changed.
    - READY or FAILED: whether that last verify passed.
    """
    result = results.get(config.profile_hash(profile))
    result = result if isinstance(result, dict) else {}
    if validate_profile(name, profile)["status"] != "ok":
        return "NOT_CONFIGURED", result
    # A new version can pin other software or a new default image.
    if result.get("version") != utils.TPU_AGENT_SUITE_VERSION:
        return "UNVERIFIED", result
    return ("READY" if result.get("passed") else "FAILED"), result


def save_profile(doc: dict[str, Any], name: str,
                 answers: dict[str, Any]) -> dict[str, Any]:
    """Merge `answers` into profile `name` of `doc`, if the result is complete.

    `doc` is an environment.json document with `default` and `profiles`. If
    `doc` has no default, or its default names a missing profile, the saved
    profile becomes it. Returns `validate_profile`'s result; `doc` is
    changed only if its status is ok. `compute setup apply` and `config set`
    both save through here.
    """
    result = validate_profile(name, doc["profiles"].get(name, {}), answers)
    if result["status"] == "ok":
        doc["profiles"][name] = {
            "backend": result["backend"],
            "params": result["params"]
        }
        if doc["default"] not in doc["profiles"]:
            doc["default"] = name
    return result


def config_show(*,
                package: str | None = None,
                workspace: Path | str | None = None,
                run_id: str | None = None,
                profile_override: str | None = None) -> dict[str, Any]:
    """The profiles, the default, and which profile this run resolves to."""
    env, path = _environment(package, workspace)
    profiles, default = config.normalize_environment_profiles(env)
    active, source = config.resolve_profile_name(env, profile_override, run_id)
    results = jsonio.load_json_or_empty(path.parent / paths.VERIFICATION_FILE)
    return {
        **{
            k: v
            for k, v in env.items() if k != "profiles"
        },
        "path": str(path),
        "revision": env.get("revision", 0),
        "default": default,
        "active_profile": active,
        "active_source": source if active else None,
        "profiles": profiles,
        "setup_state": {
            n: setup_state(n, p, results)[0]
            for n, p in profiles.items()
        },
    }


def set_answers(environment: dict[str, Any], questions: dict[str, dict[str,
                                                                       Any]],
                answers: dict[str, Any]) -> list[str]:
    """Store settings outside the compute profiles where their questions put
    them; a blank answer removes one.

    Afterwards a setting is removed if its question would not have been
    asked: its `when` names a setting whose value (or, if unset, default) is not listed, or that would
    not have been asked itself. So turning trajectory sharing off also
    removes its sink, bucket and org. Returns what was removed, with the
    reason. Raises TpuAgentSuiteError for an invalid value.
    """

    def parent(question: dict[str, Any], create: bool) -> dict | None:
        node = environment
        for segment in utils.question_path(question)[:-1]:
            if not isinstance(node.get(segment), dict):
                if not create:
                    return None
                node[segment] = {}
            node = node[segment]
        return node

    def unmet(question: dict[str, Any]) -> str:
        """Why `question` would not be asked now; empty if it would be."""
        for key, allowed in (question.get("when") or {}).items():
            if unmet(questions[key]) or value(questions[key]) not in allowed:
                return (f"only when {key} is "
                        f"{' or '.join(json.dumps(a) for a in allowed)}")
        return ""

    def value(question: dict[str, Any]) -> Any:
        node = parent(question, False) or {}
        return node.get(
            utils.question_path(question)[-1], question.get("default"))

    for key, raw in answers.items():
        blank = raw is None or str(raw).strip() == ""
        node = parent(questions[key], not blank)
        if node is not None:
            leaf = utils.question_path(questions[key])[-1]
            if blank:
                node.pop(leaf, None)
            else:
                node[leaf] = utils.parse_answer(questions[key], raw)
    removed = []
    for q in questions.values():
        node, leaf = parent(q, False) or {}, utils.question_path(q)[-1]
        if not utils.is_profile_question(q) and leaf in node and (reason :=
                                                                  unmet(q)):
            del node[leaf]
            removed.append(f"{q['name']} ({reason})")
    return removed


HARBOR_CHECK_TIMEOUT_SEC = 20


def _harbor_org_exists(name: str) -> bool:
    """Ask the Harbor hub whether an organization with this name exists.

    Reads the same `organization` table harbor's own upload uses to spot orgs
    the uploader is not in (`UploadDB.get_non_member_org_names`). Membership is
    deliberately not required: exports pass `--yes`, so sharing with an org the
    user does not belong to is allowed.
    """
    # Lazy, like the export's own harbor imports: an install without harbor
    # still works, and this check then only warns. asyncio is lazy too, as it
    # adds tens of milliseconds to every hook and runtime call.
    import asyncio

    from harbor.auth.client import create_authenticated_client

    async def lookup() -> bool:
        client = await create_authenticated_client()
        response = await (client.table("organization").select("name").eq(
            "name", name).execute())
        return bool(response.data)

    return asyncio.run(asyncio.wait_for(lookup(), HARBOR_CHECK_TIMEOUT_SEC))


def check_share_org(environment: dict[str, Any],
                    previous: dict[str, Any] | None = None) -> None:
    """Refuse a Harbor organization the hub does not have.

    Harbor checks the name only at upload, after a session has been converted
    and staged, so a wrong name saved here fails every export. A name the hub
    says does not exist is refused now; one that cannot be checked (harbor not
    importable, not logged in, offline) is kept with a warning. Only a name
    that differs from the one in `previous` is checked.
    """

    def share_org(env: dict[str, Any]) -> tuple[dict[str, Any], str]:
        params = (env.get("trajectory") or {}).get("params") or {}
        return params, str(params.get("share_org") or "").strip()

    params, org = share_org(environment)
    if (params.get("enabled") is False or params.get("sink") != "harbor"
            or not org or org == share_org(previous or {})[1]):
        return
    try:
        exists = _harbor_org_exists(org)
    except ImportError:
        # harbor is optional (not in requirements-client.txt); the export
        # itself refuses the harbor sink without it, so this is only a heads-up.
        print(
            f"WARNING: harbor is not installed here, so the Harbor"
            f" organization {org!r} was not checked, and exports to Harbor"
            " will not work until it is (`pip install harbor==0.23.0`).",
            file=sys.stderr)
        return
    except Exception as exc:  # noqa: BLE001 - reported, never blocks
        print(
            f"WARNING: could not check the Harbor organization {org!r}"
            f" ({type(exc).__name__}: {exc}). If it does not exist, every"
            " export will be refused at upload.",
            file=sys.stderr)
        return
    if not exists:
        raise utils.TpuAgentSuiteError(
            f"Harbor has no organization named {org!r}, so every export"
            " would be refused at upload. Use the exact name (`harbor auth"
            " org list` shows yours), or leave it blank to keep uploads"
            " private.")


def config_set(*,
               assignments: list[str],
               profile: str | None = None,
               package: str | None = None,
               workspace: Path | str | None = None,
               run_id: str | None = None,
               user_confirmed: bool = False) -> dict[str, Any]:
    """Change settings in environment.json from `key=value` strings.

    A profile field (e.g. `zone=us-central1-a`) changes the profile compute
    commands would use: `profile` if given, else the one run `run_id` is
    pinned to, else the default. It goes through the same checks as
    `compute setup apply`, so an incomplete or invalid result is returned as
    INCOMPLETE and nothing is saved. An empty value (`zone=`) removes the
    field.

    Any other questions.json setting, such as `xprof_interpreter`, is stored
    where its question puts it, and settings that then no longer apply are
    removed and listed in `removed` (see `set_answers`).
    Trajectory-sharing settings send the user's sessions somewhere, so they
    are refused unless `user_confirmed` says the user agreed to the change.

    All assignments are applied together or not at all.
    """
    pairs: dict[str, str] = {}
    for item in assignments:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ComputeConfigError(
                f"Invalid assignment {item!r}; expected key=value.",
                error_code="INVALID_FIELD")
        pairs[key.strip()] = value
    questions = _questions()
    top_level = {
        k: pairs.pop(k)
        for k in list(pairs) if not _is_profile_field(questions.get(k))
    }
    for key in top_level:
        if (utils.question_path(questions[key])[0] == "trajectory"
                and not user_confirmed):
            raise ComputeConfigError(
                f"{key} is a trajectory-sharing setting, so it needs the "
                "user's consent: ask them, and pass --user-confirmed only if "
                "they agreed.",
                error_code="INVALID_FIELD")
    result: dict[str, Any] = {"status": "ok"}
    if top_level:
        # The Harbor lookup can take seconds, so it runs before the lock
        # rather than holding up every other session's writes.
        current = _environment(package, workspace)[0]
        proposed = copy.deepcopy(current)
        try:
            set_answers(proposed, questions, top_level)
            check_share_org(proposed, current)
        except utils.TpuAgentSuiteError as exc:
            raise ComputeConfigError(str(exc),
                                     error_code="INVALID_FIELD") from exc

    def mutate(doc: dict[str, Any]) -> bool:
        if removed := set_answers(doc, questions, top_level):
            result["removed"] = removed
        if pairs:
            target, source = config.resolve_profile_name(doc, profile, run_id)
            if not target:
                raise ComputeConfigError(
                    "No profile to change: pass --profile <name>, or create "
                    "one with `compute setup apply`.",
                    error_code="MISSING_FIELD")
            if target not in doc["profiles"]:
                raise config.unknown_profile_error(target, source,
                                                   doc["profiles"])
            result.update(save_profile(doc, target, pairs))
        return result["status"] == "ok"

    written, path = _update_environment(package, workspace, mutate)
    if written:
        result.update(revision=written["revision"], path=str(path))
    return result


def config_use(*,
               profile: str,
               as_default: bool = False,
               package: str | None = None,
               workspace: Path | str | None = None,
               run_id: str | None = None) -> dict[str, Any]:
    """Choose which compute profile is used.

    Normally this pins run `run_id` to `profile` (as `compute_profile` in the
    run's run.json), so every later compute command in that run uses it. With
    `as_default`, it sets environment.json's `default` instead, which every
    run that isn't pinned uses. Fails if `profile` doesn't exist.
    """
    target = (profile or "").strip()
    if not target:
        raise ComputeConfigError("config use requires a profile name.",
                                 error_code="MISSING_FIELD")

    def check(profiles: dict[str, Any]) -> None:
        if target not in profiles:
            raise config.unknown_profile_error(target, "config use", profiles)

    def use(doc: dict[str, Any]) -> bool:
        check(doc["profiles"])
        doc["default"] = target
        return True

    if as_default:
        written, path = _update_environment(package, workspace, use)
        return {
            "status": "ok",
            "scope": "default",
            "profile": target,
            "revision": written["revision"],
            "path": str(path)
        }
    if not run_id:
        raise ComputeConfigError(
            "No active run to switch; pass --run-id, or --default to change "
            "the default profile.",
            error_code="MISSING_FIELD")
    run_file = paths.get_run_root(run_id) / "run.json"
    if not run_file.is_file():
        raise ComputeConfigError(f"Run {run_id!r} has no run.json.",
                                 error_code="MISSING_FIELD")
    check(
        config.normalize_environment_profiles(
            _environment(package, workspace)[0])[0])
    jsonio.write_json(
        run_file, {
            **jsonio.load_json_or_empty(run_file), "compute_profile": target
        })
    return {
        "status": "ok",
        "scope": "run",
        "run_id": run_id,
        "profile": target
    }


def config_remove(*,
                  profile: str,
                  user_confirmed: bool = False,
                  package: str | None = None,
                  workspace: Path | str | None = None,
                  run_id: str | None = None) -> dict[str, Any]:
    """Remove a named compute profile from environment.json.

    Requires `user_confirmed=True` after the user asks to remove the profile.
    If the removed profile was `default`, `default` moves to the first
    remaining profile, or `None` when none remain. If `run_id` has a `run.json`
    pinned to `profile`, `compute_profile` is cleared there too.
    """
    target = (profile or "").strip()
    if not target:
        raise ComputeConfigError("config remove requires a profile name.",
                                 error_code="MISSING_FIELD")
    if not user_confirmed:
        raise ComputeConfigError(
            "Removing a compute profile requires --user-confirmed after the "
            "user asks to remove it.",
            error_code="CONFIRMATION_REQUIRED")

    def remove(doc: dict[str, Any]) -> bool:
        if target not in doc["profiles"]:
            raise config.unknown_profile_error(target, "config remove",
                                               doc["profiles"])
        del doc["profiles"][target]
        if doc.get("default") not in doc["profiles"]:
            doc["default"] = next(iter(doc["profiles"]), None)
        return True

    written, path = _update_environment(package, workspace, remove)
    unpinned_run = False
    if run_id:
        try:
            run_file = paths.get_run_root(run_id) / "run.json"
        except utils.TpuAgentSuiteError:
            run_file = None
        if run_file is not None and run_file.is_file():
            run_doc = jsonio.load_json_or_empty(run_file)
            if run_doc.get("compute_profile") == target:
                run_doc.pop("compute_profile", None)
                jsonio.write_json(run_file, run_doc)
                unpinned_run = True
    result: dict[str, Any] = {
        "status": "ok",
        "profile": target,
        "default": written["default"],
        "revision": written["revision"],
        "path": str(path),
    }
    if unpinned_run:
        result["unpinned_run"] = run_id
    return result


def read_answers(answers_arg: str) -> dict[str, Any]:
    """`--answers`: a JSON object from a file, `-` (stdin), or inline."""
    raw = (answers_arg or "").strip()
    if raw == "-":
        text = sys.stdin.read()
    elif raw.startswith("{"):
        text = raw
    elif Path(raw).is_file():
        text = Path(raw).read_text(encoding="utf-8")
    else:
        raise ComputeConfigError(f"Answers file does not exist: {raw}",
                                 error_code="MISSING_FILE")
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise ComputeConfigError(f"Answers are not valid JSON: {exc}",
                                 error_code="INVALID_FIELD") from exc
    if not isinstance(parsed, dict):
        raise ComputeConfigError("Answers must be a JSON object.",
                                 error_code="INVALID_FIELD")
    return parsed


def host_requirements_note() -> str:
    """'' when this machine meets the package's host requirements, else the
    message asking the user to install what is missing.

    The installer runs the same check once and only warns, to whoever ran
    it; this is said to the agent at every session start until fixed. A
    build that doesn't ship the file is not checked.
    """
    try:
        path = _package_file(utils.HOST_REQUIREMENTS)
    except ComputeConfigError:
        return ""
    missing = env_check.failing_requirements(
        env_check.read_requirements(str(path)))
    return env_check.ask_to_install(missing) if missing else ""


def _reported_versions(report: str) -> dict[str, str | None] | None:
    """The package releases on env_check.py's `Versions:` line, if any."""
    for line in reversed(report.splitlines()):
        if line.startswith(env_check.VERSIONS_PREFIX):
            try:
                found = json.loads(line[len(env_check.VERSIONS_PREFIX):])
            except ValueError:
                return None
            return found if isinstance(found, dict) else None
    return None


def _run_target_check(
        driver: ComputeDriver,
        requirements: Path,
        *,
        name: str = "software",
        timeout: int = 300,
        out: Any = None
) -> tuple[PreflightCheck, dict[str, str | None] | None]:
    """Run env_check.py on the target and return the result as check `name`,
    with the package releases the check reported (None if it reported none).

    The requirements are passed as arguments, not as a file, because some
    backends can send a script but not files (gke_job without a staging
    bucket). On container backends the printed fixes point at the image. A
    failure asks again for the profile's `image` if it has one, otherwise
    `python_interpreter`. The check's report is printed to `out`, all but
    its `Versions:` line, which is for `compute versions`.
    """
    image = driver.runs_in_container
    fields = (["image"] if "image" in driver.SETUP_FIELDS.all_fields() else
              ["python_interpreter"])
    # `--require=` so a pip option line is not read as an option.
    args = [
        f"--require={line}"
        for line in env_check.read_requirements(str(requirements))
    ]
    if image:
        args.append("--image")
    try:
        result = driver.run(str(ENV_CHECK), timeout=timeout, args=args)
    except Exception as exc:  # noqa: BLE001 - a check that cannot run fails
        detail = f"could not run the check on the target: {exc}"
        print(f"[FAIL] {detail}\n", file=out)
        return PreflightCheck(name, "fail", detail, fields=fields), None
    details = result.details or {}
    if (details.get("timed_out_after_sec")
            and details.get("job_status") == "PENDING"):
        # The Job never started (queued, or waiting for a node), so the image
        # was not checked. A busy cluster is not a broken one, so only warn.
        detail = (f"the check was still pending after "
                  f"{details['timed_out_after_sec']}s, so the image and its "
                  f"software are unverified: {result.stderr or 'not started'}")
        print(
            f"[WARN] {detail}\n"
            "       Run it again later with --smoke-test.\n",
            file=out)
        return PreflightCheck(name, "warn", detail,
                              "Run it again later with --smoke-test"), None
    output = (result.stdout or "").rstrip()
    versions = _reported_versions(output)
    report = "\n".join(line for line in output.splitlines()
                       if not line.startswith(env_check.VERSIONS_PREFIX))
    print(report, file=out)
    # If the check printed a report, that explains the failure. If not, it
    # never ran, and stderr says why (for example, the image did not pull).
    if result.exit_code != 0 and result.stderr and not report:
        print(result.stderr.rstrip(), file=out)
    print(file=out)
    if result.exit_code == 0:
        return PreflightCheck(name, "ok", "the software the agents need is "
                              "installed"), versions
    return PreflightCheck(
        name, "fail", report or (result.stderr or "").strip(),
        f"Install what is marked FAIL, or set {fields[0]} to one that has it",
        fields), versions


def check_target(driver: ComputeDriver,
                 params: dict[str, Any],
                 requirements: Path | None,
                 *,
                 smoke_test: bool = False,
                 out: Any = None) -> dict[str, Any]:
    """Check that `driver`'s target works: preflight, then its software.

    The software check runs env_check.py on the target against
    `requirements`, if given. Backends with a quick `run` run it directly.
    Backends that schedule a Job (gke_job) run it as a smoke test, which takes
    minutes, so only with `smoke_test`.
    Progress is printed to `out` (default stdout).

    Returns the preflight report as a dict, with the software check, if it
    ran, added to its `checks` and counted in its `status`, and the package
    releases it reported as `target_versions` (None if it reported none).
    """
    from .cli import format_preflight  # cli imports this module
    report = driver.preflight()
    print("\nChecking the compute target...\n\n" +
          format_preflight(report.to_dict()) + "\n",
          file=out)
    versions = None
    if not (report.reachable and report.status != "fail" and requirements
            and requirements.is_file()):
        return {**report.to_dict(), "target_versions": versions}
    if driver.quick_run:
        if (driver.backend_name == "local"
                and not params.get("python_interpreter")):
            # The local driver would fall back to the interpreter running
            # this code, which need not be the workload's.
            print(
                "Skipping the software check: set python_interpreter to the "
                "workload's Python to have it checked.\n",
                file=out)
        else:
            print("Checking the software on the target...\n", file=out)
            check, versions = _run_target_check(driver, requirements, out=out)
            report.checks.append(check)
    elif driver.scheduled_run and not smoke_test:
        print(
            "Skipping the smoke test: pass --smoke-test to submit a Job that "
            "checks the image and the software in it.\n",
            file=out)
    elif driver.scheduled_run:
        print(
            "Submitting the smoke test; it waits up to "
            f"{SMOKE_TEST_TIMEOUT_SEC // 60} minutes...\n",
            file=out)
        check, versions = _run_target_check(driver,
                                            requirements,
                                            name="smoke_test",
                                            timeout=SMOKE_TEST_TIMEOUT_SEC,
                                            out=out)
        report.checks.append(check)
    return {**report.to_dict(), "target_versions": versions}


def verify_setup(*,
                 profile: str | None = None,
                 package: str | None = None,
                 workspace: Path | str | None = None,
                 run_id: str | None = None,
                 smoke_test: bool = False) -> dict[str, Any]:
    """Check that a saved profile's target works, and record the result.

    Checks the profile that compute commands would use: `profile` if given,
    else the one run `run_id` uses, else the default. Runs `check_target`,
    printing progress to stderr, and saves the result in verification.json
    under the profile's hash with the package version and any failed checks,
    replacing that profile's previous result. The smoke-test Job runs only
    with `smoke_test`; it never prompts, as the agent running it would
    answer for the user. A skipped smoke test still counts as a pass, saved
    with `smoke_test` false.

    Returns status "ok" if nothing failed (warnings are fine), or "FAILED"
    with the checks and the `fields` to ask about again. An incomplete
    profile returns INCOMPLETE, as `compute setup apply` does, and nothing
    runs.
    """
    env, path = _environment(package, workspace)
    name, source = config.resolve_profile_name(env, profile, run_id)
    if not name:
        raise ComputeConfigError(
            "No profile to verify: pass --profile <name>, or create one with "
            "`compute setup apply`.",
            error_code="MISSING_FIELD")
    backend, params = config.parse_compute_section(env, name, source)
    stored = {"backend": backend, "params": params}
    checked = validate_profile(name, stored)
    if checked["status"] != "ok":
        return checked
    backend, params = checked["backend"], checked["params"]
    requirements = None
    try:
        requirements = _package_file(utils.TARGET_REQUIREMENTS)
    except ComputeConfigError:
        print(
            f"Skipping the software check: {utils.TARGET_REQUIREMENTS} is not "
            "installed with the runtime; reinstall the package to have it "
            "checked.",
            file=sys.stderr)
    report = check_target(get_driver(backend).from_params(dict(params)),
                          params,
                          requirements,
                          smoke_test=smoke_test,
                          out=sys.stderr)
    key, checks = config.profile_hash(stored), report["checks"]
    failed = failed_checks(report)
    passed = not failed
    # A smoke test still pending at its timeout learned nothing.
    smoke = any(c["name"] == "smoke_test" and c["status"] != "warn"
                for c in checks)
    record = {"profile": name, "passed": passed, "smoke_test": smoke}
    record.update(version=utils.TPU_AGENT_SUITE_VERSION,
                  checked_at=utils.utc_now_iso())
    if report.get("target_versions") is not None:
        # For `compute versions`. Kept under the profile's hash like the rest
        # of the record, so editing the profile (its image, say) leaves it
        # behind; the image is kept too, to say which one it was read from.
        record["target_versions"] = {
            "packages": report["target_versions"],
            "image": params.get("image"),
            "checked_at": record["checked_at"],
        }
    results_file = path.parent / paths.VERIFICATION_FILE
    with _locked(path.parent):
        loaded = jsonio.load_json_or_empty(results_file)
        previous = loaded.get(key)
        if ("target_versions" not in record and isinstance(previous, dict)
                and isinstance(previous.get("target_versions"), dict)):
            # This run didn't check the software (gke_job without
            # --smoke-test, say), and the profile is unchanged: the releases
            # recorded before still hold, dated by their own check.
            record["target_versions"] = {
                "checked_at": previous.get("checked_at"),
                **previous["target_versions"],
            }
        results = {
            k: v
            for k, v in loaded.items()
            if isinstance(v, dict) and v.get("profile") != name
        }
        # The failures are kept so session start can show them (FAILED).
        jsonio.write_json(results_file, {
            **results, key: {
                **record, "failed_checks": failed
            }
        })
    return {
        "status": "ok" if passed else "FAILED",
        **record,
        "backend": backend,
        "target": report["target"],
        "profile_hash": key,
        "checks": checks,
        "fields": fields_of(failed),
    }

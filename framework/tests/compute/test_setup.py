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
"""Unit tests for `compute setup`.

Covers its fields, `apply`, `config`, and the Harbor org check.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
_repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from framework import compute, paths, run_state, runtime, utils
from framework.compute import (ComputeConfigError, GkeJobDriver,
                               get_registered_drivers)
from framework.compute.drivers import base
from framework.compute.drivers import registry as driver_registry
from framework.tests.compute.helpers import (GKE_ANSWERS, patch_home,
                                             profile_env, set_up_module,
                                             tear_down_module)

Path = pathlib.Path


def setUpModule():  # noqa: N802 - unittest's spelling
    set_up_module()


def tearDownModule():  # noqa: N802 - unittest's spelling
    tear_down_module()


# A complete gce_tpu_vm profile.
_GCE_ANSWERS = {
    "backend": "gce_tpu_vm",
    "project": "p",
    "zone": "us-east5-a",
    "tpu_name": "v",
}


class TestSetupFieldsAndValidation(unittest.TestCase):
    """SETUP_FIELDS metadata, format validation, and schema/questions consistency."""

    def test_setup_fields_categories_are_disjoint_and_consistent(self):
        pkg_root = pathlib.Path(__file__).resolve().parents[3] / "package"
        schema_path = pkg_root / "schemas" / "environment.json"
        questions_path = pkg_root / "environment" / "questions.json"
        if not schema_path.is_file() or not questions_path.is_file():
            self.skipTest("package schema/questions not available")
        param_props = json.loads(
            schema_path.read_text())["properties"]["profiles"][
                "additionalProperties"]["properties"]["params"]["properties"]
        # A missing `when` means "every backend" in questions.json, which is
        # never what a backend-specific param wants, so each one must say.
        questions = {
            q["path"].removeprefix("compute.params."): q
            for q in json.loads(questions_path.read_text())["questions"]
            if q.get("path", "").startswith("compute.params.")
        }
        required_flags = compute.describe_required_flags()
        samples = {**GKE_ANSWERS, **_GCE_ANSWERS, "pod": "p"}
        shipped = {
            k: v.SETUP_FIELDS
            for k, v in get_registered_drivers().items()
            if not k.startswith("mock")
            and k not in driver_registry.EXTERNAL_DRIVERS
        }
        # Nothing may exist only in the schema or questions.json: a param the
        # driver reads but no SETUP_FIELDS declares is invisible to setup.
        declared = set().union(*(sf.all_fields() for sf in shipped.values()))
        self.assertLessEqual(set(param_props) | set(questions), declared)
        for backend, sf in shipped.items():
            all_fields = sf.all_fields()
            with self.subTest(backend=backend):
                self.assertEqual(
                    len(sf.required) + len(sf.optional) + len(sf.conditional) +
                    len(sf.on_demand), len(all_fields))
                self.assertLessEqual(set(sf.conditional.values()), all_fields)
                self.assertLessEqual(all_fields, set(param_props))
                # on_demand fields are set when a command first needs them,
                # not at install, so they need no prompt here.
                asked = {
                    f
                    for f, q in questions.items()
                    if backend in q["when"]["backend"]
                }
                self.assertLessEqual(all_fields - set(sf.on_demand), asked)
                self.assertLessEqual(asked, all_fields)
                for field in sf.required:
                    self.assertNotEqual(questions[field].get("default"), "")
                self.assertEqual(
                    required_flags.get(backend, set()),
                    {f"--{f.replace('_', '-')}"
                     for f in sf.required})
                for missing in sf.required:
                    with self.assertRaises(ComputeConfigError) as ctx:
                        drv = get_registered_drivers()[backend].from_params({
                            f:
                            samples[f]
                            for f in sf.required if f != missing
                        })
                        if isinstance(drv, GkeJobDriver):
                            drv.config.require_dispatch()
                    self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

    def test_validate_field_format(self):
        v = compute.validate_field_format
        for field, good, bad in [
            ("zone", "us-central1-a", "not_a_zone"),
            ("tpu_topology", "2x2x1", "v6e-8"),
            ("tpu_topology", "2x4", "0x4"),
            ("priority", "low", "urgent"),
            ("staging_bucket", "gs://my-bucket/path", "my-bucket"),
            ("trajectory_bucket", "gs://my-bucket", "http://my-bucket"),
        ]:
            self.assertIsNone(v(field, good), field)
            self.assertIsNotNone(v(field, bad), field)

        region_err = v("zone", "us-central1")
        self.assertIn("is a region, not a zone", region_err["reason"])
        self.assertIn("us-central1-a", region_err["choices"])
        # Priority choices come out in a fixed order, highest first.
        self.assertEqual(
            v("priority", "urgent")["choices"],
            ["", "medium", "low", "very-low"],
        )
        # An unfamiliar accelerator label is left to preflight, which only
        # warns, so a newer TPU generation is not rejected at setup.
        self.assertIsNone(v("tpu_accelerator", "tpu-v99-slice"))

        # Images are accepted whether pinned by digest or by tag.
        self.assertIsNone(v("image", "gcr.io/p/img@sha256:deadbeef"))
        self.assertIsNone(v("image", "gcr.io/p/img:latest"))


class TestComputeSetupApplyAndConfig(unittest.TestCase):
    """`compute setup apply` and `config [show|set|use]`."""

    PKG = "pkg@1.0.0"
    GKE = GKE_ANSWERS

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = pathlib.Path(self.temp_dir.name)
        self.workspace = root / "ws"
        self.env_file = self.workspace / ".tpu-agent-suite" / "environment.json"
        self.write({"revision": 1, "default": None, "profiles": {}})
        patch_home(self)

    def write(self, doc):
        self.env_file.parent.mkdir(parents=True, exist_ok=True)
        self.env_file.write_text(json.dumps(doc), encoding="utf-8")

    def saved(self):
        return json.loads(self.env_file.read_text(encoding="utf-8"))

    def apply(self, profile, answers, **kwargs):
        return compute.apply_setup(profile=profile,
                                   answers=answers,
                                   package=self.PKG,
                                   workspace=self.workspace,
                                   **kwargs)

    def config_set(self, *assignments, profile=None):
        return compute.config_set(assignments=list(assignments),
                                  profile=profile,
                                  package=self.PKG,
                                  workspace=self.workspace)

    def test_apply_incomplete_writes_nothing(self):
        vm = {"backend": "gce_tpu_vm", "project": "p"}
        gke = self.GKE
        cases = [
            ({}, "missing", ["backend"]),
            (dict(backend="nope"), "invalid", ["backend"]),
            (vm, "missing", ["zone", "tpu_name"]),
            (dict(_GCE_ANSWERS, zone="us-central1"), "invalid", ["zone"]),
            (dict(gke, queue="q"), "newly_applicable", ["priority"]),
            (dict(gke, tpu_chips="abc"), "invalid", ["tpu_chips"]),
            (dict(gke,
                  staging_service_account="sa@p",
                  staging_service_account_key="/k"), "invalid", ["params"]),
            (dict(backend="local",
                  xprof_interpreter="py"), "invalid", ["xprof_interpreter"]),
        ]
        before = self.env_file.read_text(encoding="utf-8")
        for answers, key, names in cases:
            with self.subTest(answers=answers):
                res = self.apply("p", answers)
                self.assertEqual(res["status"], "INCOMPLETE")
                self.assertEqual(sorted(e["name"] for e in res[key]),
                                 sorted(names))
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("my profile", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        self.assertEqual(self.env_file.read_text(encoding="utf-8"), before)

    def test_config_set_validates_like_apply(self):
        # A blank field that has a default, like gke_job's image, takes it.
        self.assertEqual(
            self.apply("blank", {
                **self.GKE, "image": ""
            }, dry_run=True)["params"]["image"],
            compute.setup._questions()["image"]["default"])
        self.apply("gke", self.GKE)
        before = self.saved()
        self.assertEqual(
            self.config_set("tpu_topology=")["missing"][0]["name"],
            "tpu_topology")
        self.assertEqual(self.saved(), before)
        # slots has no question; its schema type still makes "2" an integer.
        res = self.config_set("declared_duration_minutes=30", "slots=2",
                              "xprof_interpreter=/py")
        self.assertEqual((res["status"], res["revision"]), ("ok", 3))
        saved = self.saved()["profiles"]["gke"]["params"]
        self.assertEqual((saved["declared_duration_minutes"], saved["slots"]),
                         (30, 2))
        self.assertEqual(self.saved()["xprof_interpreter"], "/py")
        for args, kwargs, code in [(("trajectory_enabled=true", ), {},
                                    "INVALID_FIELD"),
                                   (("queue=x", ), {
                                       "profile": "nope"
                                   }, "UNKNOWN_PROFILE"),
                                   (("no-equals", ), {}, "INVALID_FIELD")]:
            with self.subTest(args=args), self.assertRaises(
                    compute.ComputeConfigError) as ctx:
                self.config_set(*args, **kwargs)
            self.assertEqual(ctx.exception.error_code, code)

    def test_environment_location_is_never_guessed(self):
        self.env_file.write_text("{not json", encoding="utf-8")
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            self.apply("dev", {"backend": "local"})
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        # Other runtime commands, such as trajectory export, say so too.
        with self.assertRaisesRegex(runtime.TpuAgentSuiteError,
                                    "not valid JSON"):
            paths.get_environment(self.workspace, self.PKG)
        # An ownership ledger without environment.json is repaired ...
        self.env_file.unlink()
        (self.env_file.parent / "ownership.json").write_text(json.dumps(
            {"package": "pkg"}),
                                                             encoding="utf-8")
        self.assertEqual(
            self.apply("dev", {"backend": "local"})["status"], "ok")
        # ... but a package installed nowhere is an error.
        with self.assertRaises(compute.ComputeConfigError) as ctx:
            compute.config_show(package=self.PKG,
                                workspace=pathlib.Path(self.temp_dir.name))
        self.assertEqual(ctx.exception.error_code, "MISSING_FIELD")

    def test_cli(self):
        where = ["--package", self.PKG, "--workspace", str(self.workspace)]
        err = io.StringIO()

        def run(*argv):
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(err):
                rc = runtime.main([*argv, *where, "--json"])
            return rc, (json.loads(out.getvalue()) if out.getvalue() else None)

        answers = pathlib.Path(self.temp_dir.name) / "answers.json"
        answers.write_text(json.dumps(self.GKE), encoding="utf-8")
        self.assertEqual(
            run("compute", "setup", "apply", "--profile", "gke", "--answers",
                str(answers))[0], 0)
        self.apply("dev", self.GKE)
        self.assertEqual(run("config", "set", "queue=q")[0], 1)
        rc, shown = run("config", "show")
        self.assertEqual((rc, shown["active_profile"], shown["active_source"]),
                         (0, "gke", "default"))
        run_id = run_state.start_run(package=self.PKG,
                                     workspace=self.workspace)["run_id"]
        self.assertEqual(run("config", "use", "dev")[0], 0)
        self.assertEqual(run("config", "use", "nope")[0], 1)
        # With no --profile, `set` changes the run's profile, not the default.
        rc, res = run("config", "set", "declared_duration_minutes=45")
        self.assertEqual((rc, res["profile"]), (0, "dev"))
        shown = run("config", "show", "--run-id", run_id)[1]
        self.assertEqual((shown["active_profile"], shown["active_source"]),
                         ("dev", "run"))
        self.assertNotIn("declared_duration_minutes",
                         shown["profiles"]["gke"]["params"])
        self.assertEqual(run("config", "use", "dev", "--default")[0], 0)
        self.assertEqual(self.saved()["default"], "dev")
        # Trajectory sharing changes only once the user has agreed, and lands
        # where the export reads it.
        sharing = ("trajectory_enabled=true", "trajectory_bucket=gs://b")
        self.assertEqual(run("config", "set", *sharing)[0], 1)
        self.assertIn("--user-confirmed", err.getvalue())
        self.assertNotIn("trajectory", self.saved())
        run("config", "set", *sharing, "--user-confirmed")
        self.assertEqual(self.saved()["trajectory"]["params"], {
            "enabled": True,
            "bucket": "gs://b"
        })
        # A Harbor org is checked with the hub.
        with patch.object(SETUP, "_harbor_org_exists", return_value=False):
            rc, _ = run("config", "set", "trajectory_sink=harbor",
                        "trajectory_share_org=nope", "--user-confirmed")
        self.assertEqual(rc, 1)
        self.assertIn("no organization named 'nope'", err.getvalue())
        # Turning sharing off takes the destination with it, and the export
        # then refuses rather than uploading to the old bucket.
        _, res = run("config", "set", "trajectory_enabled=false",
                     "--user-confirmed")
        self.assertEqual(self.saved()["trajectory"]["params"],
                         {"enabled": False})
        self.assertIn("trajectory_bucket", res["removed"][0])
        with contextlib.redirect_stderr(err):
            rc = runtime.main(
                ["trajectory", "export", "--harness", "claude-code", *where])
        self.assertEqual(rc, 1)
        self.assertIn("trajectory sharing is off", err.getvalue())

    def test_config_remove(self):
        for name, answers in (("gke", self.GKE), ("dev", {
                "backend": "local"
        }), ("aux", {
                "backend": "local"
        })):
            self.apply(name, answers)

        def remove(profile, **kwargs):
            return compute.config_remove(profile=profile,
                                         package=self.PKG,
                                         workspace=self.workspace,
                                         **kwargs)

        before = self.saved()
        for profile, confirmed, code in (("  ", True, "MISSING_FIELD"),
                                         ("aux", False,
                                          "CONFIRMATION_REQUIRED"),
                                         ("nope", True, "UNKNOWN_PROFILE")):
            with self.subTest(code), self.assertRaises(
                    compute.ComputeConfigError) as ctx:
                remove(profile, user_confirmed=confirmed)
            self.assertEqual(ctx.exception.error_code, code)
        self.assertEqual(self.saved(), before)

        # Removing a non-default profile preserves default and leaves other
        # pinned runs alone.
        run_dev = run_state.start_run(package=self.PKG,
                                      workspace=self.workspace)["run_id"]
        compute.config_use(profile="dev",
                           package=self.PKG,
                           workspace=self.workspace,
                           run_id=run_dev)
        res = remove("aux", user_confirmed=True, run_id=run_dev)
        self.assertEqual((res["default"], res.get("unpinned_run")),
                         ("gke", None))
        run_file = paths.get_run_root(run_dev) / "run.json"
        self.assertEqual(
            json.loads(run_file.read_text())["compute_profile"], "dev")
        # Removing the default profile promotes a remaining profile ("dev").
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = runtime.main([
                "config", "remove", "gke", "--user-confirmed", "--package",
                self.PKG, "--workspace",
                str(self.workspace), "--json"
            ])
        self.assertEqual((rc, json.loads(out.getvalue())["default"]),
                         (0, "dev"))
        self.assertEqual(list(self.saved()["profiles"]), ["dev"])
        # Removing the last profile sets default to None, passes schema
        # validation, and unpins a run pinned to that profile.
        res = remove("dev", user_confirmed=True, run_id=run_dev)
        self.assertEqual((res["default"], res["unpinned_run"]),
                         (None, run_dev))
        self.assertEqual(
            utils.validate_instance(self.saved(), compute.setup._schema()), [])
        self.assertNotIn("compute_profile", json.loads(run_file.read_text()))

    def test_verify_records_a_result_per_profile_hash(self):
        """A result counts until the profile's settings change."""
        self.apply("vm", _GCE_ANSWERS)
        driver = MagicMock(quick_run=True, scheduled_run=False)
        driver.preflight.side_effect = lambda: base.PreflightReport(
            "gce_tpu_vm", "vm", True, [])

        def verify(stdout, *flags):
            if stdout is not None:
                driver.run.return_value = base.SynchronousExecutionResult(
                    0 if "ok" in stdout else 1, stdout, "")
            out = io.StringIO()
            with patch.object(compute.GceTpuVmDriver,
                              "from_params",
                              return_value=driver), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", "setup", "verify", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            res = json.loads(out.getvalue())
            return rc, res["status"], res.get("smoke_test")

        def state():
            return compute.config_show(package=self.PKG,
                                       workspace=self.workspace)["setup_state"]

        self.assertEqual(state(), {"vm": "UNVERIFIED"})
        self.assertEqual(verify("[FAIL] torch_tpu")[:2], (1, "FAILED"))
        self.assertEqual(state(), {"vm": "FAILED"})
        # The failing checks are kept, for session start to show.
        results = self.env_file.parent / "verification.json"
        (saved, ) = json.loads(results.read_text()).values()
        self.assertEqual([c["fields"] for c in saved["failed_checks"]],
                         [["python_interpreter"]])
        self.assertEqual(verify("Status: ok", "--profile", "vm"),
                         (0, "ok", False))
        self.assertEqual(state(), {"vm": "READY"})
        self.assertEqual(len(json.loads(results.read_text())), 1)
        # A result from another package version doesn't count.
        with patch.object(SETUP.utils, "TPU_AGENT_SUITE_VERSION", "0.0.1"):
            self.assertEqual(state(), {"vm": "UNVERIFIED"})
        # A smoke test still pending at its timeout passes, but didn't run.
        driver.configure_mock(quick_run=False, scheduled_run=True)
        driver.run.return_value = base.SynchronousExecutionResult(
            1, "", "", {
                "timed_out_after_sec": 900,
                "job_status": "PENDING"
            })
        self.assertEqual(verify(None, "--smoke-test"), (0, "ok", False))
        self.config_set("zone=us-east5-b")
        self.assertEqual(state(), {"vm": "UNVERIFIED"})
        # An incomplete profile is reported, not checked.
        self.write(profile_env("gce_tpu_vm"))
        self.assertEqual(verify("")[1], "INCOMPLETE")
        self.assertEqual(state(), {"default": "NOT_CONFIGURED"})

    def test_plan_lists_questions_with_suggestions(self):
        outputs = {
            "kubectl get nodes": "tpu7x 2x2x1\ntpu7x 2x2x1\ntpu-v6e-slice 2x4",
            "kubectl get localqueues": "lq other",
            "gcloud config get-value": "other-proj",
            "gcloud compute tpus": "vm1 us-west4-a",
            "kubectl config current-context": "gke_my-proj_us-central1_c1",
        }
        ran = []

        def run(cmd):
            ran.append(" ".join(cmd))
            return outputs.get(
                " ".join(cmd[:3]),
                ""), ("kubectl is not installed" if cmd[2] == "view" else None)

        def plan(*flags, before=()):
            out = io.StringIO()
            with patch.object(compute.probes, "_run", side_effect=run), \
                    patch.object(compute.probes.probe, "list_devices",
                                 return_value=([], [])), \
                    contextlib.redirect_stdout(out), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = runtime.main([
                    "compute", *before, "setup", "plan", "--package", self.PKG,
                    "--workspace",
                    str(self.workspace), "--json", *flags
                ])
            res = json.loads(out.getvalue() or "{}")
            return rc, res, {q["name"]: q for q in res.get("questions", [])}

        # No backend: only the backend question, and the profiles to pick.
        self.write({"default": None, "profiles": {"vm": {"backend": "local"}}})
        _, res, qs = plan()
        self.assertEqual(
            (res["profiles"], list(qs), qs["backend"]["suggested"]),
            ({
                "vm": "local"
            }, ["backend"], "gke_job"))
        # A backend's setup fields, suggested by probes or defaults. Flags
        # before the subcommand count too.
        _, res, qs = plan(before=("--profile", "gke", "--backend", "gke_job"))
        self.assertEqual((res["profile"], res["backend"]), ("gke", "gke_job"))
        nodes = "kubectl get nodes"
        self.assertEqual([(f, q.get("suggested_from")) for f, q in qs.items()],
                         [("image", "default"), ("tpu_accelerator", nodes),
                          ("tpu_topology", nodes),
                          ("kube_context", "kubectl config current-context"),
                          ("namespace", None),
                          ("queue", "kubectl get localqueues"),
                          ("hf_cache_pvc", None), ("priority", None)])
        self.assertEqual([
            qs[f]["suggested"] for f in ("tpu_accelerator", "tpu_topology",
                                         "kube_context", "queue")
        ], ["tpu7x", "2x2x1", "gke_my-proj_us-central1_c1", "lq"])
        self.assertEqual(qs["priority"]["asked_if"], "queue")
        # A missing tool is reported, deduplicated.
        self.assertEqual(res["probe_warnings"], ["kubectl is not installed"])
        # With a backend chosen, a cluster with no TPU nodes is reported.
        saved_nodes = outputs.pop(nodes)
        _, res, _ = plan("--profile", "g2", "--backend", "gke_job")
        self.assertIn("no TPU nodes", " ".join(res["probe_warnings"]))
        outputs[nodes] = saved_nodes
        # Stored values are suggested as they are, except with --fields,
        # which asks again for values verify rejected.
        self.apply("gke", {**self.GKE, "kubeconfig": "/k"})
        _, res, qs = plan("--profile", "gke")
        self.assertEqual(qs["tpu_accelerator"]["suggested_from"], "profile")
        _, res, qs = plan("--profile", "gke", "--fields",
                          "tpu_accelerator,queue")
        self.assertEqual(list(qs), ["tpu_accelerator", "queue"])
        self.assertEqual((qs["tpu_accelerator"]["current"],
                          qs["tpu_accelerator"]["suggested_from"]),
                         ("tpu7x", "kubectl get nodes"))
        # A saved kubeconfig is used by the kubectl probes.
        self.assertIn(
            "kubectl get localqueues -o jsonpath="
            "{.items[*].metadata.name} --kubeconfig /k", ran)
        # Unknown, consent-only (trajectory) or backend-less --fields fail.
        for flags in (("gke", "--fields", "zone"), ("gke", "--fields",
                                                    "trajectory_enabled"),
                      ("new", "--fields", "zone"), ("my vm", "--backend",
                                                    "gke_job")):
            self.assertEqual(plan("--profile", *flags)[0], 1, flags)
        # gcloud prints "(unset)" when it has no project. A TPU VM is only
        # suggested when it's clear which one is meant; otherwise the VMs are
        # offered as choices, each with its zone.
        outputs["gcloud config get-value"] = "(unset)"
        _, _, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual(qs["tpu_name"]["suggested"], "vm1")
        self.assertNotIn("suggested", qs["project"])
        outputs["gcloud compute tpus"] = "vm1 us-west4-a\nme-vm us-east5-a"
        with patch.dict(os.environ, {"USER": "nobody"}):
            _, res, qs = plan("--profile", "tv", "--backend", "gce_tpu_vm")
        self.assertEqual([c["tpu_name"] for c in qs["tpu_name"]["choices"]],
                         ["vm1", "me-vm"])
        self.assertNotIn("suggested", qs["zone"])
        self.assertIn("free text", res["note"])
        # After a backend change, the values apply would keep are current.
        self.apply("vm", {**_GCE_ANSWERS, "python_interpreter": "/py"})
        _, _, qs = plan("--profile", "vm", "--backend", "gke_pod")
        self.assertEqual(
            [qs[f].get("current") for f in ("python_interpreter", "pod")],
            ["/py", None])

    def test_tpu_vm_choices(self):
        """$USER's VMs come first, and the list is capped.

        A zone is suggested only when every VM, listed or not, shares it.
        """
        many = "\n".join(f"vm{i} z{i % 2}" for i in range(20))
        with patch.dict(os.environ, {"USER": "ann"}):
            found = compute.probes._tpu_vm(f"{many}\nann-a z9\nann-b z9")
        self.assertEqual([c["tpu_name"] for c in found["tpu_name"][:3]],
                         ["ann-a", "ann-b", "vm0"])
        self.assertEqual(len(found["tpu_name"]), compute.probes.MAX_CHOICES)
        self.assertNotIn("zone", found)
        self.assertEqual(compute.probes._tpu_vm(""), {})
        self.assertEqual(compute.probes._tpu_vm("a z\nb z")["zone"], "z")
        cut = "\n".join(f"vm{i} z" for i in range(20)) + "\nlast elsewhere"
        self.assertNotIn("zone", compute.probes._tpu_vm(cut))
        # The same name in two zones stays two distinct choices.
        self.assertEqual(
            len(compute.probes._tpu_vm("dup za\ndup zb")["tpu_name"]), 2)
        # Several VMs still point to gce_tpu_vm as the backend.
        self.assertEqual(
            compute.probes._backend({}, "a z\nb y", ["gce_tpu_vm"])[0],
            "gce_tpu_vm")

    def test_a_probe_reports_a_missing_tool_or_a_timeout(self):
        self.assertEqual(compute.probes._run(["no-such-tool-xyz"]),
                         ("", "no-such-tool-xyz is not installed"))
        with patch.object(compute.probes, "PROBE_TIMEOUT_SEC", 0.2):
            self.assertEqual(compute.probes._run(["sleep", "5"]),
                             ("", "`sleep 5` did not answer within 0.2s"))
        # A command that runs and fails just has nothing to suggest.
        self.assertEqual(compute.probes._run(["false"]), ("", None))


# The module `compute setup` and `config` run in.
SETUP = sys.modules[compute.config_show.__module__]


class ShareOrgCheckTest(unittest.TestCase):
    """A Harbor org the hub lacks is refused when saved, not at upload."""

    def test_an_org_that_cannot_be_checked_is_kept_with_a_warning(
            self) -> None:
        """No harbor, no login or no network is not evidence the org is wrong."""
        answers = {
            "trajectory": {
                "params": {
                    "sink": "harbor",
                    "share_org": "t"
                }
            }
        }
        for error, says in ((ImportError("harbor"), "harbor is not installed"),
                            (TimeoutError(), "could not check the Harbor")):
            with patch.object(SETUP, "_harbor_org_exists", side_effect=error), \
                    contextlib.redirect_stderr(io.StringIO()) as stderr:
                SETUP.check_share_org(answers, None)
            self.assertIn(says, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

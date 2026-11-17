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
"""The flag-optimization suite: catalog, compiled tasks, flagopt, verifier.

eval/suites/flag_optimization/ holds one toolkit (assets/flagopt/flagopt.py),
one task catalog (assets/flagopt/tasks.json), one instruction template and the
suite YAML; eval/verifiers/flag_optimization/ holds the rewardkit verifier.
These tests compile the suite the way eval/run.py does and check that every
task resolves, that the six bundles share one environment, that each case's
prompt_vars are the ones its spec implies, and that the verifier's
measurement behaves: on the paths that end before any Job is submitted, and
end to end against a fake kubectl with a recorded baseline.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
SUITE = REPO / "eval" / "suites" / "flag_optimization"
SUITE_YAML = SUITE / "flag_optimization.yaml"
ASSETS = SUITE / "assets"
FLAGOPT_HOME = ASSETS / "flagopt"
CATALOG_PATH = FLAGOPT_HOME / "tasks.json"
VERIFIERS = REPO / "eval" / "verifiers" / "flag_optimization"
GEMMA = "flagopt-gemma4-31b-v7x-2x2x2-sharding"

HAVE_SUITE_DEPS = all(
    importlib.util.find_spec(m) is not None for m in ("pydantic", "yaml"))
HAVE_REWARDKIT = importlib.util.find_spec("rewardkit") is not None
HAVE_HARBOR = importlib.util.find_spec("harbor") is not None


def _load(name: str, path: Path):
    # No bytecode: assets/flagopt/ is copied into the image as is.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flagopt = _load("flagopt", FLAGOPT_HOME / "flagopt.py")
instruction_vars = _load("instruction_vars", SUITE / "instruction_vars.py")
CATALOG = json.loads(CATALOG_PATH.read_text("utf-8"))
TASK_IDS = sorted(CATALOG["tasks"])


def _spec(task_id: str) -> dict:
    return flagopt.expand_task(CATALOG, task_id)


_COMPILED: Path | None = None
_TMP: tempfile.TemporaryDirectory | None = None


def compiled() -> Path:
    """The suite compiled once, as eval/run.py compiles it."""
    global _COMPILED, _TMP
    if _COMPILED is None:
        from eval.suite_to_harbor.suite_to_harbor import compile_suite
        _TMP = tempfile.TemporaryDirectory()
        compile_suite(SUITE_YAML, Path(_TMP.name))
        _COMPILED = Path(_TMP.name)
    return _COMPILED


def tearDownModule():  # noqa: N802 - unittest hook
    if _TMP is not None:
        _TMP.cleanup()


def _tree_digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class CatalogTest(unittest.TestCase):

    def test_every_task_resolves(self):
        self.assertEqual(len(TASK_IDS), 6)
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                spec = _spec(task_id)
                self.assertEqual(spec["task_id"], task_id)
                self.assertTrue(spec["tunables"])
                self.assertFalse(set(spec["fixed"]) & set(spec["tunables"]))
                hw, batch = spec["hardware"], spec["batch"]
                self.assertEqual(
                    batch["global_batch_size"],
                    batch["per_device_batch_size"] *
                    batch["gradient_accumulation_steps"] * hw["num_devices"])
                self.assertLessEqual(spec["time_budget_sec"],
                                     spec["agent_timeout_sec"])

    def test_catalog_names_only_known_knobs(self):
        knobs = set(CATALOG["knobs"])
        for name, model in CATALOG["models"].items():
            with self.subTest(model=name):
                self.assertLessEqual(set(model["exclude_knobs"]), knobs)
                self.assertLessEqual(set(model["baseline"]), knobs)
        for task_id, task in CATALOG["tasks"].items():
            with self.subTest(task=task_id):
                self.assertLessEqual(set(task.get("mesh", {})), knobs)
                self.assertIn(task["model"], CATALOG["models"])
                self.assertIn(task["topology"], CATALOG["topologies"])

    def test_derived_choices_follow_the_slice_and_batch(self):
        spec = _spec("flagopt-deepseek-v3-671b-v7x-8x8x8-all")
        self.assertEqual(spec["tunables"]["ici_data_parallelism"]["choices"],
                         [2**i for i in range(11)])
        qwen = _spec("flagopt-qwen3-5-397b-a17b-v7x-4x4x4-all")
        self.assertEqual(qwen["tunables"]["per_device_batch_size"]["choices"],
                         [0.25, 0.5, 1.0, 2.0])

    def test_locked_groups_pin_their_knobs(self):
        spec = _spec(GEMMA)
        self.assertEqual(spec["fixed"]["remat_policy"], "full")
        self.assertEqual(spec["fixed"]["per_device_batch_size"], 1)
        self.assertNotIn("remat_policy", spec["tunables"])
        self.assertFalse(spec["xla_flags"]["tunable"])

    def test_baseline_is_a_valid_config(self):
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                spec = _spec(task_id)
                resolved, xla = flagopt.validate_candidate(spec, {})
                self.assertEqual(resolved,
                                 spec["baseline"]["maxtext_overrides"])
                self.assertEqual(xla, spec["baseline"]["xla_flags"])

    def test_recorded_baselines_match_their_spec(self):
        """A recording the verifier would ignore is a stale recording."""
        for path in sorted((ASSETS / "baselines").glob("*.json")):
            with self.subTest(baseline=path.name):
                spec = _spec(path.stem)
                data = json.loads(path.read_text("utf-8"))
                self.assertEqual(data["task_id"], spec["task_id"])
                self.assertEqual(data["steps"], spec["measurement"]["steps"])
                self.assertEqual(data["warmup_steps"],
                                 spec["measurement"]["warmup_steps"])
                self.assertEqual(data["image"], spec["cluster"]["image"])
                self.assertIn("checkpoint", data["runs"])

    def test_flagopt_guide_matches_the_measurement(self):
        guide = " ".join(
            (FLAGOPT_HOME / "README.md").read_text("utf-8").split())
        meas = CATALOG["measurement"]
        self.assertIn(f"trains for {meas['steps']} steps", guide)
        self.assertIn(
            f"median over steps {meas['warmup_steps']}..{meas['steps'] - 1}",
            guide)
        self.assertIn(f"first {meas['gate_steps']} steps", guide)

    def test_spec_needs_a_task(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLAGOPT_TASK", None)
            with self.assertRaisesRegex(RuntimeError, "FLAGOPT_TASK"):
                flagopt.load_spec(CATALOG_PATH)
        with mock.patch.dict(os.environ, {"FLAGOPT_TASK": GEMMA}):
            self.assertEqual(flagopt.load_spec(CATALOG_PATH)["task_id"], GEMMA)


@unittest.skipUnless(HAVE_SUITE_DEPS, "suite_to_harbor needs pydantic, yaml")
class SuiteTest(unittest.TestCase):
    """The bundles eval/run.py compiles from flag_optimization.yaml."""

    @classmethod
    def setUpClass(cls):
        import yaml
        cls.root = compiled()
        cls.yaml = yaml.safe_load(SUITE_YAML.read_text("utf-8"))
        cls.cases = {c["name"]: c for c in cls.yaml["evaluation"]["cases"]}

    def test_one_case_per_task(self):
        self.assertEqual(sorted(self.cases), TASK_IDS)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir() if p.is_dir()),
            TASK_IDS)

    def test_every_task_shares_one_environment(self):
        """Identical environment/ dirs: Harbor builds and caches one image."""
        digests = {
            _tree_digest(self.root / t / "environment")
            for t in TASK_IDS
        }
        self.assertEqual(len(digests), 1)
        data = self.root / GEMMA / "environment" / "data" / ".flagopt"
        self.assertEqual(sorted(p.name for p in data.iterdir()),
                         ["README.md", "flagopt.py", "tasks.json"])
        dockerfile = (self.root / GEMMA / "environment" /
                      "Dockerfile").read_text("utf-8")
        self.assertIn(
            f"FROM --platform=linux/amd64 {CATALOG['cluster']['image']}",
            dockerfile)
        self.assertIn("install_flagopt.sh", dockerfile)

    def test_prompt_vars_come_from_the_spec(self):
        """Run instruction_vars.py and paste its output after editing
        tasks.json."""
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                self.assertEqual(self.cases[task_id]["prompt_vars"],
                                 instruction_vars.prompt_vars(_spec(task_id)))

    def test_instruction_is_complete_and_matches_the_catalog(self):
        meas, cl = CATALOG["measurement"], CATALOG["cluster"]
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                spec = _spec(task_id)
                text = (self.root / task_id /
                        "instruction.md").read_text("utf-8")
                self.assertNotRegex(text, r"\{\{\s*\w+\s*\}\}")
                for knob in spec["tunables"]:
                    self.assertIn(f"`{knob}`", text)
                self.assertIn(f"MaxText v{CATALOG['maxtext']['version']} ",
                              text)
                self.assertIn(
                    f"cluster `{cl['cluster']}` (project `{cl['project']}`, "
                    f"`{cl['location']}`), namespace `{cl['namespace']}`, "
                    f"Kueue queue `{cl['queue']}`", text)
                self.assertIn(
                    f"`{cl['accelerator']}` `{spec['cluster']['topology']}` slice",
                    text)
                self.assertIn(
                    f"over the first {meas['gate_steps']} steps, the per-step "
                    f"training loss must match the current config's within "
                    f"{meas['loss_rtol'] * 100:g}% (relative) and the "
                    f"gradient norm within {meas['grad_norm_rtol'] * 100:g}%",
                    text)
                self.assertIn("/opt/flagopt/README.md", text)

    def test_task_toml_selects_the_task(self):
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                spec = _spec(task_id)
                cfg = tomllib.loads(
                    (self.root / task_id / "task.toml").read_text("utf-8"))
                self.assertEqual(cfg["task"]["name"],
                                 f"flag-optimization/{task_id}")
                self.assertEqual(cfg["task"]["description"], spec["title"])
                self.assertEqual(cfg["environment"]["env"]["FLAGOPT_TASK"],
                                 task_id)
                venv = cfg["verifier"]["env"]
                self.assertEqual(venv["FLAGOPT_TASK"], task_id)
                self.assertEqual(cfg["agent"]["timeout_sec"],
                                 spec["agent_timeout_sec"])
                self.assertEqual(cfg["verifier"]["timeout_sec"],
                                 spec["verifier_timeout_sec"])
                self.assertEqual(int(venv["VERIFY_DEADLINE_SEC"]),
                                 spec["verifier_timeout_sec"] - 600)
                self.assertEqual(float(venv["LOSS_RTOL"]),
                                 CATALOG["measurement"]["loss_rtol"])
                self.assertEqual(float(venv["GRAD_NORM_RTOL"]),
                                 CATALOG["measurement"]["grad_norm_rtol"])
                self.assertEqual(cfg["metadata"]["gke_cluster"],
                                 spec["cluster"]["cluster"])

    def test_cluster_target_is_left_to_the_config(self):
        """GKE_* in a task would fight the config's; the trial gate is the
        only way to the cluster, so no task carries a sidecar either."""
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                cfg = tomllib.loads(
                    (self.root / task_id / "task.toml").read_text("utf-8"))
                for section in (cfg["environment"].get("env", {}),
                                cfg["verifier"].get("env", {})):
                    self.assertFalse(
                        [k for k in section if k.startswith("GKE_")])
                self.assertNotIn("services", cfg["environment"])

    @unittest.skipUnless(HAVE_HARBOR, "harbor is not installed")
    def test_task_toml_is_a_harbor_task_config(self):
        from harbor.models.task.config import TaskConfig
        for task_id in TASK_IDS:
            with self.subTest(task=task_id):
                TaskConfig.model_validate(
                    tomllib.loads((self.root / task_id /
                                   "task.toml").read_text("utf-8")))

    def test_reward_counts_only_the_throughput_improvement(self):
        tests = self.root / GEMMA / "tests"
        dims = sorted(p.name for p in tests.iterdir()
                      if p.is_dir() and (p / f"{p.name}.py").is_file())
        reward = tomllib.loads((tests / "reward.toml").read_text("utf-8"))
        entry = reward["reward"][0]
        self.assertEqual(entry["aggregation"], "weighted-sum")
        self.assertEqual(sorted(entry["weights"]), dims)
        self.assertEqual({
            k: v
            for k, v in entry["weights"].items() if v
        }, {"throughput_improvement": 1.0})
        for dim in dims:
            self.assertTrue((tests / dim / "measure.py").is_file())
        self.assertTrue((tests / "baselines").is_dir())


class ClusterSettingsTest(unittest.TestCase):

    def setUp(self):
        self.spec = _spec(GEMMA)

    def test_config_slice_does_not_override_the_task(self):
        """The shared-capacity configs default GKE_TPU_TOPOLOGY to 2x2x1."""
        with mock.patch.dict(
                os.environ, {
                    "GKE_TPU_ACCELERATOR": "tpu-v6e-slice",
                    "GKE_TPU_TOPOLOGY": "2x2x1",
                    "GKE_NAMESPACE": "team-ns",
                }):
            cluster = flagopt.cluster_settings(self.spec)
        self.assertEqual(cluster["accelerator"], "tpu7x")
        self.assertEqual(cluster["topology"], "2x2x2")
        self.assertEqual(cluster["namespace"], "team-ns")

    def test_flagopt_names_override_the_slice(self):
        with mock.patch.dict(os.environ, {"FLAGOPT_TPU_TOPOLOGY": "2x2x4"}):
            self.assertEqual(
                flagopt.cluster_settings(self.spec)["topology"], "2x2x4")

    def test_multi_host_job_is_indexed_on_the_task_slice(self):
        cluster = flagopt.cluster_settings(self.spec)
        mans = flagopt.job_manifests(self.spec, cluster, "fo-test-base",
                                     "true", "agent")
        job = mans["job"]["spec"]
        hosts = self.spec["hardware"]["hosts"]
        self.assertEqual(job["completionMode"], "Indexed")
        self.assertEqual(job["completions"], hosts)
        selector = job["template"]["spec"]["nodeSelector"]
        self.assertEqual(selector[flagopt.TOPOLOGY_LABEL], "2x2x2")
        self.assertIsNotNone(mans["service"])


FAKE_KUBECTL = r"""
import json, os, sys
argv = [a for a in sys.argv[1:]]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(" ".join(argv) + "\n")
if argv[:1] == ["apply"]:
    sys.exit(0)
if argv[:2] == ["get", "job"]:
    print(json.dumps({"metadata": {"uid": "u"}, "spec": {},
                      "status": {"ready": 1, "conditions": [
                          {"type": "Complete", "status": "True"}]}}))
    sys.exit(0)
if argv[:2] == ["get", "pods"]:
    name = argv[argv.index("-l") + 1].split("=", 1)[1]
    print(json.dumps({"items": [{"metadata": {
        "name": name + "-0", "creationTimestamp": "t",
        "labels": {"batch.kubernetes.io/job-completion-index": "0"}},
        "status": {}}]}))
    sys.exit(0)
if argv[:1] == ["logs"]:
    t = float(os.environ["FAKE_STEP_TIME"])
    scale = float(os.environ.get("FAKE_LOSS_SCALE", "1"))
    for i in range(int(os.environ["FAKE_STEPS"])):
        rec = {"step": i, "learning/loss": (10.0 - 0.01 * i) * scale,
               "learning/raw_grad_norm": 1.0, "perf/step_time_seconds": t}
        print("FLAGOPT_METRIC " + json.dumps(rec))
    print("FLAGOPT_EXIT_CODE 0")
    print("FLAGOPT_DONE now")
    sys.exit(0)
sys.exit(0)
"""


def _metrics(n: int, *, loss=lambda i: 10.0 - 0.01 * i, grad_norm=1.0):
    return [{
        "step": i,
        "loss": loss(i),
        "grad_norm": grad_norm,
        "step_time": 2.0
    } for i in range(n)]


class AccuracyGateTest(unittest.TestCase):

    def setUp(self):
        self.spec = _spec(GEMMA)
        self.n = self.spec["measurement"]["gate_steps"]

    def test_identical_runs_pass(self):
        gate = flagopt.accuracy_gate(self.spec, _metrics(self.n),
                                     _metrics(self.n))
        self.assertTrue(gate["pass"], gate["reasons"])

    def test_missing_candidate_grad_norm_fails(self):
        """Metrics scraped from stdout carry no gradient norm."""
        gate = flagopt.accuracy_gate(self.spec, _metrics(self.n),
                                     _metrics(self.n, grad_norm=None))
        self.assertFalse(gate["pass"])
        self.assertIn("candidate grad_norm None", gate["reasons"][0])

    def test_nan_in_the_baseline_fails(self):
        for kwargs in ({
                "loss": lambda i: float("nan")
        }, {
                "grad_norm": float("nan")
        }):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                gate = flagopt.accuracy_gate(self.spec,
                                             _metrics(self.n, **kwargs),
                                             _metrics(self.n))
                self.assertFalse(gate["pass"])

    def test_stdout_metrics_cannot_pass(self):
        lines = "\n".join(
            f"completed step: {i}, seconds: 2.0, TFLOP/s/device: 1.0, "
            f"Tokens/s/device: 1.0, loss: {10.0 - 0.01 * i:.3f}"
            for i in range(self.n))
        parsed = flagopt.parse_log(lines)
        self.assertEqual(len(parsed["metrics"]), self.n)
        gate = flagopt.accuracy_gate(self.spec, _metrics(self.n),
                                     parsed["metrics"])
        self.assertFalse(gate["pass"])


class RunCandidateTest(unittest.TestCase):
    """flagopt.run_candidate against a fake kubectl."""

    def _run(self, tmp: Path, *, logged: int, steps: int, label="run"):
        fake = tmp / "fake_kubectl.py"
        fake.write_text(FAKE_KUBECTL, "utf-8")
        spec = _spec(GEMMA)
        kube = flagopt.Kube(cmd=[sys.executable, "-I", "-S",
                                 str(fake)],
                            namespace="ns")
        with mock.patch.dict(
                os.environ, {
                    "FAKE_LOG": str(tmp / "calls.log"),
                    "FAKE_STEPS": str(logged),
                    "FAKE_STEP_TIME": "1.5"
                }):
            return flagopt.run_candidate(spec,
                                         flagopt.baseline_candidate(spec),
                                         label=label,
                                         steps=steps,
                                         out_dir=tmp / "runs",
                                         kube=kube,
                                         log=lambda *_: None)

    def test_complete_run_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(Path(tmp), logged=26, steps=26)
        self.assertTrue(res["ok"])
        self.assertIn("step_time_p50_s", res["summary"])

    def test_run_that_stops_early_is_not_ok(self):
        """15 of 26 steps with exit code 0 used to count as a success."""
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(Path(tmp), logged=15, steps=26)
        self.assertFalse(res["ok"])
        self.assertEqual(res["missing_steps"], list(range(15, 26)))

    def test_short_run_within_warmup_is_ok_without_a_step_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self._run(Path(tmp), logged=5, steps=5)
        self.assertTrue(res["ok"])
        self.assertNotIn("step_time_p50_s", res["summary"])

    def test_repeated_label_gets_a_new_job_and_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            with mock.patch.object(flagopt,
                                   "unique_suffix",
                                   side_effect=["120000aa", "120001bb"]):
                a = self._run(tmp, logged=26, steps=26, label="base")
                b = self._run(tmp, logged=26, steps=26, label="base")
            self.assertNotEqual(a["name"], b["name"])
            self.assertTrue(
                (tmp / "runs" / a["name"] / "result.json").is_file())
            self.assertTrue(
                (tmp / "runs" / b["name"] / "result.json").is_file())

    def test_job_name_keeps_its_suffix(self):
        name = flagopt.make_job_name("fo-abcdef", "x" * 60, "120000aa")
        self.assertLessEqual(len(name), 40)
        self.assertTrue(name.endswith("-120000aa"))
        self.assertRegex(name, flagopt.K8S_NAME_RE)


class BaselineXlaFlagsTest(unittest.TestCase):
    """A task whose baseline sets XLA flags keeps them when a config (or
    `flagopt aot baseline`) leaves xla_flags out."""

    def setUp(self):
        self.spec = _spec("flagopt-gemma4-31b-v7x-2x2x2-full-stack")
        self.spec["baseline"]["xla_flags"] = ["--xla_tpu_enable_foo=true"]

    def test_omitted_xla_flags_keep_the_baseline(self):
        _, xla = flagopt.validate_candidate(self.spec, {})
        self.assertEqual(xla, ["--xla_tpu_enable_foo=true"])
        _, xla = flagopt.validate_candidate(
            self.spec, flagopt.baseline_candidate(self.spec))
        self.assertEqual(xla, ["--xla_tpu_enable_foo=true"])

    def test_explicit_empty_list_clears_them(self):
        _, xla = flagopt.validate_candidate(self.spec, {"xla_flags": []})
        self.assertEqual(xla, [])

    def test_profile_help_names_the_traced_steps(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            flagopt.main(["--spec", str(CATALOG_PATH), "run", "--help"])
        self.assertIn("steps 2-4", " ".join(out.getvalue().split()))


DRIVER = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import measure
r = measure.evaluate(Path(sys.argv[2]))
print(json.dumps(dict(r, improvement_score=measure.improvement_score(r)), default=str))
"""


class MeasureTest(unittest.TestCase):
    """measure.py, which every criterion reads, run for real."""

    GOOD = {
        "maxtext_overrides": {
            "ici_fsdp_parallelism": 16,
            "ici_tensor_parallelism": 1
        }
    }

    @staticmethod
    def setup_tests_dir(tmp: Path, config: dict | None):
        """A tests/ dir as the suite stages it, with a recorded baseline."""
        spec = _spec(GEMMA)
        meas = spec["measurement"]
        tests = tmp / "tests"
        for dim in ("throughput_improvement", "config_valid", "candidate_ran",
                    "accuracy_gate", "infra_error"):
            (tests / dim).mkdir(parents=True)
            shutil.copy2(VERIFIERS / f"{dim}.py", tests / dim)
            shutil.copy2(VERIFIERS / "measure.py", tests / dim)
        (tests / "baselines").mkdir()
        metrics = _metrics(meas["steps"])
        (tests / "baselines" / f"{GEMMA}.json").write_text(
            json.dumps({
                "task_id": spec["task_id"],
                "image": spec["cluster"]["image"],
                "steps": meas["steps"],
                "warmup_steps": meas["warmup_steps"],
                "runs": {
                    "checkpoint": {
                        "name": "rec",
                        "metrics": metrics,
                        "summary": flagopt.summarize(spec, metrics)
                    }
                },
            }), "utf-8")
        (tmp / "fake_kubectl.py").write_text(FAKE_KUBECTL, "utf-8")
        (tmp / "driver.py").write_text(DRIVER, "utf-8")
        ws = tmp / "workspace"
        (ws / "output").mkdir(parents=True)
        if config is not None:
            (ws / "output" / "best_config.json").write_text(
                json.dumps(config), "utf-8")
        env = dict(os.environ,
                   PYTHONDONTWRITEBYTECODE="1",
                   TESTS_DIR=str(tests),
                   FLAGOPT_HOME=str(FLAGOPT_HOME),
                   FLAGOPT_TASK=GEMMA,
                   LOGS_DIR=str(tmp / "logs"),
                   WORKSPACE_DIR=str(ws),
                   FLAGOPT_LOG_DIR=str(tmp / "logs" / "verifier"),
                   FLAGOPT_ARTIFACTS_DIR=str(tmp / "logs" / "artifacts"),
                   FLAGOPT_VERIFIER_KUBECTL=(
                       f"{sys.executable} -I -S {tmp / 'fake_kubectl.py'}"),
                   FAKE_LOG=str(tmp / "calls.log"),
                   FAKE_STEPS=str(meas["steps"]),
                   FAKE_STEP_TIME="1.6")
        return tests, ws, env

    def _measure(self, config, *, loss_scale=1.0, step_time=1.6):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tests, ws, env = self.setup_tests_dir(tmp, config)
            env["FAKE_LOSS_SCALE"] = str(loss_scale)
            env["FAKE_STEP_TIME"] = str(step_time)
            out = subprocess.run([
                sys.executable, "-S",
                str(tmp / "driver.py"),
                str(tests / "throughput_improvement"),
                str(ws)
            ],
                                 env=env,
                                 check=True,
                                 capture_output=True,
                                 text=True,
                                 timeout=120)
            result = json.loads(out.stdout.strip().splitlines()[-1])
            calls_log = tmp / "calls.log"
            calls = (calls_log.read_text("utf-8").splitlines()
                     if calls_log.exists() else [])
            report = (tmp / "logs" / "verifier" /
                      "verification.md").read_text("utf-8")
        return result, calls, report

    def test_missing_deliverable_submits_nothing(self):
        result, calls, _ = self._measure(None)
        self.assertFalse(result["config_valid"])
        self.assertEqual(result["improvement_score"], 0.0)
        self.assertEqual(calls, [])

    def test_fixed_setting_in_deliverable_is_invalid(self):
        result, calls, _ = self._measure(
            {"maxtext_overrides": {
                "per_device_batch_size": 2
            }})
        self.assertFalse(result["config_valid"])
        self.assertIn("fixed", result["error"])
        self.assertEqual(calls, [])

    def test_candidate_is_scored_against_the_recording(self):
        result, calls, _ = self._measure(self.GOOD)
        self.assertEqual(result["baseline_source"], "recorded")
        # One Job (the candidate's) and its headless Service; no baseline Job.
        job_applies = [
            c for c in calls if c.startswith("apply") and "job.json" in c
        ]
        self.assertEqual(len(job_applies), 1)
        self.assertIn("vc", job_applies[0])
        self.assertAlmostEqual(result["speedup"], 1.25, places=4)
        self.assertTrue(result["accuracy_gate"])
        # The reward is the throughput improvement, speedup - 1.
        self.assertAlmostEqual(result["improvement_score"], 0.25, places=4)

    def test_reward_grows_with_the_improvement(self):
        """No cap: a 2x speedup scores 1.0, a 2.5x speedup 1.5."""
        for step_time, expected in ((1.0, 1.0), (0.8, 1.5)):
            with self.subTest(step_time=step_time):
                result, _, _ = self._measure(self.GOOD, step_time=step_time)
                self.assertAlmostEqual(result["improvement_score"],
                                       expected,
                                       places=4)

    def test_slower_config_scores_zero(self):
        result, _, _ = self._measure(self.GOOD, step_time=2.5)
        self.assertTrue(result["accuracy_gate"])
        self.assertLess(result["speedup"], 1.0)
        self.assertEqual(result["improvement_score"], 0.0)

    def test_changed_math_fails_the_gate(self):
        result, _, _ = self._measure(self.GOOD, loss_scale=1.01)
        self.assertFalse(result["accuracy_gate"])
        self.assertEqual(result["improvement_score"], 0.0)


class VerifierBudgetTest(unittest.TestCase):
    """How measure.py shares the verifier deadline between its runs."""

    def _evaluate(self, tmp: Path, statuses: list[str], *, recorded: bool,
                  deadline: str):
        tests, ws, env = MeasureTest.setup_tests_dir(tmp, MeasureTest.GOOD)
        if not recorded:
            (tests / "baselines" / f"{GEMMA}.json").unlink()
        env["VERIFY_DEADLINE_SEC"] = deadline
        calls = []
        spec = _spec(GEMMA)
        summary = flagopt.summarize(spec, _metrics(26))

        def fake_run(_spec, _cand, **kw):
            calls.append(kw)
            status = statuses[len(calls) - 1]
            ok = status == "SUCCEEDED"
            return {
                "name": kw["label"],
                "status": status,
                "ok": ok,
                "metrics": _metrics(26),
                "summary": summary if ok else {},
                "error_signatures": []
            }

        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
                flagopt, "run_candidate", side_effect=fake_run):
            sys.modules["flagopt"] = flagopt
            measure = _load("measure_under_test", VERIFIERS / "measure.py")
            measure.flagopt = flagopt
            result = measure.evaluate(ws)
        return result, calls

    def test_candidate_run_gets_the_whole_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self._evaluate(Path(tmp), ["SUCCEEDED"] * 2,
                                           recorded=False,
                                           deadline="3000")
        self.assertEqual(result["baseline_source"], "live")
        self.assertEqual(len(calls), 2)
        base, cand = calls
        self.assertLessEqual(base["run_timeout_s"], 1500)
        self.assertGreater(cand["run_timeout_s"], 2500)

    def test_shortened_run_timeout_is_an_infra_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self._evaluate(Path(tmp), ["RUN_TIMEOUT"],
                                           recorded=True,
                                           deadline="2000")
        self.assertEqual(len(calls), 1)
        self.assertLess(calls[0]["run_timeout_s"], 5400)
        self.assertFalse(result["candidate_ran"])
        self.assertTrue(result["infra_error"])

    def test_full_run_timeout_is_the_configs_fault(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, calls = self._evaluate(Path(tmp), ["RUN_TIMEOUT"],
                                           recorded=True,
                                           deadline="8400")
        self.assertEqual(calls[0]["run_timeout_s"], 5400)
        self.assertFalse(result["candidate_ran"])
        self.assertFalse(result["infra_error"])


@unittest.skipUnless(HAVE_SUITE_DEPS and HAVE_REWARDKIT,
                     "needs pydantic, yaml and harbor-rewardkit")
class RewardkitTest(unittest.TestCase):
    """The compiled tests/ directory through its generated test.sh."""

    def test_reward_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, ws, env = MeasureTest.setup_tests_dir(tmp / "unused",
                                                     MeasureTest.GOOD)
            tests = tmp / "tests"
            shutil.copytree(compiled() / GEMMA / "tests", tests)
            shutil.copytree(tmp / "unused" / "tests" / "baselines",
                            tests / "baselines",
                            dirs_exist_ok=True)
            bin_dir = Path(sys.executable).parent
            env.update(TESTS_DIR=str(tests),
                       PATH=f"{bin_dir}:{env.get('PATH', '')}")
            subprocess.run(["bash", str(tests / "test.sh")],
                           env=env,
                           check=True,
                           capture_output=True,
                           text=True,
                           timeout=300)
            reward = json.loads((tmp / "unused" / "logs" / "verifier" /
                                 "reward.json").read_text("utf-8"))
        self.assertEqual(
            sorted(reward),
            sorted([
                "accuracy_gate", "candidate_ran", "config_valid",
                "infra_error", "throughput_improvement", "reward"
            ]))
        self.assertAlmostEqual(reward["reward"], 0.25, places=3)
        self.assertEqual(reward["infra_error"], 0.0)


if __name__ == "__main__":
    unittest.main()

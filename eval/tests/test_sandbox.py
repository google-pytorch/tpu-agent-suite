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
"""sandbox.sh's host-side work, run for real with docker and tmux stubbed out.

Everything up to the container is real: the config is resolved and the package
compiled under a temporary HOME. Only `docker`, `tmux` and `claude` are
stand-ins, and the docker stub keeps what `docker cp` was handed, so what these
tests assert is the tree the container would have received and whether it was
started at all.

Adapted from the suite in #323, which found the bundle `package_dir` bug and
reproduced the missing `packages/` directory. The install now happens inside
the container, so the two tests that pinned the host install's failure modes
assert instead that the host performs no install for them to corrupt.
"""

import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SANDBOX = _REPO / "eval" / "sandbox.sh"
# The agent-suite configs install the whole package.
_CONFIG_BUNDLE = "claude-code/agent-suite"
_CONFIG_OTHER = "agy/agent-suite"
# A control arm: it is named `baseline`, so it installs nothing.
_CONFIG_BASELINE = "claude-code/baseline"


def _executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@unittest.skipUnless(shutil.which("bash"), "sandbox.sh needs bash")
class TestSandbox(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        adc = self.home / ".config" / "gcloud"
        adc.mkdir(parents=True)
        (adc / "application_default_credentials.json").write_text("{}")
        self.log = self.tmp / "calls.log"
        # What `docker cp` was handed, kept so the staged tree outlives the
        # scratch directory sandbox.sh deletes on the way out.
        self.staged = self.tmp / "staged"
        self.staged.mkdir()

        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        _executable(
            self.bin / "docker", "#!/bin/sh\n"
            f'echo "docker $*" >> {shlex.quote(str(self.log))}\n'
            'if [ "$1" = "cp" ]; then '
            f'cp -r "$2" {shlex.quote(str(self.staged))}/; fi\n'
            "exit 0\n")
        _executable(
            self.bin / "tmux", "#!/bin/sh\n"
            f'echo "tmux $*" >> {shlex.quote(str(self.log))}\n'
            "exit 0\n")
        # The installer checks the harness version before anything else.
        _executable(self.bin / "claude",
                    "#!/bin/sh\necho '2.1.241 (Claude Code)'\n")
        # The interpreter running these tests, which has pyyaml; a wrapper
        # rather than a symlink so a virtualenv still finds its packages.
        _executable(self.bin / "python3",
                    f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')

    def _run(self, *args: str, **env: str) -> subprocess.CompletedProcess:
        environ = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            **env,
        }
        # stdin is not a terminal, so the run stops after preparing the
        # container rather than attaching a shell to it.
        return subprocess.run(["bash", str(_SANDBOX), "-c", *args],
                              cwd=_REPO,
                              env=environ,
                              stdin=subprocess.DEVNULL,
                              capture_output=True,
                              text=True,
                              timeout=900,
                              check=False)

    def _calls(self) -> list[str]:
        if not self.log.exists():
            return []
        return self.log.read_text().splitlines()

    def _dist_agents(self, harness: str) -> set[str]:
        # Markdown everywhere except codex, which writes each agent as TOML.
        agents = self.staged / "sandbox" / "dist" / harness / "agents"
        return {
            p.stem
            for p in agents.iterdir() if p.suffix in (".md", ".toml")
        }

    def _explain(self, proc: subprocess.CompletedProcess) -> str:
        return (f"exit {proc.returncode}\n{proc.stdout[-3000:]}\n"
                f"{proc.stderr[-3000:]}")

    def test_an_agent_suite_config_compiles_the_whole_graph(self):
        """Every skill package.release.json releases, and every agent they reach."""
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        self.assertNotIn("SKILL.md does not exist", proc.stderr)
        # One agent from each of several families: the whole graph. Sharding is
        # not among them: nothing has routed to it since #313, so no build
        # holds it.
        for agent in ("tpu-agent", "tpu-model-onboarding",
                      "tpu-kernel-development", "tpu-diagnostics",
                      "tpu-knowledge-base"):
            self.assertIn(agent, self._dist_agents("claude-code"))

    def test_every_harness_is_compiled_and_staged(self):
        """The container has no compiler.py to produce another one later."""
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        for harness in ("claude-code", "antigravity", "codex"):
            dist = self.staged / "sandbox" / "dist" / harness
            self.assertTrue((dist / "framework" / "installer.py").is_file(),
                            harness)
            self.assertIn("tpu-agent", self._dist_agents(harness))

    def test_every_harness_gets_an_install_and_a_run_script(self):
        """The sandbox does not decide which harness you are evaluating."""
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        bootstrap = (self.staged / "sandbox" / "bootstrap.sh").read_text()
        for script in ("install_claude", "install_agy", "install_codex",
                       "run_claude", "run_agy", "run_codex"):
            self.assertIn(f"cat > /root/.local/bin/{script} <<", bootstrap)

    def test_every_agent_suite_config_builds_every_skill(self):
        """No config chooses a subset: there is one package."""
        proc = self._run(_CONFIG_OTHER)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        skills = self.staged / "sandbox" / "dist" / "claude-code" / "project"
        self.assertEqual(sorted(p.name for p in (skills / "skills").iterdir()),
                         ["tpu-agent", "tpu-agent-export-trajectory"])

    def test_skills_builds_only_that_skill(self):
        """The subset mode: a debugging subset of the package."""
        proc = self._run(_CONFIG_BUNDLE, "--skills", "tpu-agent")

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        skills = self.staged / "sandbox" / "dist" / "claude-code" / "project"
        self.assertEqual(sorted(p.name for p in (skills / "skills").iterdir()),
                         ["tpu-agent"])

    def test_skills_with_a_baseline_is_refused(self):
        proc = self._run(_CONFIG_BASELINE, "--skills", "tpu-agent")

        self.assertNotEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("control arm", proc.stderr)

    def test_skills_in_ci_is_refused(self):
        proc = self._run(_CONFIG_BUNDLE,
                         "--skills",
                         "tpu-agent",
                         GITHUB_RUN_ID="1")

        self.assertNotEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("CI runs the whole package", proc.stderr)

    def test_a_baseline_config_compiles_nothing(self):
        """A control arm is named, not configured: nothing is compiled."""
        proc = self._run(_CONFIG_BASELINE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("Control arm", proc.stdout)
        self.assertFalse((self.staged / "sandbox" / "dist").exists())

    def test_the_host_installs_nothing_for_its_environment_to_misplace(self):
        """TPU_AGENT_SUITE_HOME and a shadowing framework had moved the payload.

        Both only ever reached the install because it ran on the host, in the
        caller's environment: the central store went elsewhere, the container
        got no packages/ directory, and every hook failed on a missing
        runtime.py. The host only compiles now, so neither has an install to
        redirect -- asserted here as the host writing no install at all.
        """
        elsewhere = self.tmp / "elsewhere"
        shadow = self.tmp / "shadow"
        (shadow / "framework").mkdir(parents=True)
        (shadow /
         "tpu-agent-suite-build.json").write_text('{"version": "0.1.0"}\n')
        (shadow / "framework" / "__init__.py").write_text("")
        # The real utils first, so only get_tpu_agent_suite_home differs: the rest of
        # the framework still has to import for the host compile to run at all.
        real_utils = _REPO / "framework" / "utils.py"
        (shadow / "framework" / "utils.py").write_text(
            f"exec(compile(open({str(real_utils)!r}).read(),"
            f" {str(real_utils)!r}, 'exec'))\n"
            "def get_tpu_agent_suite_home():\n"
            f"    return Path({str(elsewhere)!r})\n")

        proc = self._run(_CONFIG_BUNDLE,
                         TPU_AGENT_SUITE_HOME=str(elsewhere),
                         PYTHONPATH=str(shadow))

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        self.assertFalse(elsewhere.exists(), sorted(self.tmp.iterdir()))
        self.assertFalse((self.home / ".tpu-agent-suite").exists())
        self.assertFalse((self.home / ".claude").exists())
        self.assertFalse((self.home / "tmp").exists())
        # The dist still crossed into the container intact.
        self.assertIn("tpu-agent", self._dist_agents("claude-code"))

    def test_the_install_scripts_refuse_an_install_without_its_hook_runtime(
            self):
        """A payload-less install must not be reported as a success.

        The check moves with the install: it runs in the container, at the end
        of bin/install_*, rather than before launch on the host.
        """
        proc = self._run(_CONFIG_BUNDLE)
        self.assertEqual(proc.returncode, 0, self._explain(proc))

        bootstrap = (self.staged / "sandbox" / "bootstrap.sh").read_text()
        for script in ("install_claude", "install_agy", "install_codex"):
            body = bootstrap.split(f"cat > /root/.local/bin/{script} <<")[1]
            self.assertIn("framework/runtime.py", body)
            self.assertIn("the install left no framework/runtime.py", body)

    def test_agy_takes_the_mounted_adc_by_default(self):
        """The credential is already in the container; agy has to be told.

        Emitted as a ${VAR:-default} so the config and env.txt, both sourced
        above it, still win.
        """
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        env = (self.staged / "sandbox" / "env.txt").read_text()
        self.assertIn('export AGY_ADC_AUTH="${AGY_ADC_AUTH:-true}"', env)
        self.assertIn(
            "export GOOGLE_APPLICATION_CREDENTIALS="
            "/root/.config/gcloud/application_default_credentials.json", env)

    def _source_env(self, **env: str) -> str:
        """KUBECONFIG and SANDBOX_TEST_VALUE after sourcing the generated file."""
        env_file = self.staged / "sandbox" / "env.txt"
        return subprocess.run([
            "env", "-i", *(f"{k}={v}" for k, v in env.items()), "bash", "-c",
            f'. {shlex.quote(str(env_file))}; '
            'printf "%s\\n%s" "$KUBECONFIG" "$SANDBOX_TEST_VALUE"'
        ],
                              capture_output=True,
                              text=True,
                              check=True).stdout

    def test_a_child_shell_keeps_the_kubeconfig_it_inherited(self):
        """BASH_ENV re-runs the env file in every shell; it must not reset this.

        The config and the env file both set KUBECONFIG, so a plain export
        anywhere in the generated file would put a child back on the mount.
        """
        (self.home / ".kube").mkdir()
        (self.home / ".kube" / "config").write_text("{}")
        env_file = self.tmp / "env.txt"
        env_file.write_text('export KUBECONFIG="/host/only/kubeconfig"\n'
                            'export PATH="$PATH:/host/only/bin"\n')

        proc = self._run(_CONFIG_BUNDLE, SANDBOX_ENV_FILE=str(env_file))

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        runs = [c for c in self._calls() if c.startswith("docker run -d ")]
        self.assertIn(":/root/.kube/config:ro", runs[0])
        self.assertNotIn("/host/only",
                         (self.staged / "sandbox" / "env.txt").read_text())
        self.assertEqual(self._source_env().splitlines()[0],
                         "/root/.kube/config")
        self.assertEqual(
            self._source_env(KUBECONFIG="/tmp/agent").splitlines()[0],
            "/tmp/agent")

    def test_the_env_file_is_written_in_not_sourced_from_the_sandbox(self):
        """The file is never copied in, so a path to it is dead and a leak."""
        env_file = self.tmp / "env.txt"
        env_file.write_text("SANDBOX_TEST_VALUE='a b $c'\n")

        proc = self._run(_CONFIG_BUNDLE, SANDBOX_ENV_FILE=str(env_file))

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        env = (self.staged / "sandbox" / "env.txt").read_text()
        self.assertNotIn(str(env_file), env)
        self.assertNotIn("/root/sandbox/eval", env)
        self.assertEqual(self._source_env().split("\n", 1)[1], "a b $c")

    def test_a_named_env_file_that_is_missing_is_an_error(self):
        """Asked for by name, so starting without its settings is wrong."""
        proc = self._run(_CONFIG_BUNDLE,
                         SANDBOX_ENV_FILE=str(self.tmp / "absent.txt"))

        self.assertNotEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("SANDBOX_ENV_FILE", proc.stderr)
        self.assertFalse([c for c in self._calls() if "run -d" in c])

    def test_no_env_file_found_is_a_warning(self):
        candidates = ("eval/env.txt", "env.txt", "eval/.env", ".env")
        if any((_REPO / c).is_file() for c in candidates):
            self.skipTest("this checkout has its own env file")

        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("WARNING: no env file found", proc.stderr)

    def test_vertex_defaults_match_the_harness(self):
        """Same project and variable name as agents/claude_code.py."""
        proc = self._run(_CONFIG_BUNDLE)
        self.assertEqual(proc.returncode, 0, self._explain(proc))
        env = (self.staged / "sandbox" / "env.txt").read_text()
        self.assertIn(
            'export ANTHROPIC_VERTEX_PROJECT_ID='
            '"${ANTHROPIC_VERTEX_PROJECT_ID:-tpu-nexus-767207}"', env)

        shutil.rmtree(self.staged / "sandbox")
        proc = self._run(_CONFIG_BUNDLE, VERTEXAI_PROJECT="from-ci")
        self.assertEqual(proc.returncode, 0, self._explain(proc))
        env = (self.staged / "sandbox" / "env.txt").read_text()
        self.assertIn(
            'export ANTHROPIC_VERTEX_PROJECT_ID='
            '"${ANTHROPIC_VERTEX_PROJECT_ID:-from-ci}"', env)

    def test_harbor_credentials_are_mounted_once(self):
        harbor = self.home / ".harbor"
        harbor.mkdir()
        (harbor / "credentials.json").write_text("{}")

        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        runs = [c for c in self._calls() if c.startswith("docker run -d ")]
        self.assertEqual(len(runs), 1, self._calls())
        self.assertEqual(runs[0].count(":/root/.harbor:ro"), 1, runs[0])

    def test_the_container_is_a_daemon_that_outlives_the_shell(self):
        """No --rm, and nothing that ends when a shell does."""
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        runs = [c for c in self._calls() if c.startswith("docker run ")]
        self.assertEqual(len(runs), 1, self._calls())
        self.assertIn("sleep infinity", runs[0])
        self.assertNotIn("--rm", runs[0])
        self.assertNotIn("/workspace", runs[0].split(" -w ")[0])

    def test_the_repository_is_never_mounted(self):
        """A control arm that can read package/ is not a control arm."""
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        for call in self._calls():
            self.assertNotIn(f"{_REPO}:/workspace", call)

    def test_no_terminal_means_no_tmux_session(self):
        proc = self._run(_CONFIG_BUNDLE)

        self.assertEqual(proc.returncode, 0, self._explain(proc))
        self.assertFalse([c for c in self._calls() if "new-session" in c],
                         self._calls())
        # The stale session is still cleared, so a later attach is not handed
        # a window onto the container this run just replaced.
        self.assertTrue([c for c in self._calls() if "kill-session" in c])

    def test_the_package_override_is_rejected(self):
        """The config decides the arm; an override contradicts it."""
        proc = self._run(_CONFIG_BUNDLE, "--package", "tpu-hello-world")

        self.assertNotEqual(proc.returncode, 0, self._explain(proc))
        self.assertIn("baseline", proc.stderr)
        self.assertFalse([c for c in self._calls() if "run -d" in c])


if __name__ == "__main__":
    unittest.main()

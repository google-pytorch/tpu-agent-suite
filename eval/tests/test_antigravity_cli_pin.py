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
"""Tests for the harness CLI pins in .ci/ and how the eval adapter applies them."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from eval.adapter import sitecustomize as _ADAPTER

try:
    from harbor.agents.installed.antigravity_cli import AntigravityCli
except ImportError:
    AntigravityCli = None


def _fake_release(version: str) -> bytes:
    """A release tarball whose `antigravity` reports `version`."""
    script = (
        f"#!/bin/sh\n"
        f"if [ \"$1\" = --version ]; then echo {version}; fi\n").encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo("antigravity")
        info.size = len(script)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(script))
    return buf.getvalue()


class TestPinRequested(unittest.TestCase):

    def test_pinned_unless_latest_is_asked_for(self) -> None:
        env = _ADAPTER.ANTIGRAVITY_CLI_VERSION_ENV
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(env, None)
            self.assertTrue(_ADAPTER.antigravity_cli_pin_requested())
        for value in ("latest", " Latest "):
            with mock.patch.dict(os.environ, {env: value}):
                self.assertFalse(_ADAPTER.antigravity_cli_pin_requested())

    def test_pin_is_a_full_release(self) -> None:
        pins = _ADAPTER.read_harness_pins()
        self.assertRegex(pins["AGY_VERSION"], r"^\d+\.\d+\.\d+$")
        self.assertIn(f"/{pins['AGY_VERSION']}-", pins["AGY_URL"])
        self.assertTrue(pins["AGY_URL"].startswith("https://"))
        self.assertRegex(pins["AGY_SHA512"], r"^[0-9a-f]{128}$")
        self.assertRegex(pins["AGY_PATCH_BEFORE"], r"^[0-9a-f]+$")
        self.assertRegex(pins["AGY_PATCH_AFTER"], r"^[0-9a-f]+$")
        self.assertEqual(
            len(bytes.fromhex(pins["AGY_PATCH_BEFORE"])),
            len(bytes.fromhex(pins["AGY_PATCH_AFTER"])),
        )
        self.assertNotEqual(pins["AGY_PATCH_BEFORE"], pins["AGY_PATCH_AFTER"])


class TestInstallScript(unittest.TestCase):
    """Runs .ci/install-agy.sh against a local fake release.

    The script text runs under `bash -c` with the pin in the environment, as
    it does inside a trial container, so the real pins.conf is never sourced.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.home = root / "home"
        self.home.mkdir()
        self.bin = root / "bin"
        self.bin.mkdir()
        self.release = root / "release.tar.gz"
        # `curl -fsSL -o <dest> <url>` copies the local release instead.
        curl = self.bin / "curl"
        curl.write_text('#!/bin/sh\nwhile [ "$1" != -o ]; do shift; done\n'
                        f'cp "{self.release}" "$2"\n')
        curl.chmod(0o755)
        self.script = _ADAPTER.ANTIGRAVITY_CLI_INSTALL_SCRIPT.read_text(
            encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _run(self, pin: dict[str, str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", "-c", self.script],
            env={
                "HOME": str(self.home),
                "PATH": f"{self.bin}:{os.environ['PATH']}",
                **pin,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    def _pin(self, served: str, expected: str) -> dict[str, str]:
        data = _fake_release(served)
        self.release.write_bytes(data)
        return {
            "AGY_VERSION": expected,
            "AGY_URL": f"https://example.invalid/{expected}-1/cli.tar.gz",
            "AGY_SHA512": hashlib.sha512(data).hexdigest(),
        }

    def test_installs_the_pinned_release(self) -> None:
        result = self._run(self._pin("9.9.9", "9.9.9"))
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.home / ".local" / "bin" / "agy"
        self.assertTrue(os.access(installed, os.X_OK))
        self.assertIn("9.9.9", result.stdout)

    def test_patches_binary_in_place(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["AGY_PATCH_BEFORE"] = b"9.9.9".hex()
        pin["AGY_PATCH_AFTER"] = b"9.9.0".hex()
        pin["AGY_VERSION"] = "9.9.0"
        result = self._run(pin)
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.home / ".local" / "bin" / "agy"
        self.assertIn(b"9.9.0", installed.read_bytes())
        self.assertNotIn(b"9.9.9", installed.read_bytes())

    def test_missing_patch_target_installs_nothing(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["AGY_PATCH_BEFORE"] = "deadbeef"
        pin["AGY_PATCH_AFTER"] = "cafebabe"
        result = self._run(pin)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / ".local" / "bin" / "agy").exists())

    def test_a_checksum_mismatch_installs_nothing(self) -> None:
        pin = self._pin("9.9.9", "9.9.9")
        pin["AGY_SHA512"] = "0" * 128
        result = self._run(pin)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.home / ".local" / "bin" / "agy").exists())

    def test_a_binary_reporting_another_version_fails(self) -> None:
        result = self._run(self._pin("9.9.8", "9.9.9"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 9.9.9", result.stderr)


@unittest.skipIf(AntigravityCli is None, "harbor is not installed")
class TestPatchedInstall(unittest.TestCase):

    def _install(self, env: dict[str, str]) -> mock.MagicMock:
        agent = mock.MagicMock()
        agent.ensure_system_dependencies = mock.AsyncMock()
        agent.exec_as_agent = mock.AsyncMock()
        with mock.patch.dict(os.environ, env):
            asyncio.run(AntigravityCli.install(agent, mock.MagicMock()))
        return agent

    def test_install_is_patched(self) -> None:
        self.assertTrue(getattr(AntigravityCli, "_eval_cli_pin_patched",
                                False))

    def test_installs_the_pin_by_default(self) -> None:
        env = {_ADAPTER.ANTIGRAVITY_CLI_VERSION_ENV: ""}
        agent = self._install(env)
        command = agent.exec_as_agent.call_args.kwargs["command"]
        pins = _ADAPTER.read_harness_pins()
        self.assertTrue(command.startswith("bash -c "))
        self.assertIn(pins["AGY_SHA512"], command)
        self.assertIn(pins["AGY_PATCH_BEFORE"], command)
        self.assertIn(pins["AGY_PATCH_AFTER"], command)
        self.assertNotIn("install.sh", command)

    def test_validate_reasoning_effort_allows_medium_for_gemini_4(
            self) -> None:
        AntigravityCli._validate_reasoning_effort("medium",
                                                  "gemini/gemini-4-argon")
        with self.assertRaises(ValueError):
            AntigravityCli._validate_reasoning_effort(
                "medium", "gemini/gemini-3.1-pro-preview")


if __name__ == "__main__":
    unittest.main()

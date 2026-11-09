# TPU Agent Suite vX.Y.Z

<!--
Copy this file to release-notes/vX.Y.Z.md (matching "version" in
package/package.release.json) in the version-bump PR.

These notes are published verbatim to the public
google-pytorch/tpu-agent-suite Release after passing scripts/check_denylist.py.
Write for external users: no internal links, bug or CL numbers, LDAPs, or
private repository/PR references.
-->

## Highlights

-

## New

-

## Changes

-

## Fixes

-

## Install

Download the archive for your harness (`claude-code` or `antigravity`) below, activate a Python 3.12+ virtual environment, then:

```bash
tar -xzf tpu-agent-suite-<harness>-vX.Y.Z.tar.gz
cd tpu-agent-suite-<harness>-vX.Y.Z
python3 -m pip install -r requirements-client.txt
python3 framework/installer.py install --destination /path/to/project
```

Pass `--scope user` instead of `--destination` for a user-wide install; see `docs/install.md` inside the archive for backend setup and upgrades. Verify downloads against `SHA256SUMS`: `sha256sum -c SHA256SUMS --ignore-missing`

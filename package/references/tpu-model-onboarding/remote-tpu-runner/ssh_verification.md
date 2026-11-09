# Connectivity Verification

Before sizing, check that a compute target (local, GCE TPU VM, or GKE) is
configured.

## Instructions

Read the compute configuration; do not run `compute inspect`:

1. Run:

   ```bash
   {{runtime}} config show --json --package {{package_tag}}
   ```

2. If it names no active profile, do not proceed. Report
   `COMPUTE_NOT_CONFIGURED` and exit. A connectivity error from the first job
   is reported the same way.

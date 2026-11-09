Profile and size the TPU accelerator hardware and topology of the GKE TPU cluster configured for this package using the `remote-tpu-runner` reference guides (`ssh_verification.md` and `hardware_profiling.md`).

Verify the active compute configuration, size the cluster hardware, and report the hardware profile.

Important execution instructions:
- Only inspect and size the remote TPU cluster configuration (do not run model benchmarks or deploy servers).
- Follow the `remote-tpu-runner` reference guides (`config show --json` and `tpu-knowledge-base` hardware sizing, or `compute inspect --json`) rather than inspecting only the local container's CPU/GPU hardware.
- Report the sized hardware profile for the cluster, including `accelerator`, `chips`, `hbm_gb`, `details.job_devices` (or `job_devices`), and `details.hbm_gb_per_device` (or `hbm_gb_per_device`).

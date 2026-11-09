# Accelerator Hardware Sizing

Size the accelerator the request targets. Do not run `compute inspect` or
query the cluster or VM yourself.

## Instructions

1. Take the accelerator from the request's `accelerator_type`.

2. Ask {{agent:tpu-knowledge-base}} for that accelerator's HBM per **chip**.
   With no page for it, return `needs_input` naming the accelerator. Devices
   per chip comes from tpu-model-onboarding's table, not the page.

3. Take the chips one deployment gets from the active compute profile
   (`{{runtime}} config show --json --package {{package_tag}}`): `tpu_chips`,
   else the product of `tpu_topology`. With neither, return `needs_input`
   asking your caller for the chip count.

4. Compute what inspect would report:
   - **Devices, for tensor parallelism**: TP is sized to devices, not chips,
   and some chips are more than one device. Devices = chips x devices per
   chip, and HBM per device = HBM per chip / devices per chip: a tpu7x chip
   is two devices of 96 GB each, so a 2x2x1 slice is 4 chips but 8 devices.
   Do not assume one device per chip.
   - **Memory limit** (`hbm_gb`): HBM per chip x chips.

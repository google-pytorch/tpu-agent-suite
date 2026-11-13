# onboarding-frontier

Onboard a frontier MoE checkpoint (DeepSeek-V4.1-Flash, GLM-5.3-Flash) on
vLLM-TorchTPU, with the nightly Qwen3.8-27B task as the control.

Run these on a config whose slice holds the weights: the staged
checkpoints are about 510 GB (DeepSeek, fp8 with fp4 experts) and 328 GB
(GLM, fp8), and onboarding success is judged on the configured
`GKE_TPU_ACCELERATOR`/`GKE_TPU_TOPOLOGY`. The `*-shared-capacity*` configs
(tpu7x 2x2x1, 768 GB of HBM) do; the default `tpu-v6e-slice` 2x4 configs
(256 GB) cannot, and a run there scores 0.

    TASK_DOCKER=<nexus-tpu digest> python3 eval/run.py \
        --config claude-code/agent-suite-shared-capacity-remote \
        --task onboarding-frontier/deepseek-v4-1-flash

All three task images are one overlay `FROM nexus-tpu:latest`; run.py
stages the base digest so a moved `:latest` builds a new image.

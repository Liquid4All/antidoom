#!/usr/bin/env bash
# antidoom runs inside the cluster's vLLM ROCm image (see training.toml): the pinned uv.lock is CUDA-only, and
# vLLM's rolling ROCm wheel index tracks a newer ROCm than the nodes run.
set -euo pipefail
docker pull -q vllm/vllm-openai-rocm:v0.33.0-rocm72

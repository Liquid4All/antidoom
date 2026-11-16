#!/usr/bin/env bash
# The pinned uv.lock is CUDA-only; on AMD the venv takes vLLM's ROCm wheels instead (README "AMD / ROCm").
set -euo pipefail
uv venv --allow-existing
uv pip install -e . --extra-index-url https://wheels.vllm.ai/rocm/
uv run --no-sync python -c "import torch, vllm; assert torch.version.hip, 'expected a ROCm torch build'; print(torch.__version__, vllm.__version__)"

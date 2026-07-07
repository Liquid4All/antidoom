# Antidoom

Antidoom generates and trains targeted preference data for reducing model
repetition loops (doom loops).

It is a narrow tool for a narrow failure mode: sample model completions, detect
where a repeated span begins, mark the first loop-starting token as rejected,
choose coherent alternative next tokens, then train a LoRA adapter with Final
Token Preference Optimization (FTPO).

The method adapts the single-token preference training idea from
[Antislop](https://arxiv.org/abs/2510.15061) and its
[auto-antislop implementation](https://github.com/sam-paech/auto-antislop) to
the specific problem of runaway repetition during reasoning.

## Prerequisites

Antidoom needs a supported GPU runtime for PyTorch and vLLM. NVIDIA/CUDA is the
default path; AMD/ROCm requires a separate environment.

- **NVIDIA / CUDA:** use the pinned `uv.lock` and
  [`configs/default.yaml`](configs/default.yaml).

- **AMD / ROCm (Instinct MI-series):** do not use the default `uv.lock`; it is
  CUDA-only. See [AMD / ROCm](#amd--rocm) to create the ROCm venv and run
  with [`configs/default_amd.yaml`](configs/default_amd.yaml), which adds the required
  `attention_backend: TRITON_ATTN` and `optim: adamw_torch` overrides.

## Quick Start

```bash
git clone https://github.com/Liquid4All/antidoom
cd antidoom
uv sync
```

Edit [`configs/default.yaml`](configs/default.yaml) and set `model_name` to the
checkpoint you want to antidoom. The default config reads prompts from
[LiquidAI/antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0), generates FTPO pairs, trains a LoRA adapter, and
merges it.

Run the full generate-and-train flow:

```bash
uv run antidoom -c configs/default.yaml -r runs/antidoom1 \
  --temp 0.01 \
  --model-name LiquidAI/LFM2.5-1.2B-Base
```

This writes generated completions and FTPO pairs to `runs/antidoom1/`, trains
on the pair file, and writes the adapter plus merged model under the same run
directory.

## Why Doom Loops Happen

Doom loops tend to appear when three forces line up:

- **Overtrained tokens.** Common reasoning tokens such as `Wait`, `So`,
  `But`, `Alternatively`, can become unusually attractive after heavy
  synthetic reasoning training. When the model is uncertain, those tokens can
  dominate the next-token distribution without moving the reasoning forward.
- **Self-reinforcing context.** Once a short sequence appears, the prior context
  makes that sequence more likely to appear again. Across repeated turns of the
  same loop, the probability of each token can climb toward certainty.
- **Low-temperature sampling.** At temperature 0 or near 0, the model keeps
  selecting the highest-probability continuation, so a locally reinforced loop
  has no natural escape route.

Antidoom attacks the failure at the token where the loop begins. Instead of
training on full gold answers, it trains only on the local preference:
do not choose the token that starts the repetition; choose one of the plausible
alternatives available at that same position.

## How It Works

For each prompt, Antidoom generates a completion and scans it for inner
repetition. When a loop is found, it refines the boundary in token space so the
rejected token is the first readable token of the repeated segment.

Each FTPO row contains:

- a prompt/context prefix ending immediately before the rejected token
- one rejected token, the token that begins the loop
- one or more chosen tokens sampled from filtered alternatives at that position
- metadata about the source prompt, generated completion, and detected loop

The training step regularises overrepresented rejected and chosen tokens so the
adapter learns a broad anti-loop preference rather than simply suppressing one
word.

## Dataset

The default config uses
[LiquidAI/antidoom-mix-v1.0](https://huggingface.co/datasets/LiquidAI/antidoom-mix-v1.0),
a prompt-only ShareGPT mixture built for this pipeline.

The dataset intentionally excludes gold answers, rationales, hidden tests,
verifier targets, and answer labels.

Dataset build scripts and license notes live in `datasets/`.

## Configure

Start from [`configs/default.yaml`](configs/default.yaml). The most common
fields to change are:

- `model_name`: base checkpoint or Hugging Face model id
- `generation.hf_dataset`: prompt dataset, or set `generation.input_jsonl`
- `generation.prompt_field`: column to read from local/HF rows
- `generation.target_pairs`: number of FTPO pairs to collect
- `generation.temperature` or `generation.temperatures`: sampling temperature
- `train.max_train_examples`: maximum FTPO rows used for training
- `train.output_dir` and `train.merged_output_dir`: adapter and merged model dirs

`prompt_field` is authoritative: generation reads that column and parses the
value by shape. A string is treated as a plain user prompt, OpenAI-style
`[{"role": ..., "content": ...}]` lists are used as chat messages, and
ShareGPT-style `[{"from": ..., "value": ...}]` lists are converted to chat
messages. The parser does not infer prompt format from the column name.

### Important Hyperparameters

`max_train_examples`

It is important to generate enough preference samples before training. The
number of samples produced depends on the number of prompts available to the
generation step, the number of temperature passes you run, and how doom-loopy
the checkpoint is. As a rule of thumb, start with at least 15k prompts and aim
to produce roughly 15k-20k preference rows.

Set `max_train_examples` below the number of generated preference rows, for
example 12k rows from a 15k-20k generated set. This gives
`rejected_regularisation_strength` room to shave off overrepresented rejected
tokens. Regularisation matters: without it, the generated set can be badly
unbalanced, which can create poor training outcomes.

Usage hint: set `max_train_examples` to at most 70% of the actual number of
preference rows in the generated dataset.

`learning_rate`

The Antidoom trainer can undertrain or overtrain. If overtrained, the model can
degrade and produce more doom loops, so the right learning rate can take a
little trial and error.

Usage hint: `0.00001`-`0.00002` is a good starting range when training on about
12k samples. Early stopping is useful to avoid overtraining.

`early_stopping_chosen_win`

This stops training when `chosen_win` exceeds the configured value.
`chosen_win` is the share of samples where the chosen token is winning against
the rejected token. Around `0.15`-`0.3`, you should usually see a strong
reduction in doom looping. Training past `0.5` may be overtraining, though more
ablations are needed.

Usage hint: try `0.4` and adjust from there.

### Other Hyperparameters

`generation.*`

These control the generation stage, where we try to elicit doom loops to build
the preference training set. The defaults are usually a reasonable starting
point.

`rejected_regularisation_strength`

This flattens the distribution of rejected-token frequency by culling samples.
Some rejected tokens occur much more frequently than others, because they are
the tokens that begin doom loops. If the training set is too unbalanced, those
tokens can be suppressed too aggressively.

Usage hint: the default `0.3` is a good starting point.

`chosen_regularisation_strength`

This flattens the distribution of chosen-token frequency. If a chosen token is
too overrepresented, its probability can be increased too much and it may start
causing loops of its own. This step usually prunes chosen alternatives rather
than whole samples, because each sample can contain multiple chosen tokens.

Usage hint: the default `0.5` is a good starting point.

`filter_rejected_stop_words`

Filters samples where the rejected token is a stop word like `the` or `a`.
Leave this set to `false`; common words can genuinely be loop-starting tokens
and the regularisation pass is the safer way to handle frequency.

`source_balance_mode`

Optionally balances the training set by source after rejected-token
regularisation. The default `off` is fine. Older configs may call this
`balance_by_source_task`.

`min_chosen_tokens`

The minimum number of chosen tokens required for a sample to be included. It
can sometimes be useful to require more than one chosen token to avoid mode
collapse on a single preferred token, but chosen-token regularisation already
mitigates this and `1` is usually fine for Antidoom training.

`lora_r`

A higher LoRA rank than usual tends to improve learnability with less
degradation for this trainer. If in doubt, use `128` or `256`.

`lora_alpha`

Set this to the same value as `lora_r`, or half of `lora_r`.

`lora_dropout`

This has not been heavily ablated, but `0` works well in current runs.

`target_modules`

Previous Antislop ablations found that training all layers learned more easily
but could also degrade more easily; restricting to `up_proj`, `down_proj`, and
`lm_head` helped there. Antidoom training appears to prefer training all layers.
The safe starting point is:

```text
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
```

`freeze_early_layers`

This can reduce model impact and degradation at the expense of learnability. In
practice, Antidoom training has worked well with freezing turned off.

`lambda_mse_target`

Strength of the MSE tether to the reference model for the chosen and rejected
logits.

`tau_mse_target`

Penalty-free logit delta from the reference before the target-logit MSE loss
kicks in.

`lambda_mse`

Strength of the MSE tether to the reference model for the remaining vocabulary,
excluding chosen and rejected logits. This is typically set higher than
`lambda_mse_target`, because chosen/rejected logits should move more freely
while the rest of the vocabulary stays close to the reference.

`clip_epsilon_logits`

Once the chosen token is beating the rejected token by this logit margin, the
preference loss turns off.

## Generate and Train

By default, `antidoom` is a one-shot pipeline. It generates preference rows
until `generation.target_pairs` is satisfied, then trains. If the target pair
JSONL already exists and has enough rows, generation returns immediately and
training starts.

```bash
uv run antidoom -c configs/default.yaml -r runs/antidoom1 \
  --temp 0.01 \
  --model-name LiquidAI/LFM2.5-1.2B-Base
```

With `-r runs/antidoom1`, generation writes:

```text
runs/antidoom1/iter_0_generations.jsonl
runs/antidoom1/iter_0_ftpo_pairs.jsonl
```

and training writes:

```text
runs/antidoom1/lora
runs/antidoom1/<auto-named-merged-model-dir>
```

## Stage Commands

Run generation only:

```bash
uv run antidoom -c configs/default.yaml -r runs/antidoom1 generate \
  --model-name LiquidAI/LFM2.5-1.2B-Base
```

Train from an existing pair file:

```bash
uv run antidoom -c configs/default.yaml -r runs/antidoom1 train \
  --dataset-jsonl runs/antidoom1/iter_0_ftpo_pairs.jsonl \
  --model-name LiquidAI/LFM2.5-1.2B-Base
```

Merge an adapter:

```bash
uv run antidoom -c configs/default.yaml -r runs/antidoom1 merge \
  --model-name LiquidAI/LFM2.5-1.2B-Base
```

## Filtering

Generation skips prompts that exceed `generation.max_prompt_tokens`. The generated preference dataset is regularized so that any given token does not occur too frequently in the *chosen* or *rejected* sets.

## Outputs

Each FTPO row contains fields like:

```json
{
  "source_task": "openai/gsm8k",
  "source_sample_id": "1234",
  "context_with_chat_template": "... text before rejected token ...",
  "rejected_decoded": " Wait",
  "multi_chosen_decoded": [" Therefore", " The"],
  "repetition": {
    "start_char": 1024,
    "repeat_start_char": 1526,
    "rejected_token_index": 342
  }
}
```

`iter_0_generations.jsonl` stores the full generated completion and detector
metadata. `iter_0_ftpo_pairs.jsonl` stores the preference rows used for
training.

## AMD / ROCm

The Quick Start above targets NVIDIA/CUDA, whose pinned `uv.lock` is CUDA-only and can't drive AMD
GPUs. On AMD you build a **separate ROCm venv** from vLLM's prebuilt ROCm wheels and run with
[`configs/default_amd.yaml`](configs/default_amd.yaml) — whose key setting is
`attention_backend: TRITON_ATTN`, since vLLM's default `ROCM_ATTN` kernel **memory-faults** on LFM2.
Run it with `uv run --no-sync antidoom …` — the `--no-sync` is required: a bare `uv run` (as in the
Quick Start) would re-sync the venv to the CUDA-only `uv.lock` and clobber the ROCm install.

Validated on **Instinct MI325 (gfx942)**. vLLM's ROCm wheels support MI200/MI300/MI350 and Radeon
RX 7900/9000; see vLLM's
[GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html) for the
support matrix and ROCm requirements (ROCm 6.3+, or 7.0+ for MI350).

### Environment setup (ROCm venv)

Requires ROCm 6.3+ on the host (`rocminfo` lists your GPU) and Python 3.12 (already pinned in
`.python-version`). From the repo root, create the venv and install antidoom + ROCm torch/vLLM in a
single command — `--extra-index-url` gives the ROCm wheel index priority, so `torch`/`vllm` resolve to
their ROCm builds instead of the CUDA PyPI wheels
([ref](https://docs.vllm.ai/en/stable/getting_started/quickstart/#installation)):

```bash
uv venv && uv pip install -e . --extra-index-url https://wheels.vllm.ai/rocm/
```

Verify:

```bash
uv run --no-sync python -c "import torch, vllm; print(torch.__version__, torch.version.hip, vllm.__version__)"
# torch should print a '+rocm...' build and a non-None torch.version.hip
```

### ROCm-specific settings (already in `configs/default_amd.yaml`)

[`configs/default_amd.yaml`](configs/default_amd.yaml) is `default.yaml` with exactly these overrides:

- **`generation.vllm_kwargs`**
  ```yaml
  vllm_kwargs:
    enforce_eager: true            # eager is stable and fast on ROCm
    attention_backend: TRITON_ATTN # the fix: ROCM_ATTN memory-faults on LFM2
  ```
  The `attention_backend` value is forwarded to the vLLM engine. The `VLLM_ATTENTION_BACKEND`
  env var is a **no-op** in recent vLLM — it must be the engine arg, which `vllm_kwargs` forwards.
- **`train.optim: adamw_torch`** — the NVIDIA default `paged_adamw_32bit` needs `bitsandbytes`
  (painful on ROCm). `adamw_torch` is the same 32-bit AdamW math; its "paged" CPU-offload is
  pointless on a 256 GB MI325.

The multi-GPU generate path additionally sets, per worker (in `generate.py`, no config needed):
`HIP_VISIBLE_DEVICES` (ROCm doesn't reliably pin GPUs via `CUDA_VISIBLE_DEVICES` alone), per-worker
node-local `VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` (shared JIT caches race →
"Memory access fault"), and a probed free port for `VLLM_PORT` (avoids `EADDRINUSE`). These are
no-ops on NVIDIA.

### Run on AMD

Same commands as the Quick Start, but with `configs/default_amd.yaml` and `uv run --no-sync` (the
`--no-sync` stops `uv run` from re-syncing the venv back to the CUDA lock):

```bash
uv run --no-sync antidoom -c configs/default_amd.yaml -r runs/antidoom1 \
  --temp 0.01 \
  --model-name <your-lfm2-checkpoint-or-hf-id>
```

Generation uses one single-GPU vLLM engine per visible GPU; set `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` to choose GPUs. Training is single-GPU LoRA + merge.

Prefer a container? vLLM also ships official ROCm images — `vllm/vllm-openai-rocm:latest` (stable) or
`:nightly` — listed on the
[GPU installation guide](https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html). Run
antidoom inside one with the **same `configs/default_amd.yaml`**; the `TRITON_ATTN` setting is still
required.

### AMD troubleshooting

| symptom | cause | fix |
|---|---|---|
| `Memory access fault by GPU` mid-generation | vLLM `ROCM_ATTN` kernel bug on LFM2 | use `configs/default_amd.yaml` (`attention_backend: TRITON_ATTN`) — mandatory, affects all cudagraph modes |
| `ImportError: You need to install bitsandbytes` during train | `optim: paged_adamw_32bit` | use `configs/default_amd.yaml` (`optim: adamw_torch`) |
| `RuntimeError: Device string must not be empty` at vLLM init | `amdsmi` missing (ROCm platform not detected) | `cp -r "${ROCM_PATH:-/opt/rocm}/share/amd_smi" /tmp/amd_smi_build && uv pip install /tmp/amd_smi_build` (copy first — the ROCm-shipped tree is root-owned) |

## Citation

```yaml
@article{liquidAI2026Antidoom,
    author = {Liquid AI},
    title = {Reducing Doom Loops with Final Token Preference Optimization},
    journal = {Liquid AI Blog},
    year = {2026},
    note = {www.liquid.ai/blog/antidoom}
}
```
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


def _coerce_paths(obj: Any) -> None:
    for f in fields(obj):
        if f.name.endswith(("_jsonl", "_dir")):
            value = getattr(obj, f.name)
            if isinstance(value, str) and value:
                setattr(obj, f.name, Path(value))


@dataclass
class GenerationConfig:
    api_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model_name: str = ""
    tokenizer_name: str = ""
    output_jsonl: Path = Path("runs/ftpo_pairs.jsonl")
    generations_jsonl: Path = Path("runs/generations.jsonl")
    hf_dataset: str | None = "LiquidAI/antidoom-mix-v1.0"
    hf_split: str = "train"
    hf_source_field: str | None = "source"
    hf_source_values: list[str] = field(default_factory=list)
    input_jsonl: Path | None = None
    prompt_field: str = "conversations"
    max_prompts: int | None = None
    target_pairs: int | None = 10_000
    target_pair_batch_size: int = 256
    threads: int = 32
    generation_backend: str = "vllm"
    request_batch_size: int = 99999
    save_token_trace: bool = False
    token_trace_statuses: list[str] = field(default_factory=lambda: ["loop_detected"])
    vllm_dtype: str = "bfloat16"
    vllm_tensor_parallel_size: int | None = None
    vllm_max_model_len: int | None = 12000
    vllm_gpu_memory_utilization: float = 0.85
    vllm_enable_prefix_caching: bool = False
    vllm_trust_remote_code: bool = True
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)
    prompt_template: str = (
        "{prompt}\n\n"
        'Think through the problem step by step, then respond with your final answer as "Answer: <your answer>".'
    )
    system_prompt: str = ""
    max_prompt_tokens: int | None = None
    max_new_tokens: int = 1000
    temperature: float = 0.1
    temperatures: list[float] | None = None
    top_p: float | None = 1.0
    top_k: int | None = 50
    min_p: float | None = 0.01
    top_logprobs: int = 20
    timeout: int = 480
    stop: list[str] = field(default_factory=list)
    repetition: dict[str, Any] = field(default_factory=dict)
    chosen_max_tokens: int = 20
    chosen_top_k: int = 50
    chosen_min_p: float | None = None
    chosen_min_decoded_chars: int = 1
    chosen_require_alnum: bool = False
    chosen_skip_raw_token_substrings: list[str] = field(default_factory=list)
    skip_decoded_shorter_than: int | None = None
    skip_eval_key_substrings: list[str] = field(
        default_factory=lambda: ["bfcl", "tau", "browse", "search"]
    )

    def __post_init__(self) -> None:
        _coerce_paths(self)


@dataclass
class TrainConfig:
    dataset_jsonl: Path | None = None
    model_name: str = ""
    output_dir: Path = Path("runs/ftpo_lora")
    merged_output_dir: Path = Path("runs/ftpo_merged")
    auto_merged_output_name: bool = True
    merge_lora: bool = True
    max_seq_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_epochs: float = 1.0
    learning_rate: float = 1e-6
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 3407
    max_train_examples: int | None = None
    rejected_regularisation_strength: float = 0.8
    chosen_regularisation_strength: float = 0.0
    source_balance_mode: str = "off"
    min_chosen_tokens: int = 1
    filter_rejected_stop_words: bool = True
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_ensure_weight_tying: bool = False
    freeze_early_layers: bool = True
    n_layers_unfrozen: int = 10
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ]
    )
    load_in_4bit: bool = False
    bf16: bool = True
    lambda_mse_target: float = 0.05
    tau_mse_target: float = 0.5
    lambda_mse: float = 0.4
    clip_epsilon_logits: float = 2.0
    early_stopping_chosen_win: float | None = 0.85
    # HF Trainer optimizer; ROCm/AMD should set "adamw_torch" to avoid the bitsandbytes dependency.
    optim: str = "paged_adamw_32bit"

    def __post_init__(self) -> None:
        _coerce_paths(self)


@dataclass
class AppConfig:
    model_name: str | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    unknown = set(data) - {"model_name", "generation", "train"}
    if unknown:
        raise ValueError(f"Unknown top-level config key(s): {sorted(unknown)}")
    for name in ("generation", "train"):
        section = data.get(name)
        if section is not None and not isinstance(section, dict):
            raise ValueError(f"config section {name!r} must be a mapping")
    model_name = data.get("model_name")
    return AppConfig(
        model_name=str(model_name) if model_name is not None else None,
        generation=GenerationConfig(**(data.get("generation") or {})),
        train=TrainConfig(**(data.get("train") or {})),
    )

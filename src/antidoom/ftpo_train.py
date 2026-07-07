from __future__ import annotations

import inspect
import logging
import math
import getpass
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import cached_file
from trl import DPOConfig, DPOTrainer

from antidoom.config import TrainConfig
from antidoom.ftpo_data import (
    FTPODataset,
    has_trimmed_source_prompt,
    read_jsonl,
    row_source_label,
    sample_rows,
)
from antidoom.reference_ftpo_trainer import FTPOTrainer, ThresholdStop

logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    text = f"{value:.8g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        if sign == "+":
            sign = ""
        text = f"{mantissa}e{sign}{exponent}"
    return text


def _format_epochs(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else _format_float(value)


def _format_counter(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counter.most_common(limit))


def _sanitize_name_part(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._+-]+", "-", value.strip())
    value = value.strip("-_")
    return value or "unknown"


def _model_basename(model_name: str) -> str:
    return model_name.rstrip("/").rsplit("/", 1)[-1]


def _parse_parent_model_name(model_name: str) -> dict[str, str]:
    name = _model_basename(model_name)
    size_match = re.search(r"(\d+B)", name, flags=re.IGNORECASE)
    parent_match = re.search(r"_(\d+)_HF(?:_|$)", name)
    if parent_match is None:
        numbers = re.findall(r"\d+", name)
        parent_id = numbers[-1] if numbers else "unknown"
    else:
        parent_id = parent_match.group(1)
    family_match = re.search(r"(liquidthink\d*)", name, flags=re.IGNORECASE)
    return {
        "size": size_match.group(1) if size_match else "model",
        "parent_id": parent_id,
        "family": family_match.group(1) if family_match else "",
    }


def build_merged_model_name(cfg: TrainConfig, *, train_examples: int) -> str:
    parent = _parse_parent_model_name(cfg.model_name)
    user = _sanitize_name_part(getpass.getuser())
    job_id = _sanitize_name_part(os.environ.get("SLURM_JOB_ID") or "local")
    parts = [
        f"{user}_antidoom{parent['size']}",
        f"from{parent['parent_id']}",
    ]
    if parent["family"]:
        parts.append(parent["family"])
    parts.extend(
        [
            f"{train_examples}-samples",
            f"lr{_format_float(cfg.learning_rate)}",
            f"ep{_format_epochs(cfg.num_epochs)}",
            job_id,
            "HF",
        ]
    )
    return _sanitize_name_part("_".join(parts))


def resolve_merged_output_dir(cfg: TrainConfig, *, train_examples: int) -> Path:
    if not cfg.auto_merged_output_name:
        return cfg.merged_output_dir
    return cfg.merged_output_dir.parent / build_merged_model_name(
        cfg,
        train_examples=train_examples,
    )


def filter_trimmed_source_prompt_rows(rows: list[dict]) -> tuple[list[dict], int]:
    kept = [row for row in rows if not has_trimmed_source_prompt(row)]
    return kept, len(rows) - len(kept)


def _make_dpo_config(cfg: TrainConfig) -> DPOConfig:
    kwargs = {
        "per_device_train_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "num_train_epochs": cfg.num_epochs,
        "learning_rate": cfg.learning_rate,
        "logging_steps": 5,
        "optim": cfg.optim,
        "seed": cfg.seed,
        "output_dir": str(cfg.output_dir),
        "max_length": cfg.max_seq_length,
        "max_prompt_length": cfg.max_seq_length // 2,
        "beta": 0.1,
        "weight_decay": cfg.weight_decay,
        "report_to": "tensorboard",
        "lr_scheduler_type": "linear",
        "bf16": cfg.bf16,
        "fp16": False,
        "remove_unused_columns": False,
        "disable_tqdm": False,
        "max_grad_norm": 2.5,
    }
    supported_params = inspect.signature(DPOConfig).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped = sorted(set(kwargs) - set(supported_kwargs))
    if dropped:
        logger.info("DPOConfig does not support these arguments; dropping: %s", ", ".join(dropped))
    return DPOConfig(**supported_kwargs)


def _make_lora_config(cfg: TrainConfig) -> LoraConfig:
    kwargs = {
        "r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "bias": "none",
        "target_modules": cfg.target_modules,
    }
    if cfg.lora_ensure_weight_tying:
        kwargs["ensure_weight_tying"] = True
    supported_params = inspect.signature(LoraConfig).parameters
    supported_kwargs = {key: value for key, value in kwargs.items() if key in supported_params}
    dropped = sorted(set(kwargs) - set(supported_kwargs))
    if dropped:
        logger.info("LoraConfig does not support these arguments; dropping: %s", ", ".join(dropped))
    return LoraConfig(**supported_kwargs)


def _discover_transformer_blocks(base_model) -> tuple[str, list | torch.nn.ModuleList]:
    candidate_paths = (
        "model.transformer.layers",
        "model.layers",
        "model.decoder.layers",
        "layers",
        "transformer.layers",
        "transformer.h",
        "backbone.layers",
    )
    for path in candidate_paths:
        obj = base_model
        found = True
        for part in path.split("."):
            if not hasattr(obj, part):
                found = False
                break
            obj = getattr(obj, part)
        if found and isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
            return path, obj

    seen: list[tuple[str, int, str]] = []
    best_name = ""
    best_mod = None
    best_score = (-1, -1)
    for name, mod in base_model.named_modules():
        if not isinstance(mod, torch.nn.ModuleList) or len(mod) < 2:
            continue
        first = mod[0]
        seen.append((name, len(mod), type(first).__name__))
        looks_like_block = any(
            hasattr(first, attr)
            for attr in ("self_attn", "attention", "self_attention", "mlp", "ffn", "mixer")
        ) or any(word in type(first).__name__.lower() for word in ("layer", "block"))
        score = (len(mod), 1 if looks_like_block else 0)
        if score > best_score:
            best_name = name
            best_mod = mod
            best_score = score

    if best_mod is not None:
        return best_name, best_mod

    details = ", ".join(
        f"{name}(len={length}, first={child_type})"
        for name, length, child_type in sorted(seen, key=lambda item: -item[1])[:15]
    )
    raise RuntimeError(f"Could not locate transformer layers list. Scanned ModuleLists: {details}")


def _compile_layer_regex(layers_path: str) -> re.Pattern[str]:
    return re.compile(rf"(?:^|\.){re.escape(layers_path)}\.(\d+)\.")


def freeze_lora_to_last_k_layers(
    peft_model,
    *,
    k: int,
    target_modules: tuple[str, ...],
) -> None:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    layers_path, blocks = _discover_transformer_blocks(base)
    num_layers = len(blocks)
    cut = max(0, num_layers - int(k))
    layer_regex = _compile_layer_regex(layers_path)

    def matches_target(name: str) -> bool:
        return any(token in name for token in target_modules)

    frozen = 0
    trainable = 0
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            continue
        match = layer_regex.search(name)
        if match is not None:
            layer_idx = int(match.group(1))
            if layer_idx < cut:
                param.requires_grad_(False)
                frozen += param.numel()
            else:
                param.requires_grad_(True)
                trainable += param.numel()
        elif matches_target(name):
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()

    offenders = []
    for name, param in peft_model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        match = layer_regex.search(name)
        if match is not None and int(match.group(1)) < cut:
            offenders.append(name)
    if offenders:
        raise AssertionError(
            "LoRA params outside the last trainable layers remained trainable: "
            + ", ".join(offenders[:20])
        )

    logger.info(
        "LoRA layer freeze: path=%s, total_layers=%d, keeping_last=%d, frozen_params=%d, trainable_params=%d",
        layers_path,
        num_layers,
        int(k),
        frozen,
        trainable,
    )


def log_trainable_lora_by_layer(peft_model) -> None:
    base = peft_model.get_base_model() if hasattr(peft_model, "get_base_model") else peft_model
    try:
        layers_path, _ = _discover_transformer_blocks(base)
        layer_regex = _compile_layer_regex(layers_path)
    except Exception as exc:
        logger.warning("could not summarize trainable LoRA by layer: %s", exc)
        return

    counts: Counter[int] = Counter()
    lm_head_trainable = False
    total = 0
    for name, param in peft_model.named_parameters():
        if "lora_" not in name or not param.requires_grad:
            continue
        total += param.numel()
        match = layer_regex.search(name)
        if match is not None:
            counts[int(match.group(1))] += param.numel()
        if "lm_head" in name:
            lm_head_trainable = True
    layer_summary = ", ".join(f"{layer}={count}" for layer, count in sorted(counts.items()))
    logger.info("trainable LoRA params by layer: %s", layer_summary or "none")
    logger.info("trainable lm_head LoRA: %s; total trainable LoRA params: %d", lm_head_trainable, total)


def _adapter_key_with_default_name(key: str) -> str:
    replacements = {
        ".lora_A.weight": ".lora_A.default.weight",
        ".lora_B.weight": ".lora_B.default.weight",
        ".lora_embedding_A.weight": ".lora_embedding_A.default",
        ".lora_embedding_B.weight": ".lora_embedding_B.default",
        ".modules_to_save.weight": ".modules_to_save.default.weight",
        ".modules_to_save.bias": ".modules_to_save.default.bias",
    }
    for old, new in replacements.items():
        key = key.replace(old, new)
    return key


def _load_adapter_state_direct(model, adapter_dir: Path) -> None:
    from safetensors.torch import load_file

    adapter_path = adapter_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        state = load_file(adapter_path)
    else:
        adapter_path = adapter_dir / "adapter_model.bin"
        state = torch.load(adapter_path, map_location="cpu")

    state = {_adapter_key_with_default_name(key): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [key for key in missing if "lora_" in key or "modules_to_save" in key]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Adapter state did not load cleanly. "
            f"bad_missing={bad_missing[:40]}, unexpected={unexpected[:40]}"
        )


def _torch_dtype_from_name(name: str):
    dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in dtypes:
        raise ValueError(f"unsupported dtype {name!r}; expected one of {sorted(dtypes)}")
    return dtypes[name]


def merge_lora_adapter(
    *,
    model_name: str,
    adapter_dir: Path,
    merged_output_dir: Path,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    max_shard_size: str = "5GB",
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_torch_dtype_from_name(dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    lora_cfg = LoraConfig.from_pretrained(adapter_dir)
    model = get_peft_model(base, lora_cfg)
    _load_adapter_state_direct(model, adapter_dir)

    logger.info("merging LoRA adapter from %s into %s", adapter_dir, model_name)
    merged = model.merge_and_unload()
    merged.eval()
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        merged_output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(merged_output_dir)
    _restore_parent_config_json(model_name, merged_output_dir)
    logger.info("saved merged model to %s", merged_output_dir)


def _restore_parent_config_json(model_name: str, output_dir: Path) -> None:
    """Keep raw parent config metadata compatible with downstream loaders.

    Some model families round-trip through `save_pretrained` with a newer
    Transformers config schema than vLLM/eval infra expects. LoRA merging does
    not change architecture, so the raw parent config is the safest metadata to
    ship alongside the merged weights.
    """
    source: Path | None = None
    model_path = Path(model_name)
    if model_path.is_dir() and (model_path / "config.json").exists():
        source = model_path / "config.json"
    else:
        try:
            cached = cached_file(
                model_name,
                "config.json",
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        except Exception as exc:
            logger.warning("could not locate parent config.json for %s: %s", model_name, exc)
            cached = None
        if cached:
            source = Path(cached)

    if source is None or not source.exists():
        logger.warning("parent config.json not found; leaving merged config as saved")
        return

    destination = output_dir / "config.json"
    shutil.copyfile(source, destination)
    logger.info("restored parent config.json from %s", source)


def _save_merged_model(trainer: FTPOTrainer, tokenizer, cfg: TrainConfig) -> None:
    if not cfg.merge_lora:
        logger.info("skipping LoRA merge because merge_lora=false")
        return
    if cfg.load_in_4bit:
        raise ValueError("merge_lora=true is not supported with load_in_4bit=true")
    accelerator = getattr(trainer, "accelerator", None)
    model_to_merge = (
        accelerator.unwrap_model(trainer.model) if accelerator is not None else trainer.model
    )
    if not hasattr(model_to_merge, "merge_and_unload"):
        raise AttributeError("trained model does not expose merge_and_unload(); cannot save merged model")

    logger.info("merging LoRA adapter into base model")
    merged_model = model_to_merge.merge_and_unload()
    merged_model.eval()
    cfg.merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(cfg.merged_output_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.merged_output_dir)
    _restore_parent_config_json(cfg.model_name, cfg.merged_output_dir)
    logger.info("saved merged model to %s", cfg.merged_output_dir)


def run_train(cfg: TrainConfig) -> None:
    if cfg.merge_lora and cfg.load_in_4bit:
        raise ValueError("merge_lora=true is not supported with load_in_4bit=true")
    if cfg.dataset_jsonl is None:
        raise ValueError("train.dataset_jsonl is required")
    if not cfg.model_name:
        raise ValueError("model_name is required for training")

    logger.info(
        "training paths: dataset_jsonl=%s, output_dir=%s, merged_output_dir=%s",
        cfg.dataset_jsonl,
        cfg.output_dir,
        cfg.merged_output_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(cfg.dataset_jsonl)
    logger.info("loaded %d FTPO rows from %s", len(rows), cfg.dataset_jsonl)
    logger.info(
        "raw rows by source: %s",
        _format_counter(Counter(row_source_label(row) for row in rows)),
    )
    logger.info(
        "raw rows by rejected token: %s",
        _format_counter(Counter(str(row.get("rejected_decoded")) for row in rows)),
    )
    logger.info(
        "raw rows by chosen candidate count: %s",
        _format_counter(Counter(str(len(row.get("multi_chosen_decoded") or [])) for row in rows)),
    )

    rows_without_trimmed_prompts, trimmed_prompt_rows = filter_trimmed_source_prompt_rows(rows)
    if trimmed_prompt_rows:
        logger.info(
            "%d/%d rows pass trimmed source_prompt filter; discarded %d rows",
            len(rows_without_trimmed_prompts),
            len(rows),
            trimmed_prompt_rows,
        )

    min_chosen_rows = [
        row
        for row in rows_without_trimmed_prompts
        if len(row.get("multi_chosen_decoded") or []) >= cfg.min_chosen_tokens
    ]
    logger.info(
        "%d/%d rows pass min_chosen_tokens=%d before tokenization",
        len(min_chosen_rows),
        len(rows_without_trimmed_prompts),
        cfg.min_chosen_tokens,
    )

    sampled_rows = sample_rows(
        rows_without_trimmed_prompts,
        max_train_examples=cfg.max_train_examples,
        rejected_strength=cfg.rejected_regularisation_strength,
        min_chosen_tokens=cfg.min_chosen_tokens,
        seed=cfg.seed,
        source_balance_mode=cfg.source_balance_mode,
    )
    logger.info(
        "sampled %d/%d rows after pre-tokenization filters "
        "(max_train_examples=%s, min_chosen_tokens=%d, source_balance_mode=%s)",
        len(sampled_rows),
        len(rows_without_trimmed_prompts),
        cfg.max_train_examples,
        cfg.min_chosen_tokens,
        cfg.source_balance_mode,
    )
    logger.info(
        "sampled rows by source: %s",
        _format_counter(Counter(row_source_label(row) for row in sampled_rows)),
    )

    train_dataset = FTPODataset(
        sampled_rows,
        tokenizer,
        max_seq_length=cfg.max_seq_length,
        filter_rejected_stop_words=cfg.filter_rejected_stop_words,
        chosen_regularisation_strength=cfg.chosen_regularisation_strength,
    )
    logger.info(
        "tokenized FTPO dataset kept %d/%d sampled rows; filter counts: %s",
        len(train_dataset),
        len(sampled_rows),
        _format_counter(train_dataset.filter_counts, limit=12),
    )
    effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps
    estimated_steps = math.ceil(len(train_dataset) * cfg.num_epochs / effective_batch)
    logger.info(
        "training schedule: examples=%d, batch_size=%d, gradient_accumulation_steps=%d, "
        "effective_batch=%d, num_epochs=%s, estimated_optimizer_steps=%d",
        len(train_dataset),
        cfg.batch_size,
        cfg.gradient_accumulation_steps,
        effective_batch,
        cfg.num_epochs,
        estimated_steps,
    )
    cfg.merged_output_dir = resolve_merged_output_dir(
        cfg,
        train_examples=len(train_dataset),
    )
    if cfg.merge_lora:
        logger.info("merged model output directory: %s", cfg.merged_output_dir)

    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=quant,
            device_map={"": 0},
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
        )

    lora_cfg = _make_lora_config(cfg)
    model = get_peft_model(model, lora_cfg)
    if cfg.freeze_early_layers:
        freeze_lora_to_last_k_layers(
            model,
            k=cfg.n_layers_unfrozen,
            target_modules=tuple(cfg.target_modules),
        )
        log_trainable_lora_by_layer(model)
    model.train()
    model.print_trainable_parameters()

    init_params = inspect.signature(DPOTrainer.__init__).parameters
    processing_kw = {"processing_class" if "processing_class" in init_params else "tokenizer": tokenizer}

    trainer = FTPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_dataset,
        **processing_kw,
        args=_make_dpo_config(cfg),
    )

    trainer.lambda_mse_target = cfg.lambda_mse_target
    trainer.tau_mse_target = cfg.tau_mse_target
    trainer.lambda_mse = cfg.lambda_mse
    trainer.clip_epsilon_logits = cfg.clip_epsilon_logits
    if cfg.early_stopping_chosen_win is not None:
        trainer.add_callback(
            ThresholdStop("chosen_win", threshold=cfg.early_stopping_chosen_win, higher_is_better=True)
        )

    trainer.train()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    logger.info("saved LoRA adapter to %s", cfg.output_dir)
    _save_merged_model(trainer, tokenizer, cfg)

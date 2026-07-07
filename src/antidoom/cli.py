from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from antidoom.config import AppConfig, load_config

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _add_common_args(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-r",
        "--run-dir",
        type=Path,
        default=None if defaults else argparse.SUPPRESS,
        help="Put all run artifacts under this directory",
    )
    parser.add_argument(
        "--log-level",
        default="INFO" if defaults else argparse.SUPPRESS,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument("--model-name", "--model", dest="model_name", default=None if defaults else argparse.SUPPRESS)


def _add_generate_args(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    parser.add_argument("--tokenizer-name", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--api-base-url", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--api-key", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--output-jsonl", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--generations-jsonl", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--hf-dataset", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--hf-split", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--hf-source-field", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--hf-source-values", type=_parse_str_list, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--input-jsonl", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--prompt-field", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--max-prompts", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--target-pairs", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--target-pair-batch-size", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--generation-backend",
        choices=["vllm", "completions", "chat"],
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--request-batch-size", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--save-token-trace",
        dest="save_token_trace",
        action="store_true",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-save-token-trace",
        dest="save_token_trace",
        action="store_false",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--vllm-max-model-len", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--vllm-dtype", default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--max-prompt-tokens", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--temp", dest="temperatures", type=_parse_float_list, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--temperatures",
        dest="temperatures",
        type=_parse_float_list,
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--skip-eval-key-substrings", type=_parse_str_list, default=None if defaults else argparse.SUPPRESS)


def _add_train_args(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    parser.add_argument("--dataset-jsonl", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--merged-output-dir", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--merge-lora",
        dest="merge_lora",
        action="store_true",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-merge-lora",
        dest="merge_lora",
        action="store_false",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--max-train-examples", type=int, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--filter-rejected-stop-words",
        dest="filter_rejected_stop_words",
        action="store_true",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-filter-rejected-stop-words",
        dest="filter_rejected_stop_words",
        action="store_false",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--chosen-regularisation-strength", type=float, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--freeze-early-layers",
        dest="freeze_early_layers",
        action="store_true",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-freeze-early-layers",
        dest="freeze_early_layers",
        action="store_false",
        default=None if defaults else argparse.SUPPRESS,
    )
    parser.add_argument("--n-layers-unfrozen", type=int, default=None if defaults else argparse.SUPPRESS)


def _add_merge_args(parser: argparse.ArgumentParser, *, defaults: bool) -> None:
    parser.add_argument("--adapter-dir", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument("--merged-output-dir", type=Path, default=None if defaults else argparse.SUPPRESS)
    parser.add_argument(
        "--dtype",
        default="bfloat16" if defaults else argparse.SUPPRESS,
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
    )
    parser.add_argument("--device-map", default="auto" if defaults else argparse.SUPPRESS)
    parser.add_argument("--max-shard-size", default="5GB" if defaults else argparse.SUPPRESS)


def apply_run_dir(cfg: AppConfig, run_dir: Path) -> None:
    cfg.generation.output_jsonl = run_dir / "iter_0_ftpo_pairs.jsonl"
    cfg.generation.generations_jsonl = run_dir / "iter_0_generations.jsonl"
    cfg.train.output_dir = run_dir / "lora"
    cfg.train.merged_output_dir = run_dir / "merged"


def _warn_ignored_generate_args(args: argparse.Namespace, backend: str) -> None:
    if backend == "vllm":
        ignored = [
            ("api_base_url", "--api-base-url"),
            ("api_key", "--api-key"),
            ("threads", "--threads"),
        ]
    elif backend == "completions":
        ignored = [
            ("vllm_tensor_parallel_size", "--vllm-tensor-parallel-size"),
            ("vllm_max_model_len", "--vllm-max-model-len"),
            ("vllm_gpu_memory_utilization", "--vllm-gpu-memory-utilization"),
            ("vllm_dtype", "--vllm-dtype"),
        ]
    else:
        ignored = [
            ("request_batch_size", "--request-batch-size"),
            ("vllm_tensor_parallel_size", "--vllm-tensor-parallel-size"),
            ("vllm_max_model_len", "--vllm-max-model-len"),
            ("vllm_gpu_memory_utilization", "--vllm-gpu-memory-utilization"),
            ("vllm_dtype", "--vllm-dtype"),
        ]
    for attr, flag in ignored:
        if getattr(args, attr, None) is not None:
            logger.warning("%s is ignored when generation_backend=%s", flag, backend)


def _apply_arg_overrides(target: Any, args: argparse.Namespace, keys: tuple[str, ...]) -> None:
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            setattr(target, key, value)


def _apply_model_name(cfg: AppConfig, args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    model_name = getattr(args, "model_name", None) or cfg.model_name
    if not model_name:
        parser.error("--model-name/--model is required when top-level model_name is null or missing in config")
    cfg.model_name = model_name
    cfg.generation.model_name = model_name
    cfg.train.model_name = model_name


def _apply_generate_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    _apply_arg_overrides(
        cfg.generation,
        args,
        (
            "tokenizer_name",
            "api_base_url",
            "api_key",
            "output_jsonl",
            "generations_jsonl",
            "hf_dataset",
            "hf_split",
            "hf_source_field",
            "hf_source_values",
            "input_jsonl",
            "prompt_field",
            "max_prompts",
            "target_pairs",
            "target_pair_batch_size",
            "threads",
            "generation_backend",
            "request_batch_size",
            "save_token_trace",
            "vllm_tensor_parallel_size",
            "vllm_max_model_len",
            "vllm_gpu_memory_utilization",
            "vllm_dtype",
            "max_prompt_tokens",
            "temperature",
            "temperatures",
            "skip_eval_key_substrings",
        ),
    )


def _apply_train_overrides(cfg: AppConfig, args: argparse.Namespace) -> None:
    _apply_arg_overrides(
        cfg.train,
        args,
        (
            "dataset_jsonl",
            "output_dir",
            "merged_output_dir",
            "merge_lora",
            "max_train_examples",
            "filter_rejected_stop_words",
            "chosen_regularisation_strength",
            "freeze_early_layers",
            "n_layers_unfrozen",
        ),
    )
    if getattr(args, "merged_output_dir", None) is not None:
        cfg.train.auto_merged_output_name = False


def _warn_run_dir_overrides(args: argparse.Namespace, command: str, keys: tuple[str, ...]) -> None:
    if getattr(args, "run_dir", None) is None:
        return
    for key in keys:
        if getattr(args, key, None) is not None:
            logger.warning("--%s overrides --run-dir for %s", key.replace("_", "-"), command)


def _run_default_pipeline(cfg: AppConfig) -> dict[str, int] | None:
    from antidoom.ftpo_train import run_train
    from antidoom.generate import run_generation

    explicit_dataset = cfg.train.dataset_jsonl is not None
    counts = None
    if explicit_dataset:
        logger.info("using explicit training dataset %s; skipping generation", cfg.train.dataset_jsonl)
    else:
        counts = run_generation(cfg.generation)
        cfg.train.dataset_jsonl = cfg.generation.output_jsonl
    run_train(cfg.train)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and train FTPO data from repetition loops.")
    _add_common_args(parser, defaults=True)
    _add_generate_args(parser, defaults=True)
    _add_train_args(parser, defaults=True)
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate doomloop FTPO JSONL")
    _add_common_args(gen, defaults=False)
    _add_generate_args(gen, defaults=False)

    train = sub.add_parser("train", help="Train a LoRA adapter with FTPO")
    _add_common_args(train, defaults=False)
    _add_train_args(train, defaults=False)

    merge = sub.add_parser("merge", help="Merge a saved LoRA adapter into a full model")
    _add_common_args(merge, defaults=False)
    _add_merge_args(merge, defaults=True)

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = getattr(args, "config", None)
    if config_path is not None:
        if not config_path.exists():
            parser.error(f"config file not found: {config_path}")
        if not config_path.is_file():
            parser.error(f"config path is not a file: {config_path}")
        cfg = load_config(config_path)
    elif DEFAULT_CONFIG_PATH.is_file():
        cfg = load_config(DEFAULT_CONFIG_PATH)
    else:
        cfg = load_config(None)
    if args.run_dir is not None:
        apply_run_dir(cfg, args.run_dir)

    _apply_model_name(cfg, args, parser)
    command = args.command or "default"

    if command == "generate":
        from antidoom.generate import run_generation

        _warn_run_dir_overrides(args, "generate", ("output_jsonl", "generations_jsonl"))
        _apply_generate_overrides(cfg, args)
        _warn_ignored_generate_args(args, cfg.generation.generation_backend)
        print(run_generation(cfg.generation))
    elif command == "train":
        from antidoom.ftpo_train import run_train

        _warn_run_dir_overrides(args, "train", ("dataset_jsonl", "output_dir", "merged_output_dir"))
        _apply_train_overrides(cfg, args)
        if cfg.train.dataset_jsonl is None:
            cfg.train.dataset_jsonl = cfg.generation.output_jsonl
        run_train(cfg.train)
    elif command == "merge":
        from antidoom.ftpo_train import merge_lora_adapter

        _warn_run_dir_overrides(args, "merge", ("adapter_dir", "merged_output_dir"))
        adapter_dir = getattr(args, "adapter_dir", None) or cfg.train.output_dir
        merged_output_dir = getattr(args, "merged_output_dir", None) or cfg.train.merged_output_dir
        if getattr(args, "merged_output_dir", None) is None and cfg.train.auto_merged_output_name:
            cfg.train.merged_output_dir = merged_output_dir
        merge_lora_adapter(
            model_name=cfg.model_name,
            adapter_dir=adapter_dir,
            merged_output_dir=merged_output_dir,
            dtype=args.dtype,
            device_map=args.device_map,
            max_shard_size=args.max_shard_size,
        )
    else:
        _warn_run_dir_overrides(
            args,
            "default pipeline",
            ("output_jsonl", "generations_jsonl", "dataset_jsonl", "output_dir", "merged_output_dir"),
        )
        _apply_generate_overrides(cfg, args)
        _apply_train_overrides(cfg, args)
        _warn_ignored_generate_args(args, cfg.generation.generation_backend)
        counts = _run_default_pipeline(cfg)
        if counts is not None:
            print(counts)


if __name__ == "__main__":
    main()

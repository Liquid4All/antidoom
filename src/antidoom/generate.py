from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import re
import secrets
import signal
import socket
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

from antidoom.client import CompletionChunk, VllmClient
from antidoom.config import GenerationConfig
from antidoom.prompts import (
    PromptRecord,
    iter_raw_prompts,
    load_prompt_tokenizer,
    render_chat_context,
)
from antidoom.repetition import find_inner_repetition
from antidoom.sampling import select_chosen_tokens
from antidoom.tokens import TokenState, decode_token
from antidoom.trimmed import text_contains_trimmed_prompt_marker

logger = logging.getLogger(__name__)

_RESTART_WORDS = {
    "actually",
    "after",
    "also",
    "alternatively",
    "because",
    "but",
    "finally",
    "first",
    "given",
    "hmm",
    "however",
    "in",
    "let",
    "looking",
    "maybe",
    "now",
    "okay",
    "perhaps",
    "second",
    "since",
    "so",
    "the",
    "then",
    "therefore",
    "this",
    "thus",
    "wait",
}


@dataclass(frozen=True)
class GenerationJob:
    record: PromptRecord
    chat_context: str
    temperature: float

    @property
    def key(self) -> str:
        return _generation_key(self.record.prompt_id, self.temperature)


def _jsonl_append(path: Path, lock: Lock, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _generation_key(prompt_id: int, temperature: float) -> str:
    return f"{prompt_id}:temp={temperature:.12g}"


def _processed_generation_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("generation_key")
            if isinstance(key, str):
                keys.add(key)
                continue
            pid = row.get("prompt_id")
            temp = row.get("temperature")
            if isinstance(pid, int) and isinstance(temp, (int, float)):
                keys.add(_generation_key(pid, float(temp)))
    return keys


def _generation_temperatures(cfg: GenerationConfig) -> list[float]:
    values = cfg.temperatures if cfg.temperatures is not None else [cfg.temperature]
    deduped: list[float] = []
    seen: set[str] = set()
    for value in values:
        temp = float(value)
        key = f"{temp:.12g}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(temp)
    if not deduped:
        raise ValueError("At least one generation temperature is required")
    return deduped


def _prompt_record_contains_trimmed_marker(record: PromptRecord) -> bool:
    if text_contains_trimmed_prompt_marker(record.raw_prompt):
        return True
    return any(text_contains_trimmed_prompt_marker(msg.get("content", "")) for msg in record.messages)


def _prompt_record_matches_skipped_eval_key(record: PromptRecord, cfg: GenerationConfig) -> bool:
    eval_key = record.metadata.get("source_eval_key")
    if not isinstance(eval_key, str):
        return False
    lowered = eval_key.lower()
    return any(substring.lower() in lowered for substring in cfg.skip_eval_key_substrings if substring)


def _source_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if "source_task" in metadata:
        fields["source_task"] = metadata["source_task"]
    if "source_dataset" in metadata:
        fields["source_dataset"] = metadata["source_dataset"]
    if "sample_id" in metadata:
        fields["source_sample_id"] = metadata["sample_id"]
    if "task_metadata" in metadata:
        fields["source_task_metadata"] = metadata["task_metadata"]
    for key in (
        "source_model_ref",
        "source_run_id",
        "source_task_id",
        "source_eval_key",
        "source_eval_harness",
        "source_eval_index",
        "source_sample_idx",
        "source_record_id",
        "source_detectors",
    ):
        if key in metadata:
            fields[key] = metadata[key]
    return fields


def select_rejected_token_index(state: TokenState, repeat_start: int, repeat_end: int) -> int | None:
    reject_char = repeat_start
    while reject_char < repeat_end and state.text[reject_char].isspace():
        reject_char += 1
    if reject_char >= repeat_end:
        return None

    reject_idx = state.char_to_token_index(reject_char)
    while reject_idx is not None and reject_idx < len(state.token_strings):
        if decode_token(state.token_strings[reject_idx]).strip():
            return reject_idx
        reject_idx += 1
    return None


def _token_key_sequence(token_ids: list[int], token_strings: list[str]) -> list[int | str]:
    return token_ids if len(token_ids) == len(token_strings) else token_strings


def _is_boundary_only(text: str) -> bool:
    return bool(text) and not any(ch.isalnum() or ch == "_" for ch in text)


def _skip_boundary_only_tokens(
    token_strings: list[str],
    token_idx: int,
    *,
    max_tokens: int = 4,
    max_chars: int = 12,
) -> int:
    idx = token_idx
    chars = 0
    while idx < len(token_strings) and idx - token_idx < max_tokens:
        text = decode_token(token_strings[idx])
        chars += len(text)
        if chars > max_chars or not _is_boundary_only(text):
            break
        idx += 1
    return idx


def _normalised_word(text: str) -> str:
    return re.sub(r"^[^\w_]+|[^\w_]+$", "", text.strip().lower())


def _lstrip_token_key_sequence(token_strings: list[str]) -> list[str]:
    return [decode_token(raw).lstrip() for raw in token_strings]


def _slide_token_period_left(
    token_keys: list[int | str],
    seed_idx: int,
    period_tokens: int,
) -> int:
    idx = seed_idx
    while idx > 0:
        prev_start = idx - 1
        prev_repeat = prev_start + period_tokens
        if prev_repeat >= len(token_keys):
            break
        if token_keys[prev_start] != token_keys[prev_repeat]:
            break
        idx = prev_start
    return idx


def select_rejected_token_index_token_aware(
    state: TokenState,
    seed_start: int,
    repeat_start: int,
    repeat_end: int,
    token_ids: list[int],
) -> int | None:
    seed_idx = state.char_to_token_index(seed_start)
    repeat_idx = state.char_to_token_index(repeat_start)
    if seed_idx is None or repeat_idx is None or repeat_idx <= seed_idx:
        return select_rejected_token_index(state, repeat_start, repeat_end)

    token_keys = _lstrip_token_key_sequence(state.token_strings)
    exact_token_keys = _token_key_sequence(token_ids, state.token_strings)
    period_tokens = repeat_idx - seed_idx
    seed_idx = _slide_token_period_left(token_keys, seed_idx, period_tokens)
    target_idx = seed_idx + period_tokens
    if target_idx >= len(state.token_strings):
        return None

    target_idx = _skip_boundary_only_tokens(state.token_strings, target_idx)
    if target_idx >= len(state.token_strings):
        return None

    word = _normalised_word(decode_token(state.token_strings[target_idx]))
    if word and any(ch.isalnum() or ch == "_" for ch in word):
        punct_start = target_idx + 1
        punct_end = _skip_boundary_only_tokens(
            state.token_strings,
            punct_start,
            max_tokens=3,
            max_chars=8,
        )
        punct_text = "".join(decode_token(tok) for tok in state.token_strings[punct_start:punct_end])
        next_word = (
            _normalised_word(decode_token(state.token_strings[punct_end]))
            if punct_end < len(state.token_strings)
            else ""
        )
        seed_prev = seed_idx - 1
        repeat_prev = seed_idx + period_tokens - 1
        if (
            punct_end > punct_start
            and any(ch in punct_text for ch in ".!?")
            and seed_prev >= 0
            and repeat_prev >= 0
            and repeat_prev < len(exact_token_keys)
            and exact_token_keys[seed_prev] != exact_token_keys[repeat_prev]
            and next_word in _RESTART_WORDS
        ):
            target_idx = punct_end

    while target_idx < len(state.token_strings):
        if decode_token(state.token_strings[target_idx]).strip():
            return target_idx
        target_idx += 1
    return None


def _repetition_config(cfg: GenerationConfig) -> dict[str, Any]:
    return {
        "min_repeats": 4,
        "max_period": 1024,
        "min_period": 1,
        "min_total_repeated": 60,
        "sample_len": 16,
        "sample_interval": 128,
        **cfg.repetition,
    }


def _failure_row(
    record: PromptRecord,
    chat_context: str,
    temperature: float,
    state: TokenState,
    start: float,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "prompt_id": record.prompt_id,
        "generation_key": _generation_key(record.prompt_id, temperature),
        "temperature": temperature,
        "prompt": chat_context,
        "messages": record.messages,
        **_source_fields(record.metadata),
        "generation": state.text,
        "status": "failed",
        "error": str(exc),
        "duration_sec": time.perf_counter() - start,
        "tokens_generated": len(state.token_strings),
    }


def _maybe_token_trace(
    cfg: GenerationConfig,
    status: str,
    completion: CompletionChunk,
) -> dict[str, Any]:
    if not cfg.save_token_trace or status not in cfg.token_trace_statuses:
        return {}
    return {
        "generated_token_ids": completion.token_ids,
        "generated_token_strings": completion.token_strings,
    }


def _analyse_completion(
    record: PromptRecord,
    chat_context: str,
    cfg: GenerationConfig,
    temperature: float,
    completion: CompletionChunk,
    start: float,
) -> dict[str, Any]:
    prompt_id = record.prompt_id
    raw_prompt = record.raw_prompt
    state = TokenState(chat_context)
    rep_cfg = _repetition_config(cfg)

    try:
        state.append(completion.token_strings, completion.logprobs)

        detected, hit = find_inner_repetition(state.text, **rep_cfg)
        if detected and hit is not None:
            reject_idx = select_rejected_token_index_token_aware(
                state,
                hit.start,
                hit.repeat_start,
                hit.end,
                completion.token_ids,
            )
            if reject_idx is not None:
                rejected_raw = state.token_strings[reject_idx]
                lp_items = state.logprobs.get(reject_idx) or []
                min_decoded_chars = (
                    cfg.skip_decoded_shorter_than
                    if cfg.skip_decoded_shorter_than is not None
                    else cfg.chosen_min_decoded_chars
                )
                chosen_raw, chosen_decoded = select_chosen_tokens(
                    lp_items,
                    rejected_token=rejected_raw,
                    temperature=temperature,
                    min_p=cfg.chosen_min_p if cfg.chosen_min_p is not None else cfg.min_p,
                    top_k=cfg.chosen_top_k,
                    max_tokens=cfg.chosen_max_tokens,
                    min_decoded_chars=min_decoded_chars,
                    require_alnum=cfg.chosen_require_alnum,
                    skip_raw_token_substrings=cfg.chosen_skip_raw_token_substrings,
                )
                if chosen_raw:
                    generation_raw = "".join(decode_token(tok) for tok in state.token_strings[:reject_idx])
                    repetition = {
                        "start_char": hit.start,
                        "repeat_start_char": hit.repeat_start,
                        "end_char": hit.end,
                        "period": hit.period,
                        "repeats": hit.repeats,
                        "snippet": hit.snippet,
                        "rejected_token_index": reject_idx,
                    }
                    ftpo = {
                        "prompt_id": prompt_id,
                        "generation_key": _generation_key(prompt_id, temperature),
                        "temperature": temperature,
                        "prompt_raw": chat_context,
                        "source_prompt": raw_prompt,
                        "messages": record.messages,
                        **_source_fields(record.metadata),
                        "generation_raw": generation_raw,
                        "context_with_chat_template": chat_context + generation_raw,
                        "chosen_decoded": chosen_decoded[0],
                        "rejected_decoded": decode_token(rejected_raw),
                        "chosen_raw": chosen_raw[0],
                        "rejected_raw": rejected_raw,
                        "multi_chosen_decoded": chosen_decoded,
                        "multi_chosen_raw": chosen_raw,
                        "multi_rejected_decoded": [decode_token(rejected_raw)],
                        "multi_rejected_raw": [rejected_raw],
                        "validator": {
                            "class": "inner_repetition",
                            "rule": hit.snippet,
                            "subtype": "doomloop",
                        },
                        "repetition": {
                            **repetition,
                        },
                        "stats": {},
                    }
                    status = "loop_detected"
                    return {
                        "prompt_id": prompt_id,
                        "generation_key": _generation_key(prompt_id, temperature),
                        "temperature": temperature,
                        "prompt": chat_context,
                        "messages": record.messages,
                        **_source_fields(record.metadata),
                        "generation": state.text,
                        "status": status,
                        "repetition": repetition,
                        "duration_sec": time.perf_counter() - start,
                        "tokens_generated": len(state.token_strings),
                        "finish_reason": completion.finish_reason,
                        "ftpo": ftpo,
                        **_maybe_token_trace(cfg, status, completion),
                    }
    except Exception as exc:
        logger.exception("prompt %s failed", prompt_id)
        return _failure_row(record, chat_context, temperature, state, start, exc)

    status = "no_loop"
    return {
        "prompt_id": prompt_id,
        "generation_key": _generation_key(prompt_id, temperature),
        "temperature": temperature,
        "prompt": chat_context,
        "messages": record.messages,
        **_source_fields(record.metadata),
        "generation": state.text,
        "status": status,
        "duration_sec": time.perf_counter() - start,
        "tokens_generated": len(state.token_strings),
        "finish_reason": completion.finish_reason,
        **_maybe_token_trace(cfg, status, completion),
    }


def generate_one(
    job: GenerationJob,
    cfg: GenerationConfig,
    client: VllmClient | None = None,
) -> dict[str, Any]:
    record = job.record
    chat_context = job.chat_context
    if client is None:
        client = VllmClient(
            api_base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            model_name=cfg.model_name,
            pool_size=max(cfg.threads + 8, 16),
        )
    start = time.perf_counter()
    state = TokenState(chat_context)
    try:
        completion = client.chat_complete(
            messages=record.messages,
            max_tokens=cfg.max_new_tokens,
            temperature=job.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            top_logprobs=cfg.top_logprobs,
            timeout=cfg.timeout,
            stop=cfg.stop,
        )
    except Exception as exc:
        logger.exception("prompt %s failed", record.prompt_id)
        return _failure_row(record, chat_context, job.temperature, state, start, exc)
    return _analyse_completion(record, chat_context, cfg, job.temperature, completion, start)


def generate_completion_batch(
    batch: list[GenerationJob],
    cfg: GenerationConfig,
) -> list[dict[str, Any]]:
    starts = [time.perf_counter()] * len(batch)
    client = VllmClient(
        api_base_url=cfg.api_base_url,
        api_key=cfg.api_key,
        model_name=cfg.model_name,
        pool_size=1,
    )
    try:
        completions = client.complete_many(
            prompts=[job.chat_context for job in batch],
            max_tokens=cfg.max_new_tokens,
            temperature=batch[0].temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            top_logprobs=cfg.top_logprobs,
            timeout=cfg.timeout,
            stop=cfg.stop,
        )
    except Exception as exc:
        logger.exception("completion batch failed")
        return [
            _failure_row(job.record, job.chat_context, job.temperature, TokenState(job.chat_context), start, exc)
            for job, start in zip(batch, starts)
        ]

    return [
        _analyse_completion(job.record, job.chat_context, cfg, job.temperature, completion, start)
        for job, completion, start in zip(batch, completions, starts)
    ]


def _logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    logprob = getattr(value, "logprob", None)
    if isinstance(logprob, (int, float)):
        return float(logprob)
    if isinstance(value, dict) and isinstance(value.get("logprob"), (int, float)):
        return float(value["logprob"])
    return 0.0


def _vllm_completion_chunk(output: Any, tokenizer: Any) -> CompletionChunk:
    choice = output.outputs[0]
    token_ids = list(getattr(choice, "token_ids", None) or [])
    token_strings = tokenizer.convert_ids_to_tokens(token_ids) if token_ids else []
    logprobs: dict[int, list[tuple[str, float]]] = {}

    for idx, item in enumerate(getattr(choice, "logprobs", None) or []):
        if not isinstance(item, dict):
            continue
        alts: list[tuple[str, float]] = []
        for token_id, value in item.items():
            try:
                alt_token_id = int(token_id)
            except (TypeError, ValueError):
                continue
            alts.append((tokenizer.convert_ids_to_tokens(alt_token_id), _logprob_value(value)))
        if alts:
            logprobs[idx] = alts

    return CompletionChunk(
        text=getattr(choice, "text", "") or "",
        token_ids=token_ids,
        token_strings=token_strings,
        logprobs=logprobs,
        finish_reason=getattr(choice, "finish_reason", None),
    )


def generate_vllm_batch(
    batch: list[GenerationJob],
    cfg: GenerationConfig,
    llm: Any,
    sampling_params: Any,
    tokenizer: Any,
    *,
    use_tqdm: bool = True,
) -> list[dict[str, Any]]:
    starts = [time.perf_counter()] * len(batch)
    try:
        outputs = llm.generate(
            [job.chat_context for job in batch],
            sampling_params,
            use_tqdm=use_tqdm,
        )
    except Exception as exc:
        logger.exception("vLLM generation failed")
        return [
            _failure_row(job.record, job.chat_context, job.temperature, TokenState(job.chat_context), start, exc)
            for job, start in zip(batch, starts)
        ]

    return [
        _analyse_completion(
            job.record,
            job.chat_context,
            cfg,
            job.temperature,
            _vllm_completion_chunk(output, tokenizer),
            start,
        )
        for job, output, start in zip(batch, outputs, starts)
    ]


def _build_vllm(cfg: GenerationConfig) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "generation_backend='vllm' requires vllm to be installed in this environment"
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": cfg.model_name,
        "dtype": cfg.vllm_dtype,
        "gpu_memory_utilization": cfg.vllm_gpu_memory_utilization,
        "enable_prefix_caching": cfg.vllm_enable_prefix_caching,
        "trust_remote_code": cfg.vllm_trust_remote_code,
        **cfg.vllm_kwargs,
    }
    if cfg.tokenizer_name:
        llm_kwargs["tokenizer"] = cfg.tokenizer_name
    if cfg.vllm_tensor_parallel_size is not None:
        llm_kwargs["tensor_parallel_size"] = cfg.vllm_tensor_parallel_size
    if cfg.vllm_max_model_len is not None:
        llm_kwargs["max_model_len"] = cfg.vllm_max_model_len

    return LLM(**llm_kwargs)


def _available_cuda_gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        visible = visible.strip()
        if not visible or visible == "-1":
            return []
        return [item.strip() for item in visible.split(",") if item.strip()]

    try:
        import torch
    except ImportError:
        return []

    try:
        return [str(idx) for idx in range(torch.cuda.device_count())]
    except Exception:
        logger.exception("failed to detect CUDA devices")
        return []


def _single_gpu_vllm_config(cfg: GenerationConfig) -> GenerationConfig:
    return replace(cfg, vllm_tensor_parallel_size=1)


def _build_sampling_params(cfg: GenerationConfig, temperature: float) -> Any:
    try:
        from vllm import SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "generation_backend='vllm' requires vllm to be installed in this environment"
        ) from exc

    sampling_kwargs: dict[str, Any] = {
        "max_tokens": cfg.max_new_tokens,
        "temperature": temperature,
        "logprobs": cfg.top_logprobs,
        "seed": secrets.randbits(31),
    }
    if cfg.top_p is not None:
        sampling_kwargs["top_p"] = cfg.top_p
    if cfg.top_k is not None:
        sampling_kwargs["top_k"] = cfg.top_k
    if cfg.min_p is not None:
        sampling_kwargs["min_p"] = cfg.min_p
    if cfg.stop:
        sampling_kwargs["stop"] = cfg.stop
    return SamplingParams(**sampling_kwargs)


def _vllm_gpu_worker(
    worker_id: int,
    gpu_id: str,
    cfg: GenerationConfig,
    task_queue: Any,
    result_queue: Any,
) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            logger.exception("failed to start vLLM worker %s in a new process group", worker_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # ROCm: CUDA_VISIBLE_DEVICES alone doesn't pin the GPU; HIP_VISIBLE_DEVICES does. No-op on NVIDIA.
    os.environ["HIP_VISIBLE_DEVICES"] = gpu_id
    # Per-worker node-local caches: engines sharing the JIT caches race -> "Memory access fault".
    _cache_root = f"/tmp/doomloop_cache/{os.environ.get('SLURM_JOB_ID', 'nojob')}/gpu{gpu_id}"
    os.environ["VLLM_CACHE_ROOT"] = f"{_cache_root}/vllm"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{_cache_root}/inductor"
    os.environ["TRITON_CACHE_DIR"] = f"{_cache_root}/triton"
    # Probe a free port so independent engines don't collide on VLLM_PORT (EADDRINUSE).
    _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _probe.bind(("", 0))
        os.environ["VLLM_PORT"] = str(_probe.getsockname()[1])
    finally:
        _probe.close()
    worker_cfg = _single_gpu_vllm_config(cfg)
    sampling_params_by_temperature: dict[float, Any] = {}
    try:
        tokenizer = load_prompt_tokenizer(worker_cfg)
        llm = _build_vllm(worker_cfg)
    except Exception as exc:
        result_queue.put(("worker_failed", worker_id, repr(exc)))
        return

    result_queue.put(("ready", worker_id, gpu_id))
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_id, batch, temperature = task
        try:
            sampling_params = sampling_params_by_temperature.get(temperature)
            if sampling_params is None:
                sampling_params = _build_sampling_params(worker_cfg, temperature)
                sampling_params_by_temperature[temperature] = sampling_params
            rows = generate_vllm_batch(
                batch,
                worker_cfg,
                llm,
                sampling_params,
                tokenizer,
                use_tqdm=False,
            )
            result_queue.put(("result", worker_id, task_id, rows))
        except Exception as exc:
            logger.exception("vLLM worker %s failed task %s", worker_id, task_id)
            result_queue.put(("task_failed", worker_id, task_id, repr(exc)))


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _effective_max_prompt_tokens(cfg: GenerationConfig) -> int | None:
    if cfg.max_prompt_tokens is not None:
        return cfg.max_prompt_tokens
    if cfg.vllm_max_model_len is None:
        return None
    return max(cfg.vllm_max_model_len - cfg.max_new_tokens, 0)


def _token_count(text: str, tokenizer: Any) -> int:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        try:
            return len(encode(text, add_special_tokens=False))
        except TypeError:
            return len(encode(text))

    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = getattr(encoded, "input_ids", None)
    if input_ids is None and isinstance(encoded, dict):
        input_ids = encoded.get("input_ids")
    if input_ids is None:
        raise TypeError(f"{type(tokenizer).__name__} cannot count prompt tokens")
    return len(input_ids)


def _target_reached(cfg: GenerationConfig, counts: dict[str, int]) -> bool:
    if cfg.target_pairs is None:
        return False
    return counts["existing_pairs"] + counts["pairs"] >= cfg.target_pairs


def _iter_generation_jobs(
    cfg: GenerationConfig,
    tokenizer: Any,
    temperature: float,
    done: set[str],
    counts: dict[str, int],
    *,
    count_skips: bool,
) -> Iterator[GenerationJob]:
    max_prompt_tokens = _effective_max_prompt_tokens(cfg)
    for record in iter_raw_prompts(cfg):
        if _prompt_record_contains_trimmed_marker(record):
            if count_skips:
                counts["skipped_trimmed_prompts"] += 1
            continue
        if _prompt_record_matches_skipped_eval_key(record, cfg):
            if count_skips:
                counts["skipped_eval_key_prompts"] += 1
            continue
        key = _generation_key(record.prompt_id, temperature)
        if key in done:
            if count_skips:
                counts["skipped_processed_generations"] += 1
            continue
        chat_context = render_chat_context(record.messages, tokenizer)
        if max_prompt_tokens is not None:
            prompt_tokens = _token_count(chat_context, tokenizer)
            if prompt_tokens > max_prompt_tokens:
                if count_skips:
                    counts["skipped_overlength_prompts"] += 1
                continue
        job = GenerationJob(record=record, chat_context=chat_context, temperature=temperature)
        yield job


def _batched(items: Iterator[GenerationJob], batch_size: int) -> Iterator[list[GenerationJob]]:
    batch: list[GenerationJob] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _effective_request_batch_size(cfg: GenerationConfig) -> int:
    if cfg.target_pairs is None:
        return cfg.request_batch_size
    return min(cfg.request_batch_size, cfg.target_pair_batch_size)


def _pair_progress_text(cfg: GenerationConfig, counts: dict[str, int]) -> str:
    total_pairs = counts["existing_pairs"] + counts["pairs"]
    if cfg.target_pairs is None:
        return f"pairs={total_pairs} (+{counts['pairs']} new)"
    remaining = max(cfg.target_pairs - total_pairs, 0)
    return (
        f"pairs={total_pairs}/{cfg.target_pairs} "
        f"(+{counts['pairs']} new, existing={counts['existing_pairs']}, remaining={remaining})"
    )


def _report_pair_progress(bar: tqdm, cfg: GenerationConfig, counts: dict[str, int]) -> None:
    text = _pair_progress_text(cfg, counts)
    bar.set_postfix_str(text, refresh=True)
    tqdm.write(f"preference samples: {text}")


def _run_vllm_single_worker(
    cfg: GenerationConfig,
    tokenizer: Any,
    temperatures: list[float],
    done: set[str],
    counts: dict[str, int],
    handle_result: Any,
    bar: tqdm,
) -> None:
    llm = _build_vllm(cfg)
    request_batch_size = _effective_request_batch_size(cfg)
    for temperature in temperatures:
        sampling_params = _build_sampling_params(cfg, temperature)
        jobs = _iter_generation_jobs(
            cfg,
            tokenizer,
            temperature,
            done,
            counts,
            count_skips=temperature == temperatures[0],
        )
        for batch in _batched(jobs, request_batch_size):
            if _target_reached(cfg, counts):
                break
            for result in generate_vllm_batch(batch, cfg, llm, sampling_params, tokenizer):
                handle_result(result)
                bar.update(1)
                if _target_reached(cfg, counts):
                    break
            _report_pair_progress(bar, cfg, counts)
            if _target_reached(cfg, counts):
                break
        if _target_reached(cfg, counts):
            break


def _check_vllm_workers_alive(processes: list[mp.Process], in_flight: int) -> None:
    if in_flight <= 0:
        return
    dead = [proc.name for proc in processes if proc.exitcode not in (None, 0)]
    if dead:
        raise RuntimeError(f"vLLM worker process exited unexpectedly: {', '.join(dead)}")


def _wait_for_vllm_workers_ready(
    result_queue: Any,
    processes: list[mp.Process],
    expected: int,
) -> None:
    ready = 0
    while ready < expected:
        try:
            message = result_queue.get(timeout=5)
        except queue.Empty:
            _check_vllm_workers_alive(processes, in_flight=expected - ready)
            continue
        kind = message[0]
        if kind == "ready":
            ready += 1
            logger.info("vLLM worker %s ready on CUDA_VISIBLE_DEVICES=%s", message[1], message[2])
        elif kind == "worker_failed":
            raise RuntimeError(f"vLLM worker {message[1]} failed to start: {message[2]}")
        else:
            raise RuntimeError(f"unexpected vLLM worker message before startup completed: {kind}")


def _signal_vllm_worker(proc: mp.Process, sig: signal.Signals) -> None:
    if proc.pid is None:
        return
    if hasattr(os, "killpg"):
        try:
            os.killpg(proc.pid, sig)
            return
        except ProcessLookupError:
            pass
        except PermissionError:
            logger.warning("no permission to signal vLLM worker process group pid=%s", proc.pid)
        except OSError:
            logger.exception("failed to signal vLLM worker process group pid=%s", proc.pid)

    if sig == signal.SIGKILL and hasattr(proc, "kill"):
        proc.kill()
    else:
        proc.terminate()


def _join_vllm_workers(processes: list[mp.Process], timeout_sec: float) -> None:
    deadline = time.perf_counter() + timeout_sec
    for proc in processes:
        remaining = max(deadline - time.perf_counter(), 0.0)
        proc.join(timeout=remaining)


def _close_mp_queue(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
    join_thread = getattr(value, "join_thread", None)
    if callable(join_thread):
        join_thread()


def _shutdown_vllm_workers(
    processes: list[mp.Process],
    task_queue: Any,
    result_queue: Any,
    *,
    terminate: bool,
) -> None:
    try:
        if terminate:
            for proc in processes:
                if proc.is_alive():
                    _signal_vllm_worker(proc, signal.SIGTERM)
            _join_vllm_workers(processes, timeout_sec=5)
        else:
            for _ in processes:
                task_queue.put(None)
            _join_vllm_workers(processes, timeout_sec=10)

        alive = [proc for proc in processes if proc.is_alive()]
        for proc in alive:
            _signal_vllm_worker(proc, signal.SIGKILL)
        if alive:
            _join_vllm_workers(alive, timeout_sec=2)
    finally:
        _close_mp_queue(task_queue)
        _close_mp_queue(result_queue)


def _run_vllm_gpu_workers(
    cfg: GenerationConfig,
    tokenizer: Any,
    temperatures: list[float],
    done: set[str],
    counts: dict[str, int],
    handle_result: Any,
    bar: tqdm,
    gpu_ids: list[str],
) -> None:
    worker_cfg = _single_gpu_vllm_config(cfg)
    if cfg.vllm_tensor_parallel_size not in (None, 1):
        logger.warning(
            "overriding vllm_tensor_parallel_size=%s to 1 for one-vLLM-instance-per-GPU generation",
            cfg.vllm_tensor_parallel_size,
        )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue(maxsize=max(len(gpu_ids) * 2, 1))
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_vllm_gpu_worker,
            args=(worker_id, gpu_id, worker_cfg, task_queue, result_queue),
            name=f"doomloop-vllm-gpu-{gpu_id}",
        )
        for worker_id, gpu_id in enumerate(gpu_ids)
    ]
    for proc in processes:
        proc.start()

    in_flight = 0
    task_id = 0
    should_terminate = True
    try:
        _wait_for_vllm_workers_ready(result_queue, processes, len(processes))
        request_batch_size = _effective_request_batch_size(cfg)
        logger.info("running vLLM generation with %d GPU workers", len(processes))

        for temperature in temperatures:
            jobs = _iter_generation_jobs(
                cfg,
                tokenizer,
                temperature,
                done,
                counts,
                count_skips=temperature == temperatures[0],
            )
            batches = _batched(jobs, request_batch_size)
            exhausted = False

            while not _target_reached(cfg, counts) and (not exhausted or in_flight):
                while not _target_reached(cfg, counts) and in_flight < len(processes) and not exhausted:
                    try:
                        batch = next(batches)
                    except StopIteration:
                        exhausted = True
                        break
                    task_id += 1
                    task_queue.put((task_id, batch, temperature))
                    in_flight += 1

                if in_flight == 0:
                    break

                try:
                    message = result_queue.get(timeout=5)
                except queue.Empty:
                    _check_vllm_workers_alive(processes, in_flight)
                    continue

                kind = message[0]
                if kind == "result":
                    in_flight -= 1
                    rows = message[3]
                    for result in rows:
                        handle_result(result)
                        bar.update(1)
                        if _target_reached(cfg, counts):
                            break
                    _report_pair_progress(bar, cfg, counts)
                elif kind == "task_failed":
                    in_flight -= 1
                    raise RuntimeError(f"vLLM worker {message[1]} failed task {message[2]}: {message[3]}")
                elif kind == "worker_failed":
                    raise RuntimeError(f"vLLM worker {message[1]} failed: {message[2]}")
                else:
                    raise RuntimeError(f"unexpected vLLM worker message: {kind}")

            if _target_reached(cfg, counts):
                break

        should_terminate = _target_reached(cfg, counts)
    finally:
        _shutdown_vllm_workers(processes, task_queue, result_queue, terminate=should_terminate)


def run_generation(cfg: GenerationConfig) -> dict[str, int]:
    if cfg.generation_backend not in {"vllm", "completions", "chat"}:
        raise ValueError("generation.generation_backend must be 'vllm', 'completions', or 'chat'")
    if cfg.request_batch_size < 1:
        raise ValueError("generation.request_batch_size must be >= 1")
    if cfg.target_pair_batch_size < 1:
        raise ValueError("generation.target_pair_batch_size must be >= 1")
    if cfg.threads < 1:
        raise ValueError("generation.threads must be >= 1")
    if cfg.max_prompt_tokens is not None and cfg.max_prompt_tokens < 0:
        raise ValueError("generation.max_prompt_tokens must be >= 0")

    logger.info(
        "generation paths: backend=%s, input_jsonl=%s, output_jsonl=%s, generations_jsonl=%s",
        cfg.generation_backend,
        cfg.input_jsonl,
        cfg.output_jsonl,
        cfg.generations_jsonl,
    )

    temperatures = _generation_temperatures(cfg)
    done = _processed_generation_keys(cfg.generations_jsonl)
    gen_lock = Lock()
    ftpo_lock = Lock()
    counts = {
        "attempted": 0,
        "pairs": 0,
        "existing_pairs": _jsonl_line_count(cfg.output_jsonl),
        "loop_detected": 0,
        "no_loop": 0,
        "failed": 0,
        "skipped_trimmed_prompts": 0,
        "skipped_eval_key_prompts": 0,
        "skipped_overlength_prompts": 0,
        "skipped_processed_generations": 0,
        "pairs_dropped_after_target": 0,
    }
    if cfg.target_pairs is not None:
        counts["target_pairs"] = cfg.target_pairs
    if _target_reached(cfg, counts):
        logger.info(
            "target pair count already satisfied: existing_pairs=%d target_pairs=%d",
            counts["existing_pairs"],
            cfg.target_pairs,
        )
        return counts

    tokenizer = load_prompt_tokenizer(cfg)
    max_prompt_tokens = _effective_max_prompt_tokens(cfg)
    if max_prompt_tokens is not None:
        logger.info("skipping generation prompts over %d tokens", max_prompt_tokens)

    def handle_result(result: dict[str, Any]) -> None:
        ftpo = result.pop("ftpo", None)
        _jsonl_append(cfg.generations_jsonl, gen_lock, result)
        counts["attempted"] += 1
        counts[result.get("status", "failed")] = counts.get(result.get("status", "failed"), 0) + 1
        if ftpo:
            if _target_reached(cfg, counts):
                counts["pairs_dropped_after_target"] += 1
            else:
                _jsonl_append(cfg.output_jsonl, ftpo_lock, ftpo)
                counts["pairs"] += 1

    with tqdm(desc="doomloop generation", unit="attempt") as bar:
        if cfg.generation_backend == "vllm":
            gpu_ids = _available_cuda_gpu_ids()
            if len(gpu_ids) > 1:
                _run_vllm_gpu_workers(
                    cfg,
                    tokenizer,
                    temperatures,
                    done,
                    counts,
                    handle_result,
                    bar,
                    gpu_ids,
                )
            else:
                if gpu_ids:
                    logger.info("running vLLM generation on one visible GPU")
                else:
                    logger.info("no CUDA GPUs detected before vLLM startup; falling back to single vLLM instance")
                _run_vllm_single_worker(cfg, tokenizer, temperatures, done, counts, handle_result, bar)

        elif cfg.generation_backend == "completions":
            request_batch_size = _effective_request_batch_size(cfg)
            for temperature in temperatures:
                jobs = _iter_generation_jobs(
                    cfg,
                    tokenizer,
                    temperature,
                    done,
                    counts,
                    count_skips=temperature == temperatures[0],
                )
                batch_group: list[list[GenerationJob]] = []
                for batch in _batched(jobs, request_batch_size):
                    if _target_reached(cfg, counts):
                        break
                    batch_group.append(batch)
                    if len(batch_group) >= cfg.threads:
                        _handle_completion_batch_group(batch_group, cfg, handle_result, bar, counts)
                        _report_pair_progress(bar, cfg, counts)
                        batch_group = []
                    if _target_reached(cfg, counts):
                        break
                if batch_group and not _target_reached(cfg, counts):
                    _handle_completion_batch_group(batch_group, cfg, handle_result, bar, counts)
                    _report_pair_progress(bar, cfg, counts)
                if _target_reached(cfg, counts):
                    break

        else:
            for temperature in temperatures:
                jobs = _iter_generation_jobs(
                    cfg,
                    tokenizer,
                    temperature,
                    done,
                    counts,
                    count_skips=temperature == temperatures[0],
                )
                for batch in _batched(jobs, cfg.threads):
                    if _target_reached(cfg, counts):
                        break
                    with ThreadPoolExecutor(max_workers=cfg.threads) as pool:
                        futures = {
                            pool.submit(generate_one, job, cfg): job.key
                            for job in batch
                        }
                        for fut in as_completed(futures):
                            handle_result(fut.result())
                            bar.update(1)
                            if _target_reached(cfg, counts):
                                break
                    _report_pair_progress(bar, cfg, counts)
                    if _target_reached(cfg, counts):
                        break
                if _target_reached(cfg, counts):
                    break

    if counts["skipped_trimmed_prompts"]:
        logger.info("skipped %d prompts containing trimmed prompt marker", counts["skipped_trimmed_prompts"])
    if counts["skipped_eval_key_prompts"]:
        logger.info("skipped %d prompts with excluded eval_key substrings", counts["skipped_eval_key_prompts"])
    if counts["skipped_overlength_prompts"]:
        logger.info("skipped %d prompts over max_prompt_tokens", counts["skipped_overlength_prompts"])
    return counts


def _handle_completion_batch_group(
    batch_group: list[list[GenerationJob]],
    cfg: GenerationConfig,
    handle_result: Any,
    bar: tqdm,
    counts: dict[str, int],
) -> None:
    with ThreadPoolExecutor(max_workers=cfg.threads) as pool:
        futures = {
            pool.submit(generate_completion_batch, batch, cfg): len(batch)
            for batch in batch_group
        }
        for fut in as_completed(futures):
            for result in fut.result():
                handle_result(result)
                bar.update(1)
                if _target_reached(cfg, counts):
                    break
            if _target_reached(cfg, counts):
                break

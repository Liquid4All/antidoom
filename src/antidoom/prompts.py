from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from datasets import interleave_datasets, load_dataset
from transformers import AutoTokenizer

from antidoom.config import GenerationConfig

Message = dict[str, str]


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: int
    raw_prompt: str
    messages: list[Message]
    metadata: dict[str, Any]


def _source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("source_task", "sample_id", "task_metadata"):
        if key in row:
            metadata[key] = row[key]

    source_dataset = row.get("source_dataset")
    source = row.get("source")
    source_name = None
    if isinstance(source_dataset, str) and source_dataset:
        metadata["source_dataset"] = source_dataset
        source_name = source_dataset
    elif isinstance(source, str) and source:
        metadata["source_dataset"] = source
        source_name = source

    if "source_task" not in metadata:
        eval_index = row.get("eval_index")
        eval_harness = row.get("eval_harness")
        eval_key = row.get("eval_key")
        if isinstance(eval_index, str) and eval_index:
            metadata["source_task"] = eval_index
        elif isinstance(eval_harness, str) and isinstance(eval_key, str):
            metadata["source_task"] = f"{eval_harness}/{eval_key}"
        elif isinstance(eval_key, str) and eval_key:
            metadata["source_task"] = eval_key
        elif source_name:
            metadata["source_task"] = source_name

    if "sample_id" not in metadata:
        source_sample_id = row.get("source_sample_id")
        record_id = row.get("record_id")
        sample_idx = row.get("sample_idx")
        if isinstance(source_sample_id, (str, int)):
            metadata["sample_id"] = str(source_sample_id)
        elif isinstance(record_id, (str, int)):
            metadata["sample_id"] = str(record_id)
        elif isinstance(sample_idx, (str, int)):
            metadata["sample_id"] = str(sample_idx)

    scheduler_fields = {
        "model_ref": "source_model_ref",
        "run_id": "source_run_id",
        "task_id": "source_task_id",
        "eval_key": "source_eval_key",
        "eval_harness": "source_eval_harness",
        "eval_index": "source_eval_index",
        "sample_idx": "source_sample_idx",
        "record_id": "source_record_id",
        "detectors": "source_detectors",
    }
    for row_key, meta_key in scheduler_fields.items():
        if row_key in row:
            metadata[meta_key] = row[row_key]
    return metadata


def _normalise_messages(value: Any) -> list[Message] | None:
    if not isinstance(value, list):
        return None
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    return messages or None


def _messages_from_conversations(value: Any) -> list[Message] | None:
    if not isinstance(value, list):
        return None
    messages: list[Message] = []
    role_map = {"human": "user", "gpt": "assistant", "assistant": "assistant", "user": "user"}
    for turn in value:
        if not isinstance(turn, dict):
            return None
        role = role_map.get(str(turn.get("from", "")).lower())
        content = turn.get("value")
        if role is None or not isinstance(content, str):
            return None
        messages.append({"role": role, "content": content})
    return messages or None


def _apply_prompt_config(prompt: str, cfg: GenerationConfig) -> list[Message]:
    content = cfg.prompt_template.format(prompt=prompt)
    messages: list[Message] = []
    if cfg.system_prompt:
        messages.append({"role": "system", "content": cfg.system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def _apply_prompt_config_to_messages(messages: list[Message], cfg: GenerationConfig) -> list[Message]:
    templated = [dict(message) for message in messages]
    for idx in range(len(templated) - 1, -1, -1):
        if templated[idx]["role"] == "user":
            templated[idx]["content"] = cfg.prompt_template.format(prompt=templated[idx]["content"])
            break
    else:
        return _apply_prompt_config(_raw_prompt_from_messages(messages), cfg)

    if cfg.system_prompt:
        templated.insert(0, {"role": "system", "content": cfg.system_prompt})
    return templated


def _raw_prompt_from_messages(messages: list[Message]) -> str:
    return "\n".join(msg["content"] for msg in messages if msg["role"] == "user")


def _messages_from_prompt_value(value: Any, cfg: GenerationConfig) -> tuple[str, list[Message]] | None:
    if isinstance(value, str):
        return value, _apply_prompt_config(value, cfg)

    messages = _normalise_messages(value)
    if messages is None:
        messages = _messages_from_conversations(value)
    if messages is None:
        return None

    raw_prompt = _raw_prompt_from_messages(messages)
    return raw_prompt, _apply_prompt_config_to_messages(messages, cfg)


def _record_from_row(prompt_id: int, row: dict[str, Any], cfg: GenerationConfig) -> PromptRecord:
    if cfg.prompt_field not in row:
        available = ", ".join(sorted(row.keys()))
        raise ValueError(
            f"Configured prompt_field={cfg.prompt_field!r} is not present in row. "
            f"Available fields: {available}"
        )

    parsed = _messages_from_prompt_value(row[cfg.prompt_field], cfg)
    if parsed is None:
        raise ValueError(
            f"Could not parse prompt_field={cfg.prompt_field!r}. Expected a string prompt, "
            "OpenAI-style messages [{'role', 'content'}], or ShareGPT conversations "
            "[{'from', 'value'}]."
        )

    raw_prompt, messages = parsed
    return PromptRecord(prompt_id, raw_prompt, messages, _source_metadata(row))


def _source_value_key(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(value)


def _dedupe_source_values(values: list[Any]) -> list[Any]:
    deduped = []
    seen = set()
    for value in values:
        if value is None or value == "":
            continue
        key = _source_value_key(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _infer_hf_source_values(ds: Any, source_field: str) -> list[Any]:
    values = []
    if hasattr(ds, "unique"):
        try:
            values = list(ds.unique(source_field))
        except (KeyError, TypeError, ValueError):
            values = []

    if not values:
        values = [row.get(source_field) for row in ds if isinstance(row, dict)]

    return _dedupe_source_values(values)


def _load_hf_prompt_dataset(cfg: GenerationConfig):
    ds = load_dataset(cfg.hf_dataset, split=cfg.hf_split)
    if not cfg.hf_source_field:
        return ds

    source_field = cfg.hf_source_field
    source_values = _dedupe_source_values(list(cfg.hf_source_values or []))
    if not source_values:
        source_values = _infer_hf_source_values(ds, source_field)
        cfg.hf_source_values = list(source_values)
    if not source_values:
        return ds

    subsets = [
        ds.filter(lambda row, source=source: row.get(source_field) == source)
        for source in source_values
    ]
    return interleave_datasets(subsets, stopping_strategy="all_exhausted_without_replacement")


def iter_raw_prompts(cfg: GenerationConfig) -> Iterator[PromptRecord]:
    count = 0
    if cfg.input_jsonl is not None:
        with cfg.input_jsonl.open(encoding="utf-8") as fh:
            for line in fh:
                if cfg.max_prompts is not None and count >= cfg.max_prompts:
                    break
                row = json.loads(line)
                record = _record_from_row(count, row, cfg)
                yield record
                count += 1
        return

    if cfg.hf_dataset is None:
        raise ValueError("Set either generation.hf_dataset or generation.input_jsonl")

    ds = _load_hf_prompt_dataset(cfg)
    for item in ds:
        if cfg.max_prompts is not None and count >= cfg.max_prompts:
            break
        record = _record_from_row(count, dict(item), cfg)
        yield record
        count += 1


def render_chat_context(messages: list[Message], tokenizer: AutoTokenizer) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return "\n\n".join(
            message["content"].strip()
            for message in messages
            if message.get("content", "").strip()
        ).rstrip() + "\n"
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_prompt_tokenizer(cfg: GenerationConfig):
    name = cfg.tokenizer_name or cfg.model_name
    if not name:
        raise ValueError("model_name or generation.tokenizer_name is required")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)

from __future__ import annotations

from typing import Any

TRIMMED_PROMPT_PREFIX = "[...trimmed...]"


def text_contains_trimmed_prompt_marker(text: str) -> bool:
    return TRIMMED_PROMPT_PREFIX in text


def row_has_trimmed_source_prompt(row: dict[str, Any]) -> bool:
    source_prompt = row.get("source_prompt")
    return isinstance(source_prompt, str) and source_prompt.startswith(TRIMMED_PROMPT_PREFIX)

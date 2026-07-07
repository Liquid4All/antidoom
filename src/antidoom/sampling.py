from __future__ import annotations

import math

from antidoom.tokens import decode_token


def _normalise(logprob_items: list[tuple[str, float]], temperature: float) -> list[tuple[str, float]]:
    if not logprob_items:
        return []
    temp = max(float(temperature), 1e-6)
    vals = [(tok, math.exp(lp) ** (1.0 / temp)) for tok, lp in logprob_items if lp is not None]
    total = sum(v for _, v in vals)
    if total <= 0:
        return []
    return [(tok, val / total) for tok, val in vals]


def select_chosen_tokens(
    logprob_items: list[tuple[str, float]],
    *,
    rejected_token: str,
    temperature: float,
    min_p: float | None,
    top_k: int | None,
    max_tokens: int,
    min_decoded_chars: int = 1,
    require_alnum: bool = False,
    skip_raw_token_substrings: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    pairs = _normalise(logprob_items, temperature)
    if not pairs:
        return [], []

    rejected_decoded = decode_token(rejected_token).lower()

    def is_rejected(tok: str) -> bool:
        return tok == rejected_token or decode_token(tok).lower() == rejected_decoded

    pairs = [(tok, p) for tok, p in pairs if not is_rejected(tok)]
    if not pairs:
        return [], []

    if min_p is not None:
        floor = min_p * max(p for _, p in pairs)
        pairs = [(tok, p) for tok, p in pairs if p >= floor]
    if top_k is not None and len(pairs) > top_k:
        pairs = sorted(pairs, key=lambda item: item[1], reverse=True)[:top_k]

    pairs.sort(key=lambda item: item[1])
    raw: list[str] = []
    decoded: list[str] = []
    skip_raw_token_substrings = skip_raw_token_substrings or []
    for tok, _ in pairs:
        surf = decode_token(tok)
        if any(substring in tok for substring in skip_raw_token_substrings):
            continue
        if len(surf.strip()) < min_decoded_chars:
            continue
        if require_alnum and not any(c.isalnum() for c in surf):
            continue
        raw.append(tok)
        decoded.append(surf)
        if len(raw) >= max_tokens:
            break
    return raw, decoded

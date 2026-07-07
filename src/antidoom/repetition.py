from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepeatHit:
    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> tuple[bool, RepeatHit | None]:
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return False, None

    pattern = text[start_pos : start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos

    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period

    total = reps * period
    if reps >= min_repeats and total >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return True, RepeatHit(start_pos, end_pos, period, reps, snippet)
    return False, None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = 4,
    max_period: int = 1024,
    min_period: int = 1,
    min_total_repeated: int = 60,
    sample_len: int = 16,
    sample_interval: int = 128,
) -> tuple[bool, RepeatHit | None]:
    if not text or len(text) < min_total_repeated:
        return False, None

    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]

        other_pos = text.find(fingerprint, sample_pos + sample_len)
        if other_pos != -1:
            candidate_period = other_pos - sample_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    sample_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

        other_pos = text.rfind(fingerprint, 0, sample_pos)
        if other_pos != -1:
            candidate_period = sample_pos - other_pos
            if min_period <= candidate_period <= max_period:
                found, hit = _verify_repetition_at(
                    text,
                    other_pos,
                    candidate_period,
                    min_repeats=min_repeats,
                    min_total_repeated=min_total_repeated,
                )
                if found:
                    return True, hit

    return False, None

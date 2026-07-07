from __future__ import annotations

from dataclasses import dataclass, field


def _build_u2b_full() -> dict[int, int]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    u2b = {ord(chr(c)): b for b, c in zip(bs, cs)}
    for b in range(256):
        u2b[0x2500 + b] = b
    return u2b


_U2B = _build_u2b_full()


def fix_mojibake(text: str) -> str:
    buf = bytearray()
    changed = False
    for ch in text:
        cp = ord(ch)
        if cp in _U2B:
            buf.append(_U2B[cp])
            changed = True
        else:
            buf.extend(ch.encode("utf-8"))
    if not changed:
        return text
    try:
        return buf.decode("utf-8")
    except UnicodeDecodeError:
        return text


def decode_token(token: str) -> str:
    if not token:
        return token
    token = fix_mojibake(token)
    return token.replace("Ċ", "\n").replace("Ġ", " ").replace("▁", " ")


@dataclass
class TokenState:
    prompt: str
    token_strings: list[str] = field(default_factory=list)
    decoded_lens: list[int] = field(default_factory=list)
    text: str = ""
    logprobs: dict[int, list[tuple[str, float]]] = field(default_factory=dict)

    def append(self, token_strings: list[str], logprobs: dict[int, list[tuple[str, float]]]) -> None:
        start = len(self.token_strings)
        for raw in token_strings:
            decoded = decode_token(raw)
            self.token_strings.append(raw)
            self.decoded_lens.append(len(decoded))
            self.text += decoded
        for rel_idx, alts in logprobs.items():
            abs_idx = start + rel_idx
            if abs_idx < len(self.token_strings):
                self.logprobs[abs_idx] = alts

    def char_to_token_index(self, char_pos: int) -> int | None:
        if char_pos < 0 or char_pos >= len(self.text):
            return None
        running = 0
        for idx, length in enumerate(self.decoded_lens):
            running += length
            if char_pos < running:
                return idx
        return None

    def context_before(self, token_idx: int) -> str:
        return self.prompt + "".join(decode_token(tok) for tok in self.token_strings[:token_idx])

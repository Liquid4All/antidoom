from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class CompletionChunk:
    text: str
    token_ids: list[int] = field(default_factory=list)
    token_strings: list[str] = field(default_factory=list)
    logprobs: dict[int, list[tuple[str, float]]] = field(default_factory=dict)
    finish_reason: str | None = None


class VllmClient:
    def __init__(self, *, api_base_url: str, api_key: str, model_name: str, pool_size: int = 32) -> None:
        base = api_base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"
        self.chat_endpoint = base + "/chat/completions"
        self.completions_endpoint = base + "/completions"
        self.api_key = api_key
        self.model_name = model_name
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=pool_size,
            pool_maxsize=pool_size,
            max_retries=Retry(
                total=3,
                backoff_factor=0.3,
                status_forcelist=(502, 503, 504),
                allowed_methods=frozenset(["POST"]),
            ),
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def chat_complete(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        min_p: float | None,
        top_logprobs: int,
        timeout: int,
        stop: list[str] | None = None,
    ) -> CompletionChunk:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self.session.post(self.chat_endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise requests.HTTPError(str(data["error"]), response=resp)
        choices = data.get("choices") or []
        if not choices:
            raise requests.HTTPError("Upstream returned no choices", response=resp)
        choice = choices[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        lp_obj = choice.get("logprobs") or {}
        content_logprobs = lp_obj.get("content") or []
        token_strings: list[str] = []
        logprobs: dict[int, list[tuple[str, float]]] = {}
        for idx, item in enumerate(content_logprobs):
            if not isinstance(item, dict):
                continue
            token = item.get("token")
            if isinstance(token, str):
                token_strings.append(token)
            alts = []
            for alt in item.get("top_logprobs") or []:
                if isinstance(alt, dict) and isinstance(alt.get("token"), str):
                    alts.append((alt["token"], float(alt.get("logprob", 0.0))))
            if alts:
                logprobs[idx] = alts
        return CompletionChunk(
            text=text,
            token_strings=token_strings,
            logprobs=logprobs,
            finish_reason=choice.get("finish_reason"),
        )

    def complete_many(
        self,
        *,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float | None,
        top_k: int | None,
        min_p: float | None,
        top_logprobs: int,
        timeout: int,
        stop: list[str] | None = None,
    ) -> list[CompletionChunk]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "logprobs": top_logprobs,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = self.session.post(self.completions_endpoint, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise requests.HTTPError(str(data["error"]), response=resp)
        choices = data.get("choices") or []
        if len(choices) != len(prompts):
            raise requests.HTTPError(
                f"Upstream returned {len(choices)} choices for {len(prompts)} prompts",
                response=resp,
            )

        ordered: list[CompletionChunk | None] = [None] * len(prompts)
        for fallback_idx, choice in enumerate(choices):
            idx = choice.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(prompts):
                idx = fallback_idx
            lp_obj = choice.get("logprobs") or {}
            tokens = lp_obj.get("tokens") or []
            top_logprobs = lp_obj.get("top_logprobs") or []
            token_strings = [tok for tok in tokens if isinstance(tok, str)]
            logprobs: dict[int, list[tuple[str, float]]] = {}
            for tok_idx, alt_map in enumerate(top_logprobs):
                if not isinstance(alt_map, dict):
                    continue
                alts = [
                    (str(token), float(logprob))
                    for token, logprob in alt_map.items()
                    if isinstance(logprob, (int, float))
                ]
                if alts:
                    logprobs[tok_idx] = alts
            ordered[idx] = CompletionChunk(
                text=choice.get("text") or "",
                token_strings=token_strings,
                logprobs=logprobs,
                finish_reason=choice.get("finish_reason"),
            )

        missing = [idx for idx, item in enumerate(ordered) if item is None]
        if missing:
            raise requests.HTTPError(f"Upstream response missing choices for prompt indexes: {missing}", response=resp)
        return [item for item in ordered if item is not None]

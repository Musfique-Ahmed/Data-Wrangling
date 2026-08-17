"""Thin Ollama HTTP client.

We use the raw HTTP API (`/api/chat`) so we get:
  - exact control over the `options` block (temperature, num_predict, …)
  - exact control over `keep_alive` so the model unloads after each call
  - explicit access to `prompt_eval_count` and `eval_count` for
    prompt/completion token accounting
  - access to the `think` field for Qwen3 thinking-mode control

This module deliberately does NOT use any third-party Ollama SDK.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ChatResponse:
    """Result of one `/api/chat` request.

    Token counts come from Ollama itself (exact, not estimated from a
    tokenizer). `thinking` holds the hidden chain-of-thought block for
    Qwen3 models when `think=true`; empty for other models. `total_duration_ns`
    is the model's reported wall time inside Ollama; the Python-side
    latency is captured separately by `inference.run` via `time.perf_counter`.
    See METRICS.md §1–§2 for details.
    """
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_duration_ns: int
    load_duration_ns: int
    eval_count: int
    prompt_eval_count: int
    thinking: str = ""          # chain-of-thought block (Qwen3)
    raw: dict[str, Any] = None


class OllamaError(RuntimeError):
    """Wraps any non-2xx response or transport failure."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        request_timeout_s: int = 120,
        retries: int = 3,
        backoff_initial_s: float = 2.0,
        backoff_max_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = request_timeout_s
        self.retries = max(0, retries)
        self.backoff_initial_s = backoff_initial_s
        self.backoff_max_s = backoff_max_s

    # ------------------------------ health ------------------------------

    def health(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return r.ok
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        r = requests.get(f"{self.base_url}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]

    def has_model(self, tag: str) -> bool:
        return any(name == tag or name.startswith(tag + ":") for name in self.list_models())

    # ------------------------------ chat -------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        keep_alive: str | None = "0s",
        stream: bool = False,
        think: bool | None = None,
    ) -> ChatResponse:
        """One-shot chat completion with retries.

        `think` (bool or None): when True/False, force Qwen3 thinking
        mode on/off. When None, the model uses its default.

        Returns a ChatResponse. Raises OllamaError after retries are exhausted.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }
        if options is not None:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        if think is not None:
            payload["think"] = think

        backoff = self.backoff_initial_s
        last_err: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                t0 = time.perf_counter()
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.request_timeout_s,
                )
                r.raise_for_status()
                data = r.json()
                latency = time.perf_counter() - t0
                msg = data.get("message", {}) or {}
                content = msg.get("content", "") or ""
                thinking = msg.get("thinking", "") or ""
                return ChatResponse(
                    content=content,
                    prompt_tokens=int(data.get("prompt_eval_count", 0) or 0),
                    completion_tokens=int(data.get("eval_count", 0) or 0),
                    total_duration_ns=int(data.get("total_duration", 0) or 0),
                    load_duration_ns=int(data.get("load_duration", 0) or 0),
                    eval_count=int(data.get("eval_count", 0) or 0),
                    prompt_eval_count=int(data.get("prompt_eval_count", 0) or 0),
                    thinking=thinking,
                    raw={**data, "_wall_latency_s": latency},
                )
            except (requests.RequestException, ValueError, json.JSONDecodeError) as e:
                last_err = e
                if attempt >= self.retries:
                    break
                time.sleep(min(backoff, self.backoff_max_s))
                backoff *= 2

        raise OllamaError(f"Ollama chat failed for {model!r} after retries: {last_err!r}")
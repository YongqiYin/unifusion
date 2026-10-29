"""Answer extraction over an assembled context.

The reader is deliberately narrow: it renders a numbered evidence block, asks
for a short answer phrase, and post-processes the raw completion. Reasoning
models prepend a scratchpad, and that scratchpad is where extraction silently
fails when the generation budget is too small, so
:func:`strip_reasoning` is part of the contract rather than an afterthought.
"""
from __future__ import annotations

import re

INSTRUCTION = (
    "Answer the question using only the provided evidence. "
    "Return a short answer phrase only, no explanation."
)


def format_evidence(contexts, max_chars_per_passage: int = 1200) -> str:
    """Render passages as ``[doc1] ...`` blocks truncated to a fixed budget."""
    contexts = list(contexts)
    if not contexts:
        return "(no evidence retrieved)"
    return "\n\n".join(
        f"[doc{i + 1}] {str(text)[:max_chars_per_passage]}"
        for i, text in enumerate(contexts)
    )


def build_prompt(query: str, contexts, max_chars_per_passage: int = 1200) -> str:
    """Single shared extraction prompt for every method being compared."""
    return (
        f"{INSTRUCTION}\n\n"
        f"Question: {query}\n\n"
        f"Evidence:\n{format_evidence(contexts, max_chars_per_passage)}\n\n"
        "Short Answer:"
    )


_REASONING_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_REASONING = re.compile(r"<think\b[^>]*>.*", re.DOTALL | re.IGNORECASE)
_BARE_REASONING = re.compile(r"^\s*(thinking process|reasoning)\s*:.*?(?=\n\s*\n|$)",
                             re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str, max_chars: int = 200) -> str:
    """Drop a reasoning scratchpad and collapse the remaining answer.

    Handles three shapes: a closed ``<think>...</think>`` block, an unclosed
    block (the generation budget ran out before the answer started, which
    yields an empty string), and a bare ``Thinking process:`` prefix.
    """
    out = str(text)
    out = _REASONING_BLOCK.sub(" ", out)
    out = _UNCLOSED_REASONING.sub(" ", out)
    out = _BARE_REASONING.sub(" ", out)
    out = " ".join(out.split())
    return out[:max_chars]


class OllamaGenerator:
    """Minimal blocking client for an Ollama ``/api/generate`` endpoint."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434",
                 num_predict: int = 2048, num_ctx: int = 4096,
                 temperature: float = 0.0, timeout: float = 600.0) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str, num_predict: int | None = None) -> str:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "OllamaGenerator needs `requests`; install it with "
                "`pip install unifusion[llm]`."
            ) from exc
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": int(num_predict or self.num_predict),
                    "num_ctx": self.num_ctx,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("error"):
            raise RuntimeError(f"generation failed: {payload['error']}")
        return strip_reasoning(payload.get("response", ""))


def answer(generator, query: str, contexts, max_chars_per_passage: int = 1200,
           retry_num_predict: int | None = None) -> str:
    """Build the prompt, generate, and retry once if the answer came back empty.

    An empty answer almost always means the scratchpad consumed the whole
    generation budget, so a single longer retry recovers it at no cost to
    queries that already succeeded.
    """
    prompt = build_prompt(query, contexts, max_chars_per_passage)
    prediction = generator.generate(prompt)
    if not prediction.strip() and retry_num_predict:
        prediction = generator.generate(prompt, num_predict=retry_num_predict)
    return prediction

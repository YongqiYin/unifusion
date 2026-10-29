"""Reader-side helpers: prompt construction, generation, post-processing."""

from .reader import (OllamaGenerator, answer, build_prompt, format_evidence,
                     strip_reasoning)

__all__ = ["OllamaGenerator", "answer", "build_prompt", "format_evidence",
           "strip_reasoning"]

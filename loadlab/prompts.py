"""Synthetic prompt generation with a controlled (approximate) token length.

Two properties matter for a fair prefill measurement:

1. Length control: we want prompts of roughly N tokens. We use a fixed vocabulary
   of common English words that tokenize to ~1 token each and calibrate with a
   words-per-token ratio. The *actual* prompt token count is always taken from the
   server's `usage.prompt_tokens` when available, so calibration error does not
   leak into the analysis.

2. No prefix-cache hits: llama.cpp, vLLM and SGLang can reuse the KV cache for a
   shared prompt prefix. If every request started with the same text, we would be
   measuring cache hits, not prefill. Each prompt therefore starts with a unique
   nonce and uses a per-request random word sequence.
"""
from __future__ import annotations

import random

VOCAB = (
    "alpha bravo river stone cloud metric kernel packet vector signal orbit lattice "
    "harbor garden copper falcon window bridge summer winter market planet engine "
    "silver forest yellow marble rocket butter castle dragon pencil candle mirror "
    "anchor basket circle desert finger hammer island jacket ladder needle orange "
    "pepper rabbit saddle tunnel violin wallet zipper meadow throne velvet"
).split()

# Measured on qwen2.5 (Ollama chat template): ~1.17 tokens per vocabulary word,
# plus ~30 tokens of chat-template / instruction overhead.
TOKENS_PER_WORD = 1.17
TEMPLATE_OVERHEAD_TOKENS = 30

INSTRUCTION = "Summarize the following list of words in two sentences."


def make_prompt(target_tokens: int, seed: int) -> str:
    """Return a prompt of approximately `target_tokens` tokens, unique per seed."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    rng = random.Random(seed)
    body_tokens = max(target_tokens - TEMPLATE_OVERHEAD_TOKENS, 1)
    n_words = max(int(round(body_tokens / TOKENS_PER_WORD)), 1)
    nonce = f"req-{seed}-{rng.getrandbits(48):012x}"
    words = " ".join(rng.choice(VOCAB) for _ in range(n_words))
    return f"[{nonce}] {INSTRUCTION}\n\n{words}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate used only when the server does not report usage."""
    return int(round(len(text.split()) * TOKENS_PER_WORD)) + TEMPLATE_OVERHEAD_TOKENS

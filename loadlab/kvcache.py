"""KV-cache capacity estimate: how many sequences fit in the memory left after the weights?

Per token, a transformer stores one key and one value vector per layer per KV head:

    kv_bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes

A sequence of `ctx` tokens therefore needs ctx * kv_bytes_per_token. Paged engines allocate in
fixed-size blocks, so the context is rounded up to a whole number of blocks.

    max_seqs = floor(free_bytes / bytes_per_sequence)

`free_bytes` is the memory that is really available for cache: device memory * utilisation limit
minus weights, activations and CUDA/runtime overhead. Measure it from the engine's startup log or
the device monitor; do not use the card's nominal size.

This is an upper bound for *concurrent sequences at that average context*. It ignores fragmentation,
prefix sharing (which raises capacity) and speculative-decoding or multimodal buffers.

Example (illustrative, generic 8B-class model with grouped-query attention):
    python -m loadlab.kvcache --layers 32 --kv-heads 8 --head-dim 128 --dtype-bytes 2 \
        --free-gb 40 --avg-ctx 4096
    -> 131072 bytes (128 KiB) per token, 512 MiB per 4096-token sequence, 80 sequences.
"""
from __future__ import annotations

import argparse
import math

GIB = 1024 ** 3


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: float) -> float:
    """Bytes of KV cache one token occupies (K and V, all layers)."""
    for name, v in (("layers", layers), ("kv_heads", kv_heads), ("head_dim", head_dim), ("dtype_bytes", dtype_bytes)):
        if v <= 0:
            raise ValueError(f"{name} must be > 0")
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def max_concurrent_sequences(layers: int, kv_heads: int, head_dim: int, dtype_bytes: float,
                             free_gb: float, avg_ctx_tokens: int, block_tokens: int = 1) -> int:
    """Max sequences of `avg_ctx_tokens` that fit in `free_gb` GiB (0 if not even one fits).

    block_tokens: paged-attention block size (e.g. 16); context is rounded up to whole blocks.
    """
    if free_gb < 0:
        raise ValueError("free_gb must be >= 0")
    if avg_ctx_tokens <= 0 or block_tokens <= 0:
        raise ValueError("avg_ctx_tokens and block_tokens must be > 0")
    blocks = math.ceil(avg_ctx_tokens / block_tokens)
    per_seq = blocks * block_tokens * kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes)
    return int((free_gb * GIB) // per_seq)


def describe(layers: int, kv_heads: int, head_dim: int, dtype_bytes: float,
             free_gb: float, avg_ctx_tokens: int, block_tokens: int = 1) -> dict:
    per_tok = kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes)
    n = max_concurrent_sequences(layers, kv_heads, head_dim, dtype_bytes, free_gb, avg_ctx_tokens, block_tokens)
    blocks = math.ceil(avg_ctx_tokens / block_tokens)
    return {"bytes_per_token": per_tok, "bytes_per_sequence": blocks * block_tokens * per_tok,
            "max_sequences": n, "free_gb": free_gb, "avg_ctx_tokens": avg_ctx_tokens,
            "block_tokens": block_tokens}


def add_cli_args(p: argparse.ArgumentParser, prefix: str = "") -> None:
    p.add_argument(f"--{prefix}layers", type=int)
    p.add_argument(f"--{prefix}kv-heads", type=int, help="KV heads (not attention heads; GQA/MQA models have fewer)")
    p.add_argument(f"--{prefix}head-dim", type=int)
    p.add_argument(f"--{prefix}dtype-bytes", type=float, default=2.0, help="2 = fp16/bf16, 1 = fp8/int8 KV")
    p.add_argument(f"--{prefix}free-gb", type=float, help="GiB left for KV cache after weights/activations/overhead")
    p.add_argument(f"--{prefix}avg-ctx", type=int, help="average context length (prompt + output) in tokens")
    p.add_argument(f"--{prefix}block-tokens", type=int, default=1, help="paged-attention block size, e.g. 16")


def from_args(a, prefix: str = "") -> dict | None:
    """Return describe(...) if the required flags were given, else None."""
    g = lambda n: getattr(a, (prefix + n).replace("-", "_"))
    vals = [g("layers"), g("kv_heads"), g("head_dim"), g("free_gb"), g("avg_ctx")]
    if all(v is None for v in vals):
        return None
    if any(v is None for v in vals):
        raise SystemExit("KV estimate needs layers, kv-heads, head-dim, free-gb and avg-ctx together")
    return describe(g("layers"), g("kv_heads"), g("head_dim"), g("dtype_bytes"), g("free_gb"), g("avg_ctx"), g("block_tokens"))


def format_estimate(d: dict) -> str:
    return (f"{d['bytes_per_token']:.0f} bytes ({d['bytes_per_token'] / 1024:.0f} KiB) of KV cache per token; "
            f"{d['bytes_per_sequence'] / 1024 ** 2:.0f} MiB per {d['avg_ctx_tokens']}-token sequence; "
            f"{d['free_gb']:g} GiB free -> at most {d['max_sequences']} concurrent sequences")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_cli_args(p)
    d = from_args(p.parse_args(argv))
    if d is None:
        p.error("give --layers --kv-heads --head-dim --free-gb --avg-ctx")
    print(format_estimate(d))


if __name__ == "__main__":
    main()

import pytest

from loadlab.kvcache import describe, kv_bytes_per_token, main, max_concurrent_sequences

# Generic 8B-class GQA shape: 32 layers, 8 KV heads, head_dim 128, fp16.
SHAPE = dict(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2)


def test_bytes_per_token_formula():
    assert kv_bytes_per_token(**SHAPE) == 2 * 32 * 8 * 128 * 2 == 131072


def test_max_sequences_known_value():
    # 4096 tokens * 128 KiB = 512 MiB per sequence; 40 GiB / 512 MiB = 80
    assert max_concurrent_sequences(**SHAPE, free_gb=40, avg_ctx_tokens=4096) == 80


def test_fp8_cache_doubles_capacity():
    fp16 = max_concurrent_sequences(**SHAPE, free_gb=40, avg_ctx_tokens=4096)
    fp8 = max_concurrent_sequences(32, 8, 128, 1, free_gb=40, avg_ctx_tokens=4096)
    assert fp8 == 2 * fp16


def test_block_rounding_and_floor():
    # 100 tokens in 16-token blocks occupy 112 tokens
    d = describe(**SHAPE, free_gb=1, avg_ctx_tokens=100, block_tokens=16)
    assert d["bytes_per_sequence"] == 112 * 131072
    assert d["max_sequences"] == (1024 ** 3) // (112 * 131072)


def test_not_even_one_fits_is_zero():
    assert max_concurrent_sequences(**SHAPE, free_gb=0.1, avg_ctx_tokens=8192) == 0


@pytest.mark.parametrize("bad", [dict(layers=0), dict(kv_heads=-1), dict(head_dim=0), dict(dtype_bytes=0)])
def test_invalid_model_params_raise(bad):
    with pytest.raises(ValueError):
        kv_bytes_per_token(**{**SHAPE, **bad})


def test_cli_prints_estimate(capsys):
    main(["--layers", "32", "--kv-heads", "8", "--head-dim", "128", "--free-gb", "40", "--avg-ctx", "4096"])
    assert "at most 80 concurrent sequences" in capsys.readouterr().out

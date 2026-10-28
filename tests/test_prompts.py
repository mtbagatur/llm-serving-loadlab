from loadlab.prompts import estimate_tokens, make_prompt


def test_prompt_length_scales_with_target():
    short, long = make_prompt(128, seed=1), make_prompt(2048, seed=1)
    assert estimate_tokens(long) > 8 * estimate_tokens(short) / 2
    assert abs(estimate_tokens(make_prompt(1024, seed=3)) - 1024) < 1024 * 0.05


def test_prompts_are_unique_so_prefix_cache_cannot_hit():
    a, b = make_prompt(512, seed=1), make_prompt(512, seed=2)
    assert a != b
    assert a[:20] != b[:20]  # different from the very first characters


def test_prompts_are_deterministic_per_seed():
    assert make_prompt(256, seed=7) == make_prompt(256, seed=7)

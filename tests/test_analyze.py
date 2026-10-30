import csv
import itertools
import math
import random

import pytest

from loadlab.analyze import analyze, classify, fit_elasticities, Fit

LENGTHS = [128, 512, 1024, 2048]
CONCS = [1, 2, 4, 8]


def _rows(latency_fn, ttft_share=0.8, scenario="s", reps=5, seed=0):
    rng = random.Random(seed)
    rows = []
    for p, c in itertools.product(LENGTHS, CONCS):
        for i in range(reps):
            lat = latency_fn(p, c) * math.exp(rng.gauss(0, 0.03))
            rows.append({"scenario": scenario, "base_url": "x", "target_prompt_tokens": p, "concurrency": c,
                         "request_index": i, "ok": True, "error": "", "prompt_tokens": p, "completion_tokens": 64,
                         "ttft_s": lat * ttft_share, "latency_s": lat, "decode_tok_s": 30.0, "started_at": 0})
    return rows


def _write(rows, path):
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_elasticities_recover_known_exponents():
    rows = _rows(lambda p, c: 0.002 * p ** 0.9 * c ** 0.2)
    f = fit_elasticities([r["prompt_tokens"] for r in rows], [r["concurrency"] for r in rows], [r["latency_s"] for r in rows])
    assert f.e_prompt == pytest.approx(0.9, abs=0.05)
    assert f.e_conc == pytest.approx(0.2, abs=0.05)
    assert f.r2_prompt_only > f.r2_conc_only


def test_prompt_driven_latency_is_classified_as_prefill(tmp_path):
    rows = _rows(lambda p, c: 0.003 * p * (1 + 0.05 * c), ttft_share=0.85)
    s = analyze([_write(rows, tmp_path / "a.csv")], tmp_path / "out")
    assert s["scenarios"]["s"]["verdict"] == "prefill"
    memo = (tmp_path / "out" / "decision_memo.md").read_text()
    assert "prefill" in memo and "second model replica" in memo
    assert (tmp_path / "out" / "s_latency_vs_prompt.png").stat().st_size > 1000


def test_concurrency_driven_latency_is_classified_as_queueing():
    lat = Fit(0, 0.05, 0.9, 0.95, 0.01, 0.94, 80)
    ttft = Fit(0, 0.05, 1.0, 0.95, 0.01, 0.94, 80)
    assert classify(lat, ttft, ttft_share_longest=0.7, error_rate=0.0)[0] == "queueing"


def test_errors_take_priority():
    f = Fit(0, 1, 0, 0.9, 0.9, 0.0, 80)
    assert classify(f, f, 0.9, error_rate=0.05)[0] == "reliability"


def test_two_scenarios_produce_comparison(tmp_path):
    one = _rows(lambda p, c: 0.003 * p * c ** 0.5, scenario="one-replica")
    two = _rows(lambda p, c: 0.003 * p * (c / 2) ** 0.5 if c > 1 else 0.003 * p, scenario="two-replicas", seed=1)
    s = analyze([_write(one, tmp_path / "1.csv"), _write(two, tmp_path / "2.csv")], tmp_path / "out")
    comp = s["comparison"]
    assert comp["baseline"] == "one-replica" and comp["candidate"] == "two-replicas"
    assert comp["median_speedup"] > 1.1
    assert (tmp_path / "out" / "scenario_comparison.png").exists()


# ---- sweep table, ITL columns, component check, KV memo section

from loadlab.analyze import cell_table, component_check, load_rows, sweep_table, BOTTLENECK_RULE


def _scaled_rows(base_fn, tail_factor, decode_tok_s, scenario, ttft_share=0.8, seed=0):
    """Rows whose p95 latency is multiplied by tail_factor for the slowest requests, decode speed fixed."""
    rows = _rows(base_fn, ttft_share=ttft_share, scenario=scenario, seed=seed)
    for i, r in enumerate(rows):
        r["decode_tok_s"] = decode_tok_s
        r["itl_p50_s"] = 1 / decode_tok_s
        r["itl_p90_s"] = 1.5 / decode_tok_s
        if i % 5 == 4:  # the 5th request of every cell is the tail
            r["latency_s"] *= tail_factor
    return rows


def test_sweep_table_has_all_columns_and_itl(tmp_path):
    rows = _scaled_rows(lambda p, c: 0.003 * p * (1 + 0.05 * c), 1.0, 30.0, "s")
    cells = cell_table(load_rows([_write(rows, tmp_path / "a.csv")]))
    table = sweep_table(cells, "s")
    head = table.splitlines()[0]
    for col in ("p50 TTFT", "p50 total", "decode tok/s per req", "aggregate tok/s", "ITL p50", "ITL p90"):
        assert col in head
    assert len(table.splitlines()) == 2 + len(LENGTHS) * len(CONCS)
    assert all(c["aggregate_tok_s"] > 0 for c in cells)


def test_old_csv_without_itl_columns_still_works(tmp_path):
    rows = _rows(lambda p, c: 0.003 * p)
    s = analyze([_write(rows, tmp_path / "a.csv")], tmp_path / "out")
    table = sweep_table(s["cells"], "s")
    assert "ITL" not in table.splitlines()[0]
    assert "Sweep table" in (tmp_path / "out" / "decision_memo.md").read_text()


def test_scaling_a_non_bottleneck_changes_tail_only():
    fn = lambda p, c: 0.003 * p * (1 + 0.05 * c)
    base = _scaled_rows(fn, 2.0, 30.0, "gw1", seed=0)  # slow tail from queueing
    scaled = _scaled_rows(fn, 1.0, 30.0, "gw2", seed=1)  # tail shrinks, per-request speed identical
    cells = cell_table(load_rows_from(base + scaled))
    r = component_check(cells, "gw1", "gw2", "gateway")
    assert r["verdict"] == "not-bottleneck"
    assert r["latency_p95_ratio"] < 0.85  # the tail did change


def test_scaling_the_real_bottleneck_is_detected():
    base = _scaled_rows(lambda p, c: 0.003 * p * (1 + 0.05 * c), 1.0, 30.0, "r1", seed=0)
    scaled = _scaled_rows(lambda p, c: 0.0015 * p * (1 + 0.05 * c), 1.0, 60.0, "r2", seed=1)  # prefill and decode 2x faster
    r = component_check(cell_table(load_rows_from(base + scaled)), "r1", "r2", "model replica")
    assert r["verdict"] == "on-critical-path"


def test_memo_documents_rule_and_kv(tmp_path):
    fn = lambda p, c: 0.003 * p * (1 + 0.05 * c)
    a = _write(_scaled_rows(fn, 2.0, 30.0, "gw1"), tmp_path / "a.csv")
    b = _write(_scaled_rows(fn, 1.0, 30.0, "gw2", seed=1), tmp_path / "b.csv")
    kv = {"bytes_per_token": 131072, "bytes_per_sequence": 4096 * 131072, "max_sequences": 80, "free_gb": 40,
          "avg_ctx_tokens": 4096, "block_tokens": 1}
    s = analyze([a, b], tmp_path / "out", baseline="gw1", component="gateway", kv_cache=kv)
    memo = (tmp_path / "out" / "decision_memo.md").read_text()
    assert BOTTLENECK_RULE in memo
    assert "not-bottleneck" in memo and "at most 80 concurrent sequences" in memo
    assert s["component_check"]["component"] == "gateway"


def load_rows_from(rows):
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    return load_rows([_write(rows, d / "x.csv")])

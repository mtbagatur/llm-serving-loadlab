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

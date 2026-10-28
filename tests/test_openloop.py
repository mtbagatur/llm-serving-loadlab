"""Open-loop mode: Poisson arrivals, in-flight accounting, queue build-up detection."""
import random
import statistics

import pytest
from aiohttp.test_utils import TestServer

from loadlab.loadgen import RunConfig, poisson_arrivals, run_matrix, write_results
from loadlab.mock_server import MockParams, make_app
from loadlab.openloop import (analyze_open, cell_table, growth_ratio, inflight_timeline, max_sustainable_rate,
                              queue_growth)


def test_poisson_arrivals_rate_and_shape():
    arr = poisson_arrivals(2.0, 4000, random.Random(1))
    gaps = [b - a for a, b in zip([0.0] + arr[:-1], arr)]
    assert all(g > 0 for g in gaps)
    assert statistics.mean(gaps) == pytest.approx(0.5, rel=0.05)  # mean gap = 1/rate
    # exponential gaps: standard deviation ~= mean (a fixed-interval schedule would have ~0)
    assert statistics.pstdev(gaps) == pytest.approx(0.5, rel=0.1)
    # conditioned on the count: all n arrivals fall inside the window n / rate
    assert arr[-1] <= 4000 / 2.0 and arr == sorted(arr)
    assert poisson_arrivals(2.0, 10, random.Random(7)) == poisson_arrivals(2.0, 10, random.Random(7))
    with pytest.raises(ValueError):
        poisson_arrivals(0, 3, random.Random(0))


def _row(i, start, latency, ttft, rate=1.0, ok=True):
    return {"scenario": "s", "target_prompt_tokens": 512, "rate_rps": rate, "request_index": i, "ok": ok,
            "started_at": 1000.0 + start, "arrival_offset_s": start, "latency_s": latency, "ttft_s": ttft,
            "prompt_tokens": 500, "inflight_at_arrival": 0}


def test_inflight_timeline_counts_overlaps():
    rows = [_row(0, 0.0, 3.0, 1.0), _row(1, 1.0, 1.0, 0.5), _row(2, 4.0, 1.0, 0.5)]
    t, n = inflight_timeline(rows)
    at = lambda x: int(n[(t <= x).nonzero()[0][-1]])
    assert at(0.5) == 1 and at(1.5) == 2 and at(2.5) == 1 and at(3.5) == 0 and at(4.5) == 1
    assert n.max() == 2 and n[-1] == 0


def test_queue_growth_from_inflight_counts():
    flat = [dict(_row(i, i * 1.0, 1.0, 0.5), inflight_at_arrival=i % 2) for i in range(20)]
    rising = [dict(_row(i, i * 1.0, 1.0, 0.5, rate=2.0), inflight_at_arrival=i // 2) for i in range(20)]
    assert abs(queue_growth(flat)) < 1
    assert queue_growth(rising) == pytest.approx(9.5, abs=0.5)
    # TTFT is flat in both, but the backlog alone marks the second cell as overloaded
    cells = {c["offered_rps"]: c for c in cell_table(flat + rising)}
    assert not cells[1.0]["queue_growing"] and cells[2.0]["queue_growing"]


def test_growth_detects_a_building_queue():
    # Stable: TTFT flat. Overloaded: every request waits one service time longer than the previous one.
    stable = [_row(i, i * 1.0, 1.2, 0.5) for i in range(12)]
    growing = [_row(i, i * 0.2, 1.0 + 0.8 * i, 0.5 + 0.8 * i, rate=5.0) for i in range(12)]
    assert growth_ratio(stable) == pytest.approx(1.0)
    assert growth_ratio(growing) > 2
    cells = {c["offered_rps"]: c for c in cell_table(stable + growing)}
    assert not cells[1.0]["queue_growing"] and cells[5.0]["queue_growing"]
    assert cells[5.0]["max_inflight"] > cells[1.0]["max_inflight"]
    assert max_sustainable_rate(list(cells.values()), None) == 1.0
    assert max_sustainable_rate(list(cells.values()), slo_ttft_s=0.1) is None  # SLO nobody meets


async def test_open_loop_against_mock_server(tmp_path):
    # Mock prefill is serialised: ~512 tokens / 10k tok/s ~= 0.05 s per request -> capacity ~20 req/s.
    server = TestServer(make_app(MockParams(prefill_tok_s=10000, decode_tok_s=2000, overhead_s=0.001)))
    await server.start_server()
    try:
        cfg = RunConfig(base_urls=[str(server.make_url("/v1"))], model="mock", prompt_lengths=[512],
                        concurrency=[], rates=[4.0, 100.0], mode="open", requests_per_cell=15,
                        max_tokens=4, scenario="mock-open")
        results = await run_matrix(cfg, progress=False)
    finally:
        await server.close()
    assert len(results) == 30 and all(r.ok for r in results)
    assert {r.mode for r in results} == {"open"} and {r.rate_rps for r in results} == {4.0, 100.0}
    fast = [r for r in results if r.rate_rps == 100.0]
    # open loop: later arrivals do not wait for earlier ones, so requests pile up
    assert max(r.inflight_at_arrival for r in fast) >= 5
    csv_path, _ = write_results(results, cfg, tmp_path / "open")
    summary = analyze_open([csv_path], tmp_path / "an", source_note="MOCK", slo_ttft_s=1.0)
    cells = {c["offered_rps"]: c for c in summary["cells"]}
    assert not cells[4.0]["queue_growing"]
    assert cells[100.0]["queue_growing"]
    assert cells[100.0]["ttft_p95_s"] > cells[4.0]["ttft_p95_s"]
    for name in ("mock-open_openloop_queue.png", "mock-open_openloop_ttft_over_time.png",
                 "mock-open_openloop_ttft_vs_rate.png", "openloop_memo.md", "openloop_cells.csv"):
        assert (tmp_path / "an" / name).exists()
    assert "MOCK DATA" in (tmp_path / "an" / "openloop_memo.md").read_text()


def test_closed_loop_rows_are_ignored(tmp_path):
    from loadlab.openloop import load_open_rows
    p = tmp_path / "closed.csv"
    p.write_text("scenario,mode,ok\ns,closed,True\n")
    with pytest.raises(ValueError):
        load_open_rows([p])

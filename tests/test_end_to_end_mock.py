"""Load generator against the in-process mock server (no model, no network)."""
import math

from aiohttp.test_utils import TestServer

from loadlab.loadgen import RunConfig, run_matrix, write_results
from loadlab.mock_server import MockParams, make_app


async def test_loadgen_against_mock_server(tmp_path):
    params = MockParams(prefill_tok_s=20000, decode_tok_s=2000, overhead_s=0.001)
    server = TestServer(make_app(params))
    await server.start_server()
    try:
        cfg = RunConfig(base_urls=[str(server.make_url("/v1"))], model="mock", prompt_lengths=[128, 2048],
                        concurrency=[1, 4], requests_per_cell=4, max_tokens=8, scenario="mock")
        results = await run_matrix(cfg, progress=False)
    finally:
        await server.close()
    assert len(results) == 16
    assert all(r.ok for r in results), [r.error for r in results if not r.ok]
    for r in results:
        assert 0 < r.ttft_s <= r.latency_s
        assert r.completion_tokens == 8
        assert not math.isnan(r.decode_tok_s)
        assert 0 < r.itl_p50_s <= r.itl_p90_s <= r.latency_s  # inter-token latency is recorded
    long_ttft = [r.ttft_s for r in results if r.target_prompt_tokens == 2048 and r.concurrency == 1]
    short_ttft = [r.ttft_s for r in results if r.target_prompt_tokens == 128 and r.concurrency == 1]
    assert min(long_ttft) > max(short_ttft)  # prefill cost is visible in TTFT
    csv_path, json_path = write_results(results, cfg, tmp_path / "run")
    assert csv_path.read_text().count("\n") == 17 and json_path.exists()


async def test_errors_are_recorded_not_raised():
    cfg = RunConfig(base_urls=["http://127.0.0.1:9/v1"], model="m", prompt_lengths=[128], concurrency=[2],
                    requests_per_cell=2, warmup_requests=0, timeout_s=2)
    results = await run_matrix(cfg, progress=False)
    assert len(results) == 2 and not any(r.ok for r in results) and all(r.error for r in results)

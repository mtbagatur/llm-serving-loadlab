"""Async load generator for OpenAI-compatible chat completion endpoints.

Runs a (prompt length x concurrency) matrix. For every request it records:

- ttft_s:        time to first content token (includes server-side queueing + prefill)
- latency_s:     total wall time of the request
- decode_tok_s:  output tokens per second after the first token
- prompt_tokens / completion_tokens (from `usage` when the server reports it)

Two load models:

- closed loop (default, `--mode closed`): `c` workers each send one request, wait
  for it to finish, then send the next. The number of in-flight requests is
  exactly `c`. Clean for diagnosis ("what does one more request in flight cost?").
- open loop (`--mode open --rates 0.1,0.2`): requests arrive at a fixed average
  rate (requests/s) as a Poisson process, whether or not earlier requests have
  finished. This is how independent users behave. If the offered rate is above
  what the server can complete, the in-flight count (the queue) keeps growing and
  TTFT climbs with time: that is the capacity limit. Each open-loop row records the
  arrival time and how many requests were already in flight at that moment.

Usage:
    python -m loadlab.loadgen --base-url http://127.0.0.1:11501/v1 \
        --model qwen2.5:0.5b --prompt-lengths 128,512,1024,2048 \
        --concurrency 1,2,4,8 --requests-per-cell 8 --out results/run

    python -m loadlab.loadgen --mode open --base-url http://127.0.0.1:11501/v1 \
        --model qwen2.5:0.5b --prompt-lengths 512 --rates 0.05,0.1,0.2 \
        --requests-per-cell 16 --out results/openloop
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import itertools
import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import numpy as np

from .prompts import estimate_tokens, make_prompt


@dataclass
class RequestResult:
    scenario: str
    base_url: str
    target_prompt_tokens: int
    concurrency: int
    request_index: int
    ok: bool
    error: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = float("nan")
    latency_s: float = float("nan")
    decode_tok_s: float = float("nan")
    started_at: float = 0.0
    # inter-token latency: gaps between consecutive content chunks (one chunk ~ one token on
    # most servers; servers that batch several tokens per chunk make this an upper bound on ITL)
    itl_p50_s: float = float("nan")
    itl_p90_s: float = float("nan")
    # open-loop only (closed-loop rows keep the defaults)
    mode: str = "closed"
    rate_rps: float = 0.0              # offered arrival rate of the cell
    arrival_offset_s: float = float("nan")  # scheduled arrival, seconds since cell start
    inflight_at_arrival: int = 0       # requests already in flight when this one was sent


@dataclass
class RunConfig:
    base_urls: list[str]
    model: str
    prompt_lengths: list[int]
    concurrency: list[int]
    requests_per_cell: int
    max_tokens: int = 64
    timeout_s: float = 600.0
    scenario: str = "default"
    warmup_requests: int = 1
    seed: int = 0
    api_key: str = "not-needed"
    mode: str = "closed"               # "closed" or "open"
    rates: list[float] = field(default_factory=list)  # open loop: requests/s per cell
    force_close: bool = False          # new TCP connection per request (lets a k8s Service balance per request)
    extra: dict = field(default_factory=dict)


async def send_one(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    api_key: str,
) -> dict:
    """Send one streaming chat completion and return raw timing measurements."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    usage = None
    stamps: list[float] = []
    async with session.post(url, json=payload, headers=headers) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
        async for raw in resp.content:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            if msg.get("usage"):
                usage = msg["usage"]
            for choice in msg.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks += 1
                    stamps.append(time.perf_counter())
    latency = time.perf_counter() - t0
    if ttft is None:  # no content at all (e.g. immediate EOS)
        ttft = latency
    completion_tokens = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens") or estimate_tokens(prompt)
    decode_time = latency - ttft
    decode_tok_s = (completion_tokens - 1) / decode_time if completion_tokens > 1 and decode_time > 0 else float("nan")
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "itl_p50_s": float(np.percentile(gaps, 50)) if gaps else float("nan"),
        "itl_p90_s": float(np.percentile(gaps, 90)) if gaps else float("nan"),
        "ttft_s": ttft,
        "latency_s": latency,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "decode_tok_s": decode_tok_s,
    }


async def run_cell(
    session: aiohttp.ClientSession,
    cfg: RunConfig,
    target_tokens: int,
    concurrency: int,
    url_cycle,
    seed_base: int,
) -> list[RequestResult]:
    """Run `requests_per_cell` requests with exactly `concurrency` in flight."""
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(cfg.requests_per_cell):
        queue.put_nowait(i)
    results: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            base_url = next(url_cycle)
            prompt = make_prompt(target_tokens, seed=seed_base + i)
            rec = RequestResult(
                scenario=cfg.scenario,
                base_url=base_url,
                target_prompt_tokens=target_tokens,
                concurrency=concurrency,
                request_index=i,
                ok=False,
                started_at=time.time(),
            )
            try:
                m = await send_one(session, base_url, cfg.model, prompt, cfg.max_tokens, cfg.api_key)
                rec.ok = True
                for k, v in m.items():
                    setattr(rec, k, v)
            except Exception as exc:  # recorded, never raised: errors are data
                rec.error = f"{type(exc).__name__}: {exc}"[:300]
            results.append(rec)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


def poisson_arrivals(rate_rps: float, n: int, rng: random.Random) -> list[float]:
    """Arrival offsets (s) for `n` requests at average rate `rate_rps`.

    A Poisson process *conditioned on* n arrivals in the window T = n / rate: the
    arrival times are n sorted uniform draws on [0, T]. Gaps are still exponential-
    like (bursts and lulls, as with real users), but the realised rate is exactly
    the offered one. Unconditioned exponential gaps over only ~20 requests can miss
    the target badly (in a first run here, 0.4 req/s offered came out as 0.27 req/s).
    """
    if rate_rps <= 0:
        raise ValueError("rate must be > 0")
    window = n / rate_rps
    return sorted(rng.uniform(0.0, window) for _ in range(n))


async def run_open_cell(
    session: aiohttp.ClientSession,
    cfg: RunConfig,
    target_tokens: int,
    rate_rps: float,
    url_cycle,
    seed_base: int,
) -> list[RequestResult]:
    """Open loop: send `requests_per_cell` requests at Poisson arrival times, never waiting
    for earlier ones. Returns after the last request has finished."""
    arrivals = poisson_arrivals(rate_rps, cfg.requests_per_cell, random.Random(seed_base))
    results: list[RequestResult] = []
    inflight = 0

    async def one(i: int, offset: float) -> None:
        nonlocal inflight
        base_url = next(url_cycle)
        rec = RequestResult(
            scenario=cfg.scenario, base_url=base_url, target_prompt_tokens=target_tokens,
            concurrency=0, request_index=i, ok=False, started_at=time.time(),
            mode="open", rate_rps=rate_rps, arrival_offset_s=offset, inflight_at_arrival=inflight,
        )
        inflight += 1
        try:
            m = await send_one(session, base_url, cfg.model, make_prompt(target_tokens, seed=seed_base + i),
                               cfg.max_tokens, cfg.api_key)
            rec.ok = True
            for k, v in m.items():
                setattr(rec, k, v)
        except Exception as exc:  # recorded, never raised: errors are data
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            inflight -= 1
        results.append(rec)

    t0 = time.perf_counter()
    tasks = []
    for i, offset in enumerate(arrivals):
        delay = t0 + offset - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i, offset)))
    await asyncio.gather(*tasks)
    results.sort(key=lambda r: r.request_index)
    return results


async def run_matrix(cfg: RunConfig, progress: bool = True) -> list[RequestResult]:
    timeout = aiohttp.ClientTimeout(total=cfg.timeout_s)
    connector = aiohttp.TCPConnector(limit=0, force_close=cfg.force_close)
    url_cycle = itertools.cycle(cfg.base_urls)
    all_results: list[RequestResult] = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warm-up: load the model into memory so the first cell is not penalised.
        for i in range(cfg.warmup_requests):
            for url in cfg.base_urls:
                await send_one(session, url, cfg.model, make_prompt(64, seed=10_000_000 + i), 8, cfg.api_key)
        seed = cfg.seed * 1_000_000
        # Interleave cells (prompt length outer, concurrency inner) so slow drift
        # of the machine (thermals, background jobs) does not line up with one factor.
        levels = cfg.rates if cfg.mode == "open" else cfg.concurrency
        for target in cfg.prompt_lengths:
            for level in levels:
                t0 = time.perf_counter()
                if cfg.mode == "open":
                    res = await run_open_cell(session, cfg, target, level, url_cycle, seed)
                    label = f"rate={level:g}/s"
                else:
                    res = await run_cell(session, cfg, target, level, url_cycle, seed)
                    label = f"conc={level:>2}"
                seed += cfg.requests_per_cell
                all_results.extend(res)
                if progress:
                    ok = [r for r in res if r.ok]
                    p50 = sorted(r.latency_s for r in ok)[len(ok) // 2] if ok else float("nan")
                    extra = (f" max_inflight={max(r.inflight_at_arrival for r in res) + 1}"
                             if cfg.mode == "open" else "")
                    print(
                        f"[{cfg.scenario}] prompt~{target:>5} {label} "
                        f"ok={len(ok)}/{len(res)} p50_latency={p50:6.2f}s{extra} "
                        f"cell_time={time.perf_counter() - t0:6.1f}s",
                        flush=True,
                    )
    return all_results


CSV_FIELDS = [f for f in RequestResult.__dataclass_fields__]


def write_results(results: list[RequestResult], cfg: RunConfig, out_prefix: Path) -> tuple[Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    json_path = out_prefix.with_suffix(".json")
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    meta = {
        "config": asdict(cfg),
        "host": {"platform": platform.platform(), "python": platform.python_version(), "machine": platform.machine()},
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_requests": len(results),
        "n_errors": sum(not r.ok for r in results),
    }
    with json_path.open("w") as fh:
        json.dump({"meta": meta, "results": [asdict(r) for r in results]}, fh, indent=1, default=str)
    return csv_path, json_path


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", action="append", required=True,
                   help="OpenAI-compatible base URL (repeat for client-side round robin across replicas)")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt-lengths", type=_ints, default=[128, 512, 1024, 2048])
    p.add_argument("--mode", choices=["closed", "open"], default="closed",
                   help="closed: fixed requests in flight (--concurrency); open: Poisson arrivals at --rates req/s")
    p.add_argument("--concurrency", type=_ints, default=[1, 2, 4, 8])
    p.add_argument("--rates", type=_floats, default=[],
                   help="open loop only: offered arrival rates in requests/s, e.g. 0.05,0.1,0.2")
    p.add_argument("--no-keepalive", action="store_true",
                   help="open a new connection per request (a k8s Service balances connections, not requests)")
    p.add_argument("--requests-per-cell", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--scenario", default="default", help="label stored with every row, e.g. one-replica")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", required=True, help="output prefix; writes <out>.csv and <out>.json")
    a = p.parse_args(argv)
    cfg = RunConfig(
        base_urls=a.base_url, model=a.model, prompt_lengths=a.prompt_lengths,
        concurrency=a.concurrency, requests_per_cell=a.requests_per_cell,
        max_tokens=a.max_tokens, timeout_s=a.timeout, scenario=a.scenario,
        warmup_requests=a.warmup, seed=a.seed, mode=a.mode, rates=a.rates,
        force_close=a.no_keepalive,
    )
    if cfg.mode == "open" and not cfg.rates:
        p.error("--mode open needs --rates")
    results = asyncio.run(run_matrix(cfg))
    csv_path, json_path = write_results(results, cfg, Path(a.out))
    n_err = sum(not r.ok for r in results)
    print(f"wrote {csv_path} and {json_path}: {len(results)} requests, {n_err} errors")


if __name__ == "__main__":
    main()

"""Analysis of open-loop runs (`loadgen --mode open`): does the queue build up?

In an open-loop test requests arrive at a fixed average rate (Poisson process),
independent of how fast the server answers. Two regimes:

- stable: the server completes requests about as fast as they arrive. The number
  of requests in flight goes up and down around a small value and TTFT does not
  trend upward over the run.
- overloaded: arrivals outpace completions. In-flight requests pile up, every new
  request waits behind a longer queue, and TTFT climbs with arrival time. A
  longer test would only make it worse; there is no steady state.

Per cell (scenario x prompt length x offered rate) this module reports offered
and realised arrival rate, completed throughput, TTFT/latency percentiles, peak
in-flight requests and two queue build-up signals:

- queue growth: the fitted increase in requests in flight (sampled at each
  arrival) across the arrival window. ~0 when stable, ~(arrival rate - service
  rate) x window when overloaded. This is the primary signal: an engine with
  parallel slots (Ollama NUM_PARALLEL, vLLM batching) admits several requests at
  once, so the backlog shows up as more requests in flight and longer total
  latency before it shows up in TTFT;
- TTFT growth: median TTFT of the last third of arrivals / first third. Charts:

- <scenario>_openloop_queue.png: requests in flight over time, one line per rate
  (the queue build-up chart);
- <scenario>_openloop_ttft_over_time.png: TTFT of each request vs its arrival time;
- <scenario>_openloop_ttft_vs_rate.png: p50/p95 TTFT vs offered rate (+ SLO line).

Usage:
    python -m loadlab.openloop results/x/open.csv --out-dir results/x/analysis-openloop \
        --slo-ttft 10 --source-note "..."
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze import INK_2, SERIES, SURFACE, _fmt, _style

GROWTH_THRESHOLD = 2.0       # last-third median TTFT >= 2x first-third -> queue is growing
QUEUE_GROWTH_MIN = 3.0       # ... or in-flight count rises by >= 3 requests over the window
QUEUE_GROWTH_FRACTION = 0.2  # ... and by >= 20% of the cell's requests


def load_open_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        with Path(p).open() as fh:
            for r in csv.DictReader(fh):
                if r.get("mode") != "open":
                    continue
                r["ok"] = r["ok"] in ("True", "true", "1")
                for k in ("target_prompt_tokens", "prompt_tokens", "completion_tokens", "request_index", "inflight_at_arrival"):
                    r[k] = int(float(r[k] or 0))
                for k in ("ttft_s", "latency_s", "decode_tok_s", "started_at", "rate_rps", "arrival_offset_s"):
                    r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
                rows.append(r)
    if not rows:
        raise ValueError("no open-loop rows (mode=open) in the input CSVs")
    return rows


def inflight_timeline(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Step function of requests in flight. Returns (t, n) with t relative to the
    first send; n[i] holds from t[i] until t[i+1]. Failed requests count until
    they failed (latency is nan for them, so they are skipped)."""
    events = []
    t_start = min(r["started_at"] for r in rows)
    for r in rows:
        if math.isnan(r["latency_s"]):
            continue
        events.append((r["started_at"] - t_start, +1))
        events.append((r["started_at"] + r["latency_s"] - t_start, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # finish before start at the same instant
    t, n, cur = [0.0], [0], 0
    for ts, d in events:
        cur += d
        t.append(ts)
        n.append(cur)
    return np.asarray(t), np.asarray(n)


def growth_ratio(rows: list[dict]) -> float:
    """Median TTFT of the last third of arrivals / median TTFT of the first third."""
    ok = sorted((r for r in rows if r["ok"]), key=lambda r: r["arrival_offset_s"])
    k = len(ok) // 3
    if k < 1:
        return float("nan")
    first = np.median([r["ttft_s"] for r in ok[:k]])
    last = np.median([r["ttft_s"] for r in ok[-k:]])
    return float(last / first) if first > 0 else float("nan")


def queue_growth(rows: list[dict]) -> float:
    """Least-squares slope of in-flight-at-arrival vs arrival time, times the
    arrival window: by how many requests the backlog rose while requests arrived."""
    pts = sorted((r["arrival_offset_s"], r["inflight_at_arrival"]) for r in rows if not math.isnan(r["arrival_offset_s"]))
    if len(pts) < 3:
        return float("nan")
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts], dtype=float)
    if np.ptp(x) == 0:
        return float("nan")
    slope = np.polyfit(x, y, 1)[0]
    return float(slope * np.ptp(x))


def is_growing(ttft_growth: float, q_growth: float, n: int) -> bool:
    by_ttft = not math.isnan(ttft_growth) and ttft_growth >= GROWTH_THRESHOLD
    by_queue = (not math.isnan(q_growth) and q_growth >= QUEUE_GROWTH_MIN
                and q_growth >= QUEUE_GROWTH_FRACTION * n)
    return by_ttft or by_queue


def cell_table(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["target_prompt_tokens"], r["rate_rps"])].append(r)
    out = []
    for (scen, tgt, rate), rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        arr = sorted(r["arrival_offset_s"] for r in rs)
        realised = (len(arr) - 1) / (arr[-1] - arr[0]) if len(arr) > 1 and arr[-1] > arr[0] else float("nan")
        if ok:
            wall = max(r["started_at"] + r["latency_s"] for r in ok) - min(r["started_at"] for r in rs)
            thr = len(ok) / wall if wall > 0 else float("nan")
        else:
            thr = float("nan")
        _, n = inflight_timeline(rs)
        ttft = [r["ttft_s"] for r in ok]
        lat = [r["latency_s"] for r in ok]
        g = growth_ratio(rs)
        q = queue_growth(rs)
        out.append({
            "scenario": scen, "target_prompt_tokens": tgt, "offered_rps": rate,
            "n": len(rs), "errors": len(rs) - len(ok),
            "realised_arrival_rps": realised, "throughput_rps": thr,
            "prompt_tokens_median": float(np.median([r["prompt_tokens"] for r in ok])) if ok else float("nan"),
            "ttft_p50_s": float(np.median(ttft)) if ttft else float("nan"),
            "ttft_p95_s": float(np.percentile(ttft, 95)) if ttft else float("nan"),
            "latency_p50_s": float(np.median(lat)) if lat else float("nan"),
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
            "max_inflight": int(n.max()) if len(n) else 0,
            "queue_growth_requests": q,
            "ttft_growth_last_vs_first_third": g,
            "queue_growing": is_growing(g, q, len(rs)),
        })
    return out


def max_sustainable_rate(cells: list[dict], slo_ttft_s: float | None) -> float | None:
    """Highest offered rate (within the tested ones) whose queue did not grow and,
    if an SLO is given, whose p95 TTFT met it. None if no tested rate qualifies."""
    good = [c["offered_rps"] for c in cells
            if not c["queue_growing"] and c["errors"] == 0
            and (slo_ttft_s is None or c["ttft_p95_s"] <= slo_ttft_s)]
    return max(good) if good else None


# ------------------------------------------------------------------------ charts

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def queue_chart(rows: list[dict], cells: list[dict], path: Path, title: str) -> None:
    plt = _plt()
    tgts = sorted({c["target_prompt_tokens"] for c in cells})
    fig, axes = plt.subplots(len(tgts), 1, figsize=(7.5, 3.6 * len(tgts)), facecolor=SURFACE, squeeze=False)
    for ax, tgt in zip(axes[:, 0], tgts):
        rates = sorted({c["offered_rps"] for c in cells if c["target_prompt_tokens"] == tgt})
        for i, rate in enumerate(rates):
            rs = [r for r in rows if r["target_prompt_tokens"] == tgt and r["rate_rps"] == rate]
            t, n = inflight_timeline(rs)
            cell = next(c for c in cells if c["target_prompt_tokens"] == tgt and c["offered_rps"] == rate)
            label = f"{rate:g} req/s" + ("  (queue growing)" if cell["queue_growing"] else "")
            ax.step(t, n, where="post", color=SERIES[i % len(SERIES)], linewidth=2, label=label)
        ax.set_ylim(bottom=0)
        _style(ax, f"{title}: requests in flight, prompt ~{tgt} tokens", "seconds since first arrival", "requests in flight")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ttft_over_time_chart(rows: list[dict], cells: list[dict], path: Path, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    keys = sorted({(c["target_prompt_tokens"], c["offered_rps"]) for c in cells})
    for i, (tgt, rate) in enumerate(keys):
        ok = sorted((r for r in rows if r["ok"] and r["target_prompt_tokens"] == tgt and r["rate_rps"] == rate),
                    key=lambda r: r["arrival_offset_s"])
        if not ok:
            continue
        lbl = f"{rate:g} req/s" if len({k[0] for k in keys}) == 1 else f"{tgt} tok, {rate:g} req/s"
        ax.plot([r["arrival_offset_s"] for r in ok], [r["ttft_s"] for r in ok], color=SERIES[i % len(SERIES)],
                linewidth=1.5, marker="o", markersize=4, label=lbl)
    ax.set_ylim(bottom=0)
    _style(ax, f"{title}: TTFT of each request by arrival time", "arrival time (s since first arrival)", "TTFT (s)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def ttft_vs_rate_chart(cells: list[dict], path: Path, title: str, slo_ttft_s: float | None) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    for i, tgt in enumerate(sorted({c["target_prompt_tokens"] for c in cells})):
        cs = sorted((c for c in cells if c["target_prompt_tokens"] == tgt), key=lambda c: c["offered_rps"])
        x = [c["offered_rps"] for c in cs]
        ax.plot(x, [c["ttft_p95_s"] for c in cs], color=SERIES[i % len(SERIES)], linewidth=2, marker="o", label=f"{tgt} tok p95")
        ax.plot(x, [c["ttft_p50_s"] for c in cs], color=SERIES[i % len(SERIES)], linewidth=1.5, linestyle="--",
                marker="o", markersize=4, label=f"{tgt} tok p50")
    if slo_ttft_s:
        ax.axhline(slo_ttft_s, color=INK_2, linewidth=1, linestyle=":")
        ax.annotate(f"TTFT SLO {slo_ttft_s:g}s", (ax.get_xlim()[0], slo_ttft_s), xytext=(4, 4),
                    textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_ylim(bottom=0)
    _style(ax, f"{title}: TTFT vs offered load", "offered arrival rate (req/s)", "TTFT (s)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------- memo

def write_memo(cells: list[dict], out_dir: Path, source_note: str, slo_ttft_s: float | None) -> Path:
    L = ["# Open-loop memo: at what arrival rate does the queue start to grow?", ""]
    if "mock" in source_note.lower():
        L += ["> **MOCK DATA.** Simulated server, not a model.", ""]
    L += [f"_Data: {source_note}. Generated by `loadlab.openloop`; every number comes from the CSVs listed in openloop_summary.json._", ""]
    for scen in sorted({c["scenario"] for c in cells}):
        cs = [c for c in cells if c["scenario"] == scen]
        L += [f"## `{scen}`", ""]
        L += ["| prompt | offered req/s | realised arrivals req/s | completed req/s | TTFT p50 / p95 (s) | latency p50 / p95 (s) | peak in flight | in-flight growth over window | TTFT growth (last/first third) | queue |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for c in cs:
            L.append(f"| {c['target_prompt_tokens']} | {c['offered_rps']:g} | {_fmt(c['realised_arrival_rps'], 3)} | {_fmt(c['throughput_rps'], 3)} | "
                     f"{_fmt(c['ttft_p50_s'])} / {_fmt(c['ttft_p95_s'])} | {_fmt(c['latency_p50_s'])} / {_fmt(c['latency_p95_s'])} | {c['max_inflight']} | "
                     f"{_fmt(c['queue_growth_requests'], 1)} | "
                     f"x{_fmt(c['ttft_growth_last_vs_first_third'])} | {'**growing**' if c['queue_growing'] else 'stable'}"
                     + (f" ({c['errors']} errors)" if c["errors"] else "") + " |")
        L.append("")
        for tgt in sorted({c["target_prompt_tokens"] for c in cs}):
            sub = [c for c in cs if c["target_prompt_tokens"] == tgt]
            r = max_sustainable_rate(sub, slo_ttft_s)
            slo = f" and p95 TTFT <= {slo_ttft_s:g}s" if slo_ttft_s else ""
            if r is None:
                L.append(f"- Prompt ~{tgt} tokens: no tested rate kept the queue stable{slo}.")
            else:
                L.append(f"- Prompt ~{tgt} tokens: highest tested rate with a stable queue{slo}: **{r:g} req/s**. "
                         "The true limit lies between this and the next tested rate.")
        L.append("")
    L += ["## How to read this", "",
          f"- `queue growing` = requests in flight rose by at least {QUEUE_GROWTH_MIN:g} (and {QUEUE_GROWTH_FRACTION:.0%} of the cell) "
          f"over the arrival window, or median TTFT of the last third of arrivals is at least x{GROWTH_THRESHOLD:g} the first third. "
          "Arrivals outpace completions. A heuristic on short cells, not a statistical test.",
          "- Engines with parallel slots (Ollama NUM_PARALLEL, vLLM batching) admit several requests at once, so overload shows up "
          "first as more requests in flight and longer total latency, and only later in TTFT.",
          "- Arrivals: seeded Poisson process conditioned on the request count, so the realised rate equals the offered rate "
          "while gaps stay bursty. Cells are short (tens of requests): read trends, not single cells.",
          "- Capacity for planning = the highest rate that stays stable under your TTFT SLO, with headroom; "
          "not the closed-loop peak throughput.", ""]
    path = out_dir / "openloop_memo.md"
    path.write_text("\n".join(L))
    return path


def analyze_open(paths: list[Path], out_dir: Path, source_note: str = "open-loop load test",
                 slo_ttft_s: float | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_open_rows(paths)
    cells = cell_table(rows)
    with (out_dir / "openloop_cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cells[0]))
        w.writeheader()
        w.writerows(cells)
    summary = {"inputs": [str(p) for p in paths], "slo_ttft_s": slo_ttft_s, "cells": cells}
    (out_dir / "openloop_summary.json").write_text(json.dumps(summary, indent=1))
    for scen in sorted({c["scenario"] for c in cells}):
        rs = [r for r in rows if r["scenario"] == scen]
        cs = [c for c in cells if c["scenario"] == scen]
        queue_chart(rs, cs, out_dir / f"{scen}_openloop_queue.png", scen)
        ttft_over_time_chart(rs, cs, out_dir / f"{scen}_openloop_ttft_over_time.png", scen)
        ttft_vs_rate_chart(cs, out_dir / f"{scen}_openloop_ttft_vs_rate.png", scen, slo_ttft_s)
    write_memo(cells, out_dir, source_note, slo_ttft_s)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-note", default="open-loop load test")
    p.add_argument("--slo-ttft", type=float, default=None, help="TTFT SLO in seconds (p95), drawn on the chart")
    a = p.parse_args(argv)
    s = analyze_open(a.csv, a.out_dir, a.source_note, a.slo_ttft)
    for c in s["cells"]:
        print(f"{c['scenario']} prompt~{c['target_prompt_tokens']} rate={c['offered_rps']:g}/s: "
              f"done={_fmt(c['throughput_rps'], 3)}/s ttft_p95={_fmt(c['ttft_p95_s'])}s peak_inflight={c['max_inflight']} "
              f"q+{_fmt(c['queue_growth_requests'], 1)} ttft_growth=x{_fmt(c['ttft_growth_last_vs_first_third'])} {'GROWING' if c['queue_growing'] else 'stable'}")


if __name__ == "__main__":
    main()

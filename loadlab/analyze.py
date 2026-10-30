"""Separate the two usual suspects for slow LLM responses: load vs prompt length.

Input: one or more CSV files written by `loadlab.loadgen` (each row = one request,
tagged with a `scenario`, e.g. one-replica / two-replicas).

Method
------
The load test is a full factorial design: every prompt length is crossed with
every concurrency level. For each scenario we fit

    log(latency) = a + e_p * log(prompt_tokens) + e_c * log(concurrency)

`e_p` and `e_c` are elasticities: doubling the prompt multiplies latency by
2**e_p, doubling concurrency multiplies it by 2**e_c. Because the design is
balanced, the two predictors are (nearly) uncorrelated, so the variance each one
explains on its own adds up and can be compared directly.

We fit the same model for TTFT (time to first token = queueing + prefill) and
report how much of total latency is TTFT. If most of the latency is TTFT and TTFT
scales with prompt length, the bottleneck is prefill on the serving device, not
the number of requests hitting the gateway.

Also: a sweep table (TTFT, latency, decode and aggregate tok/s, ITL), an optional component
bottleneck check (`--component`, two scenarios) and an optional KV-cache capacity estimate (`--kv-*`).

Outputs (in --out-dir): summary.json, cell_summary.csv, PNG charts, decision_memo.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Validated categorical palette (fixed order, never cycled) + light chart surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------- data

def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with Path(p).open() as fh:
            for r in csv.DictReader(fh):
                r["ok"] = r["ok"] in ("True", "true", "1")
                for k in ("target_prompt_tokens", "concurrency", "prompt_tokens", "completion_tokens", "request_index"):
                    r[k] = int(float(r[k] or 0))
                for k in ("ttft_s", "latency_s", "decode_tok_s", "started_at", "itl_p50_s", "itl_p90_s"):
                    # older CSVs have no ITL columns: r.get -> NaN
                    r[k] = float(r[k]) if r.get(k) not in ("", None) else float("nan")
                rows.append(r)
    return rows


@dataclass
class Fit:
    intercept: float
    e_prompt: float
    e_conc: float
    r2_full: float
    r2_prompt_only: float
    r2_conc_only: float
    n: int


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, _r2(y, X @ beta)


def fit_elasticities(prompt_tokens, concurrency, y) -> Fit:
    """OLS of log(y) on log(prompt_tokens) and log(concurrency)."""
    p = np.log(np.asarray(prompt_tokens, dtype=float))
    c = np.log(np.asarray(concurrency, dtype=float))
    ly = np.log(np.asarray(y, dtype=float))
    ones = np.ones_like(p)
    beta, r2 = _ols(np.column_stack([ones, p, c]), ly)
    _, r2_p = _ols(np.column_stack([ones, p]), ly) if np.ptp(p) > 0 else (None, 0.0)
    _, r2_c = _ols(np.column_stack([ones, c]), ly) if np.ptp(c) > 0 else (None, 0.0)
    return Fit(float(beta[0]), float(beta[1]), float(beta[2]), r2, r2_p, r2_c, len(ly))


def cell_table(rows: list[dict]) -> list[dict]:
    """Median metrics per (scenario, target prompt length, concurrency)."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["target_prompt_tokens"], r["concurrency"])].append(r)
    out = []
    for (scen, tgt, conc), rs in sorted(groups.items()):
        ok = [r for r in rs if r["ok"]]
        def med(key):
            vals = [r[key] for r in ok if not math.isnan(r[key])]
            return float(np.median(vals)) if vals else float("nan")
        def p95(key):
            vals = [r[key] for r in ok if not math.isnan(r[key])]
            return float(np.percentile(vals, 95)) if vals else float("nan")
        # Throughput of the cell: completed requests / wall time from first send to last finish.
        if ok:
            wall = max(r["started_at"] + r["latency_s"] for r in ok) - min(r["started_at"] for r in ok)
            rps = len(ok) / wall if wall > 0 else float("nan")
        else:
            rps = float("nan")
        if ok:
            wall_tok = max(r["started_at"] + r["latency_s"] for r in ok) - min(r["started_at"] for r in ok)
            agg_tok = sum(r["completion_tokens"] for r in ok) / wall_tok if wall_tok > 0 else float("nan")
        else:
            agg_tok = float("nan")
        out.append({
            "scenario": scen, "target_prompt_tokens": tgt, "concurrency": conc,
            "n": len(rs), "errors": len(rs) - len(ok),
            "prompt_tokens_median": med("prompt_tokens"),
            "ttft_p50_s": med("ttft_s"), "latency_p50_s": med("latency_s"), "latency_p95_s": p95("latency_s"),
            "decode_tok_s_p50": med("decode_tok_s"),
            "ttft_share_p50": (med("ttft_s") / med("latency_s")) if ok else float("nan"),
            "throughput_rps": rps,
            "aggregate_tok_s": agg_tok,   # completion tokens of all requests / cell wall time
            "itl_p50_s": med("itl_p50_s"),  # median over requests of each request's p50 gap
            "itl_p90_s": med("itl_p90_s"),  # median over requests of each request's p90 gap
        })
    return out


# ----------------------------------------------------------------------- verdict

def classify(lat: Fit, ttft: Fit, ttft_share_longest: float, error_rate: float) -> tuple[str, str]:
    """Return (verdict_code, one-line explanation)."""
    if error_rate > 0.01:
        return "reliability", f"{error_rate:.1%} of requests failed; fix errors before tuning latency."
    if lat.r2_full < 0.3:
        return "inconclusive", "Neither prompt length nor concurrency explains latency well (R^2 < 0.3); look at fixed overheads (network, auth, gateway)."
    if lat.r2_prompt_only >= lat.r2_conc_only:
        if ttft_share_longest >= 0.5:
            return "prefill", ("Latency tracks prompt length, and for the longest prompts most of it is time-to-first-token: "
                               "the bottleneck is prefill on the serving device.")
        return "prompt-decode", "Latency tracks prompt length, but most time is spent after the first token (decode / long outputs)."
    if ttft.e_conc > lat.e_conc * 0.5 and ttft_share_longest >= 0.5:
        return "queueing", ("Latency tracks concurrency and the extra time shows up before the first token: "
                            "requests queue for a saturated serving device.")
    return "decode-contention", "Latency tracks concurrency and the extra time is in decode: batch contention on the serving device."


RECOMMENDATIONS = {
    "prefill": [
        "Do not scale stateless front services (gateway, auth); they are not where the time goes.",
        "Add serving capacity where prefill runs: a second model replica on an idle accelerator, behind the same Service.",
        "Reduce prefill work: trim/cap prompt and RAG context length, enable prefix caching for shared system prompts.",
        "If the engine supports it, enable chunked prefill so long prompts do not stall decode of other requests.",
        "Alert on TTFT p95 split by prompt-length bucket, not only on request rate.",
    ],
    "queueing": [
        "Add replicas of the model server (scale on queue depth / in-flight requests, not CPU).",
        "Set a max in-flight limit per replica and shed or queue at the gateway rather than overloading the engine.",
    ],
    "decode-contention": [
        "Tune batch size / max concurrent sequences; consider a larger accelerator or more replicas.",
        "Cap max_tokens for interactive endpoints.",
    ],
    "prompt-decode": [
        "Cap max_tokens and check whether long prompts also cause long answers (prompt design).",
    ],
    "reliability": ["Investigate error logs and timeouts first; latency numbers with failures are not trustworthy."],
    "inconclusive": ["Trace a single request end to end (gateway, auth, network, engine) to find the fixed cost."],
}


# ------------------------------------------------------------------------- plots

def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def _line_chart(cells, x_key, series_key, title, xlabel, path, series_fmt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    series_vals = sorted({c[series_key] for c in cells})
    for i, sv in enumerate(series_vals):
        pts = sorted((c[x_key], c["latency_p50_s"]) for c in cells if c[series_key] == sv)
        xs, ys = zip(*pts)
        color = SERIES[i % len(SERIES)]
        label = series_fmt(sv)
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({c[x_key] for c in cells}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(bottom=0)
    _style(ax, title, xlabel, "median latency (s)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _ttft_split_chart(cells, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    cells = sorted(cells, key=lambda c: (c["target_prompt_tokens"], c["concurrency"]))
    labels = [f"{c['target_prompt_tokens']}\nc={c['concurrency']}" for c in cells]
    ttft = [c["ttft_p50_s"] for c in cells]
    rest = [max(c["latency_p50_s"] - c["ttft_p50_s"], 0) for c in cells]
    x = np.arange(len(cells))
    ax.bar(x, ttft, width=0.7, color=SERIES[0], label="time to first token (queue + prefill)", edgecolor=SURFACE, linewidth=1)
    ax.bar(x, rest, width=0.7, bottom=ttft, color=SERIES[1], label="after first token (decode)", edgecolor=SURFACE, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    _style(ax, title, "target prompt tokens / concurrency", "median seconds")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _scenario_chart(cells, scenarios, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = sorted({(c["target_prompt_tokens"], c["concurrency"]) for c in cells})
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    width = 0.8 / len(scenarios)
    x = np.arange(len(keys))
    for i, s in enumerate(scenarios):
        lookup = {(c["target_prompt_tokens"], c["concurrency"]): c["latency_p50_s"] for c in cells if c["scenario"] == s}
        ax.bar(x + i * width - 0.4 + width / 2, [lookup.get(k, np.nan) for k in keys], width=width * 0.92,
               color=SERIES[i % len(SERIES)], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\nc={c}" for p, c in keys], fontsize=7)
    _style(ax, "Median latency by scenario", "target prompt tokens / concurrency", "median latency (s)")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)



# ------------------------------------------------------ sweep table, component check

def sweep_table(cells: list[dict], scenario: str) -> str:
    """One markdown table: prompt length x concurrency -> TTFT, latency, decode speed, aggregate speed, ITL."""
    sc = sorted((c for c in cells if c["scenario"] == scenario),
                key=lambda c: (c["target_prompt_tokens"], c["concurrency"]))
    has_itl = any(not math.isnan(c.get("itl_p50_s", float("nan"))) for c in sc)
    head = ["prompt tok", "conc", "p50 TTFT (s)", "p50 total (s)", "decode tok/s per req", "aggregate tok/s"]
    if has_itl:
        head += ["ITL p50 (ms)", "ITL p90 (ms)"]
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    ms = lambda v: "n/a" if v is None or math.isnan(v) else f"{1000 * v:.0f}"
    for c in sc:
        row = [f"{c['prompt_tokens_median']:.0f}", str(c["concurrency"]), _fmt(c["ttft_p50_s"]), _fmt(c["latency_p50_s"]),
               _fmt(c["decode_tok_s_p50"], 1), _fmt(c.get("aggregate_tok_s"), 1)]
        if has_itl:
            row += [ms(c.get("itl_p50_s")), ms(c.get("itl_p90_s"))]
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L)


BOTTLENECK_RULE = (
    "Rule: scale one component (gateway/auth replicas, proxy, ...) and rerun the same matrix. "
    "If tail latency changes but per-request decode tok/s and TTFT-vs-prompt-length stay the same, "
    "that component was not the bottleneck: it only moved queueing around. "
    "If decode tok/s or TTFT at a given prompt length change, it was on the critical path."
)


def _ttft_prompt_slope(cells: list[dict], scenario: str) -> float:
    """Slope of log(TTFT) vs log(prompt tokens) at the lowest concurrency (queue-free)."""
    sc = [c for c in cells if c["scenario"] == scenario]
    cmin = min(c["concurrency"] for c in sc)
    pts = [(c["prompt_tokens_median"], c["ttft_p50_s"]) for c in sc
           if c["concurrency"] == cmin and c["ttft_p50_s"] > 0 and not math.isnan(c["ttft_p50_s"])]
    if len({p for p, _ in pts}) < 2:
        return float("nan")
    x, y = np.log([p for p, _ in pts]), np.log([t for _, t in pts])
    return float(np.polyfit(x, y, 1)[0])


def component_check(cells: list[dict], baseline: str, scaled: str, component: str, tol: float = 0.15) -> dict:
    """Did scaling `component` (baseline -> scaled scenario) change what the model server delivers?

    Compared per matrix cell, median across cells: per-request decode tok/s, p50 TTFT, p95 latency.
    Also the TTFT-vs-prompt-length slope (queue-free, lowest concurrency). `tol` is the relative change
    treated as noise (cells are small; 0.15 is a judgement, not a statistical test).
    Also reports the per-hop overhead: scaled minus baseline p50 TTFT/latency at the lowest concurrency
    (meaningful when `scaled` is "same server, but reached through the extra hop").
    """
    b = {(c["target_prompt_tokens"], c["concurrency"]): c for c in cells if c["scenario"] == baseline}
    n = {(c["target_prompt_tokens"], c["concurrency"]): c for c in cells if c["scenario"] == scaled}
    common = sorted(set(b) & set(n))
    if not common:
        raise ValueError("scenarios share no matrix cells")
    def med_ratio(key):
        r = [n[k][key] / b[k][key] for k in common
             if b[k][key] > 0 and not math.isnan(b[k][key]) and not math.isnan(n[k][key])]
        return float(np.median(r)) if r else float("nan")
    decode_ratio, ttft_ratio, tail_ratio = med_ratio("decode_tok_s_p50"), med_ratio("ttft_p50_s"), med_ratio("latency_p95_s")
    slope_b, slope_n = _ttft_prompt_slope(cells, baseline), _ttft_prompt_slope(cells, scaled)
    cmin = min(k[1] for k in common)
    d_ttft = [n[k]["ttft_p50_s"] - b[k]["ttft_p50_s"] for k in common if k[1] == cmin]
    d_lat = [n[k]["latency_p50_s"] - b[k]["latency_p50_s"] for k in common if k[1] == cmin]
    near = lambda r: not math.isnan(r) and abs(r - 1) <= tol
    decode_same, ttft_same = near(decode_ratio), near(ttft_ratio)
    slope_same = math.isnan(slope_b) or math.isnan(slope_n) or abs(slope_n - slope_b) <= 0.15
    tail_moved = not math.isnan(tail_ratio) and abs(tail_ratio - 1) > tol
    if decode_same and ttft_same and slope_same:
        verdict = "not-bottleneck"
        why = (f"Scaling {component} left per-request decode tok/s (x{decode_ratio:.2f}) and TTFT (x{ttft_ratio:.2f}) unchanged, "
               f"and the TTFT-vs-prompt slope ({slope_b:.2f} -> {slope_n:.2f}). "
               + (f"Tail latency did move (p95 x{tail_ratio:.2f}), but that is queueing shifting, not the cost per request: "
                  if tail_moved else "Tail latency did not move either: ")
               + f"{component} is not the bottleneck.")
    else:
        verdict = "on-critical-path"
        why = (f"Scaling {component} changed per-request speed (decode tok/s x{decode_ratio:.2f}, TTFT x{ttft_ratio:.2f}, "
               f"TTFT-vs-prompt slope {slope_b:.2f} -> {slope_n:.2f}): it sits on the critical path.")
    return {"component": component, "baseline": baseline, "scaled": scaled, "tolerance": tol, "verdict": verdict,
            "why": why, "decode_tok_s_ratio": decode_ratio, "ttft_p50_ratio": ttft_ratio, "latency_p95_ratio": tail_ratio,
            "ttft_prompt_slope": [slope_b, slope_n], "n_cells": len(common),
            "hop_overhead_ttft_s_at_min_concurrency": float(np.median(d_ttft)) if d_ttft else float("nan"),
            "hop_overhead_latency_s_at_min_concurrency": float(np.median(d_lat)) if d_lat else float("nan"),
            "min_concurrency": cmin}


# -------------------------------------------------------------------------- memo

def _fmt(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def build_summary(rows: list[dict]) -> dict:
    cells = cell_table(rows)
    scenarios = sorted({r["scenario"] for r in rows})
    per = {}
    for s in scenarios:
        rs = [r for r in rows if r["scenario"] == s]
        ok = [r for r in rs if r["ok"] and r["latency_s"] > 0 and r["ttft_s"] > 0]
        lat = fit_elasticities([r["prompt_tokens"] for r in ok], [r["concurrency"] for r in ok], [r["latency_s"] for r in ok])
        tt = fit_elasticities([r["prompt_tokens"] for r in ok], [r["concurrency"] for r in ok], [r["ttft_s"] for r in ok])
        sc = [c for c in cells if c["scenario"] == s]
        longest = max(c["target_prompt_tokens"] for c in sc)
        share_longest = float(np.nanmedian([c["ttft_share_p50"] for c in sc if c["target_prompt_tokens"] == longest]))
        err = (len(rs) - len(ok)) / len(rs) if rs else 0.0
        # Saturation: how much does throughput grow from the lowest to the highest concurrency?
        # ~1.0 means extra in-flight requests only queue: the serving device is already full.
        cmin, cmax = min(c["concurrency"] for c in sc), max(c["concurrency"] for c in sc)
        gains = []
        for tgt in sorted({c["target_prompt_tokens"] for c in sc}):
            lo = [c["throughput_rps"] for c in sc if c["target_prompt_tokens"] == tgt and c["concurrency"] == cmin]
            hi = [c["throughput_rps"] for c in sc if c["target_prompt_tokens"] == tgt and c["concurrency"] == cmax]
            if lo and hi and lo[0] > 0:
                gains.append(hi[0] / lo[0])
        throughput_gain = float(np.median(gains)) if gains else float("nan")
        verdict, why = classify(lat, tt, share_longest, err)
        per[s] = {
            "n_requests": len(rs), "n_errors": len(rs) - len(ok), "error_rate": err,
            "latency_fit": lat.__dict__, "ttft_fit": tt.__dict__,
            "ttft_share_at_longest_prompt": share_longest,
            "throughput_gain_min_to_max_concurrency": throughput_gain,
            "concurrency_range": [cmin, cmax],
            "verdict": verdict, "why": why,
        }
    comparison = None
    if len(scenarios) == 2:
        a, b = scenarios
        la = {(c["target_prompt_tokens"], c["concurrency"]): c["latency_p50_s"] for c in cells if c["scenario"] == a}
        lb = {(c["target_prompt_tokens"], c["concurrency"]): c["latency_p50_s"] for c in cells if c["scenario"] == b}
        common = sorted(set(la) & set(lb))
        ratios = {f"{k[0]}x{k[1]}": la[k] / lb[k] for k in common if lb[k] > 0}
        ta = {(c["target_prompt_tokens"], c["concurrency"]): c["throughput_rps"] for c in cells if c["scenario"] == a}
        tb = {(c["target_prompt_tokens"], c["concurrency"]): c["throughput_rps"] for c in cells if c["scenario"] == b}
        cmax = max(k[1] for k in common) if common else 0
        tp_max = [tb[k] / ta[k] for k in common if k[1] == cmax and ta[k] > 0]
        loaded = [la[k] / lb[k] for k in common if k[1] > 1 and lb[k] > 0]
        comparison = {"baseline": a, "candidate": b, "p50_speedup_by_cell": ratios,
                      "throughput_by_cell": {f"{k[0]}x{k[1]}": [ta[k], tb[k]] for k in common},
                      "median_speedup": float(np.median(list(ratios.values()))) if ratios else float("nan"),
                      "median_p50_speedup_concurrency_gt1": float(np.median(loaded)) if loaded else float("nan"),
                      "max_concurrency": cmax,
                      "median_throughput_gain_at_max_concurrency": float(np.median(tp_max)) if tp_max else float("nan")}
    return {"scenarios": per, "cells": cells, "comparison": comparison}


def write_memo(summary: dict, out_dir: Path, source_note: str, baseline: str | None) -> Path:
    per = summary["scenarios"]
    base = baseline if baseline in per else sorted(per)[0]
    s = per[base]
    lat, tt = s["latency_fit"], s["ttft_fit"]
    cells = [c for c in summary["cells"] if c["scenario"] == base]
    L = []
    L.append("# Decision memo: where is the latency?")
    L.append("")
    if "mock" in source_note.lower():
        L.append("> **MOCK DATA.** These numbers come from the simulated server in `loadlab/mock_server.py`, not from a model.")
        L.append("")
    L.append(f"_Data: {source_note}. Generated by `loadlab.analyze`; every number below comes from the CSVs listed in summary.json._")
    L.append("")
    L.append(f"**Scenario analysed:** `{base}`  |  **requests:** {s['n_requests']}  |  **errors:** {s['n_errors']}")
    L.append("")
    L.append("## Finding")
    L.append("")
    L.append(f"**Verdict: `{s['verdict']}`.** {s['why']}")
    L.append("")
    L.append("| driver | elasticity of latency | variance explained alone (R^2, log scale) |")
    L.append("|---|---|---|")
    L.append(f"| prompt length | {_fmt(lat['e_prompt'])} (2x prompt -> x{_fmt(2 ** lat['e_prompt'])}) | {_fmt(lat['r2_prompt_only'])} |")
    L.append(f"| concurrency | {_fmt(lat['e_conc'])} (2x in-flight -> x{_fmt(2 ** lat['e_conc'])}) | {_fmt(lat['r2_conc_only'])} |")
    L.append("")
    L.append(f"TTFT elasticities: prompt {_fmt(tt['e_prompt'])}, concurrency {_fmt(tt['e_conc'])}. "
             f"At the longest prompt, TTFT is {_fmt(100 * s['ttft_share_at_longest_prompt'], 0)}% of total latency (median over concurrency levels).")
    L.append("")
    g = s.get("throughput_gain_min_to_max_concurrency")
    if g is not None and not math.isnan(g):
        cmin, cmax = s["concurrency_range"]
        L.append(f"Throughput changes by x{_fmt(g)} when in-flight requests go from {cmin} to {cmax} (x{cmax // cmin} more load, median over prompt lengths). "
                 + ("The serving device is already saturated: extra requests wait in line, they do not get processed faster."
                    if g < 1.5 else "The device still has headroom at higher concurrency."))
        L.append("")
    c1 = sorted([c for c in cells if c["concurrency"] == min(x["concurrency"] for x in cells)], key=lambda c: c["target_prompt_tokens"])
    if len(c1) >= 2:
        L.append(f"With a single request in flight, going from ~{c1[0]['prompt_tokens_median']:.0f} to ~{c1[-1]['prompt_tokens_median']:.0f} prompt tokens "
                 f"moves median latency from {_fmt(c1[0]['latency_p50_s'])}s to {_fmt(c1[-1]['latency_p50_s'])}s: that cost exists with no load at all.")
        L.append("")
    comp = summary.get("comparison")
    if comp:
        L.append("## Scenario comparison")
        L.append("")
        L.append(f"- Throughput at concurrency {comp['max_concurrency']}: `{comp['candidate']}` completes "
                 f"**x{_fmt(comp['median_throughput_gain_at_max_concurrency'])}** the requests/s of `{comp['baseline']}` (median over prompt lengths).")
        L.append(f"- Median p50 latency speedup where more than one request is in flight: **x{_fmt(comp['median_p50_speedup_concurrency_gt1'])}** "
                 f"(all cells incl. concurrency 1: x{_fmt(comp['median_speedup'])}; a single request cannot get faster from a second replica).")
        L.append("")
        L.append(f"| cell (prompt x concurrency) | p50 speedup | throughput {comp['baseline']} -> {comp['candidate']} (req/s) |")
        L.append("|---|---|---|")
        for k, v in comp["p50_speedup_by_cell"].items():
            t0, t1 = comp["throughput_by_cell"][k]
            L.append(f"| {k} | x{_fmt(v)} | {_fmt(t0, 3)} -> {_fmt(t1, 3)} |")
        L.append("")
    L.append("## Sweep table (prompt length x concurrency)")
    L.append("")
    L.append("Medians per cell. `decode tok/s per req` = what one user sees after the first token; "
             "`aggregate tok/s` = all completion tokens / cell wall time (what the device delivers). "
             "If aggregate stays flat while concurrency grows, the device is saturated; "
             "if ITL p90 grows faster than p50, requests are interfering in decode.")
    L.append("")
    for sc_name in sorted(per):
        if len(per) > 1:
            L.append(f"**{sc_name}**")
            L.append("")
        L.append(sweep_table(summary["cells"], sc_name))
        L.append("")
    L.append("## Which component is the bottleneck?")
    L.append("")
    L.append(BOTTLENECK_RULE)
    L.append("")
    cc = summary.get("component_check")
    if cc:
        L.append(f"**Check on this data** (`{cc['baseline']}` -> `{cc['scaled']}`, component: {cc['component']}): "
                 f"**{cc['verdict']}**. {cc['why']}")
        L.append("")
        L.append(f"Per-hop overhead at concurrency {cc['min_concurrency']} (scaled minus baseline, median over prompt lengths): "
                 f"TTFT {_fmt(cc['hop_overhead_ttft_s_at_min_concurrency'], 3)} s, total {_fmt(cc['hop_overhead_latency_s_at_min_concurrency'], 3)} s "
                 "(only an overhead if `scaled` is the same server reached through an extra hop).")
        L.append("")
    kv = summary.get("kv_cache")
    if kv:
        from .kvcache import format_estimate
        L.append("## KV-cache capacity estimate")
        L.append("")
        L.append(format_estimate(kv) + ". Upper bound: ignores fragmentation and prefix sharing. "
                 "If the number is far below your concurrency target, decode batches will be capped by memory, not compute.")
        L.append("")
    L.append("## Recommended action")
    L.append("")
    for i, rec in enumerate(RECOMMENDATIONS[s["verdict"]], 1):
        L.append(f"{i}. {rec}")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- Elasticities come from a log-log OLS fit on a balanced factorial design; they summarise, they do not prove causality.")
    L.append("- Closed-loop load: concurrency = requests in flight, not requests per second.")
    L.append("- Results are specific to the hardware, engine and model in the run metadata (results/*.json).")
    L.append("")
    path = out_dir / "decision_memo.md"
    path.write_text("\n".join(L))
    return path


def analyze(paths: list[Path], out_dir: Path, source_note: str = "load test", baseline: str | None = None,
            component: str | None = None, kv_cache: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(paths)
    summary = build_summary(rows)
    if component:
        names = sorted(summary["scenarios"])
        if len(names) != 2:
            raise SystemExit("--component needs exactly two scenarios (baseline and scaled)")
        base = baseline if baseline in names else names[0]
        scaled = [x for x in names if x != base][0]
        summary["component_check"] = component_check(summary["cells"], base, scaled, component)
    if kv_cache:
        summary["kv_cache"] = kv_cache
    summary["inputs"] = [str(p) for p in paths]
    with (out_dir / "cell_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary["cells"][0].keys()))
        w.writeheader()
        w.writerows(summary["cells"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    scenarios = sorted(summary["scenarios"])
    for s in scenarios:
        cells = [c for c in summary["cells"] if c["scenario"] == s]
        _line_chart(cells, "target_prompt_tokens", "concurrency", f"[{s}] latency vs prompt length",
                    "target prompt tokens (log2)", out_dir / f"{s}_latency_vs_prompt.png", lambda v: f"c={v}")
        _line_chart(cells, "concurrency", "target_prompt_tokens", f"[{s}] latency vs concurrency",
                    "requests in flight (log2)", out_dir / f"{s}_latency_vs_concurrency.png", lambda v: f"{v} tok")
        _ttft_split_chart(cells, out_dir / f"{s}_ttft_vs_decode.png", f"[{s}] where the time goes")
    if len(scenarios) > 1:
        _scenario_chart(summary["cells"], scenarios, out_dir / "scenario_comparison.png")
    write_memo(summary, out_dir, source_note, baseline)
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-note", default="load test")
    p.add_argument("--baseline", default=None, help="scenario the memo's main finding is about")
    p.add_argument("--component", default=None,
                   help="name of the component scaled between the two scenarios (e.g. gateway); adds the bottleneck check to the memo")
    from .kvcache import add_cli_args, from_args
    add_cli_args(p, prefix="kv-")
    a = p.parse_args(argv)
    summary = analyze(a.csv, a.out_dir, a.source_note, a.baseline, a.component, from_args(a, prefix="kv-"))
    for name, s in summary["scenarios"].items():
        f = s["latency_fit"]
        print(f"{name}: verdict={s['verdict']} e_prompt={f['e_prompt']:.2f} e_conc={f['e_conc']:.2f} "
              f"R2(prompt)={f['r2_prompt_only']:.2f} R2(conc)={f['r2_conc_only']:.2f} errors={s['n_errors']}")
    if summary["comparison"]:
        c = summary["comparison"]
        print(f"{c['candidate']} vs {c['baseline']}: throughput gain at c={c['max_concurrency']} "
              f"x{c['median_throughput_gain_at_max_concurrency']:.2f}, p50 speedup (c>1) x{c['median_p50_speedup_concurrency_gt1']:.2f}")
    if summary.get("component_check"):
        print("component check:", summary["component_check"]["verdict"], "-", summary["component_check"]["why"])
    for name in sorted(summary["scenarios"]):
        print(f"\n[{name}] sweep table\n" + sweep_table(summary["cells"], name))
    print(f"memo: {a.out_dir / 'decision_memo.md'}")


if __name__ == "__main__":
    main()

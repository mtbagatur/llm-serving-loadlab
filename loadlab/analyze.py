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
                for k in ("ttft_s", "latency_s", "decode_tok_s", "started_at"):
                    r[k] = float(r[k]) if r[k] not in ("", None) else float("nan")
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
        out.append({
            "scenario": scen, "target_prompt_tokens": tgt, "concurrency": conc,
            "n": len(rs), "errors": len(rs) - len(ok),
            "prompt_tokens_median": med("prompt_tokens"),
            "ttft_p50_s": med("ttft_s"), "latency_p50_s": med("latency_s"), "latency_p95_s": p95("latency_s"),
            "decode_tok_s_p50": med("decode_tok_s"),
            "ttft_share_p50": (med("ttft_s") / med("latency_s")) if ok else float("nan"),
            "throughput_rps": rps,
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


def analyze(paths: list[Path], out_dir: Path, source_note: str = "load test", baseline: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(paths)
    summary = build_summary(rows)
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
    a = p.parse_args(argv)
    summary = analyze(a.csv, a.out_dir, a.source_note, a.baseline)
    for name, s in summary["scenarios"].items():
        f = s["latency_fit"]
        print(f"{name}: verdict={s['verdict']} e_prompt={f['e_prompt']:.2f} e_conc={f['e_conc']:.2f} "
              f"R2(prompt)={f['r2_prompt_only']:.2f} R2(conc)={f['r2_conc_only']:.2f} errors={s['n_errors']}")
    if summary["comparison"]:
        c = summary["comparison"]
        print(f"{c['candidate']} vs {c['baseline']}: throughput gain at c={c['max_concurrency']} "
              f"x{c['median_throughput_gain_at_max_concurrency']:.2f}, p50 speedup (c>1) x{c['median_p50_speedup_concurrency_gt1']:.2f}")
    print(f"memo: {a.out_dir / 'decision_memo.md'}")


if __name__ == "__main__":
    main()

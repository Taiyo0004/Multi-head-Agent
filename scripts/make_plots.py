#!/usr/bin/env python3
"""Render the README figures from results/torch/*.json.

Each figure is written for light and dark themes so the README can serve
the right one with a <picture> element. Colours are a CVD-checked
categorical set used in fixed order.

    python scripts/make_plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "torch"
FIGS = RESULTS / "figures"

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e2e1dd",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "grid": "#383835",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}
MODE_LABEL = {
    "baseline": "baseline",
    "shared_kv": "shared KV",
    "latent": "latent",
    "shared_kv+latent": "shared KV\n+ latent",
}
EVAL_LABEL = {
    "direct": "base model\n(no pipeline)",
    "pipeline_text": "text\nhand-off",
    "pipeline_shared_kv": "shared KV",
    "pipeline_latent": "latent\nhand-off",
    "pipeline_fast": "shared KV\n+ latent",
}


def style(ax, t, *, xlabel="", ylabel="", title="", grid_axis="y"):
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    ax.grid(True, axis=grid_axis, color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["ink2"], fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=t["ink2"], fontsize=10)
    if title:
        ax.set_title(title, color=t["ink"], fontsize=12, fontweight="bold", loc="left", pad=12)


def save(fig, name, theme):
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"{name}.{theme}.png"
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def fig_accuracy_vs_cost(ev: Dict, theme: str) -> None:
    """The headline: what each optimisation costs in accuracy."""
    t = THEME[theme]
    order = ["direct", "pipeline_text", "pipeline_shared_kv", "pipeline_latent", "pipeline_fast"]
    acc = [ev[k]["accuracy_pct"] for k in order]
    n_ok = [ev[k]["n_correct"] for k in order]
    n = ev["pipeline_text"]["n"]
    colors = [t["ink2"], t["series"][0], t["series"][1], t["series"][2], t["series"][3]]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    bars = ax.bar([EVAL_LABEL[k] for k in order], acc, 0.56, color=colors, zorder=3)
    for rect, a, c in zip(bars, acc, n_ok):
        ax.annotate(f"{a:.1f}%\n{c}/{n}", (rect.get_x() + rect.get_width() / 2, a),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=t["ink2"])
    ax.set_ylim(0, max(acc) * 1.3)
    style(ax, t, ylabel="GSM8K accuracy (%)",
          title="What each optimisation costs in accuracy")
    ax.text(0.015, 0.94, "Qwen2.5-0.5B-Instruct, 200 held-out questions",
            transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="top")
    save(fig, "01_accuracy_by_mode", theme)


def fig_cost_breakdown(summary, theme: str) -> None:
    """Producer decode steps and prefill tokens, at fixed decode budgets."""
    t = THEME[theme]
    modes = [s["mode"] for s in summary]
    prod = [s["producer_decode_steps"] for s in summary]
    enc = [s["tokens_encoded"] for s in summary]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.9))
    for ax, vals, color, label, title in (
        (ax1, prod, t["series"][0], "producer decode steps",
         "Latent hand-off: producers stop writing prose"),
        (ax2, enc, t["series"][1], "prefill tokens encoded",
         "Prefix sharing: the prompt is encoded once"),
    ):
        bars = ax.bar([MODE_LABEL[m] for m in modes], vals, 0.56, color=color, zorder=3)
        for rect, v in zip(bars, vals):
            ax.annotate(f"{v:.0f}", (rect.get_x() + rect.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 4), ha="center",
                        fontsize=9, color=t["ink2"])
        ax.set_ylim(0, max(vals) * 1.22)
        style(ax, t, ylabel=label, title=title)
        ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    save(fig, "02_cost_breakdown", theme)


def fig_fork(fork: Dict, theme: str) -> None:
    """Reference fork against HuggingFace's deepcopy recipe."""
    t = THEME[theme]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6))
    labels = ["copy.deepcopy\n(HF recipe)", "fork by\nreference"]

    ms = [fork["deepcopy_seconds"] * 1000, fork["fork_seconds"] * 1000]
    bars = ax1.bar(labels, ms, 0.5, color=[t["ink2"], t["series"][0]], zorder=3)
    for rect, v in zip(bars, ms):
        ax1.annotate(f"{v:.2f} ms", (rect.get_x() + rect.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=9, color=t["ink2"])
    ax1.set_ylim(0, max(ms) * 1.25)
    style(ax1, t, ylabel="time to make 3 forks (ms)",
          title=f"{fork['fork_speedup_x']:.0f}x faster")

    mb = [fork["deepcopy_bytes"] / 2**20, fork["fork_bytes"] / 2**20]
    bars = ax2.bar(labels, mb, 0.5, color=[t["ink2"], t["series"][1]], zorder=3)
    for rect, v in zip(bars, mb):
        ax2.annotate(f"{v:.1f} MiB", (rect.get_x() + rect.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 4), ha="center",
                     fontsize=9, color=t["ink2"])
    ax2.set_ylim(0, max(mb) * 1.25)
    style(ax2, t, ylabel="distinct KV storage (MiB)",
          title=f"{fork['memory_reduction_pct']:.0f}% less memory")
    fig.suptitle(f"Forking a {fork['prefix_tokens']}-token prefix cache, 3 heads",
                 color=t["ink"], fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, "03_cache_fork", theme)


def main() -> None:
    bench_path, eval_path = RESULTS / "bench_summary.json", RESULTS / "gsm8k_eval.json"
    if not bench_path.exists() or not eval_path.exists():
        sys.exit("run mha_torch.bench and mha_torch.evaluate first")
    bench = json.loads(bench_path.read_text())
    ev = json.loads(eval_path.read_text())["summary"]
    for theme in ("light", "dark"):
        fig_accuracy_vs_cost(ev, theme)
        fig_cost_breakdown(bench["summary"], theme)
        fig_fork(bench["cache_fork_vs_deepcopy"], theme)


if __name__ == "__main__":
    main()

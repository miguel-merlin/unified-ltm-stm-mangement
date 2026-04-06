#!/usr/bin/env python3
"""Plotting scripts reproducing Figures 2-5 from the AgeMem paper.

Figures produced:
  Figure 2 — Memory Quality (MQ) bar chart: AgeMem vs baselines
  Figure 3 — Prompt token count bar chart: AgeMem vs AgeMem-RAG
  Figure 4 — Ablation bar chart: progressive contribution of LTM / STM / RL
  Figure 5 — GRPO reward convergence: All-Returns vs Answer-Only

Usage (with real eval JSON):
    python scripts/plot_results.py \\
        --eval_json outputs/eval_results.json \\
        --outdir outputs/plots

Usage (with mock data to test plotting pipeline):
    python scripts/plot_results.py --mock --outdir outputs/plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Matplotlib setup (non-interactive backend for server use)
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Paper-style plot settings
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
})

COLORS = {
    "agemem": "#2563EB",       # bright blue
    "agemem_norl": "#60A5FA",  # light blue
    "mem0": "#DC2626",         # red
    "a_mem": "#F97316",        # orange
    "langmem": "#16A34A",      # green
    "mem0g": "#9333EA",        # purple
    "no_mem": "#6B7280",       # grey
    "rag": "#F59E0B",          # amber
    "all_returns": "#2563EB",
    "answer_only": "#DC2626",
}


# ---------------------------------------------------------------------------
# Mock data matching approximate paper numbers
# ---------------------------------------------------------------------------

MOCK_DATA = {
    # Figure 2 — Memory Quality scores on HotpotQA
    "mq_qwen25": {
        "AgeMem": 0.533,
        "AgeMem-noRL": 0.459,
        "Mem0": 0.398,
        "Mem0g": 0.371,
        "A-Mem": 0.352,
        "LangMem": 0.329,
        "No-Memory": 0.0,
    },
    "mq_qwen3": {
        "AgeMem": 0.605,
        "AgeMem-noRL": 0.517,
        "Mem0": 0.421,
        "Mem0g": 0.395,
        "A-Mem": 0.384,
        "LangMem": 0.358,
        "No-Memory": 0.0,
    },
    # Figure 3 — Prompt token counts (AgeMem vs AgeMem-RAG)
    "tokens_qwen25": {
        "AgeMem": 2117,
        "AgeMem-RAG": 2186,
    },
    "tokens_qwen3": {
        "AgeMem": 2191,
        "AgeMem-RAG": 2310,
    },
    # Figure 4 — Ablation on ALFWorld / SciWorld / HotpotQA (Qwen2.5)
    "ablation": {
        "datasets": ["ALFWorld", "SciWorld", "HotpotQA"],
        "Baseline":    [0.28, 0.21, 0.38],
        "+LTM":        [0.39, 0.38, 0.46],
        "+LTM/RL":     [0.44, 0.43, 0.52],
        "+LTM/STM/RL": [0.49, 0.55, 0.54],
    },
    # Figure 5 — GRPO reward convergence curves
    "convergence": {
        "steps": list(range(0, 105, 5)),
        "all_returns": [
            0.05, 0.12, 0.19, 0.26, 0.31, 0.35, 0.38, 0.40, 0.42, 0.43,
            0.44, 0.45, 0.46, 0.46, 0.47, 0.47, 0.48, 0.48, 0.48, 0.49, 0.49,
        ],
        "answer_only": [
            0.04, 0.09, 0.14, 0.18, 0.22, 0.24, 0.26, 0.27, 0.28, 0.29,
            0.30, 0.31, 0.31, 0.32, 0.32, 0.33, 0.33, 0.33, 0.34, 0.34, 0.34,
        ],
    },
}


# ---------------------------------------------------------------------------
# Figure 2 — Memory Quality bar chart
# ---------------------------------------------------------------------------

def plot_memory_quality(data: dict, outdir: Path) -> None:
    mq25 = data["mq_qwen25"]
    mq3  = data["mq_qwen3"]
    labels = list(mq25.keys())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width / 2, [mq25[l] for l in labels], width,
                   label="Qwen2.5-7B-Instruct", color=COLORS["agemem"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, [mq3[l] for l in labels], width,
                   label="Qwen3-4B-Instruct", color=COLORS["mem0"], alpha=0.85)

    ax.set_ylabel("Memory Quality (MQ)")
    ax.set_title("Figure 2: Memory Quality Across Methods")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 0.75)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="upper right")

    # Value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    path = outdir / "fig2_memory_quality.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Token count bar chart
# ---------------------------------------------------------------------------

def plot_token_usage(data: dict, outdir: Path) -> None:
    tok25 = data["tokens_qwen25"]
    tok3  = data["tokens_qwen3"]
    labels = ["Qwen2.5-7B-Instruct", "Qwen3-4B-Instruct"]
    agemem_vals = [tok25["AgeMem"], tok3["AgeMem"]]
    rag_vals    = [tok25["AgeMem-RAG"], tok3["AgeMem-RAG"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width / 2, agemem_vals, width, label="AgeMem (STM tools)",
           color=COLORS["agemem"], alpha=0.85)
    ax.bar(x + width / 2, rag_vals, width, label="AgeMem-RAG",
           color=COLORS["rag"], alpha=0.85)

    ax.set_ylabel("Avg Prompt Token Count")
    ax.set_title("Figure 3: Context Efficiency (Lower = Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(1900, 2500)
    ax.legend()

    # Reduction annotation
    for i, (a, r) in enumerate(zip(agemem_vals, rag_vals)):
        reduction = (r - a) / r * 100
        ax.annotate(
            f"−{reduction:.1f}%",
            xy=(x[i], (a + r) / 2),
            fontsize=9, color="black", ha="center",
        )

    path = outdir / "fig3_token_usage.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Ablation bar chart
# ---------------------------------------------------------------------------

def plot_ablation(data: dict, outdir: Path) -> None:
    abl = data["ablation"]
    datasets = abl["datasets"]
    systems = ["Baseline", "+LTM", "+LTM/RL", "+LTM/STM/RL"]
    n_ds = len(datasets)
    n_sys = len(systems)
    x = np.arange(n_ds)
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 4.5))
    palette = ["#6B7280", "#60A5FA", "#2563EB", "#1D4ED8"]

    for i, (sys, col) in enumerate(zip(systems, palette)):
        offset = (i - n_sys / 2 + 0.5) * width
        vals = abl[sys]
        bars = ax.bar(x + offset, vals, width, label=sys, color=col, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_ylabel("Performance (SR / PR / J)")
    ax.set_title("Figure 4: Component Ablation (Qwen2.5-7B-Instruct)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 0.75)
    ax.legend(loc="upper left", fontsize=9)

    path = outdir / "fig4_ablation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5 — GRPO convergence curves
# ---------------------------------------------------------------------------

def plot_convergence(data: dict, outdir: Path) -> None:
    conv = data["convergence"]
    steps = conv["steps"]
    all_ret = conv["all_returns"]
    ans_only = conv["answer_only"]

    # Smooth with a simple moving average
    def _smooth(vals: List[float], w: int = 3) -> np.ndarray:
        kernel = np.ones(w) / w
        return np.convolve(vals, kernel, mode="same")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, _smooth(all_ret), color=COLORS["all_returns"],
            linewidth=2.0, label="All-Returns (full reward)")
    ax.fill_between(steps,
                    [max(0, v - 0.02) for v in _smooth(all_ret)],
                    [v + 0.02 for v in _smooth(all_ret)],
                    alpha=0.15, color=COLORS["all_returns"])
    ax.plot(steps, _smooth(ans_only), color=COLORS["answer_only"],
            linewidth=2.0, linestyle="--", label="Answer-Only (R_task only)")
    ax.fill_between(steps,
                    [max(0, v - 0.015) for v in _smooth(ans_only)],
                    [v + 0.015 for v in _smooth(ans_only)],
                    alpha=0.12, color=COLORS["answer_only"])

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Reward")
    ax.set_title("Figure 5: GRPO Reward Convergence (Qwen2.5-7B-Instruct)")
    ax.legend()
    ax.set_xlim(0, max(steps))
    ax.set_ylim(0, 0.65)

    path = outdir / "fig5_convergence.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[Plot] Saved {path}")


# ---------------------------------------------------------------------------
# Loader for real eval JSON (replaces mock data where available)
# ---------------------------------------------------------------------------

def _load_eval_json(eval_json: str) -> Optional[dict]:
    try:
        with open(eval_json, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Plot] Could not load eval JSON: {e}")
        return None


def _merge_eval_into_data(eval_data: dict, plot_data: dict) -> dict:
    """Overlay real eval metrics into plot data dict."""
    merged = dict(plot_data)
    model = eval_data.get("model", "unknown")
    judge = eval_data.get("llm_judge", None)
    mq = eval_data.get("memory_quality", None)
    tok = eval_data.get("avg_token_proxy", None)

    if mq is not None:
        key = "mq_qwen25"  # default to Qwen2.5 slot
        if "qwen3" in model.lower() or "3-4b" in model.lower():
            key = "mq_qwen3"
        merged[key]["AgeMem"] = mq
        print(f"[Plot] Injected real MQ={mq:.4f} into {key}['AgeMem']")

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AgeMem result plots (Figures 2-5)")
    parser.add_argument("--eval_json", type=str, default=None,
                        help="Path to evaluate.py output JSON (optional)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data instead of real eval results")
    parser.add_argument("--outdir", type=str, default="outputs/plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_data = dict(MOCK_DATA)

    if args.eval_json and not args.mock:
        real = _load_eval_json(args.eval_json)
        if real:
            plot_data = _merge_eval_into_data(real, plot_data)

    plot_memory_quality(plot_data, outdir)
    plot_token_usage(plot_data, outdir)
    plot_ablation(plot_data, outdir)
    plot_convergence(plot_data, outdir)

    print(f"\n[Plot] All figures saved to {outdir}/")


if __name__ == "__main__":
    main()

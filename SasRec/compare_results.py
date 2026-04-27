"""
Comparison plots for all QAT experiments on SASRec.
Run from SasRec/ directory:  python compare_results.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


RESULTS_DIR = Path("results")
SAVE_DIR = Path("results")

# ── Collected test-split metrics (from individual result JSONs) ──────────────
ALL_RUNS: dict[str, dict] = {
    # FP32 baselines
    "FP32\n(base)":          {"method": "FP32",     "ndcg": 0.01022, "hit": 0.02036},
    "FP32\n(large)":         {"method": "FP32",     "ndcg": 0.00882, "hit": 0.01854},
    "FP32\n(low_drop)":      {"method": "FP32",     "ndcg": 0.00637, "hit": 0.01275},
    # LSQ
    "LSQ\n(8-bit sym)":      {"method": "LSQ",      "ndcg": 0.00314, "hit": 0.00778},
    "LSQ\n(8-bit run)":      {"method": "LSQ",      "ndcg": 0.00400, "hit": 0.00977},
    "LSQ\n(8-bit asym)":     {"method": "LSQ",      "ndcg": 0.00434, "hit": 0.01043},
    "LSQ\n(4-bit asym)":     {"method": "LSQ",      "ndcg": 0.00334, "hit": 0.00795},
    # APoT
    "APoT\n(8-bit)":         {"method": "APoT",     "ndcg": 0.00904, "hit": 0.01937},
    "APoT\n(per-ch)":        {"method": "APoT",     "ndcg": 0.00904, "hit": 0.01937},
    "APoT\n(4-bit)":         {"method": "APoT",     "ndcg": 0.00986, "hit": 0.02136},
    # QDrop
    "QDrop\n(8-bit p=0.5)":  {"method": "QDrop",    "ndcg": 0.01046, "hit": 0.02152},
    "QDrop\n(8-bit p=0.8)":  {"method": "QDrop",    "ndcg": 0.01026, "hit": 0.02103},
    "QDrop\n(4-bit p=0.5)":  {"method": "QDrop",    "ndcg": 0.01158, "hit": 0.02235},
    # AdaRound
    "AdaRound\n(8-bit)":     {"method": "AdaRound", "ndcg": 0.01034, "hit": 0.02086},
    "AdaRound\n(4-bit)":     {"method": "AdaRound", "ndcg": 0.01030, "hit": 0.02086},
    "AdaRound\n(v2 8-bit)":  {"method": "AdaRound", "ndcg": 0.01034, "hit": 0.02086},
}

BENCHMARK: list[dict] = [
    {"method": "FP32",     "ndcg_val": 0.01279, "hit_val": 0.02731,
     "throughput": 12.63,  "avg_lat_ms": 79.2,  "size_mb": 3.54},
    {"method": "LSQ",      "ndcg_val": 0.00449, "hit_val": 0.01010,
     "throughput": 13.34,  "avg_lat_ms": 75.0,  "size_mb": 3.57},
    {"method": "APoT",     "ndcg_val": 0.00706, "hit_val": 0.01507,
     "throughput": 12.99,  "avg_lat_ms": 77.0,  "size_mb": 3.55},
    {"method": "QDrop",    "ndcg_val": 0.01324, "hit_val": 0.02815,
     "throughput": 14.55,  "avg_lat_ms": 68.7,  "size_mb": 3.56},
    {"method": "AdaRound", "ndcg_val": 0.01279, "hit_val": 0.02731,
     "throughput": 14.12,  "avg_lat_ms": 70.8,  "size_mb": 1.38},
]

METHOD_COLORS = {
    "FP32":     "#4C72B0",
    "LSQ":      "#DD8452",
    "APoT":     "#55A868",
    "QDrop":    "#C44E52",
    "AdaRound": "#8172B2",
}

BEST_PER_METHOD = {
    "FP32":     {"ndcg": 0.01022, "hit": 0.02036},
    "LSQ":      {"ndcg": 0.00434, "hit": 0.01043},
    "APoT":     {"ndcg": 0.00986, "hit": 0.02136},
    "QDrop":    {"ndcg": 0.01158, "hit": 0.02235},
    "AdaRound": {"ndcg": 0.01034, "hit": 0.02086},
}


# ── Plot 1: best-per-method NDCG & Hit bar chart ─────────────────────────────
def plot_best_per_method(save_path: Path) -> None:
    methods = list(BEST_PER_METHOD)
    ndcg_vals = [BEST_PER_METHOD[m]["ndcg"] for m in methods]
    hit_vals  = [BEST_PER_METHOD[m]["hit"]  for m in methods]
    colors    = [METHOD_COLORS[m] for m in methods]

    x = np.arange(len(methods))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Best result per QAT method  (SASRec · ML-1M · test split)",
                 fontsize=13, fontweight="bold")

    for ax, vals, metric in zip(axes, [ndcg_vals, hit_vals], ["NDCG@10", "Hit@10"]):
        bars = ax.bar(x, vals, width=0.55, color=colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        fp32_line = vals[methods.index("FP32")]
        ax.axhline(fp32_line, color="#4C72B0", linestyle="--", linewidth=1.2,
                   label=f"FP32 baseline = {fp32_line:.4f}")

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.012,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=10)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"{metric} — best run per method", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.22)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 2: all runs scatter (hyperparameter effect) ─────────────────────────
def plot_all_runs_scatter(save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("All hyperparameter runs  (SASRec · ML-1M · test NDCG@10 & Hit@10)",
                 fontsize=12, fontweight="bold")

    run_labels = list(ALL_RUNS.keys())
    methods_seq = [ALL_RUNS[r]["method"] for r in run_labels]

    for ax, metric_key, ylabel in zip(
        axes,
        ["ndcg", "hit"],
        ["NDCG@10", "Hit@10"],
    ):
        vals   = [ALL_RUNS[r][metric_key] for r in run_labels]
        colors = [METHOD_COLORS[m] for m in methods_seq]
        x_pos  = np.arange(len(run_labels))
        bars   = ax.bar(x_pos, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.6)

        fp32_best = BEST_PER_METHOD["FP32"][metric_key]
        ax.axhline(fp32_best, color="#4C72B0", linestyle="--", linewidth=1.2,
                   label=f"FP32 best = {fp32_best:.4f}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_labels, rotation=45, ha="right", fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — all runs")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(vals) * 1.22)

    legend_patches = [mpatches.Patch(color=METHOD_COLORS[m], label=m) for m in METHOD_COLORS]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 3: speed vs quality (benchmark) ─────────────────────────────────────
def plot_speed_vs_quality(save_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CPU fake-quant benchmark: speed vs quality  (val split)",
                 fontsize=12, fontweight="bold")

    for ax, x_key, xlabel in zip(
        axes,
        ["throughput", "avg_lat_ms"],
        ["Throughput (batches / sec)", "Avg batch latency (ms)"],
    ):
        for rec in BENCHMARK:
            m   = rec["method"]
            xv  = rec[x_key]
            yv  = rec["ndcg_val"]
            col = METHOD_COLORS[m]
            ax.scatter(xv, yv, s=140, color=col, zorder=3, edgecolors="white", linewidths=0.8)
            ax.annotate(m, (xv, yv), textcoords="offset points",
                        xytext=(6, 4), fontsize=9, color=col)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("NDCG@10 (val)", fontsize=10)
        ax.set_title(f"NDCG@10 vs {xlabel}")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 4: QAT degradation heatmap relative to FP32 ────────────────────────
def plot_degradation_table(save_path: Path) -> None:
    fp32_ndcg = BEST_PER_METHOD["FP32"]["ndcg"]
    fp32_hit  = BEST_PER_METHOD["FP32"]["hit"]

    methods  = [m for m in BEST_PER_METHOD if m != "FP32"]
    ndcg_rel = [(BEST_PER_METHOD[m]["ndcg"] - fp32_ndcg) / fp32_ndcg * 100 for m in methods]
    hit_rel  = [(BEST_PER_METHOD[m]["hit"]  - fp32_hit)  / fp32_hit  * 100 for m in methods]

    x  = np.arange(len(methods))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))

    b1 = ax.bar(x - w / 2, ndcg_rel, width=w, label="NDCG@10 Δ%",
                color=[METHOD_COLORS[m] for m in methods], alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + w / 2, hit_rel,  width=w, label="Hit@10 Δ%",
                color=[METHOD_COLORS[m] for m in methods], alpha=0.55, edgecolor="white",
                hatch="//")

    ax.axhline(0, color="black", linewidth=1.0)
    for bar, v in [*zip(b1, ndcg_rel), *zip(b2, hit_rel)]:
        offset = 0.4 if v >= 0 else -1.2
        ax.text(bar.get_x() + bar.get_width() / 2, v + offset,
                f"{v:+.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Relative change vs FP32 (%)", fontsize=11)
    ax.set_title("QAT quality degradation vs FP32 baseline  (best run per method)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── Plot 5: training curves overlay for QAT runs ─────────────────────────────
def _load_val_ndcg_history(json_path: Path) -> list[float] | None:
    if not json_path.exists():
        return None
    with open(json_path) as f:
        data = json.load(f)
    if "history" not in data:
        return None
    return data["history"].get("val_ndcg")


def plot_training_curves_overlay(save_path: Path) -> None:
    curve_files = {
        "QDrop 8-bit (p=0.5)":  RESULTS_DIR / "sasrec_qdrop_results.json",
        "QDrop 4-bit (p=0.5)":  RESULTS_DIR / "sasrec_qdrop_4bit_results.json",
        "QDrop 8-bit (p=0.8)":  RESULTS_DIR / "sasrec_qdrop_highp_results.json",
        "APoT 8-bit":            RESULTS_DIR / "sasrec_apot_results.json",
        "APoT 4-bit":            RESULTS_DIR / "sasrec_apot_4bit_results.json",
        "LSQ 8-bit asym":        RESULTS_DIR / "sasrec_lsq_asym_results.json",
        "FP32 baseline":         RESULTS_DIR / "sasrec_fp32_results.json",
    }
    curve_colors = {
        "QDrop 8-bit (p=0.5)":  "#C44E52",
        "QDrop 4-bit (p=0.5)":  "#E07070",
        "QDrop 8-bit (p=0.8)":  "#F4A0A0",
        "APoT 8-bit":            "#55A868",
        "APoT 4-bit":            "#88CC88",
        "LSQ 8-bit asym":        "#DD8452",
        "FP32 baseline":         "#4C72B0",
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False
    for label, path in curve_files.items():
        hist = _load_val_ndcg_history(path)
        if hist:
            ax.plot(range(1, len(hist) + 1), hist,
                    label=label, color=curve_colors.get(label, "grey"),
                    linewidth=1.8, alpha=0.85)
            any_plotted = True

    if any_plotted:
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("NDCG@10 (val)", fontsize=11)
        ax.set_title("Validation NDCG@10 training curves", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8.5, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.close(fig)
        print("No training history data found; skipping training curves overlay.")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    plot_best_per_method(SAVE_DIR / "comparison_best_per_method.png")
    plot_all_runs_scatter(SAVE_DIR / "comparison_all_runs.png")
    plot_speed_vs_quality(SAVE_DIR / "comparison_speed_vs_quality.png")
    plot_degradation_table(SAVE_DIR / "comparison_degradation.png")
    plot_training_curves_overlay(SAVE_DIR / "comparison_training_curves.png")

    print("\n=== Summary: Best NDCG@10 (test) per method ===")
    fp32 = BEST_PER_METHOD["FP32"]["ndcg"]
    for m, v in sorted(BEST_PER_METHOD.items(), key=lambda kv: -kv[1]["ndcg"]):
        delta = (v["ndcg"] - fp32) / fp32 * 100
        winner = " ← BEST QAT" if m == "QDrop" else ""
        print(f"  {m:12s}  NDCG@10 = {v['ndcg']:.5f}  ({delta:+.1f}% vs FP32){winner}")


if __name__ == "__main__":
    main()

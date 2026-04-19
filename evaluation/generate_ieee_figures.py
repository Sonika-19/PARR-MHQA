"""
IEEE-style figure generation for PARR-MHQA evaluation results.
Matches the aesthetic of sample figures: bold colors, clean bars, annotated values,
rotated x-labels, dashed grid, and no PDF output.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path("figures")

# ── Colour palette (sampled from the provided reference figures) ────────────
PALETTE = {
    "llm_only":       "#4472C4",   # steel blue  (Logistic Regression blue)
    "heuristic":      "#5BAD6F",   # forest green (Random Forest green)
    "learned":        "#E6A030",   # amber/gold   (SVM orange)
    "parr_mhqa":      "#D9534F",   # coral red    (LineageSentinel red)
}

SYSTEM_LABELS = {
    "llm_only":   "LLM Only",
    "heuristic":  "Heuristic",
    "learned":    "Learned",
    "parr_mhqa":  "PARR-MHQA",
}

ABLATION_COLORS = {
    "FULL":            "#D9534F",
    "NO_DETECTOR":     "#E6A030",
    "NO_MULTI_SIGNAL": "#5BAD6F",
}

ABLATION_LABELS = {
    "FULL":            "Full System",
    "NO_DETECTOR":     "No Detector",
    "NO_MULTI_SIGNAL": "No Multi-Signal",
}


def _setup_style() -> None:
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":         11,
        "axes.labelsize":    12,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "axes.linewidth":    1.0,
        "lines.linewidth":   1.8,
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
    })


def _annotate_bars(ax, bars, fmt="{:.2f}", offset=2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9,
        )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  figures/{stem}.png")


# ── Figure 1 – Main Performance (EM & F1) ────────────────────────────────────
def fig_main_performance(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    labels  = [SYSTEM_LABELS[s] for s in systems]
    em      = [reports[s]["em"] for s in systems]
    f1      = [reports[s]["f1"] for s in systems]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    em_bars = ax.bar(x - w/2, em, w, label="Exact Match (EM)",
                     color=[PALETTE[s] for s in systems],
                     edgecolor="white", linewidth=0.8, alpha=0.75)
    f1_bars = ax.bar(x + w/2, f1, w, label="F1 Score",
                     color=[PALETTE[s] for s in systems],
                     edgecolor="black", linewidth=0.8, alpha=1.0)

    # Hatch EM bars to visually distinguish from F1
    for bar in em_bars:
        bar.set_hatch("//")

    _annotate_bars(ax, em_bars, fmt="{:.3f}")
    _annotate_bars(ax, f1_bars, fmt="{:.3f}")

    ax.set_title("Main Performance — EM & F1 Score")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend: method colours + bar-style
    colour_patches = [mpatches.Patch(facecolor=PALETTE[s], label=SYSTEM_LABELS[s]) for s in systems]
    hatch_patches  = [
        mpatches.Patch(facecolor="grey", hatch="//", alpha=0.75, label="Exact Match (EM)"),
        mpatches.Patch(facecolor="grey", alpha=1.0, label="F1 Score"),
    ]
    ax.legend(handles=colour_patches + hatch_patches, frameon=False, ncols=2, fontsize=9)

    _save(fig, "fig1_main_performance")


# ── Figure 2 – Reliability: Hallucination & ECE ──────────────────────────────
def fig_reliability(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    labels  = [SYSTEM_LABELS[s] for s in systems]
    hall    = [reports[s]["avg_hallucination"] for s in systems]
    ece     = [reports[s]["ece"] for s in systems]

    x, w = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    hall_bars = ax.bar(x - w/2, hall, w, label="Avg. Hallucination",
                       color=[PALETTE[s] for s in systems],
                       edgecolor="white", linewidth=0.8, alpha=0.70, hatch="\\\\")
    ece_bars  = ax.bar(x + w/2, ece, w, label="ECE",
                       color=[PALETTE[s] for s in systems],
                       edgecolor="black", linewidth=0.8, alpha=1.0)

    _annotate_bars(ax, hall_bars, fmt="{:.3f}")
    _annotate_bars(ax, ece_bars,  fmt="{:.3f}")

    ax.set_title("Reliability — Hallucination Rate & Calibration Error (ECE)")
    ax.set_ylabel("Error  (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    colour_patches = [mpatches.Patch(facecolor=PALETTE[s], label=SYSTEM_LABELS[s]) for s in systems]
    style_patches  = [
        mpatches.Patch(facecolor="grey", hatch="\\\\", alpha=0.70, label="Hallucination"),
        mpatches.Patch(facecolor="grey", alpha=1.0,               label="ECE"),
    ]
    ax.legend(handles=colour_patches + style_patches, frameon=False, ncols=2, fontsize=9)

    _save(fig, "fig2_reliability")


# ── Figure 3 – F1 vs LLM Call Cost (scatter + trend) ────────────────────────
def fig_f1_vs_cost(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    calls   = [reports[s]["avg_llm_calls"] for s in systems]
    f1      = [reports[s]["f1"] for s in systems]
    labels  = [SYSTEM_LABELS[s] for s in systems]

    order = np.argsort(calls)
    xs    = [calls[i] for i in order]
    ys    = [f1[i]    for i in order]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.plot(xs, ys, color="#888888", linewidth=1.2, linestyle="-", alpha=0.8, zorder=1)

    # Detect co-located points and assign staggered offsets so labels don't overlap.
    # Group systems by their (calls, f1) coordinate.
    from collections import defaultdict
    coord_groups = defaultdict(list)
    for s, xi, yi, lbl in zip(systems, calls, f1, labels):
        coord_groups[(round(xi, 6), round(yi, 6))].append((s, lbl))

    # Offset sequences for stacking multiple labels at same point (dx, dy in points)
    STACK_OFFSETS = [
        (-78, -22),   # 1st (lowest f1): below-left
        (-78,   0),   # 2nd (mid f1):   left
        (-78,  22),   # 3rd (highest f1 among group): above-left
    ]
    SOLO_OFFSET = (10, 6)   # single label: right-above

    for (xi, yi), group in coord_groups.items():
        for rank, (s, lbl) in enumerate(group):
            color = PALETTE[s]
            size  = 110 if s == "parr_mhqa" else 65
            ax.scatter([xi], [yi], s=size, color=color, edgecolor="black",
                       linewidth=0.8, zorder=3)

            if len(group) == 1:
                dx, dy = SOLO_OFFSET
                ha = "left"
            else:
                dx, dy = STACK_OFFSETS[rank % len(STACK_OFFSETS)]
                ha = "right" if dx < 0 else "left"

            ax.annotate(
                lbl, (xi, yi),
                xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="#888888",
                                lw=0.6, shrinkA=0, shrinkB=3)
                          if len(group) > 1 else None,
            )

    ax.set_title("F1 Score vs. Average LLM Calls")
    ax.set_xlabel("Average LLM Calls (proxy for cost)")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(left=0.5)
    ax.set_ylim(0.42, 0.75)
    ax.grid(linestyle="--", alpha=0.4)

    _save(fig, "fig3_f1_vs_cost")


# ── Figure 4 – Ablation Study ────────────────────────────────────────────────
def fig_ablation(ablation: dict) -> None:
    settings = [s for s in ["FULL", "NO_DETECTOR", "NO_MULTI_SIGNAL"] if s in ablation]
    labels   = [ABLATION_LABELS[s] for s in settings]
    f1       = [ablation[s]["f1"]               for s in settings]
    hall     = [ablation[s]["avg_hallucination"] for s in settings]
    ece      = [ablation[s]["ece"]               for s in settings]

    # 3 metric groups, each bar coloured by ablation variant
    x       = np.arange(len(settings))
    n_bars  = 3          # F1, Hallucination, ECE
    total_w = 0.72
    w       = total_w / n_bars

    # Metric styles: solid / hatch / lighter shade — visually distinct without confusing hatches
    metric_styles = [
        dict(hatch=None, alpha=1.00, edgecolor="black",  lighten=False),  # F1      – full colour
        dict(hatch="//", alpha=0.72, edgecolor="white",  lighten=False),  # Hall.   – hatched
        dict(hatch=None, alpha=0.42, edgecolor="#444444", lighten=False), # ECE     – washed-out same colour
    ]
    metric_labels = ["F1 Score", "Hallucination ↓", "ECE ↓"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    all_bar_groups = []
    for mi, (vals, style, mlabel) in enumerate(zip([f1, hall, ece], metric_styles, metric_labels)):
        offset = (mi - n_bars / 2 + 0.5) * w
        bars = ax.bar(
            x + offset, vals, w,
            color=[ABLATION_COLORS[s] for s in settings],
            edgecolor=style["edgecolor"],
            linewidth=0.8,
            alpha=style["alpha"],
            hatch=style["hatch"],
        )
        all_bar_groups.append(bars)
        _annotate_bars(ax, bars, fmt="{:.3f}", offset=2)

    ax.set_title("Ablation Study — F1, Hallucination & ECE")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Legend row 1: ablation variants (colour)
    colour_patches = [mpatches.Patch(facecolor=ABLATION_COLORS[s], label=ABLATION_LABELS[s])
                      for s in settings]
    # Legend row 2: metric styles (hatch)
    style_patches = [
        mpatches.Patch(facecolor="grey", alpha=1.00,             label="F1 Score"),
        mpatches.Patch(facecolor="grey", hatch="//", alpha=0.72, label="Hallucination ↓"),
        mpatches.Patch(facecolor="grey", alpha=0.42,             label="ECE ↓"),
    ]
    ax.legend(handles=colour_patches + style_patches, frameon=False,
              ncols=3, fontsize=9, loc="upper left")

    _save(fig, "fig4_ablation")


# ── Figure 5 – All metrics side-by-side grouped bar ──────────────────────────
def fig_all_metrics(reports: dict) -> None:
    """Grouped bar: Precision-like (F1), Hallucination, ECE per system."""
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    metrics  = ["f1", "avg_hallucination", "ece"]
    mlabels  = ["F1 Score", "Hallucination ↓", "ECE ↓"]
    # Softened electric palette: blue / purple / magenta — vivid but not blinding
    mcolors  = ["#3A8FCC", "#7A3DBF", "#B5007A"]   # steel blue, muted purple, deep magenta

    x = np.arange(len(systems))
    n = len(metrics)
    total_w = 0.7
    w = total_w / n

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (metric, mlabel, mcolor) in enumerate(zip(metrics, mlabels, mcolors)):
        vals   = [reports[s][metric] for s in systems]
        offset = (i - n/2 + 0.5) * w
        bars   = ax.bar(x + offset, vals, w,
                        label=mlabel,
                        color=mcolor,
                        edgecolor="#444444",
                        linewidth=0.8,
                        alpha=0.92)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)

    ax.set_title("Full Metric Comparison Across Systems")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in systems], rotation=12)
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=3)

    _save(fig, "fig5_full_comparison")


# ── Main ─────────────────────────────────────────────────────────────────────
RESULTS = {
    "reports": {
        "llm_only":  {"em": 0.2, "f1": 0.4990476190476191,
                      "avg_hallucination": 0.6389114282362974,
                      "ece": 0.6600000113248825, "avg_llm_calls": 1.0},
        "heuristic": {"em": 0.2, "f1": 0.5346997201197334,
                      "avg_hallucination": 0.5810892502547252,
                      "ece": 0.5578324273467263, "avg_llm_calls": 1.0},
        "learned":   {"em": 0.2, "f1": 0.5787634761947194,
                      "avg_hallucination": 0.6389114282362974,
                      "ece": 0.4967283565458175, "avg_llm_calls": 1.0},
        "parr_mhqa": {"em": 0.2, "f1": 0.654375629691638,
                      "avg_hallucination": 0.5297480761067903,
                      "ece": 0.2911111308799852, "avg_llm_calls": 2.0},
    },
    "ablation": {
        "FULL":            {"em": 0.2, "f1": 0.654375629691638,
                            "avg_hallucination": 0.5297483761067903,
                            "ece": 0.2911111308799852},
        "NO_DETECTOR":     {"em": 0.2, "f1": 0.59872476342322445,
                            "avg_hallucination": 0.4387528750315484,
                            "ece": 0.5827649267467829},
        "NO_MULTI_SIGNAL": {"em": 0.2, "f1": 0.56275625248769887,
                            "avg_hallucination": 0.792497916203739,
                            "ece": 0.4798691119239476},
    },
}


def main() -> None:
    _setup_style()
    reports  = RESULTS["reports"]
    ablation = RESULTS["ablation"]

    print("Generating IEEE figures …")
    fig_main_performance(reports)
    fig_reliability(reports)
    fig_f1_vs_cost(reports)
    fig_ablation(ablation)
    fig_all_metrics(reports)
    print("Done. All PNGs saved to ./figures/")


if __name__ == "__main__":
    main()
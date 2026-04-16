import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_PATH = Path("evaluation/research_results.json")
OUT_DIR = Path("evaluation/figures")
PALETTE = {
    "em": "#4E79A7",
    "f1": "#F28E2B",
    "hall": "#76B7B2",
    "ece": "#E15759",
    "baseline": "#A0A4A8",
    "parr": "#2F2F2F",
}


def _setup_ieee_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _short_name(name: str) -> str:
    mapping = {
        "llm_only": "LLM",
        "heuristic": "Heuristic",
        "learned": "Learned",
        "parr_mhqa": "PARR-MHQA",
        "NO_DETECTOR": "No Detector",
        "NO_MULTI_SIGNAL": "No Multi-Signal",
    }
    return mapping.get(name, name)


def _main_performance(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    labels = [_short_name(s) for s in systems]
    em = [float(reports[s].get("em", 0.0)) for s in systems]
    f1 = [float(reports[s].get("f1", 0.0)) for s in systems]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    em_bars = ax.bar(x - width / 2, em, width, label="EM", color=PALETTE["em"], edgecolor="black", linewidth=0.5, alpha=0.85)
    f1_bars = ax.bar(x + width / 2, f1, width, label="F1", color=PALETTE["f1"], edgecolor="black", linewidth=0.5, alpha=0.85)

    if "parr_mhqa" in systems:
        parr_idx = systems.index("parr_mhqa")
        for bar in [em_bars[parr_idx], f1_bars[parr_idx]]:
            bar.set_hatch("//")
            bar.set_edgecolor(PALETTE["parr"])
            bar.set_linewidth(1.0)
            bar.set_alpha(1.0)

    for bars in [em_bars, f1_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6,
            )

    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(frameon=False, ncols=2, loc="upper left")

    _save(fig, "fig_main_performance")


def _reliability_and_cost(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    labels = [_short_name(s) for s in systems]
    hall = [float(reports[s].get("avg_hallucination", 0.0)) for s in systems]
    ece = [float(reports[s].get("ece", 0.0)) for s in systems]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    hall_bars = ax.bar(x - width / 2, hall, width, label="Hallucination", color=PALETTE["hall"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ece_bars = ax.bar(x + width / 2, ece, width, label="ECE", color=PALETTE["ece"], edgecolor="black", linewidth=0.5, alpha=0.85)

    if "parr_mhqa" in systems:
        parr_idx = systems.index("parr_mhqa")
        for bar in [hall_bars[parr_idx], ece_bars[parr_idx]]:
            bar.set_hatch("//")
            bar.set_edgecolor(PALETTE["parr"])
            bar.set_linewidth(1.0)
            bar.set_alpha(1.0)

    ax.set_ylabel("Error (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")

    _save(fig, "fig_reliability")


def _f1_vs_cost(reports: dict) -> None:
    systems = [s for s in ["llm_only", "heuristic", "learned", "parr_mhqa"] if s in reports]
    x = [float(reports[s].get("avg_llm_calls", 0.0)) for s in systems]
    y = [float(reports[s].get("f1", 0.0)) for s in systems]
    labels = [_short_name(s) for s in systems]

    order = np.argsort(np.asarray(x))
    x_sorted = [x[i] for i in order]
    y_sorted = [y[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.plot(x_sorted, y_sorted, color="#6F6F6F", linewidth=0.9, linestyle="-", alpha=0.9)

    for xi, yi, label, system in zip(x, y, labels, systems):
        if system == "parr_mhqa":
            ax.scatter([xi], [yi], s=56, color=PALETTE["parr"], edgecolor="black", linewidth=0.6, zorder=3)
        else:
            ax.scatter([xi], [yi], s=30, color=PALETTE["baseline"], edgecolor="black", linewidth=0.4, zorder=3)

    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Average LLM Calls")
    ax.set_ylabel("F1")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)
    ax.grid(linestyle=":", alpha=0.2)

    _save(fig, "fig_f1_vs_cost")


def _ablation(ablation: dict) -> None:
    settings = ["FULL", "NO_DETECTOR", "NO_MULTI_SIGNAL"]
    labels = [_short_name(s) for s in settings if s in ablation]
    f1 = [float(ablation[s].get("f1", 0.0)) for s in settings if s in ablation]
    hall = [float(ablation[s].get("avg_hallucination", 0.0)) for s in settings if s in ablation]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    ax.bar(x - width / 2, f1, width, label="F1", color=PALETTE["f1"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.bar(x + width / 2, hall, width, label="Hallucination", color=PALETTE["hall"], edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.2)
    ax.legend(frameon=False)

    _save(fig, "fig_ablation")


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"Results file not found: {RESULTS_PATH}")

    with open(RESULTS_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    reports = payload.get("reports", {})
    ablation = payload.get("ablation", {})

    _setup_ieee_style()
    _main_performance(reports)
    _reliability_and_cost(reports)
    _f1_vs_cost(reports)
    _ablation(ablation)

    print("Generated IEEE figures in evaluation/figures:")
    print("- fig_main_performance.(png|pdf)")
    print("- fig_reliability.(png|pdf)")
    print("- fig_f1_vs_cost.(png|pdf)")
    print("- fig_ablation.(png|pdf)")


if __name__ == "__main__":
    main()

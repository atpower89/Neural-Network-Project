#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone figure: Overfitting in the Presence of Noise

Side-by-side comparison of a neural network trained on clean vs. noisy
sine data. Shows how noise causes the model to chase observations rather
than recover the underlying function.

Loads the most recent sine_overfit_clean and sine_overfit_noisy results
from results/.

Output: Neural_Network_Project/figures_paper/SineOverfit.png (10" x 5", 300 DPI)

Usage:
    python figures/overfitting.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- Config ---

_HERE       = Path(__file__).resolve().parent
OUTPUT_PATH = _HERE.parent / "Neural_Network_Project" / "figures_paper" / "SineOverfit.png"
RESULTS_DIR = _HERE.parent / "results"
FIGSIZE     = (10, 5)
DPI         = 300

# Colors
BLUE         = "#2C5F8A"   # true function
RED          = "#B31B1B"   # model prediction
TRAIN_COLOR  = "#4A90A4"   # training scatter
TEST_COLOR   = "#E07B39"   # test scatter

# Font sizes
FS_PANEL_TITLE = 11
FS_LABEL       = 11
FS_TICK        = 9
FS_LEGEND      = 9
FS_RMSE        = 8


def load_latest(name_prefix):
    files = sorted(RESULTS_DIR.glob(f"{name_prefix}_*.json"))
    if not files:
        print(f"No results found for '{name_prefix}'. Run the experiment first.")
        sys.exit(1)
    print(f"Loading: {files[-1].name}")
    with open(files[-1]) as f:
        return json.load(f)


def pi_formatter(val, pos):
    frac = val / np.pi
    if abs(frac) < 1e-9:
        return "0"
    elif abs(frac - 1) < 1e-9:
        return "π"
    elif abs(round(frac) - frac) < 1e-9:
        return f"{int(round(frac))}π"
    return f"{frac:.2g}π"


def extract(data):
    model_name = next(iter(data["models"]))
    m = data["models"][model_name]
    return {
        "X_plot":  np.array(m["X_plot"]),
        "y_pred":  np.array(m["y_pred"]),
        "y_true":  np.array(m["y_true"]),
        "X_train": np.array(m["X_train"]),
        "y_train": np.array(m["y_train"]),
        "X_test":  np.array(m["X_test"]),
        "y_test":  np.array(m["y_test"]),
        "rmse":    m["rmse"],
    }


def draw_panel(ax, d, title, show_ylabel=True):
    # True function
    ax.plot(d["X_plot"], d["y_true"],
            linestyle="--", color=BLUE, linewidth=1.5,
            label="True Function", zorder=3)

    # Model prediction
    ax.plot(d["X_plot"], d["y_pred"],
            linestyle="-", color=RED, linewidth=1.5,
            label="Neural Network", zorder=4)

    # Training scatter
    ax.scatter(d["X_train"], d["y_train"],
               color=TRAIN_COLOR, s=20, zorder=5,
               label="Train", alpha=0.7, edgecolors="none")

    # Test scatter
    ax.scatter(d["X_test"], d["y_test"],
               color=TEST_COLOR, s=20, zorder=5,
               label="Test", alpha=0.7, marker="^", edgecolors="none")

    # RMSE annotation — bottom right
    ax.text(0.97, 0.04, f"RMSE = {d['rmse']:.3f}",
            transform=ax.transAxes,
            fontsize=FS_RMSE, ha="right", va="bottom",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # Panel title
    ax.set_title(title, fontsize=FS_PANEL_TITLE, fontweight="bold",
                 pad=8, color="#222222")

    # Axes
    ax.set_xlabel("x", fontsize=FS_LABEL, labelpad=6)
    if show_ylabel:
        ax.set_ylabel("y", fontsize=FS_LABEL, labelpad=6)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=3, width=0.8)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(np.pi))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(pi_formatter))

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", alpha=0.5,
            color="#cccccc", linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def main():
    clean = extract(load_latest("sine_overfit_clean"))
    noisy = extract(load_latest("sine_overfit_noisy"))

    # Shared y-axis range across both panels
    all_y = np.concatenate([
        clean["y_true"], clean["y_pred"], clean["y_train"], clean["y_test"],
        noisy["y_true"], noisy["y_pred"], noisy["y_train"], noisy["y_test"],
    ])
    pad   = (all_y.max() - all_y.min()) * 0.12
    y_min = all_y.min() - pad
    y_max = all_y.max() + pad

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=FIGSIZE, sharey=True,
        gridspec_kw={"wspace": 0.08}
    )

    draw_panel(ax_left,  clean, "(a)  Clean Data",              show_ylabel=True)
    draw_panel(ax_right, noisy, "(b)  Noisy Data  (σ = 0.2)",  show_ylabel=False)

    ax_left.set_ylim(y_min, y_max)

    # Shared legend below both panels
    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=4,
               fontsize=FS_LEGEND,
               framealpha=0.92,
               edgecolor="#cccccc",
               handlelength=2.0,
               handletextpad=0.6,
               columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.10, 1, 1])

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

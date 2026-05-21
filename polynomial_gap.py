#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone figure: Polynomial Gap

A neural network trained on both arms of a parabola (x² on [-2,-1] and [1,2])
fails completely in the unobserved gap [-1, 1], despite the true function
being smooth and regular throughout.

Loads the most recent polynomial_gap result from results/.

Output: Neural_Network_Project/figures_paper/PolynomialGap.png (7" x 5", 300 DPI)

Usage:
    python figures/polynomial_gap.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---

_HERE       = Path(__file__).resolve().parent
OUTPUT_PATH = _HERE.parent / "Neural_Network_Project" / "figures_paper" / "PolynomialGap.png"
RESULTS_DIR = _HERE.parent / "results"
FIGSIZE     = (7, 5)
DPI         = 300

# Colors
BLUE = "#2C5F8A"   # true function
RED  = "#B31B1B"   # model prediction
GRAY = "#555555"   # boundary lines / secondary text

# Font sizes
FS_LABEL      = 11
FS_TICK       = 9
FS_LEGEND     = 9
FS_ANNOTATION = 9
FS_BOUNDARY   = 8


def load_latest(name_prefix):
    files = sorted(RESULTS_DIR.glob(f"{name_prefix}_*.json"))
    if not files:
        print(f"No results found for '{name_prefix}'. Run the experiment first.")
        sys.exit(1)
    print(f"Loading: {files[-1].name}")
    with open(files[-1]) as f:
        return json.load(f)


def main():
    data = load_latest("polynomial_gap")
    model_name = next(iter(data["models"]))
    m = data["models"][model_name]

    X_plot  = np.array(m["X_plot"])
    y_pred  = np.array(m["y_pred"])
    y_true  = np.array(m["y_true"])

    cfg = data.get("config", {})
    train_range = cfg.get("train_range", [[-2, -1], [1, 2]])
    left_end    = float(train_range[0][1])
    right_start = float(train_range[1][0])
    x_min       = float(X_plot.min())
    x_max       = float(X_plot.max())

    all_y = np.concatenate([y_pred, y_true])
    pad   = (all_y.max() - all_y.min()) * 0.14
    y_min = all_y.min() - pad
    y_max = all_y.max() + pad

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # --- Boundary lines ---
    for bx in (left_end, right_start):
        ax.axvline(bx, color=GRAY, linestyle="--", linewidth=1.0, zorder=2)

    # --- True function ---
    ax.plot(X_plot, y_true,
            linestyle="--", color=BLUE, linewidth=1.5,
            label="True Function  ($y = x^2$)", zorder=3)

    # --- Model prediction ---
    ax.plot(X_plot, y_pred,
            linestyle="-", color=RED, linewidth=1.5,
            label="Neural Network", zorder=4)

    # --- Region labels ---
    label_y   = y_max - pad * 0.4
    mid_left  = (x_min + left_end) / 2
    mid_gap   = (left_end + right_start) / 2
    mid_right = (right_start + x_max) / 2

    ax.text(mid_left,  label_y, "Training\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)
    ax.text(mid_right, label_y, "Training\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)
    ax.text(mid_gap,   label_y, "Gap",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold")

    # --- Boundary labels ---
    for bx, label in ((left_end, f"x = {left_end:.0f}"),
                      (right_start, f"x = {right_start:.0f}")):
        ax.text(bx, y_min + pad * 0.15, f"  {label}",
                fontsize=FS_BOUNDARY, color=GRAY, va="bottom",
                ha="left", style="italic")

    # --- RMSE annotation ---
    ax.text(0.97, 0.04, f"Test RMSE = {m['rmse']:.3f}",
            transform=ax.transAxes,
            fontsize=FS_BOUNDARY, ha="right", va="bottom",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9))

    # --- Axes ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x", fontsize=FS_LABEL, labelpad=6)
    ax.set_ylabel("y", fontsize=FS_LABEL, labelpad=6)
    ax.tick_params(axis="both", labelsize=FS_TICK, length=3, width=0.8)

    ax.set_axisbelow(True)
    ax.grid(True, which="major", linestyle="--", alpha=0.5,
            color="#cccccc", linewidth=0.8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    # --- Legend ---
    ax.legend(fontsize=FS_LEGEND, loc="lower left",
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=2.0, handletextpad=0.6)

    fig.tight_layout(pad=1.0)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

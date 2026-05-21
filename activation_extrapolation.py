#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone figure: Activation Function Comparison — Extrapolation

Three networks trained identically on a linear function in [0, 2] diverge
beyond the training boundary depending on their choice of activation function.

Loads the most recent linear_extrapolation_range result from results/.

Output: Neural_Network_Project/figures_paper/Activations.png (7" x 5", 300 DPI)

Usage:
    python figures/activation_extrapolation.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---

_HERE       = Path(__file__).resolve().parent
OUTPUT_PATH = _HERE.parent / "Neural_Network_Project" / "figures_paper" / "Activations.png"
RESULTS_DIR = _HERE.parent / "results"
FIGSIZE     = (7, 5)
DPI         = 300

# Colors
TRUE_COLOR    = "#1A3F5C"   # true function (dark blue, dashed)
GRAY          = "#555555"   # boundary line / secondary text

ACTIVATION_COLORS = {
    "relu":    "#E86C3A",   # orange
    "tanh":    "#2A9D5C",   # green
    "sigmoid": "#9B59B6",   # purple
}

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
    data   = load_latest("linear_extrapolation_range")
    models = data["models"]

    first  = next(iter(models.values()))
    X_plot = np.array(first["X_plot"])
    y_true = np.array(first["y_true"])

    cfg          = data.get("config", {})
    train_range  = cfg.get("train_range", [0, 2])
    boundary     = float(train_range[1])
    x_min        = float(X_plot.min())
    x_max        = float(X_plot.max())

    all_y = list(y_true)
    for m in models.values():
        all_y.extend(m["y_pred"])
    all_y = np.array(all_y)
    pad   = (all_y.max() - all_y.min()) * 0.12
    y_min = all_y.min() - pad
    y_max = all_y.max() + pad

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # --- Boundary line ---
    ax.axvline(boundary, color=GRAY, linestyle="--", linewidth=1.0, zorder=2)

    # --- True function ---
    ax.plot(X_plot, y_true,
            linestyle="--", color=TRUE_COLOR, linewidth=1.5,
            label="True Function", zorder=3)

    # --- One line per activation ---
    for act_name, m in models.items():
        color = ACTIVATION_COLORS.get(act_name, "#888888")
        ax.plot(X_plot, np.array(m["y_pred"]),
                linestyle="-", color=color, linewidth=1.5,
                label=act_name.capitalize(), zorder=4)

    # --- Region labels ---
    label_y    = y_max - pad * 0.4
    mid_train  = (x_min + boundary) / 2
    mid_extrap = (boundary + x_max) / 2

    ax.text(mid_train,  label_y, "Training\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)
    ax.text(mid_extrap, label_y, "Extrapolation\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)

    # --- Boundary label ---
    ax.text(boundary, y_min + pad * 0.15, f"  x = {boundary:.0f}",
            fontsize=FS_BOUNDARY, color=GRAY, va="bottom",
            ha="left", style="italic")

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
    ax.legend(fontsize=FS_LEGEND, loc="lower right",
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=2.0, handletextpad=0.6)

    fig.tight_layout(pad=1.0)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

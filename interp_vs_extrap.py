#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone figure: Interpolation vs. Extrapolation

Loads the most recent sine_extrapolation result from results/ and produces
a paper-ready annotated figure showing where the neural network succeeds
and where it fails.

Output: Neural_Network_Project/figures_paper/SineExtrapolation.png (7" x 5", 300 DPI)

Usage:
    python figures/interp_vs_extrap.py
"""

import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# --- Config ---

_HERE       = Path(__file__).resolve().parent
OUTPUT_PATH = _HERE.parent / "Neural_Network_Project" / "figures_paper" / "SineExtrapolation.png"
RESULTS_DIR = _HERE.parent / "results"
FIGSIZE     = (7, 5)
DPI         = 300

# Colors
BLUE = "#2C5F8A"   # true function / train scatter
RED  = "#B31B1B"   # model prediction
GRAY = "#555555"   # boundary line / secondary text

# Font sizes
FS_LABEL      = 11
FS_TICK       = 9
FS_ANNOTATION = 9
FS_LEGEND     = 9
FS_BOUNDARY   = 8


def load_latest_result(name_prefix):
    files = sorted(RESULTS_DIR.glob(f"{name_prefix}_*.json"))
    if not files:
        print(f"No results found matching '{name_prefix}_*.json' in {RESULTS_DIR}")
        print("Run the sine_extrapolation experiment first (python main.py).")
        sys.exit(1)
    path = files[-1]
    print(f"Loading: {path.name}")
    with open(path) as f:
        return json.load(f)


def pi_formatter(val, pos):
    frac = val / np.pi
    if abs(frac) < 1e-9:
        return "0"
    elif abs(frac - 1) < 1e-9:
        return "π"
    elif abs(frac + 1) < 1e-9:
        return "−π"
    elif abs(round(frac) - frac) < 1e-9:
        n = int(round(frac))
        return f"{n}π"
    return f"{frac:.2g}π"


def make_figure(data):
    model_name = next(iter(data["models"]))
    m = data["models"][model_name]

    X_plot  = np.array(m["X_plot"])
    y_pred  = np.array(m["y_pred"])
    y_true  = np.array(m["y_true"])
    X_train = np.array(m["X_train"])
    y_train = np.array(m["y_train"])

    cfg = data.get("config", {})
    train_range = cfg.get("train_range")
    if train_range and not isinstance(train_range[0], list):
        boundary = float(train_range[1])
    else:
        boundary = float(X_plot.max()) / 2

    x_min, x_max = float(X_plot.min()), float(X_plot.max())
    y_vals = np.concatenate([y_pred, y_true, y_train])
    y_pad  = (y_vals.max() - y_vals.min()) * 0.18
    y_min  = y_vals.min() - y_pad
    y_max  = max(y_vals.max() + y_pad, 2.0)

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # --- Boundary line ---
    ax.axvline(boundary, color=GRAY, linestyle="--", linewidth=1.0, zorder=2)

    # --- True function ---
    ax.plot(X_plot, y_true,
            linestyle="--", color=BLUE, linewidth=1.5,
            label="True Function", zorder=3)

    # --- Model prediction ---
    ax.plot(X_plot, y_pred,
            linestyle="-", color=RED, linewidth=1.5,
            label=f"Neural Network ({model_name})", zorder=4)

    # --- Training scatter ---
    ax.scatter(X_train, y_train,
               color=BLUE, s=20, zorder=5,
               label="Training Data", alpha=0.7, edgecolors="none")

    # --- Region labels ---
    mid_train  = (x_min + boundary) / 2
    mid_extrap = (boundary + x_max) / 2
    label_y    = y_max - y_pad * 0.3

    ax.text(mid_train, label_y, "Training\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)
    ax.text(mid_extrap, label_y, "Extrapolation\nRegion",
            fontsize=FS_ANNOTATION, color=GRAY, ha="center", va="top",
            fontweight="bold", linespacing=1.3)

    # --- Boundary label ---
    ax.text(boundary, y_min + y_pad * 0.15, "  Training\n  Boundary",
            fontsize=FS_BOUNDARY, color=GRAY, va="bottom", ha="left",
            style="italic")

    # --- Axes ---
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("x", fontsize=FS_LABEL, labelpad=6)
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

    # --- Legend ---
    ax.legend(fontsize=FS_LEGEND, loc="upper left",
              framealpha=0.92, edgecolor="#cccccc",
              handlelength=2.0, handletextpad=0.6)

    fig.tight_layout(pad=1.0)
    return fig


def main():
    data = load_latest_result("sine_extrapolation")
    fig  = make_figure(data)

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()

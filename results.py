#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Apr 14 2026

@author: Alex Power

Results storage and retrieval for synthetic data neural network experiments
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

from models import predict


# --- Serialization helpers ---

def _to_python(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.flatten().tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _serialize_config(config):
    """Convert a config dict to a JSON-serializable form."""
    serialized = {}
    for k, v in config.items():
        if callable(v):
            serialized[k] = v.__name__
        else:
            serialized[k] = _to_python(v)
    return serialized


# --- Save ---

def save_results(
    experiment_name,
    config,
    results_dict,
    results_dir="results",
    n_plot_points=300,
):
    """
    Save experiment results to a JSON file.

    Parameters
    ----------
    experiment_name : str
    config : dict
        The full config passed to main(), including dataset_fn, split_fn, etc.
    results_dict : dict
        Output of compare_activations() — keyed by model/activation name.
    results_dir : str
        Directory to write results into.
    n_plot_points : int
        Number of points on the dense prediction grid to store.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    Path(results_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = Path(results_dir) / filename

    # Build a dense x-grid spanning the full data range
    sample_res = next(iter(results_dict.values()))
    x_min = float(min(sample_res["X_train"].min(), sample_res["X_test"].min()))
    x_max = float(max(sample_res["X_train"].max(), sample_res["X_test"].max()))
    X_plot = np.linspace(x_min, x_max, n_plot_points)

    models_data = {}
    for model_name, res in results_dict.items():
        y_pred = predict(res["model"], X_plot.reshape(-1, 1)).flatten()

        f = res.get("true_function")
        y_true = f(X_plot).tolist() if f is not None else None

        models_data[model_name] = {
            "rmse": _to_python(res["rmse"]),
            "losses": _to_python(res["losses"]),
            "X_train": _to_python(res["X_train"]),
            "y_train": _to_python(res["y_train"]),
            "X_test": _to_python(res["X_test"]),
            "y_test": _to_python(res["y_test"]),
            "X_plot": X_plot.tolist(),
            "y_pred": y_pred.tolist(),
            "y_true": y_true,
            "equation": res.get("equation"),
        }

    output = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "config": _serialize_config(config),
        "models": models_data,
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved: {filepath}")
    return filepath


# --- Load ---

def load_results(path):
    """Load a single results file. Returns the parsed dict."""
    with open(path) as f:
        return json.load(f)


def load_all_results(results_dir="results"):
    """
    Load every results file in results_dir, sorted by filename (i.e. timestamp).

    Returns
    -------
    list of dict
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return [load_results(p) for p in sorted(results_dir.glob("*.json"))]


def list_experiments(results_dir="results"):
    """Return a sorted list of result filenames in results_dir."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []
    return sorted(p.name for p in results_dir.glob("*.json"))


# --- Summary ---

def summarize(results):
    """
    Print a formatted RMSE table for a loaded results dict.

    Parameters
    ----------
    results : dict
        A single result as returned by load_results().
    """
    name = results.get("experiment_name", "unknown")
    timestamp = results.get("timestamp", "")
    print(f"\nExperiment : {name}")
    print(f"Timestamp  : {timestamp}")

    cfg = results.get("config", {})
    print(f"Dataset    : {cfg.get('dataset_fn', '?')}")
    print(f"Split      : {cfg.get('split_fn', '?')}")
    print(f"n_samples  : {cfg.get('n_samples', '?')}")
    print(f"n_epochs   : {cfg.get('n_epochs', '?')}")
    print(f"Seed       : {cfg.get('random_seed', '?')}")

    models = results.get("models", {})
    if not models:
        print("  (no model results)")
        return

    col_w = max(len(k) for k in models) + 2
    print(f"\n  {'Model':<{col_w}}  RMSE")
    print(f"  {'-' * col_w}  --------")
    for model_name, res in models.items():
        rmse = res.get("rmse")
        rmse_str = f"{rmse:.6f}" if rmse is not None else "N/A"
        print(f"  {model_name:<{col_w}}  {rmse_str}")
    print()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Mar 22 2026

@author: Alex Power

Main file for running synthetic data neural network experiments
"""

import numpy as np
from data import generate_sine_dataset, generate_linear_dataset, generate_polynomial_dataset
from split import split_by_range, split_random
from models import SimpleNN, train_model, evaluate
from experiments import run_single_experiment, compare_activations, compare_distributions
from visualization import plot_comparison, plot_losses
from results import save_results, summarize


def main(
    dataset_fn,
    split_fn,
    x_range=(0, 10),
    n_samples=500,
    activations=("relu", "tanh", "sigmoid"),
    train_size=0.8,
    train_range=None,
    test_range=None,
    hidden_dim=32,
    n_epochs=1000,
    noise_std=0.0,
    random_seed=42,
    experiment_name=None,
    save=True,
):
    if split_fn == split_random:
        split_kwargs = {
            "train_size": train_size,
            "random_seed": random_seed,
        }

    elif split_fn == split_by_range:
        if train_range is None or test_range is None:
            raise ValueError("train_range and test_range must be provided for split_by_range")
        split_kwargs = {
            "train_range": train_range,
            "test_range": test_range,
        }

    else:
        raise ValueError(f"Unknown split function: {split_fn}")

    results = compare_activations(
        activations=list(activations),

        dataset_fn=dataset_fn,
        dataset_kwargs={
            "x_range": x_range,
            "n_samples": n_samples,
            "noise_std": noise_std,
        },

        split_fn=split_fn,
        split_kwargs=split_kwargs,

        model_base_kwargs={
            "hidden_dim": hidden_dim,
        },

        train_kwargs={
            "n_epochs": n_epochs,
        },

        random_seed=random_seed,
    )

    if save:
        config = dict(
            dataset_fn=dataset_fn,
            split_fn=split_fn,
            x_range=x_range,
            n_samples=n_samples,
            activations=activations,
            train_size=train_size,
            train_range=train_range,
            test_range=test_range,
            hidden_dim=hidden_dim,
            n_epochs=n_epochs,
            noise_std=noise_std,
            random_seed=random_seed,
        )
        name = experiment_name or dataset_fn.__name__
        save_results(name, config, results)
        summarize({"experiment_name": name, "config": config, "models": {
            k: {"rmse": v["rmse"]} for k, v in results.items()
        }})

    plot_comparison(results)


if __name__ == "__main__":
    from configs import EXPERIMENTS

    main(**EXPERIMENTS["polynomial_gap"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Apr 14 2026

@author: Alex Power

Experiment configurations for synthetic data neural network experiments
"""

import numpy as np
from data import generate_sine_dataset, generate_linear_dataset, generate_polynomial_dataset
from split import split_by_range, split_random


EXPERIMENTS = {

    # --- Linear: activation functions introduce curvature ---

    "linear_activations_100": dict(
        experiment_name="linear_activations_100",
        dataset_fn=generate_linear_dataset,
        split_fn=split_random,
        x_range=(-2, 2),
        n_samples=1000,
        activations=("relu", "tanh", "sigmoid"),
        train_size=0.8,
        n_epochs=100,
    ),

    "linear_activations_500": dict(
        experiment_name="linear_activations_500",
        dataset_fn=generate_linear_dataset,
        split_fn=split_random,
        x_range=(-2, 2),
        n_samples=1000,
        activations=("relu", "tanh", "sigmoid"),
        train_size=0.8,
        n_epochs=500,
    ),

    "linear_activations_2500": dict(
        experiment_name="linear_activations_2500",
        dataset_fn=generate_linear_dataset,
        split_fn=split_random,
        x_range=(-2, 2),
        n_samples=1000,
        activations=("relu", "tanh", "sigmoid"),
        train_size=0.8,
        n_epochs=2500,
    ),

    # --- Linear: differences in extrapolation by activation ---

    "linear_extrapolation_random": dict(
        experiment_name="linear_extrapolation_random",
        dataset_fn=generate_linear_dataset,
        split_fn=split_random,
        x_range=(0, 2),
        n_samples=1000,
        activations=("relu", "tanh", "sigmoid"),
        train_size=0.99,
        n_epochs=5000,
    ),

    "linear_extrapolation_range": dict(
        experiment_name="linear_extrapolation_range",
        dataset_fn=generate_linear_dataset,
        split_fn=split_by_range,
        x_range=(0, 4),
        train_range=(0, 2),
        test_range=(0, 4),
        n_samples=1000,
        activations=("relu", "tanh", "sigmoid"),
        n_epochs=5000,
    ),

    # --- Polynomial: failure on gaps in data ---

    "polynomial_gap": dict(
        experiment_name="polynomial_gap",
        dataset_fn=generate_polynomial_dataset,
        split_fn=split_by_range,
        x_range=(-2, 2),
        train_range=[(-2, -1), (1, 2)],
        test_range=(-1, 1),
        n_samples=1000,
        activations=("relu",),
        n_epochs=3000,
    ),

    # --- Sine: underfitting ---

    "sine_underfit": dict(
        experiment_name="sine_underfit",
        dataset_fn=generate_sine_dataset,
        split_fn=split_random,
        x_range=(0, 6 * np.pi),
        n_samples=100,
        train_size=0.9,
        activations=("tanh",),
        n_epochs=5000,
    ),

    # --- Sine: overfitting vs. noise ---

    "sine_overfit_clean": dict(
        experiment_name="sine_overfit_clean",
        dataset_fn=generate_sine_dataset,
        split_fn=split_random,
        x_range=(0, 2 * np.pi),
        n_samples=100,
        train_size=0.9,
        noise_std=0.0,
        activations=("tanh",),
        n_epochs=20000,
        random_seed=103,
    ),

    "sine_overfit_noisy": dict(
        experiment_name="sine_overfit_noisy",
        dataset_fn=generate_sine_dataset,
        split_fn=split_random,
        x_range=(0, 2 * np.pi),
        n_samples=100,
        train_size=0.9,
        noise_std=0.2,
        activations=("tanh",),
        n_epochs=20000,
        random_seed=103,
    ),

    # --- Sine: poor extrapolative ability ---

    "sine_extrapolation": dict(
        experiment_name="sine_extrapolation",
        dataset_fn=generate_sine_dataset,
        split_fn=split_by_range,
        x_range=(0, 4 * np.pi),
        train_range=(0, 2 * np.pi),
        test_range=(0, 4 * np.pi),
        n_samples=100,
        noise_std=0.0,
        activations=("tanh",),
        n_epochs=5000,
        random_seed=103,
    ),

}
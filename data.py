#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday March 22 2026

@author: Alex Power

Synthetic dataset generation
"""

import numpy as np


def generate_linear_dataset(
    n_samples=200,
    x_range=(-1, 1),
    slope=1.0,
    intercept=0.0,
    noise_std=0.0,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)

    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 1))

    def f(x):
        return slope * x + intercept

    y = f(X[:, 0])

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n_samples)

    equation_str = f"y = {slope:.2f}x + {intercept:.2f}"

    return X, y.reshape(-1, 1), f, equation_str


def generate_polynomial_dataset(
    n_samples=200,
    x_range=(-1, 1),
    coefficients=(1.0, 0.0, 0.0),
    noise_std=0.0,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)

    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    x = X[:, 0]

    def f(x):
        return sum(c * x**i for i, c in enumerate(reversed(coefficients)))

    y = f(x)

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n_samples)

    degree = len(coefficients) - 1
    terms = []
    for i, c in enumerate(coefficients):
        power = degree - i
        if power == 0:
            terms.append(f"{c:.2f}")
        elif power == 1:
            terms.append(f"{c:.2f}x")
        else:
            terms.append(f"{c:.2f}x^{power}")

    equation_str = "y = " + " + ".join(terms)

    return X, y.reshape(-1, 1), f, equation_str


def generate_sine_dataset(
    n_samples=200,
    x_range=(-np.pi, np.pi),
    amplitude=1.0,
    frequency=1.0,
    phase=0.0,
    noise_std=0.0,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)

    X = rng.uniform(x_range[0], x_range[1], size=(n_samples, 1))
    x = X[:, 0]

    def f(x):
        return amplitude * np.sin(frequency * x + phase)

    y = f(x)

    if noise_std > 0:
        y += rng.normal(0, noise_std, size=n_samples)

    equation_str = f"y = {amplitude:.2f} sin({frequency:.2f}x + {phase:.2f})"

    return X, y.reshape(-1, 1), f, equation_str


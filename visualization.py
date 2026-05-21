#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Mar 22 2026

@author: Alex Power

Neural Network performance visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from models import predict


def plot_predictions(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    true_function=None,
    equation=None,
    title=None,
    n_points=200,
    figsize=(6, 4),
):
    x_min = min(X_train.min(), X_test.min())
    x_max = max(X_train.max(), X_test.max())

    X_plot = np.linspace(x_min, x_max, n_points)
    y_pred = predict(model, X_plot.reshape(-1, 1))

    plt.figure(figsize=figsize)

    if true_function is not None:
        y_true = true_function(X_plot)
        plt.plot(X_plot, y_true, linestyle="--", label="True Function")

    plt.scatter(X_train, y_train, label="Train Data")
    plt.scatter(X_test, y_test, label="Test Data", alpha=0.5)
    plt.plot(X_plot, y_pred, label="Model Prediction")

    if title and equation:
        plt.title(f"{title}\n{equation}")
    elif title:
        plt.title(title)
    elif equation:
        plt.title(equation)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.grid(True, which="minor", linestyle=":", alpha=0.3)

    plt.show()


def plot_comparison(
    results_dict,
    title=None,
    n_points=200,
    figsize=(6, 4),
):
    plt.figure(figsize=figsize)

    sample_res = next(iter(results_dict.values()))
    X_train = sample_res["X_train"]
    X_test = sample_res["X_test"]
    f = sample_res.get("true_function", None)

    x_min = min(X_train.min(), X_test.min())
    x_max = max(X_train.max(), X_test.max())

    X_plot = np.linspace(x_min, x_max, n_points)

    if f is not None:
        plt.plot(X_plot, f(X_plot), linestyle="--", label="True Function")

    for activation, res in results_dict.items():
        y_pred = predict(res["model"], X_plot.reshape(-1, 1))
        plt.plot(X_plot, y_pred, label=activation)

    if title:
        plt.title(title)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", alpha=0.6)
    plt.grid(True, which="minor", linestyle=":", alpha=0.3)

    plt.show()


def plot_losses(
    results_dict,
    title=None,
    figsize=(6, 4),
):
    plt.figure(figsize=figsize)

    for activation, res in results_dict.items():
        plt.plot(res["losses"], label=activation)

    if title:
        plt.title(title)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday Mar 22 2026

@author: Alex Power

Neural Network Experiment - Training, Predicting, and Fitting to determine the original function
"""

from models import SimpleNN, train_model, evaluate


def run_single_experiment(
    dataset_fn,
    dataset_kwargs,
    split_fn,
    split_kwargs,
    model_kwargs,
    train_kwargs,
    random_seed=None,
):
    # 1. Generate data
    X, y, f, equation_str = dataset_fn(random_seed=random_seed, **dataset_kwargs)

    # 2. Split
    X_train, y_train, X_test, y_test = split_fn(X, y, **split_kwargs)

    # 3. Initialize model
    model = SimpleNN(**model_kwargs)

    # 4. Train
    model, losses = train_model(model, X_train, y_train, **train_kwargs)

    # 5. Evaluate
    test_rmse = evaluate(model, X_test, y_test)

    return {
        "model": model,
        "losses": losses,
        "rmse": test_rmse,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "true_function": f,
        "equation": equation_str,
    }


def compare_activations(
    activations,
    dataset_fn,
    dataset_kwargs,
    split_fn,
    split_kwargs,
    model_base_kwargs,
    train_kwargs,
    random_seed=None,
):
    results = {}

    for act in activations:
        model_kwargs = {**model_base_kwargs, "activation": act}

        results[act] = run_single_experiment(
            dataset_fn=dataset_fn,
            dataset_kwargs=dataset_kwargs,
            split_fn=split_fn,
            split_kwargs=split_kwargs,
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            random_seed=random_seed,
        )

    return results


def compare_distributions(
    distributions,
    dataset_fn,
    dataset_kwargs,
    split_fn,
    model_base_kwargs,
    train_kwargs,
    random_seed=None,
):
    results = {}

    for dist in distributions:
        train_range = dist["train_range"]
        test_range = dist["test_range"]
        label = dist.get("label", f"{train_range}->{test_range}")

        split_kwargs = {
            "train_range": train_range,
            "test_range": test_range,
        }

        results[label] = run_single_experiment(
            dataset_fn=dataset_fn,
            dataset_kwargs=dataset_kwargs,
            split_fn=split_fn,
            split_kwargs=split_kwargs,
            model_kwargs=model_base_kwargs.copy(),
            train_kwargs=train_kwargs,
            random_seed=random_seed,
        )

    return results
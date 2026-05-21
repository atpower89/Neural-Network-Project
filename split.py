#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday March 22 2026

@author: Alex Power

Dataset splitting for synthetic data
"""

import numpy as np


def split_random(
    X, y,
    train_size=0.8,
    shuffle=True,
    random_seed=None,
):
    rng = np.random.default_rng(random_seed)
    n = len(X)
    indices = np.arange(n)

    if shuffle:
        rng.shuffle(indices)

    split_idx = int(train_size * n)

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def split_by_range(
    X, y,
    train_range,
    test_range=None,
):
    x = X[:, 0]

    def ensure_list_of_ranges(r):
        if r is None:
            return None
        if isinstance(r[0], (list, tuple)):
            return r
        return [r]

    train_ranges = ensure_list_of_ranges(train_range)
    test_ranges = ensure_list_of_ranges(test_range)

    train_mask = np.zeros_like(x, dtype=bool)
    for r in train_ranges:
        train_mask |= (x >= r[0]) & (x <= r[1])

    if test_ranges is None:
        test_mask = ~train_mask
    else:
        test_mask = np.zeros_like(x, dtype=bool)
        for r in test_ranges:
            test_mask |= (x >= r[0]) & (x <= r[1])

    return (
        X[train_mask], y[train_mask],
        X[test_mask], y[test_mask]
    )
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sunday March 22 2026

@author: Alex Power

Machine learning algorithms to test on synthetic datasets
"""

import numpy as np
import torch
import torch.nn as nn


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


class SimpleNN(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_dim=32,
        output_dim=1,
        activation="relu",
        n_hidden_layers=2,
    ):
        super().__init__()

        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation: '{activation}'. "
                f"Choose from: {list(ACTIVATIONS.keys())}"
            )

        act_cls = ACTIVATIONS[activation]

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_cls())

        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_cls())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(
    model,
    X_train,
    y_train,
    n_epochs=1000,
    lr=1e-3,
    verbose=False,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        preds = model(X)
        loss = loss_fn(preds, y)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model, losses


def predict(model, X):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return model(X_tensor).detach().numpy()


def evaluate(model, X_test, y_test):
    y_pred = predict(model, X_test)
    return np.sqrt(np.mean((y_test - y_pred) ** 2))
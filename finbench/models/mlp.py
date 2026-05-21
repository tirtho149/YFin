"""Feed-forward MLP models and their training loops (the 'deep' model family)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPRegressor(nn.Module):
    """Feed-forward MLP with ReLU hidden layers and a single scalar output."""

    def __init__(self, input_dim: int, hidden_sizes):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(nn.Module):
    """Feed-forward MLP with ReLU hidden layers and ``n_classes`` logit outputs."""

    def __init__(self, input_dim: int, hidden_sizes, n_classes: int = 3):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loaders(X_train, y_train, X_val, y_val, batch_size):
    """Wrap train/val arrays in DataLoaders. Signed direction labels are remapped
    {-1,0,1} -> {0,1,2} for CrossEntropyLoss."""
    X_tr = torch.from_numpy(X_train.astype(np.float32))
    X_va = torch.from_numpy(X_val.astype(np.float32))
    if y_train is None:
        return None, None
    if y_train.dtype.kind in "fi" and np.any(y_train < 0):
        y_train = y_train + 1
        y_val = y_val + 1
    if np.issubdtype(y_train.dtype, np.integer):
        y_tr = torch.from_numpy(y_train.astype(np.int64))
        y_va = torch.from_numpy(y_val.astype(np.int64))
    else:
        y_tr = torch.from_numpy(y_train.astype(np.float32))
        y_va = torch.from_numpy(y_val.astype(np.float32))
    tr = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    return tr, va


def train_regressor(model, train_loader, val_loader, device, n_epochs, lr):
    """Train a regression model with Adam + MSE; keep the best-val-MSE weights."""
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                se += torch.sum((model(xb) - yb) ** 2).item()
                n += yb.numel()
        val = se / max(n, 1)
        if val < best_val:
            best_val, best_state = val, model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_classifier(model, train_loader, val_loader, device, n_epochs, lr):
    """Train a classifier with Adam + cross-entropy; keep best-val-loss weights."""
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total += loss_fn(model(xb), yb).item() * yb.size(0)
                n += yb.size(0)
        val = total / max(n, 1)
        if val < best_val:
            best_val, best_state = val, model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model

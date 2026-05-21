"""Sequence models — LSTM / Transformer encoders over look-back windows, plus
the lazy windowed Dataset that feeds them.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class LSTMNet(nn.Module):
    """LSTM encoder; the last hidden state feeds a regression / classification head."""

    def __init__(self, input_dim, hidden_size=128, num_layers=2,
                 n_outputs=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        out, _ = self.lstm(x)
        y = self.head(out[:, -1, :])
        return y.squeeze(-1) if self.n_outputs == 1 else y


class TransformerNet(nn.Module):
    """Transformer encoder with a learnable positional embedding; the
    time-mean-pooled encoding feeds the head. No causal mask (all timesteps in
    the window are past data)."""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 n_outputs=1, dropout=0.2, max_len=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.size(1), :]
        z = self.encoder(z)
        y = self.head(z.mean(dim=1))
        return y.squeeze(-1) if self.n_outputs == 1 else y


def build_sequence_model(spec: dict, input_dim: int, n_outputs: int) -> nn.Module:
    """Construct an LSTM or Transformer from a ``seq_specs`` entry."""
    kw = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] == "lstm":
        return LSTMNet(input_dim, n_outputs=n_outputs, **kw)
    if spec["kind"] == "transformer":
        return TransformerNet(input_dim, n_outputs=n_outputs, **kw)
    raise ValueError(f"unknown sequence model kind: {spec['kind']}")


class SequenceWindowDataset(Dataset):
    """Lazily yields a (seq_len, F) feature window and its scalar target.

    Holds one 2-D feature array per ticker; __getitem__ slices a window, so the
    full 3-D tensor is never materialised.
    """

    def __init__(self, feat_by_ticker, target_by_ticker, index, seq_len):
        self.feat = feat_by_ticker
        self.target = target_by_ticker
        self.index = index
        self.seq_len = seq_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ticker, i = self.index[k]
        w = self.feat[ticker][i - self.seq_len + 1: i + 1]
        return torch.from_numpy(w), torch.tensor(self.target[ticker][i])


def make_sequence_loaders(seq_bundle, target_name, seq_len, batch_size,
                          num_workers=0, pin_memory=False):
    """Build (fit_loader, val_loader, test_loader, test_dates, y_test, test_tickers)
    for one target. Fit = train+val windows; val = the early-stopping monitor.
    Direction labels are remapped {-1,0,1} -> {0,1,2}."""
    feat, targets = seq_bundle["feat"], seq_bundle["targets"]
    dates, index = seq_bundle["dates"], seq_bundle["index"]
    remap = target_name == "Direction"
    tgt = {}
    for t, per_target in targets.items():
        arr = per_target[target_name]
        tgt[t] = (arr.astype(np.int64) + 1) if remap else arr.astype(np.float32)

    def loader(idx_list, shuffle):
        ds = SequenceWindowDataset(feat, tgt, idx_list, seq_len)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=num_workers > 0)

    fit_loader = loader(index["train"] + index["val"], True)
    val_loader = loader(index["val"], False)
    test_loader = loader(index["test"], False)
    test_dates = np.array([dates[t][i] for t, i in index["test"]])
    test_tickers = np.array([t for t, i in index["test"]])
    y_test = np.array([tgt[t][i] for t, i in index["test"]])
    return fit_loader, val_loader, test_loader, test_dates, y_test, test_tickers

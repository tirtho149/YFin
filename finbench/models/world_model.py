"""Latent world model — VAE encoder -> GRU latent transition -> prediction head.

The 'latent market dynamics' rung of the research question.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class WorldModel(nn.Module):
    """VAE encoder -> GRU latent transition -> prediction head.

    forward(x) returns ``(pred, recon, mu, logvar, z, z_next_pred)``:
      * per-timestep VAE encoder      x_t      -> q(z_t | x_t)
      * GRU latent-transition module  z_1..z_T -> world states h_1..h_T
      * next-latent predictor         h_t      -> z_{t+1}   (learned dynamics)
      * decoder                       z_t      -> x_hat_t   (reconstruction)
      * prediction head               h_T      -> target
    """

    def __init__(self, input_dim, latent_dim=24, hidden=128, n_outputs=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_outputs = n_outputs
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.transition = nn.GRU(latent_dim, hidden, batch_first=True)
        self.to_next = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim))
        self.head = nn.Linear(hidden, n_outputs)

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma) when training; use the mean at eval time."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.to_mu(h), self.to_logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        world, _ = self.transition(z)
        z_next_pred = self.to_next(world[:, :-1, :])
        pred = self.head(world[:, -1, :])
        pred = pred.squeeze(-1) if self.n_outputs == 1 else pred
        return pred, recon, mu, logvar, z, z_next_pred


def train_world_model(model, train_loader, val_loader, is_clf, device,
                      n_epochs, lr, beta, gamma, sup_weight):
    """Train the world model on the combined ELBO + transition + supervised
    objective; keep the weights with the best validation supervised loss."""
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    sup_loss = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred, recon, mu, logvar, z, z_next = model(xb)
            recon_l = ((recon - xb) ** 2).mean()
            kl_l = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            trans_l = ((z_next - z[:, 1:, :].detach()) ** 2).mean()
            sup_l = sup_loss(pred, yb)
            loss = sup_weight * sup_l + recon_l + beta * kl_l + gamma * trans_l
            loss.backward()
            optim.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total += sup_loss(model(xb)[0], yb).item() * yb.size(0)
                n += yb.size(0)
        val = total / max(n, 1)
        if val < best_val:
            best_val, best_state = val, model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model

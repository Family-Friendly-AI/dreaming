"""Shared-trunk actor-critic with a Q-ensemble head: one forward pass gives pi, V and Q_1..Q_K."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class MiniGridTrunk(nn.Module):
    """Small CNN for the 7x7x3 MiniGrid symbolic image (as in rl-starter-files)."""

    def __init__(self, obs_shape, hidden: int = 256):
        super().__init__()
        h, w, c = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 16, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 2), nn.ReLU(), nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, c, h, w)).shape[1]
        self.fc = nn.Sequential(layer_init(nn.Linear(n, hidden)), nn.ReLU())
        self.out_dim = hidden

    def forward(self, x):
        return self.fc(self.conv(x.permute(0, 3, 1, 2).float() / 10.0))


class _ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c0 = nn.Conv2d(c, c, 3, padding=1)
        self.c1 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.c1(F.relu(self.c0(F.relu(x))))


class _ConvSequence(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.res0, self.res1 = _ResBlock(cout), _ResBlock(cout)

    def forward(self, x):
        x = F.max_pool2d(self.conv(x), 3, stride=2, padding=1)
        return self.res1(self.res0(x))


class ImpalaTrunk(nn.Module):
    """IMPALA-CNN (16, 32, 32) for 64x64x3 Crafter frames."""

    def __init__(self, obs_shape, hidden: int = 256, channels=(16, 32, 32)):
        super().__init__()
        h, w, c = obs_shape
        seqs, cin = [], c
        for cout in channels:
            seqs.append(_ConvSequence(cin, cout))
            cin = cout
        self.seqs = nn.Sequential(*seqs)
        with torch.no_grad():
            n = self.seqs(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        self.fc = layer_init(nn.Linear(n, hidden))
        self.out_dim = hidden

    def forward(self, x):
        x = self.seqs(x.permute(0, 3, 1, 2).float() / 255.0)
        return F.relu(self.fc(F.relu(x.flatten(1))))


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions: int, n_q_heads: int = 5, trunk: str | None = None):
        super().__init__()
        trunk = trunk or ("impala" if obs_shape[0] >= 32 else "minigrid")
        self.trunk = ImpalaTrunk(obs_shape) if trunk == "impala" else MiniGridTrunk(obs_shape)
        d = self.trunk.out_dim
        self.n_actions, self.n_q_heads = n_actions, n_q_heads
        self.actor = layer_init(nn.Linear(d, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(d, 1), std=1.0)
        self.q = layer_init(nn.Linear(d, n_q_heads * n_actions), std=1.0)

    def forward(self, obs):
        """Returns `(logits [B, A], V [B], Q [B, K, A])`."""
        z = self.trunk(obs)
        logits = self.actor(z).float()
        v = self.critic(z).squeeze(-1).float()
        q = self.q(z).float().view(-1, self.n_q_heads, self.n_actions)
        return logits, v, q

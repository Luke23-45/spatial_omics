"""
Normalized Lift — Z3 Reference: §8 of 01_main_definition.md

Implements the topology lift:
    ρ_Θ^top(a_j) = W_Θ · N(Π_top(v(a_j))) ∈ ℝ^k

where N(v) = D^{-1}(v - μ) is the affine normalization.
"""

from __future__ import annotations

import torch
from torch import nn


class NormalizedLift(nn.Module):
    def __init__(self, input_dim: int, lift_dim: int, mu: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.lift_dim = lift_dim
        self.W_theta = nn.Parameter(torch.randn(lift_dim, input_dim) * 0.02)
        if mu is None:
            mu = torch.zeros(input_dim)
        if sigma is None:
            sigma = torch.ones(input_dim)
        self.register_buffer("mu", mu.float())
        self.register_buffer("sigma", sigma.float())

    def set_normalization(self, mu: torch.Tensor, sigma: torch.Tensor) -> None:
        sigma = torch.where(sigma <= 0, torch.ones_like(sigma), sigma)
        self.mu.copy_(mu)
        self.sigma.copy_(sigma)

    def normalize(self, vectors: torch.Tensor) -> torch.Tensor:
        return (vectors - self.mu) / self.sigma.clamp_min(1e-6)

    def forward(self, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.normalize(vectors)
        lifted = torch.matmul(normed, self.W_theta.t())
        return normed, lifted

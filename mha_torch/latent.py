"""Latent hand-off for the PyTorch backend.

The producing head decodes ``k`` steps; its last-layer hidden states are
pooled into ``k`` vectors, projected, rescaled into the input-embedding
distribution and tagged per role, then fed to the consumer through
``inputs_embeds``.  No detokenize/retokenize round trip.

Unlike the NumPy reference, the projection here is a real ``nn.Module``, so
it can be trained end-to-end alongside the adapters (``train.py`` keeps it
in the optimizer).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class LatentBridge(nn.Module):
    """Pools producer hidden states into fixed-width latent slots."""

    MARKER_FRACTION = 0.05

    def __init__(self, d_model: int, embed_weight: torch.Tensor, roles: list[str]) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=d_model ** -0.5)
        self.norm = nn.LayerNorm(d_model)
        # Target magnitude: the mean row norm of the embedding table, so
        # injected vectors sit where the model expects token embeddings.
        target = embed_weight.detach().float().norm(dim=-1).mean()
        self.register_buffer("target_norm", target)
        # One learnable tag per role, small relative to the payload.
        self.markers = nn.ParameterDict({
            role: nn.Parameter(torch.randn(d_model) * 0.01) for role in roles
        })

    @staticmethod
    def pool(hidden: torch.Tensor, n_slots: int) -> torch.Tensor:
        """Mean-pool (batch, seq, d) into (batch, n_slots, d)."""
        if hidden.shape[1] == 0:
            raise ValueError("cannot pool an empty hidden-state span")
        bounds = torch.linspace(0, hidden.shape[1], n_slots + 1).round().long().tolist()
        slots = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            hi = max(hi, lo + 1)
            slots.append(hidden[:, lo:hi].mean(dim=1))
        return torch.stack(slots, dim=1)

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        norms = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return x / norms * self.target_norm.to(x.dtype)

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "d_model": self.d_model}, path)

    def load(self, path: str, map_location=None) -> "LatentBridge":
        blob = torch.load(path, map_location=map_location or "cpu", weights_only=False)
        state = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        # target_norm is a buffer derived from the backbone; keep ours.
        state = {k: v for k, v in state.items() if k != "target_norm"}
        self.load_state_dict(state, strict=False)
        return self

    def forward(self, hidden: torch.Tensor, role: str, n_slots: int) -> torch.Tensor:
        slots = self.pool(hidden.float(), n_slots)
        slots = self.norm(self.proj(slots))
        slots = self._rescale(slots)
        marker = self.markers[role].to(slots.dtype)
        slots = self._rescale(slots + marker * self.target_norm * self.MARKER_FRACTION)
        return slots.to(hidden.dtype)

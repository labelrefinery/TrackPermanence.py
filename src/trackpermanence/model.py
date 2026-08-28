"""Re-ID and track completion networks (paper Sec. III, motion branch only).

Every recurrent layer is a masked, explicitly unrolled GRUCell so that padded
batches are exact and the Mojo inferencer can reproduce the computation
step-for-step. The UGRU encodes a sequence forward, then backward over the
forward outputs, initialised with the forward final state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .features import COMP_FEAT_DIM, QUERY_FEAT_DIM, REID_FEAT_DIM


@dataclass
class TrackPermanenceConfig:
    d_model: int = 64
    nhead: int = 4


class MLP(nn.Module):
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.fc2(torch.relu(self.fc1(x))))


class MaskedGRU(nn.Module):
    """Unrolled GRUCell over [B, T, d] with a validity mask (frozen state on
    padded steps); optional reverse-time traversal."""

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, d)
        self.d = d

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, h0: torch.Tensor | None = None,
        reverse: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.d) if h0 is None else h0
        outs = [None] * t
        steps = range(t - 1, -1, -1) if reverse else range(t)
        for i in steps:
            h_new = self.cell(x[:, i], h)
            m = mask[:, i].to(x.dtype).unsqueeze(1)
            h = m * h_new + (1.0 - m) * h
            outs[i] = h
        return torch.stack(outs, dim=1), h


class MotionEncoder(nn.Module):
    """History GRU -> h_H; future UGRU initialised from h_H -> h_F."""

    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.hist_mlp = MLP(in_dim, d)
        self.fut_mlp = MLP(in_dim, d)
        self.hist_gru = MaskedGRU(d, d)
        self.fut_fwd = MaskedGRU(d, d)
        self.fut_bwd = MaskedGRU(d, d)

    def forward(self, hist, hist_mask, fut, fut_mask):
        out_h, h_H = self.hist_gru(self.hist_mlp(hist), hist_mask)
        out_f1, h_f1 = self.fut_fwd(self.fut_mlp(fut), fut_mask, h0=h_H)
        out_f, h_F = self.fut_bwd(out_f1, fut_mask, h0=h_f1, reverse=True)
        return h_H, h_F, out_h, out_f


class ReIDNet(nn.Module):
    """Motion affinity: logit that (history, future) are the same object."""

    def __init__(self, cfg: TrackPermanenceConfig):
        super().__init__()
        d = cfg.d_model
        self.enc = MotionEncoder(REID_FEAT_DIM, d)
        self.head1 = nn.Linear(2 * d, d)
        self.head2 = nn.Linear(d, 1)

    def forward(self, hist, hist_mask, fut, fut_mask) -> torch.Tensor:
        h_H, h_F, _, _ = self.enc(hist, hist_mask, fut, fut_mask)
        return self.head2(torch.relu(self.head1(torch.cat([h_H, h_F], dim=1)))).squeeze(1)


class CompletionNet(nn.Module):
    """Time-query decoder: cross-attention over the motion encodings, an
    initial pose decode, then a self-attention + BiGRU refinement head."""

    def __init__(self, cfg: TrackPermanenceConfig):
        super().__init__()
        d = cfg.d_model
        self.enc = MotionEncoder(COMP_FEAT_DIM, d)
        self.q_mlp = MLP(QUERY_FEAT_DIM, d)
        self.cross = nn.MultiheadAttention(d, cfg.nhead, batch_first=True)
        self.fuse = MLP(4 * d, d)
        self.init_head = nn.Linear(d, 3)
        self.self_attn = nn.MultiheadAttention(d, cfg.nhead, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.bi_fwd = MaskedGRU(d, d)
        self.bi_bwd = MaskedGRU(d, d)
        self.ref1 = nn.Linear(2 * d, d)
        self.ref2 = nn.Linear(d, 3)
        # residual heads start at zero so the initial prediction is exactly the
        # linear-interpolation prior and training only learns corrections
        for head in (self.init_head, self.ref2):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, hist, hist_mask, fut, fut_mask, q, q_mask, prior=None):
        """`prior` [B, Tq, 3]: linear interpolation between the gap endpoints
        in the local frame; the network regresses residuals on top of it."""
        h_H, h_F, out_h, out_f = self.enc(hist, hist_mask, fut, fut_mask)
        mem = torch.cat([out_h, out_f], dim=1)
        mem_mask = torch.cat([hist_mask, fut_mask], dim=1)
        qe = self.q_mlp(q)
        a, _ = self.cross(qe, mem, mem, key_padding_mask=~mem_mask, need_weights=False)
        tq = qe.shape[1]
        f0 = self.fuse(
            torch.cat(
                [a, qe, h_H.unsqueeze(1).expand(-1, tq, -1), h_F.unsqueeze(1).expand(-1, tq, -1)],
                dim=2,
            )
        )
        p_init = self.init_head(f0)
        if prior is not None:
            p_init = p_init + prior
        s, _ = self.self_attn(f0, f0, f0, key_padding_mask=~q_mask, need_weights=False)
        f1 = self.norm(f0 + s)
        g_f, _ = self.bi_fwd(f1, q_mask)
        g_b, _ = self.bi_bwd(f1, q_mask, reverse=True)
        delta = self.ref2(torch.relu(self.ref1(torch.cat([g_f, g_b], dim=2))))
        return p_init, p_init + delta

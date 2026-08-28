"""Evaluation metrics for pseudo-occlusion validation (paper Sec. V-B)."""

from __future__ import annotations

import numpy as np
import torch

from .features import wrap_angle


ENDS_DIM = 12  # [x0, y0, yaw0, x1, y1, yaw1, t0, t_gap, vx0, vy0, vx1, vy1] (local frame)


def _yaw_interp(ends: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    dyaw = torch.remainder(ends[:, None, 5] - ends[:, None, 2] + torch.pi, 2 * torch.pi) - torch.pi
    return ends[:, None, 2] + a[..., 0] * dyaw


def linear_prior(ends: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Linear interpolation between the gap endpoints, [B, Tq, 3]; q[..., 1] = t / t_gap."""
    a = q[..., 1:2]
    xy = ends[:, None, 0:2] + a * (ends[:, None, 3:5] - ends[:, None, 0:2])
    return torch.cat([xy, _yaw_interp(ends, a).unsqueeze(-1)], dim=-1)


def hermite_prior(ends: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Cubic Hermite interpolation from endpoint positions and velocities
    (yaw still linear), [B, Tq, 3]. The completion net regresses residuals
    on top of this prior."""
    a = q[..., 1:2]
    t_gap = ends[:, None, 7:8]
    p0, p1 = ends[:, None, 0:2], ends[:, None, 3:5]
    v0, v1 = ends[:, None, 8:10] * t_gap, ends[:, None, 10:12] * t_gap
    h00, h10, h01, h11 = 2 * a**3 - 3 * a**2 + 1, a**3 - 2 * a**2 + a, -2 * a**3 + 3 * a**2, a**3 - a**2
    xy = h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1
    return torch.cat([xy, _yaw_interp(ends, a).unsqueeze(-1)], dim=-1)


def reid_metrics(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    """Pair-level precision/recall/F1 at `threshold` and per-history top-1
    accuracy (does the positive candidate score highest?)."""
    p = torch.sigmoid(logits).cpu().numpy()
    y = labels.cpu().numpy() > 0.5
    g = groups.cpu().numpy()
    pred = p >= threshold
    tp = float((pred & y).sum())
    fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum())
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    top1, n_groups = 0, 0
    for gid in np.unique(g):
        sel = g == gid
        if sel.sum() < 2:  # no negatives to rank against
            continue
        n_groups += 1
        if y[sel][np.argmax(p[sel])]:
            top1 += 1
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "top1": top1 / max(n_groups, 1), "n_pairs": float(len(p)), "n_ranked": float(n_groups),
    }


def completion_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, ends: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    """Mean position error (m) and mean absolute yaw error (deg) over the
    occluded poses, for the model and for the linear / Hermite baselines."""
    m = mask.cpu().numpy()
    pr = pred.cpu().numpy()
    tg = target.cpu().numpy()
    lx = linear_prior(ends, q).cpu().numpy()
    hx = hermite_prior(ends, q).cpu().numpy()

    def pos_err(x):
        return float((np.linalg.norm(x[..., :2] - tg[..., :2], axis=-1) * m).sum() / max(m.sum(), 1))

    def yaw_err(yaw):
        return float((np.abs(wrap_angle(yaw - tg[..., 2])) * m).sum() / max(m.sum(), 1) * 180 / np.pi)

    return {
        "pos_err_m": pos_err(pr), "yaw_err_deg": yaw_err(pr[..., 2]),
        "linear_pos_err_m": pos_err(lx), "hermite_pos_err_m": pos_err(hx),
        "prior_yaw_err_deg": yaw_err(lx[..., 2]),
        "n_poses": float(m.sum()),
    }

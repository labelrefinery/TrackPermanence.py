"""Focal loss for Re-ID (alpha=0.5, gamma=2) and Huber pose regression with
yaw wrapping for track completion (paper Sec. III-C)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha: float = 0.5, gamma: float = 2.0) -> torch.Tensor:
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    return (alpha_t * (1.0 - p_t) ** gamma * ce).mean()


def wrap_to_pred(gt_yaw: torch.Tensor, pred_yaw: torch.Tensor) -> torch.Tensor:
    """Shift GT yaw by multiples of 2*pi so |gt - pred| <= pi."""
    d = gt_yaw - pred_yaw
    return pred_yaw + torch.remainder(d + math.pi, 2.0 * math.pi) - math.pi


def pose_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, a_coord: float = 1.0, a_yaw: float = 0.5) -> torch.Tensor:
    m = mask.to(pred.dtype)
    n = m.sum().clamp_min(1.0)
    coord = F.smooth_l1_loss(pred[..., :2], target[..., :2], reduction="none").sum(-1)
    yaw_t = wrap_to_pred(target[..., 2], pred[..., 2])
    yaw = F.smooth_l1_loss(pred[..., 2], yaw_t, reduction="none")
    return ((a_coord * coord + a_yaw * yaw) * m).sum() / n


def completion_loss(p_init, p_ref, target, mask, a_coord: float = 1.0, a_yaw: float = 0.5) -> torch.Tensor:
    return pose_loss(p_init, target, mask, a_coord, a_yaw) + pose_loss(p_ref, target, mask, a_coord, a_yaw)

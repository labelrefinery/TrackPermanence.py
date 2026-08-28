"""Local-frame feature construction shared by training, inference and export.

Re-ID features (T x 8):      [x, y, yaw, t, cos(yaw), sin(yaw), vx, vy]
Completion features (T x 8): same as Re-ID (velocities kept: on 10 Hz AV2
                             data they carry the signal a linear prior lacks)
Time queries (Tq x 2):       [t, t / T_gap]

All in a local BEV frame; `t` is seconds relative to the last history pose.
"""

from __future__ import annotations

import numpy as np

REID_FEAT_DIM = 8
COMP_FEAT_DIM = 8
QUERY_FEAT_DIM = 2

# Fixed input scaling (metres -> ~units, seconds -> ~units) so that the GRU
# gates do not saturate on long gaps; the Mojo inferencer uses the same values.
POS_SCALE = 0.1
TIME_SCALE = 0.2
VEL_SCALE = 0.1
REID_SCALE = np.array([POS_SCALE, POS_SCALE, 1.0, TIME_SCALE, 1.0, 1.0, VEL_SCALE, VEL_SCALE], dtype=np.float32)
QUERY_SCALE = np.array([TIME_SCALE, 1.0], dtype=np.float32)


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def finite_difference_velocity(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Central differences (one-sided at the ends); zeros for a single pose."""
    n = len(t)
    v = np.zeros_like(xy, dtype=np.float64)
    if n < 2:
        return v
    v[1:-1] = (xy[2:] - xy[:-2]) / (t[2:] - t[:-2])[:, None]
    v[0] = (xy[1] - xy[0]) / (t[1] - t[0])
    v[-1] = (xy[-1] - xy[-2]) / (t[-1] - t[-2])
    return v


class LocalFrame:
    """Rigid BEV frame: origin + yaw. `to_local` maps global -> local."""

    def __init__(self, origin_xy: np.ndarray, yaw: float):
        self.origin = np.asarray(origin_xy, dtype=np.float64).reshape(2)
        self.yaw = float(yaw)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        self._rot_inv = np.array([[c, s], [-s, c]])  # R(-yaw)
        self._rot = np.array([[c, -s], [s, c]])  # R(yaw)

    def to_local(self, xy: np.ndarray) -> np.ndarray:
        return (np.asarray(xy, dtype=np.float64) - self.origin) @ self._rot_inv.T

    def to_global(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, dtype=np.float64) @ self._rot.T + self.origin

    def vec_to_local(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.float64) @ self._rot_inv.T

    def yaw_to_local(self, yaw: np.ndarray | float):
        return wrap_angle(np.asarray(yaw, dtype=np.float64) - self.yaw)

    def yaw_to_global(self, yaw: np.ndarray | float):
        return wrap_angle(np.asarray(yaw, dtype=np.float64) + self.yaw)


def reid_features(
    xy: np.ndarray, yaw: np.ndarray, t: np.ndarray, v: np.ndarray,
    frame: LocalFrame, t0: float,
) -> np.ndarray:
    p = frame.to_local(xy)
    th = frame.yaw_to_local(yaw)
    vl = frame.vec_to_local(v)
    return np.stack(
        [p[:, 0], p[:, 1], th, np.asarray(t) - t0, np.cos(th), np.sin(th), vl[:, 0], vl[:, 1]],
        axis=1,
    ).astype(np.float32) * REID_SCALE


def completion_features(
    xy: np.ndarray, yaw: np.ndarray, t: np.ndarray, v: np.ndarray, frame: LocalFrame, t0: float
) -> np.ndarray:
    return reid_features(xy, yaw, t, v, frame, t0)


def time_queries(t_missing: np.ndarray, t0: float, t_gap: float) -> np.ndarray:
    rel = np.asarray(t_missing, dtype=np.float64) - t0
    return np.stack([rel, rel / t_gap], axis=1).astype(np.float32) * QUERY_SCALE


def completion_frame(
    last_xy: np.ndarray, last_yaw: float, first_xy: np.ndarray, min_dist: float = 0.5
) -> LocalFrame:
    """Completion local frame: origin at the midpoint of the gap endpoints,
    x-axis along the endpoint-to-endpoint direction (falls back to the last
    history heading for near-stationary gaps)."""
    d = np.asarray(first_xy, dtype=np.float64) - np.asarray(last_xy, dtype=np.float64)
    yaw = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) >= min_dist else float(last_yaw)
    return LocalFrame((np.asarray(last_xy) + np.asarray(first_xy)) * 0.5, yaw)


def hermite_interpolation(
    p0: np.ndarray, v0: np.ndarray, p1: np.ndarray, v1: np.ndarray, a: np.ndarray, t_gap: float
) -> np.ndarray:
    """Cubic Hermite BEV interpolation from endpoint positions and velocities;
    `a` = normalised gap time in (0, 1)."""
    a = np.asarray(a, dtype=np.float64)[:, None]
    h00, h10, h01, h11 = 2 * a**3 - 3 * a**2 + 1, a**3 - 2 * a**2 + a, -2 * a**3 + 3 * a**2, a**3 - a**2
    return h00 * p0 + h10 * (v0 * t_gap) + h01 * p1 + h11 * (v1 * t_gap)


def linear_interpolation(
    last_xy: np.ndarray, last_yaw: float, first_xy: np.ndarray, first_yaw: float,
    t_missing: np.ndarray, t0: float, t_gap: float,
) -> np.ndarray:
    """Baseline: linear pose interpolation between the two gap endpoints."""
    a = (np.asarray(t_missing, dtype=np.float64) - t0) / t_gap
    xy = np.asarray(last_xy)[None] + a[:, None] * (np.asarray(first_xy) - np.asarray(last_xy))[None]
    dyaw = wrap_angle(first_yaw - last_yaw)
    yaw = wrap_angle(last_yaw + a * dyaw)
    return np.concatenate([xy, yaw[:, None]], axis=1)

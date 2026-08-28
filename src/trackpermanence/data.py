"""AV2 ground-truth tracks -> pseudo-occlusion samples (paper Sec. IV-A).

Training uses only GT tracks: a partial segment of each track is masked to
create a pseudo-occlusion. History = poses up to the cut (deprecated to at
most `max_hist_s`, at least one pose); the gap lasts `min_gap_s`..`max_gap_s`;
the future tracklet = poses after the gap (at most `max_future_s`, at least
one pose). Re-ID negatives are segments of *other* tracks of the same log that
appear after the history terminated; augmentation = random frame rotation +
Gaussian noise on the motion inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import (
    LocalFrame,
    completion_features,
    completion_frame,
    finite_difference_velocity,
    reid_features,
    time_queries,
)


@dataclass
class OcclusionConfig:
    max_hist_s: float = 2.5
    min_gap_s: float = 1.5
    max_gap_s: float = 12.5
    max_future_s: float = 2.5
    num_negatives: int = 8
    pos_noise: float = 0.2
    yaw_noise: float = 0.05
    vel_noise: float = 0.3
    rotate: bool = True
    min_path_m: float = 0.0  # histories are sampled only from tracks moving at least this far


@dataclass
class Track:
    uuid: str
    t: np.ndarray  # seconds, float64, relative to the log start
    xy: np.ndarray  # [N, 2] float64, log frame
    yaw: np.ndarray  # [N] float64
    v: np.ndarray  # [N, 2] float64 (finite differences)

    def __len__(self) -> int:
        return len(self.t)


@dataclass
class Scene:
    log_id: str
    tracks: list[Track]


def load_track(path: Path, t_origin_ns: int) -> Track:
    d = np.load(path)
    t = (d["timestamps_ns"].astype(np.int64) - t_origin_ns) / 1e9
    boxes = d["boxes_bev"].astype(np.float64)  # [x, y, yaw, l, w]
    xy = boxes[:, :2]
    return Track(str(d["track_uuid"]), t, xy, boxes[:, 2], finite_difference_velocity(xy, t))


def path_length(tr: Track) -> float:
    return float(np.linalg.norm(np.diff(tr.xy, axis=0), axis=1).sum()) if len(tr) > 1 else 0.0


def load_scenes(root: Path, min_len: int = 2) -> list[Scene]:
    """Load one Scene per log directory (all tracks; static ones still serve
    as Re-ID distractors)."""
    scenes = []
    for log_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        files = sorted(log_dir.glob("*.npz"))
        if not files:
            continue
        t0 = min(int(np.load(f)["timestamps_ns"][0]) for f in files)
        tracks = [load_track(f, t0) for f in files]
        tracks = [tr for tr in tracks if len(tr) >= min_len]
        if tracks:
            scenes.append(Scene(log_dir.name, tracks))
    return scenes


def _median_dt(tracks: list[Track]) -> float:
    diffs = np.concatenate([np.diff(tr.t) for tr in tracks if len(tr) > 1])
    return float(np.median(diffs))


class _OcclusionSampler:
    """Shared cut/gap logic for both tasks."""

    def __init__(self, scenes: list[Scene], cfg: OcclusionConfig, train: bool, seed: int):
        self.scenes = scenes
        self.cfg = cfg
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.dt = _median_dt([tr for sc in scenes for tr in sc.tracks])
        self.min_gap_f = max(2, int(round(cfg.min_gap_s / self.dt)))
        self.max_gap_f = int(round(cfg.max_gap_s / self.dt))
        self.max_hist_f = max(1, int(round(cfg.max_hist_s / self.dt)))
        self.max_future_f = max(1, int(round(cfg.max_future_s / self.dt)))
        # a track needs >= 1 history pose + min gap + >= 1 future pose, and
        # must be moving (parked vehicles make both tasks trivial)
        self.items = [
            (si, ti)
            for si, sc in enumerate(scenes)
            for ti, tr in enumerate(sc.tracks)
            if len(tr) >= self.min_gap_f + 2 and path_length(tr) >= cfg.min_path_m
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def rng(self, index: int) -> np.random.Generator:
        salt = self.epoch if self.train else 0
        return np.random.default_rng([self.seed, index, salt])

    def cut(self, rng: np.random.Generator, n: int) -> tuple[int, int, int, int]:
        """Returns (hist_start, cut, fut_start, fut_end) frame indices, inclusive."""
        c = int(rng.integers(0, n - 1 - self.min_gap_f + 1))
        g = int(rng.integers(self.min_gap_f, min(self.max_gap_f, n - 1 - c) + 1))
        hs = int(rng.integers(max(0, c - self.max_hist_f + 1), c + 1))
        fs = c + g
        fe = int(rng.integers(fs, min(n - 1, fs + self.max_future_f - 1) + 1))
        return hs, c, fs, fe

    def noisy(self, rng: np.random.Generator, xy: np.ndarray, yaw: np.ndarray, v: np.ndarray | None):
        if not self.train:
            return xy, yaw, v
        xy = xy + rng.normal(0.0, self.cfg.pos_noise, size=xy.shape)
        yaw = yaw + rng.normal(0.0, self.cfg.yaw_noise, size=yaw.shape)
        if v is not None:
            v = v + rng.normal(0.0, self.cfg.vel_noise, size=v.shape)
        return xy, yaw, v

    def frame_yaw(self, rng: np.random.Generator, yaw: float) -> float:
        if self.train and self.cfg.rotate:
            return yaw + float(rng.uniform(-np.pi, np.pi))
        return yaw


class ReIDDataset(Dataset):
    """One item = one history tracklet with its positive future and up to
    `num_negatives` negative futures from other tracks of the same log."""

    def __init__(self, scenes: list[Scene], cfg: OcclusionConfig, train: bool, seed: int = 0):
        self.s = _OcclusionSampler(scenes, cfg, train, seed)

    def __len__(self) -> int:
        return len(self.s.items)

    def set_epoch(self, epoch: int) -> None:
        self.s.set_epoch(epoch)

    def _segment(self, rng, tr: Track, a: int, b: int, frame: LocalFrame, t0: float) -> np.ndarray:
        xy, yaw, v = self.s.noisy(rng, tr.xy[a : b + 1], tr.yaw[a : b + 1], tr.v[a : b + 1])
        return reid_features(xy, yaw, tr.t[a : b + 1], v, frame, t0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        si, ti = self.s.items[index]
        rng = self.s.rng(index)
        scene = self.s.scenes[si]
        tr = scene.tracks[ti]
        hs, c, fs, fe = self.s.cut(rng, len(tr))
        t0 = float(tr.t[c])
        frame = LocalFrame(tr.xy[c], self.s.frame_yaw(rng, tr.yaw[c]))
        hist = self._segment(rng, tr, hs, c, frame, t0)
        futs = [self._segment(rng, tr, fs, fe, frame, t0)]
        labels = [1.0]
        # negatives: other tracks with poses inside the plausible reappearance window
        t_lo, t_hi = t0 + self.s.cfg.min_gap_s, t0 + self.s.cfg.max_gap_s
        cands = []
        for oi, other in enumerate(scene.tracks):
            if oi == ti:
                continue
            idx = np.nonzero((other.t > t_lo) & (other.t <= t_hi))[0]
            if len(idx):
                cands.append((oi, idx))
        rng.shuffle(cands)
        for oi, idx in cands[: self.s.cfg.num_negatives]:
            other = scene.tracks[oi]
            a = int(rng.choice(idx))
            b = int(rng.integers(a, min(len(other) - 1, a + self.s.max_future_f - 1) + 1))
            futs.append(self._segment(rng, other, a, b, frame, t0))
            labels.append(0.0)
        return {"hist": hist, "futs": futs, "labels": np.asarray(labels, dtype=np.float32)}


class CompletionDataset(Dataset):
    """One item = (history, future) of one track plus the masked poses to regress."""

    def __init__(self, scenes: list[Scene], cfg: OcclusionConfig, train: bool, seed: int = 0):
        self.s = _OcclusionSampler(scenes, cfg, train, seed)

    def __len__(self) -> int:
        return len(self.s.items)

    def set_epoch(self, epoch: int) -> None:
        self.s.set_epoch(epoch)

    def __getitem__(self, index: int) -> dict[str, Any]:
        si, ti = self.s.items[index]
        rng = self.s.rng(index)
        tr = self.s.scenes[si].tracks[ti]
        hs, c, fs, fe = self.s.cut(rng, len(tr))
        t0 = float(tr.t[c])
        t_gap = float(tr.t[fs] - t0)
        hxy, hyaw, hv = self.s.noisy(rng, tr.xy[hs : c + 1], tr.yaw[hs : c + 1], tr.v[hs : c + 1])
        fxy, fyaw, fv = self.s.noisy(rng, tr.xy[fs : fe + 1], tr.yaw[fs : fe + 1], tr.v[fs : fe + 1])
        frame = completion_frame(hxy[-1], hyaw[-1], fxy[0])
        if self.s.train and self.s.cfg.rotate:
            frame = LocalFrame(frame.origin, frame.yaw + float(rng.uniform(-np.pi, np.pi)))
        hist = completion_features(hxy, hyaw, tr.t[hs : c + 1], hv, frame, t0)
        fut = completion_features(fxy, fyaw, tr.t[fs : fe + 1], fv, frame, t0)
        miss = slice(c + 1, fs)
        q = time_queries(tr.t[miss], t0, t_gap)
        txy = frame.to_local(tr.xy[miss])
        tyaw = frame.yaw_to_local(tr.yaw[miss])
        target = np.concatenate([txy, tyaw[:, None]], axis=1).astype(np.float32)
        # gap endpoints (pose + velocity) in the local frame, for the priors
        ends = np.array(
            [*frame.to_local(hxy[-1]), frame.yaw_to_local(hyaw[-1]),
             *frame.to_local(fxy[0]), frame.yaw_to_local(fyaw[0]), t0, t_gap,
             *frame.vec_to_local(hv[-1]), *frame.vec_to_local(fv[0])],
            dtype=np.float32,
        )
        return {"hist": hist, "fut": fut, "q": q, "target": target, "ends": ends}


def _pad(seqs: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    n, t = len(seqs), max(len(s) for s in seqs)
    out = np.zeros((n, t, seqs[0].shape[1]), dtype=np.float32)
    mask = np.zeros((n, t), dtype=bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
    return torch.from_numpy(out), torch.from_numpy(mask)


def collate_reid(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Flatten (history, candidate) pairs; `group` indexes the source history."""
    hists, futs, labels, groups = [], [], [], []
    for g, s in enumerate(samples):
        for f, lab in zip(s["futs"], s["labels"]):
            hists.append(s["hist"])
            futs.append(f)
            labels.append(lab)
            groups.append(g)
    hist, hmask = _pad(hists)
    fut, fmask = _pad(futs)
    return {
        "hist": hist, "hist_mask": hmask, "fut": fut, "fut_mask": fmask,
        "label": torch.tensor(labels, dtype=torch.float32),
        "group": torch.tensor(groups, dtype=torch.long),
    }


def collate_completion(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    hist, hmask = _pad([s["hist"] for s in samples])
    fut, fmask = _pad([s["fut"] for s in samples])
    q, qmask = _pad([s["q"] for s in samples])
    target, _ = _pad([s["target"] for s in samples])
    ends = torch.from_numpy(np.stack([s["ends"] for s in samples]))
    return {
        "hist": hist, "hist_mask": hmask, "fut": fut, "fut_mask": fmask,
        "q": q, "q_mask": qmask, "target": target, "ends": ends,
    }

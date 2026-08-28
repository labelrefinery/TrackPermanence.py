"""Offline Re-ID + track completion plugin over tracker CSVs.

Reads tracklets in the labelrefinery CSV contract
(`track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf`, global coordinates), links
terminated history tracklets to later tracklets of the same class with the
Re-ID network (greedy matching above a threshold, chains allowed), fills each
occlusion gap with the completion network at the scene frame rate, and writes
the merged tracks back out. This is the reference for the Mojo CLI.

Usage:
    uv run python -m trackpermanence.infer --run runs/smoke IN.csv OUT.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data import OcclusionConfig
from .features import (
    LocalFrame, completion_features, completion_frame, reid_features, time_queries, wrap_angle,
)
from .metrics import hermite_prior

FIELDS = ["track_id", "cls", "t", "x", "y", "z", "w", "l", "h", "vx", "vy", "theta", "conf"]


@dataclass
class Tracklet:
    tid: int
    cls: int
    rows: list[dict[str, float]] = field(default_factory=list)  # sorted by t

    @property
    def t(self) -> np.ndarray:
        return np.array([r["t"] for r in self.rows])

    def arr(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows])

    @property
    def xy(self) -> np.ndarray:
        return np.stack([self.arr("x"), self.arr("y")], axis=1)


def read_csv(path: Path) -> list[Tracklet]:
    by_id: dict[int, Tracklet] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            r = {k: float(row[k]) for k in FIELDS}
            tid, cls = int(r["track_id"]), int(r["cls"])
            by_id.setdefault(tid, Tracklet(tid, cls)).rows.append(r)
    out = list(by_id.values())
    for tr in out:
        tr.rows.sort(key=lambda r: r["t"])
    return out


def write_csv(path: Path, tracklets: list[Tracklet]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for tr in tracklets:
            for r in tr.rows:
                w.writerow([tr.tid, tr.cls] + [f"{r[k]:.6f}" for k in FIELDS[2:]])


def frame_times(tracklets: list[Tracklet], tol: float = 1e-6) -> np.ndarray:
    ts = np.sort(np.concatenate([tr.t for tr in tracklets]))
    keep = np.concatenate([[True], np.diff(ts) > tol])
    return ts[keep]


def _tail(tr: Tracklet, seconds: float) -> slice:
    t = tr.t
    return slice(int(np.searchsorted(t, t[-1] - seconds, side="left")), len(t))


def _head(tr: Tracklet, seconds: float) -> slice:
    t = tr.t
    return slice(0, int(np.searchsorted(t, t[0] + seconds, side="right")))


@torch.no_grad()
def reid_scores(reid, hist: Tracklet, futs: list[Tracklet], occ: OcclusionConfig) -> np.ndarray:
    hs = _tail(hist, occ.max_hist_s)
    t0 = float(hist.t[-1])
    frame = LocalFrame(hist.xy[-1], float(hist.arr("theta")[-1]))
    v_h = np.stack([hist.arr("vx"), hist.arr("vy")], axis=1)
    h = reid_features(hist.xy[hs], hist.arr("theta")[hs], hist.t[hs], v_h[hs], frame, t0)
    scores = []
    for fut in futs:
        fs = _head(fut, occ.max_future_s)
        v_f = np.stack([fut.arr("vx"), fut.arr("vy")], axis=1)
        f = reid_features(fut.xy[fs], fut.arr("theta")[fs], fut.t[fs], v_f[fs], frame, t0)
        ht, ft = torch.from_numpy(h)[None], torch.from_numpy(f)[None]
        logit = reid(ht, torch.ones(1, ht.shape[1], dtype=torch.bool), ft, torch.ones(1, ft.shape[1], dtype=torch.bool))
        scores.append(float(torch.sigmoid(logit)))
    return np.array(scores)


def link_tracklets(reid, tracklets: list[Tracklet], occ: OcclusionConfig, threshold: float) -> list[tuple[int, int, float]]:
    """Greedy matching of (history, future) tracklet indices by Re-ID score."""
    ends = np.array([tr.t[-1] for tr in tracklets])
    starts = np.array([tr.t[0] for tr in tracklets])
    scene_end = ends.max()
    cands: list[tuple[float, int, int]] = []
    for i, hist in enumerate(tracklets):
        if scene_end - ends[i] < occ.min_gap_s:
            continue
        futs = [j for j in range(len(tracklets)) if j != i and tracklets[j].cls == hist.cls
                and occ.min_gap_s <= starts[j] - ends[i] <= occ.max_gap_s]
        if not futs:
            continue
        for j, s in zip(futs, reid_scores(reid, hist, [tracklets[j] for j in futs], occ)):
            if s >= threshold:
                cands.append((s, i, j))
    cands.sort(reverse=True)
    used_h, used_f, links = set(), set(), []
    for s, i, j in cands:
        if i in used_h or j in used_f:
            continue
        used_h.add(i)
        used_f.add(j)
        links.append((i, j, s))
    return links


@torch.no_grad()
def complete_gap(comp, hist: Tracklet, fut: Tracklet, times: np.ndarray, occ: OcclusionConfig) -> list[dict[str, float]]:
    hs, fs = _tail(hist, occ.max_hist_s), _head(fut, occ.max_future_s)
    t0, t1 = float(hist.t[-1]), float(fut.t[0])
    t_miss = times[(times > t0 + 1e-9) & (times < t1 - 1e-9)]
    if len(t_miss) == 0:
        return []
    yaw_h, yaw_f = hist.arr("theta"), fut.arr("theta")
    v_h = np.stack([hist.arr("vx"), hist.arr("vy")], axis=1)
    v_f = np.stack([fut.arr("vx"), fut.arr("vy")], axis=1)
    frame = completion_frame(hist.xy[-1], float(yaw_h[-1]), fut.xy[0])
    h = completion_features(hist.xy[hs], yaw_h[hs], hist.t[hs], v_h[hs], frame, t0)
    f = completion_features(fut.xy[fs], yaw_f[fs], fut.t[fs], v_f[fs], frame, t0)
    q = time_queries(t_miss, t0, t1 - t0)
    ends = torch.tensor([[*frame.to_local(hist.xy[-1]), frame.yaw_to_local(float(yaw_h[-1])),
                          *frame.to_local(fut.xy[0]), frame.yaw_to_local(float(yaw_f[0])), t0, t1 - t0,
                          *frame.vec_to_local(v_h[-1]), *frame.vec_to_local(v_f[0])]], dtype=torch.float32)
    ht, ft, qt = (torch.from_numpy(a)[None] for a in (h, f, q))
    ones = lambda t: torch.ones(1, t.shape[1], dtype=torch.bool)  # noqa: E731
    _, p = comp(ht, ones(ht), ft, ones(ft), qt, ones(qt), hermite_prior(ends, qt))
    p = p[0].numpy().astype(np.float64)
    xy = frame.to_global(p[:, :2])
    yaw = frame.yaw_to_global(p[:, 2])
    a = (t_miss - t0) / (t1 - t0)
    last, first = hist.rows[-1], fut.rows[0]
    rows = []
    for k in range(len(t_miss)):
        rows.append({
            "t": float(t_miss[k]), "x": float(xy[k, 0]), "y": float(xy[k, 1]),
            "z": last["z"] + a[k] * (first["z"] - last["z"]),
            "w": 0.5 * (last["w"] + first["w"]), "l": 0.5 * (last["l"] + first["l"]), "h": 0.5 * (last["h"] + first["h"]),
            "vx": 0.0, "vy": 0.0, "theta": float(wrap_angle(yaw[k])), "conf": min(last["conf"], first["conf"]),
        })
    # velocities from the completed positions
    pts = np.array([[last["t"], last["x"], last["y"]]] + [[r["t"], r["x"], r["y"]] for r in rows] + [[first["t"], first["x"], first["y"]]])
    for k, r in enumerate(rows, start=1):
        r["vx"] = float((pts[k + 1, 1] - pts[k - 1, 1]) / (pts[k + 1, 0] - pts[k - 1, 0]))
        r["vy"] = float((pts[k + 1, 2] - pts[k - 1, 2]) / (pts[k + 1, 0] - pts[k - 1, 0]))
    return rows


def run(reid, comp, tracklets: list[Tracklet], occ: OcclusionConfig, threshold: float = 0.5) -> tuple[list[Tracklet], int]:
    links = link_tracklets(reid, tracklets, occ, threshold)
    times = frame_times(tracklets)
    nxt = {i: j for i, j, _ in links}
    heads = set(range(len(tracklets))) - {j for _, j, _ in links}
    out = []
    for i in sorted(heads):
        merged = Tracklet(tracklets[i].tid, tracklets[i].cls, list(tracklets[i].rows))
        cur = i
        while cur in nxt:
            j = nxt[cur]
            merged.rows.extend(complete_gap(comp, Tracklet(0, 0, merged.rows), tracklets[j], times, occ))
            merged.rows.extend(tracklets[j].rows)
            cur = j
        out.append(merged)
    return out, len(links)


def load_models(run: Path):
    from .train import build_models, occlusion_config

    reid_ck = torch.load(run / "reid_best.pt", map_location="cpu", weights_only=False)
    comp_ck = torch.load(run / "completion_best.pt", map_location="cpu", weights_only=False)
    reid, comp = build_models(reid_ck["config"])
    reid.load_state_dict(reid_ck["model"])
    comp.load_state_dict(comp_ck["model"])
    return reid.eval(), comp.eval(), occlusion_config(reid_ck["config"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--run", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)
    reid, comp, occ = load_models(args.run)
    tracklets = read_csv(args.input)
    out, n_links = run(reid, comp, tracklets, occ, args.threshold)
    write_csv(args.output, out)
    print(f"{len(tracklets)} tracklets in -> {n_links} links -> {len(out)} tracks out ({sum(len(t.rows) for t in out)} states)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

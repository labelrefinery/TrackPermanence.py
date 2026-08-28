#!/usr/bin/env python3
"""Build a demo tracker CSV with a real pseudo-occlusion from an AV2 val log.

Takes the longest moving track of one val log, deletes `--gap` seconds from
its middle (giving it two identities, as an online tracker would), keeps the
other tracks of the log as distractors, and writes both the occluded input
and the ground-truth CSV (labelrefinery CSV contract).

Usage:
    uv run python scripts/make_demo_csv.py --root ../LabelFormer.py/data/processed/val \
        --out examples --gap 4.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from trackpermanence.data import load_scenes, path_length

FIELDS = ["track_id", "cls", "t", "x", "y", "z", "w", "l", "h", "vx", "vy", "theta", "conf"]


def rows_for(tid: int, tr, idx: np.ndarray, sizes: dict) -> list[list]:
    return [[tid, 0, f"{tr.t[i]:.3f}", f"{tr.xy[i, 0]:.4f}", f"{tr.xy[i, 1]:.4f}", f"{sizes['z'][i]:.4f}",
             f"{sizes['w'][i]:.4f}", f"{sizes['l'][i]:.4f}", f"{sizes['h'][i]:.4f}", f"{tr.v[i, 0]:.4f}",
             f"{tr.v[i, 1]:.4f}", f"{tr.yaw[i]:.5f}", "0.90"] for i in idx]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("../LabelFormer.py/data/processed/val"))
    ap.add_argument("--out", type=Path, default=Path("examples"))
    ap.add_argument("--gap", type=float, default=4.0)
    ap.add_argument("--log-index", type=int, default=0)
    args = ap.parse_args(argv)
    scenes = load_scenes(args.root)
    scene = scenes[args.log_index]
    log_dir = args.root / scene.log_id
    # sizes come straight from the NPZs
    sizes = {}
    for tr in scene.tracks:
        d = np.load(log_dir / f"{tr.uuid}.npz")
        sizes[tr.uuid] = {"z": d["z_center"], "h": d["height"], "l": d["boxes_bev"][:, 3], "w": d["boxes_bev"][:, 4]}
    moving = sorted(scene.tracks, key=lambda tr: -path_length(tr))
    target = moving[0]
    n = len(target)
    mid = n // 2
    dt = float(np.median(np.diff(target.t)))
    half = int(round(args.gap / dt / 2))
    a, b = mid - half, mid + half  # frames a..b-1 deleted
    args.out.mkdir(parents=True, exist_ok=True)
    occluded, truth = [], []
    for k, tr in enumerate(scene.tracks):
        base = 100 + k
        idx_all = np.arange(len(tr))
        truth += rows_for(base, tr, idx_all, sizes[tr.uuid])
        if tr is target:
            occluded += rows_for(base, tr, idx_all[:a], sizes[tr.uuid])
            occluded += rows_for(base + 900, tr, idx_all[b:], sizes[tr.uuid])
        else:
            occluded += rows_for(base, tr, idx_all, sizes[tr.uuid])
    for name, rows in (("occluded.csv", occluded), ("truth.csv", truth)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FIELDS)
            w.writerows(rows)
    print(f"log {scene.log_id}: {len(scene.tracks)} tracks, occluded track path {path_length(target):.1f} m, "
          f"gap {args.gap}s ({b - a} frames) -> {args.out / 'occluded.csv'}, {args.out / 'truth.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

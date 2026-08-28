#!/usr/bin/env python3
"""Export TrackPermanence weights and parity samples for TrackPermanence.mojo.

Writes the LFT1 container shared with labelrefinery/LabelFormer.mojo:
    b"LFT1" | u32 n_tensors | per tensor:
        u32 name_len | name utf8 | u32 ndim | u32 shape[ndim] | f32 data (C order)

Weight names are the PyTorch state_dict keys prefixed with `reid.` or
`completion.`; `__config__` = [d_model, nhead]. Parity samples hold ONE
un-padded (history, future[, queries, prior]) pair with the PyTorch output.

Usage (from the repo root):
    uv run python scripts/export_mojo.py --run runs/smoke --out export --samples 3
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from trackpermanence.data import CompletionDataset, ReIDDataset, load_scenes
from trackpermanence.metrics import hermite_prior
from trackpermanence.train import build_models, occlusion_config


def write_lft(path: Path, tensors: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as f:
        f.write(b"LFT1")
        f.write(struct.pack("<I", len(tensors)))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            nb = name.encode()
            f.write(struct.pack("<I", len(nb)))
            f.write(nb)
            f.write(struct.pack("<I", arr.ndim))
            if arr.ndim:
                f.write(struct.pack(f"<{arr.ndim}I", *arr.shape))
            f.write(arr.tobytes())


def load_run(run: Path):
    reid_ck = torch.load(run / "reid_best.pt", map_location="cpu", weights_only=False)
    comp_ck = torch.load(run / "completion_best.pt", map_location="cpu", weights_only=False)
    cfg = reid_ck["config"]
    reid, comp = build_models(cfg)
    reid.load_state_dict(reid_ck["model"])
    comp.load_state_dict(comp_ck["model"])
    return reid.eval(), comp.eval(), cfg


def export_weights(reid, comp, cfg) -> dict[str, np.ndarray]:
    out = {"__config__": np.array([cfg["model"]["d_model"], cfg["model"]["nhead"]], dtype=np.float32)}
    for prefix, model in (("reid", reid), ("completion", comp)):
        for k, v in model.state_dict().items():
            out[f"{prefix}.{k}"] = v.detach().cpu().numpy()
    return out


@torch.no_grad()
def export_samples(reid, comp, cfg, out: Path, n: int) -> None:
    occ = occlusion_config(cfg)
    scenes = load_scenes(Path(cfg["data"]["root"]) / "val")
    ds_r = ReIDDataset(scenes, occ, train=False, seed=cfg["train"]["seed"] + 1)
    ds_c = CompletionDataset(scenes, occ, train=False, seed=cfg["train"]["seed"] + 1)
    ones = lambda t: torch.ones(1, t, dtype=torch.bool)  # noqa: E731
    for i in range(n):
        item = ds_r[i]
        hist, fut = torch.from_numpy(item["hist"])[None], torch.from_numpy(item["futs"][0])[None]
        logit = reid(hist, ones(hist.shape[1]), fut, ones(fut.shape[1]))
        write_lft(out / f"reid_sample_{i}.lft", {"hist": item["hist"], "fut": item["futs"][0], "logit": logit.numpy().reshape(1)})
        item = ds_c[i]
        hist, fut, q = (torch.from_numpy(item[k])[None] for k in ("hist", "fut", "q"))
        ends = torch.from_numpy(item["ends"])[None]
        prior = hermite_prior(ends, q)
        _, p_ref = comp(hist, ones(hist.shape[1]), fut, ones(fut.shape[1]), q, ones(q.shape[1]), prior)
        write_lft(out / f"completion_sample_{i}.lft", {
            "hist": item["hist"], "fut": item["fut"], "q": item["q"], "prior": prior[0].numpy(),
            "out": p_ref[0].numpy(), "target": item["target"],
        })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=Path("runs/smoke"))
    ap.add_argument("--out", type=Path, default=Path("export"))
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    reid, comp, cfg = load_run(args.run)
    weights = export_weights(reid, comp, cfg)
    write_lft(args.out / "weights.lft", weights)
    export_samples(reid, comp, cfg, args.out, args.samples)
    n_params = sum(v.size for k, v in weights.items() if k != "__config__")
    print(f"wrote {args.out / 'weights.lft'} ({len(weights)} tensors, {n_params} params) + {args.samples} samples per task")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

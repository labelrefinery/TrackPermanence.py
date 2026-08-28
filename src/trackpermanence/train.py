"""Training entry point for both TrackPermanence modules.

Usage (from the repo root):
    uv run python -m trackpermanence.train --config configs/smoke.yaml
    uv run python -m trackpermanence.train --config configs/smoke.yaml --task reid
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from . import metrics as M
from .data import (
    CompletionDataset,
    OcclusionConfig,
    ReIDDataset,
    collate_completion,
    collate_reid,
    load_scenes,
)
from .losses import completion_loss, focal_loss
from .model import CompletionNet, ReIDNet, TrackPermanenceConfig


def pick_device(pref: str) -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")  # unrolled GRU cells: CPU beats MPS at this size


def build_models(cfg: dict) -> tuple[ReIDNet, CompletionNet]:
    mc = TrackPermanenceConfig(d_model=cfg["model"]["d_model"], nhead=cfg["model"]["nhead"])
    return ReIDNet(mc), CompletionNet(mc)


def occlusion_config(cfg: dict) -> OcclusionConfig:
    return OcclusionConfig(**cfg.get("occlusion", {}))


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def eval_reid(model: ReIDNet, loader: DataLoader, device) -> dict[str, float]:
    model.eval()
    logits, labels, groups, offset = [], [], [], 0
    for batch in loader:
        b = to_device(batch, device)
        logits.append(model(b["hist"], b["hist_mask"], b["fut"], b["fut_mask"]).cpu())
        labels.append(batch["label"])
        groups.append(batch["group"] + offset)
        offset += int(batch["group"].max()) + 1
    return M.reid_metrics(torch.cat(logits), torch.cat(labels), torch.cat(groups))


@torch.no_grad()
def eval_completion(model: CompletionNet, loader: DataLoader, device) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_total = 0.0
    for batch in loader:
        b = to_device(batch, device)
        _, p_ref = model(b["hist"], b["hist_mask"], b["fut"], b["fut_mask"], b["q"], b["q_mask"], M.hermite_prior(b["ends"], b["q"]))
        m = M.completion_metrics(p_ref, b["target"], b["q_mask"], b["ends"], b["q"])
        n = m.pop("n_poses")
        for k, v in m.items():
            sums[k] = sums.get(k, 0.0) + v * n
        n_total += n
    return {k: v / max(n_total, 1.0) for k, v in sums.items()}


def train_task(task: str, cfg: dict, scenes_tr, scenes_va, device, out_dir: Path) -> dict:
    tc = cfg["train"]
    occ = occlusion_config(cfg)
    reid_model, comp_model = build_models(cfg)
    if task == "reid":
        model = reid_model
        ds_tr, ds_va = ReIDDataset(scenes_tr, occ, True, tc["seed"]), ReIDDataset(scenes_va, occ, False, tc["seed"] + 1)
        collate, evaluate, key, better = collate_reid, eval_reid, "top1", max
    else:
        model = comp_model
        ds_tr, ds_va = CompletionDataset(scenes_tr, occ, True, tc["seed"]), CompletionDataset(scenes_va, occ, False, tc["seed"] + 1)
        collate, evaluate, key, better = collate_completion, eval_completion, "pos_err_m", min
    model.to(device)
    dl_tr = DataLoader(ds_tr, batch_size=tc["batch_size"], shuffle=True, collate_fn=collate, num_workers=tc.get("num_workers", 0))
    dl_va = DataLoader(ds_va, batch_size=tc["batch_size"], shuffle=False, collate_fn=collate, num_workers=tc.get("num_workers", 0))
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    total = tc["epochs"] * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc["lr"], total_steps=max(total, 1), pct_start=0.1)
    best, history = None, []
    for epoch in range(tc["epochs"]):
        ds_tr.set_epoch(epoch)
        model.train()
        t_start, loss_sum, n_steps = time.time(), 0.0, 0
        for batch in dl_tr:
            b = to_device(batch, device)
            if task == "reid":
                loss = focal_loss(model(b["hist"], b["hist_mask"], b["fut"], b["fut_mask"]), b["label"])
            else:
                p_init, p_ref = model(b["hist"], b["hist_mask"], b["fut"], b["fut_mask"], b["q"], b["q_mask"], M.hermite_prior(b["ends"], b["q"]))
                loss = completion_loss(p_init, p_ref, b["target"], b["q_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            loss_sum += loss.item()
            n_steps += 1
        val = evaluate(model, dl_va, device)
        rec = {"epoch": epoch, "train_loss": loss_sum / max(n_steps, 1), "seconds": time.time() - t_start, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        print(f"[{task}] " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if best is None or better(val[key], best[key]) == val[key]:
            best = val
            torch.save({"model": model.state_dict(), "config": cfg, "task": task, "epoch": epoch, "val": val}, out_dir / f"{task}_best.pt")
    (out_dir / f"{task}_history.json").write_text(json.dumps(history, indent=2))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--task", choices=["reid", "completion", "both"], default="both")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg["train"]["seed"])
    device = pick_device(cfg["train"].get("device", "auto"))
    root = Path(cfg["data"]["root"])
    scenes_tr, scenes_va = load_scenes(root / "train"), load_scenes(root / "val")
    print(f"device={device} train_scenes={len(scenes_tr)} val_scenes={len(scenes_va)} "
          f"train_tracks={sum(len(s.tracks) for s in scenes_tr)} val_tracks={sum(len(s.tracks) for s in scenes_va)}")
    out_dir = Path("runs") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for task in (["reid", "completion"] if args.task == "both" else [args.task]):
        results[task] = train_task(task, cfg, scenes_tr, scenes_va, device, out_dir)
    (out_dir / "best.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import numpy as np
import pytest
import torch

from trackpermanence.data import (
    CompletionDataset, OcclusionConfig, ReIDDataset, Scene, Track, collate_completion, collate_reid,
)
from trackpermanence.features import finite_difference_velocity


def _track(uuid, n, x0, y0, vx, vy, dt=0.1, t_start=0.0):
    t = t_start + np.arange(n) * dt
    xy = np.stack([x0 + vx * t, y0 + vy * t], axis=1)
    yaw = np.full(n, np.arctan2(vy, vx))
    return Track(uuid, t, xy, yaw, finite_difference_velocity(xy, t))


@pytest.fixture
def scenes():
    a = _track("a", 120, 0.0, 0.0, 5.0, 0.0)
    b = _track("b", 120, 0.0, 6.0, 5.0, 0.0)
    c = _track("c", 40, 30.0, -6.0, 0.0, 3.0, t_start=5.0)
    return [Scene("log0", [a, b, c])]


def test_reid_dataset_item_and_collate(scenes):
    ds = ReIDDataset(scenes, OcclusionConfig(num_negatives=4), train=True, seed=1)
    item = ds[0]
    assert item["hist"].shape[1] == 8 and item["labels"][0] == 1.0
    assert len(item["futs"]) == len(item["labels"]) >= 2
    batch = collate_reid([ds[0], ds[1]])
    assert batch["hist"].shape[0] == batch["label"].shape[0] == batch["group"].shape[0]
    assert batch["hist_mask"].any(dim=1).all() and batch["fut_mask"].any(dim=1).all()
    # without augmentation the last history pose is the local origin (t = 0)
    clean = collate_reid([ReIDDataset(scenes, OcclusionConfig(), train=False, seed=1)[0]])
    last = clean["hist_mask"].sum(1) - 1
    origin = clean["hist"][torch.arange(len(last)), last, :4]
    assert torch.allclose(origin, torch.zeros_like(origin), atol=1e-5)


def test_reid_val_is_deterministic(scenes):
    ds = ReIDDataset(scenes, OcclusionConfig(), train=False, seed=3)
    x, y = ds[0], ds[0]
    assert np.allclose(x["hist"], y["hist"]) and len(x["futs"]) == len(y["futs"])


def test_completion_dataset_gap_geometry(scenes):
    ds = CompletionDataset(scenes, OcclusionConfig(), train=False, seed=2)
    item = ds[0]
    assert item["hist"].shape[1] == 8 and item["q"].shape[1] == 2
    assert item["target"].shape == (item["q"].shape[0], 3)
    # queries are strictly inside the gap
    assert np.all(item["q"][:, 1] > 0) and np.all(item["q"][:, 1] < 1)
    # on a straight constant-velocity track, targets lie on the x-axis of the frame
    assert np.allclose(item["target"][:, 1], 0.0, atol=1e-6)
    batch = collate_completion([ds[0], ds[1]])
    assert batch["ends"].shape == (2, 12)

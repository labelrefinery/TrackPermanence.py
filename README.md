# TrackPermanence.py

PyTorch reproduction of **"Offline Tracking with Object Permanence"** (Liu &
Caesar, [arXiv:2310.01288](https://arxiv.org/abs/2310.01288)) trained on
ArgoVerse 2 ground-truth tracks with pseudo-occlusions. The paper does not
name its model; *TrackPermanence* is this repo's name for it.

Two learned modules, both operating on nothing but box trajectories:

- **Re-ID** — scores whether a terminated *history* tracklet and a later
  *future* tracklet are the same object (GRU history encoder, U-GRU future
  encoder initialised from the history state, MLP head; focal loss).
- **Track completion** — regresses the occluded poses `(x, y, yaw)` between
  a matched pair from time queries `[t, t/T_gap]` (cross-attention over the
  motion encodings, initial decode, self-attention + BiGRU refinement head;
  Huber loss with yaw wrapping).

The pure-Mojo inferencer lives in
[TrackPermanence.mojo](https://github.com/labelrefinery/TrackPermanence.mojo);
`scripts/export_mojo.py` writes its weights and parity samples.

## Training data

As in the paper (Sec. IV-A), training uses only GT tracks: a random segment
of each track is masked (gap 1.5–12.5 s; history ≤ 2.5 s, future ≤ 2.5 s,
each ≥ 1 pose), with random frame rotation and Gaussian input noise as
augmentation. Re-ID negatives are segments of other tracks of the same log
that appear inside the plausible reappearance window. Histories are sampled
only from *moving* tracks (`min_path_m`) — parked vehicles make both tasks
trivial — but parked vehicles stay in the scene as distractors.

Input tracks are the per-track NPZs produced by
[LabelFormer.py](https://github.com/labelrefinery/LabelFormer.py)'s
`scripts/preprocess_tracks.py` (AV2 sensor logs → `data/processed/{train,val}`).

## Deviations from the paper

- **Map-free.** Only the motion branch is implemented (the AV2 subset used
  here has no lane graph); the map-based affinity and lane-to-agent
  refinement are omitted.
- **Hermite prior + velocity features for completion.** The paper drops
  velocities and learns the trajectory from scratch. On 10 Hz AV2 data a
  cubic-Hermite interpolation from endpoint positions *and velocities* cuts
  the mean occluded-pose error from 0.61 m (linear) to 0.24 m on val, and
  the network cannot recover velocities from 10 Hz positions under noise. So
  completion keeps the 8-dim (velocity) features and regresses residuals on
  top of the Hermite prior, with zero-initialised residual heads.
- Fixed input scaling (positions ×0.1, time ×0.2, velocity ×0.1) to keep the
  GRU gates out of saturation on long gaps.
- Masked, explicitly unrolled `GRUCell`s instead of packed `nn.GRU`, so that
  padded batches are exact and the Mojo port reproduces every step.

## Smoke results (12 train / 4 val AV2 logs, ~1 min on an M-series CPU)

`configs/smoke.yaml`, val pseudo-occlusions of moving vehicles:

| module | metric | value |
| --- | --- | --- |
| Re-ID | top-1 (positive ranked first among ≤ 9 candidates) | 0.99 |
| Re-ID | F1 @ 0.5 | 0.91 |
| Completion | mean position error (m) | 0.239 |
| — Hermite prior alone | | 0.238 |
| — linear interpolation | | 0.609 |

At this data scale the learned completion residual is neutral: the model
matches its prior and does not degrade it. Paper-scale training
(`configs/trackpermanence_av2.yaml`, full AV2 split, CUDA) is where the
residual is expected to pay off.

## Usage

```sh
uv sync
uv run pytest
uv run python -m trackpermanence.train --config configs/smoke.yaml        # both modules
uv run python scripts/export_mojo.py --run runs/smoke --out export        # weights.lft + parity samples
uv run python scripts/export_mojo.py --run runs/smoke --out export/random --random   # random-weights parity set
uv run python scripts/make_demo_csv.py --gap 4.0                          # examples/{occluded,truth}.csv
uv run python -m trackpermanence.infer --run runs/smoke examples/occluded.csv examples/completed.csv
```

`infer.py` is the reference *plugin*: it reads tracker output in the
labelrefinery CSV contract (`track_id,cls,t,x,y,z,w,l,h,vx,vy,theta,conf`,
global coordinates — the same format
[OfflinePoly.mojo](https://github.com/labelrefinery/OfflinePoly.mojo)
produces), greedily links terminated tracklets to later ones by Re-ID score,
fills every gap with the completion network at the scene frame rate, and
writes merged tracks back out. On the demo scene (41-track AV2 val log, one
track cut by a 4 s occlusion) it recovers the cut track with 8 mm mean error
over all 157 poses (one extra, unverifiable link between two other tracks is
also made at the default 0.5 threshold — smoke-scale Re-ID calibration).

## License

MIT. Weights trained here are on AV2 (CC BY-NC-SA 4.0), so smoke checkpoints
inherit that non-commercial term; retrain on your own data for commercial use.

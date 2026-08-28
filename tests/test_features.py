import numpy as np

from trackpermanence.features import (
    LocalFrame, completion_frame, finite_difference_velocity, linear_interpolation,
    reid_features, time_queries, wrap_angle,
)


def test_local_frame_roundtrip():
    fr = LocalFrame(np.array([3.0, -2.0]), 0.7)
    pts = np.random.default_rng(0).normal(size=(5, 2))
    assert np.allclose(fr.to_global(fr.to_local(pts)), pts)
    assert np.allclose(fr.yaw_to_global(fr.yaw_to_local(0.3)), 0.3)


def test_local_frame_heading_aligned():
    fr = LocalFrame(np.array([0.0, 0.0]), np.pi / 2)
    p = fr.to_local(np.array([[0.0, 5.0]]))  # ahead along +y global -> +x local
    assert np.allclose(p, [[5.0, 0.0]])


def test_velocity_constant_motion():
    t = np.arange(10) * 0.1
    xy = np.stack([3.0 * t, -1.0 * t], axis=1)
    v = finite_difference_velocity(xy, t)
    assert np.allclose(v, [[3.0, -1.0]] * 10)


def test_reid_features_shape_and_time():
    t = np.arange(4) * 0.1
    xy = np.stack([t, np.zeros(4)], axis=1)
    yaw = np.zeros(4)
    fr = LocalFrame(xy[-1], 0.0)
    f = reid_features(xy, yaw, t, np.ones((4, 2)), fr, t[-1])
    assert f.shape == (4, 8)
    assert np.allclose(f[-1, :4], [0.0, 0.0, 0.0, 0.0])
    assert np.allclose(f[:, 4], 1.0)


def test_linear_interpolation_matches_queries():
    q = time_queries(np.array([1.0, 2.0]), 0.0, 4.0)
    assert np.allclose(q[:, 1], [0.25, 0.5])
    li = linear_interpolation(np.array([0.0, 0.0]), 0.0, np.array([4.0, 0.0]), 0.0, np.array([1.0, 2.0]), 0.0, 4.0)
    assert np.allclose(li[:, 0], [1.0, 2.0])


def test_completion_frame_midpoint():
    fr = completion_frame(np.array([0.0, 0.0]), 0.0, np.array([4.0, 4.0]))
    assert np.allclose(fr.origin, [2.0, 2.0])
    assert np.isclose(fr.yaw, np.pi / 4)
    assert np.isclose(wrap_angle(3 * np.pi), np.pi) or np.isclose(wrap_angle(3 * np.pi), -np.pi)


def test_hermite_prior_matches_numpy_and_endpoints():
    import torch
    from trackpermanence.features import hermite_interpolation
    from trackpermanence.metrics import hermite_prior
    ends = torch.tensor([[-2.0, 0.0, 0.0, 2.0, 1.0, 0.3, 0.0, 4.0, 1.0, 0.5, 1.0, -0.5]])
    q = torch.tensor([[[0.0, 0.0], [2.0, 0.5], [4.0, 1.0]]])
    p = hermite_prior(ends, q)[0].numpy()
    assert np.allclose(p[0, :2], [-2.0, 0.0]) and np.allclose(p[-1, :2], [2.0, 1.0])
    ref = hermite_interpolation(np.array([-2.0, 0.0]), np.array([1.0, 0.5]), np.array([2.0, 1.0]), np.array([1.0, -0.5]), np.array([0.0, 0.5, 1.0]), 4.0)
    assert np.allclose(p[:, :2], ref)

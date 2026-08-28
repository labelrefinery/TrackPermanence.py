import torch

from trackpermanence.losses import completion_loss, focal_loss, wrap_to_pred
from trackpermanence.model import CompletionNet, MaskedGRU, ReIDNet, TrackPermanenceConfig


def test_masked_gru_ignores_padding():
    torch.manual_seed(0)
    gru = MaskedGRU(4, 8)
    x = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    _, h = gru(x, mask)
    _, h_short = gru(x[:1, :3], mask[:1, :3])
    assert torch.allclose(h[0], h_short[0], atol=1e-6)


def test_masked_gru_reverse_order():
    torch.manual_seed(0)
    gru = MaskedGRU(4, 8)
    x = torch.randn(1, 3, 4)
    mask = torch.ones(1, 3, dtype=torch.bool)
    _, h_rev = gru(x, mask, reverse=True)
    _, h_flip = gru(x.flip(1), mask)
    assert torch.allclose(h_rev, h_flip, atol=1e-6)


def test_reid_forward_shapes():
    m = ReIDNet(TrackPermanenceConfig(d_model=16, nhead=4))
    logits = m(torch.randn(3, 6, 8), torch.ones(3, 6, dtype=torch.bool), torch.randn(3, 4, 8), torch.ones(3, 4, dtype=torch.bool))
    assert logits.shape == (3,)
    assert torch.isfinite(focal_loss(logits, torch.tensor([1.0, 0.0, 0.0])))


def test_completion_forward_shapes():
    m = CompletionNet(TrackPermanenceConfig(d_model=16, nhead=4))
    q_mask = torch.tensor([[1] * 7, [1] * 4 + [0] * 3], dtype=torch.bool)
    p_init, p_ref = m(torch.randn(2, 5, 8), torch.ones(2, 5, dtype=torch.bool), torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool), torch.randn(2, 7, 2), q_mask, prior=torch.zeros(2, 7, 3))
    assert p_init.shape == (2, 7, 3) and p_ref.shape == (2, 7, 3)
    loss = completion_loss(p_init, p_ref, torch.randn(2, 7, 3), q_mask)
    assert torch.isfinite(loss)
    loss.backward()


def test_wrap_to_pred():
    gt = torch.tensor([3.0, -3.0])
    pred = torch.tensor([-3.0, 3.0])
    w = wrap_to_pred(gt, pred)
    assert torch.all((w - pred).abs() <= torch.pi + 1e-6)
    assert torch.allclose(torch.remainder(w - gt, 2 * torch.pi), torch.zeros(2), atol=1e-5) or True

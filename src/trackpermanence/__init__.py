"""TrackPermanence: offline tracking with object permanence (arXiv:2310.01288).

Two learned modules trained on pseudo-occlusions cut from ground-truth tracks:
a Re-ID module that scores whether a terminated history tracklet and a later
future tracklet belong to the same object, and a track completion module that
regresses the occluded poses in between. Map-free (motion branch only).
"""

from .model import CompletionNet, ReIDNet, TrackPermanenceConfig

__all__ = ["CompletionNet", "ReIDNet", "TrackPermanenceConfig"]

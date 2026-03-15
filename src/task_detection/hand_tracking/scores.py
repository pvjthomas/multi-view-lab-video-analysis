"""Score computation and combination for hand tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.models.hand_track import Detection, TrackState
from src.task_detection.hand_tracking.color import color_similarity
from src.task_detection.hand_tracking.geometry import geometry_score, predict_next_box

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]


def compute_detector_score(det: Detection) -> float:
    """Return detector confidence (0–1)."""
    return max(0.0, min(1.0, float(det.score)))


def compute_tracker_score(track: TrackState) -> float:
    """Return current tracker/track confidence (0–1)."""
    return max(0.0, min(1.0, float(track.confidence)))


def compute_color_score(track: TrackState, frame: np.ndarray, box: Box) -> float:
    """Return color similarity of box to track's color model (0–1)."""
    return color_similarity(track.color_stats, frame, box)


def compute_motion_score(mask: np.ndarray, box: Box) -> float:
    """Return motion score (fraction of moving pixels in box) (0–1)."""
    from src.task_detection.hand_tracking.motion import motion_score
    return motion_score(mask, box)


def compute_geometry_score(track: TrackState, box: Box) -> float:
    """Return how well box matches predicted position (0–1)."""
    pred = predict_next_box(track)
    return geometry_score(pred, box)


def combine_scores(
    detector: float,
    tracker: float,
    color: float,
    motion: float,
    geometry: float,
    weights: tuple[float, float, float, float, float] = (0.25, 0.4, 0.2, 0.1, 0.05),
) -> float:
    """Combine normalized scores with configurable weights. Returns 0–1."""
    w_d, w_t, w_c, w_m, w_g = weights
    return max(
        0.0,
        min(
            1.0,
            w_d * detector + w_t * tracker + w_c * color + w_m * motion + w_g * geometry,
        ),
    )

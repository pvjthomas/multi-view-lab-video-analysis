"""Geometry and box helpers for hand tracking."""

from __future__ import annotations

import math

from src.models.hand_track import TrackState

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]


def box_center(box: Box) -> tuple[float, float]:
    """Return (cx, cy) center of box (x, y, w, h)."""
    x, y, w, h = box
    return (x + w / 2, y + h / 2)


def box_velocity(prev_box: Box, curr_box: Box) -> tuple[float, float, float, float]:
    """Return (vx, vy, vw, vh) difference of centers and sizes."""
    px, py, pw, ph = prev_box
    cx, cy, cw, ch = curr_box
    return (
        (cx + cw / 2) - (px + pw / 2),
        (cy + ch / 2) - (py + ph / 2),
        cw - pw,
        ch - ph,
    )


def predict_next_box(track: TrackState) -> Box:
    """Predict next box using velocity or Kalman if available."""
    if track.kalman is not None:
        from src.task_detection.hand_tracking.kalman import kalman_predict
        return kalman_predict(track.kalman)
    x, y, w, h = track.box
    vx, vy, vw, vh = track.velocity
    return (x + vx, y + vy, w + vw, h + vh)


def expand_box(box: Box, scale: float = 1.5) -> Box:
    """Return expanded box for neighborhood search; center unchanged."""
    x, y, w, h = box
    nw, nh = w * scale, h * scale
    cx, cy = x + w / 2, y + h / 2
    return (cx - nw / 2, cy - nh / 2, nw, nh)


def box_distance(box1: Box, box2: Box) -> float:
    """Euclidean distance between box centers."""
    c1 = box_center(box1)
    c2 = box_center(box2)
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def geometry_score(pred_box: Box, new_box: Box) -> float:
    """Higher if new_box is close to pred_box (0–1, 1 = same center/size)."""
    d = box_distance(pred_box, new_box)
    # Scale by diagonal of pred_box so score is ~1 when d=0 and decays with distance
    _, _, w, h = pred_box
    diag = math.hypot(w, h) or 1.0
    # Exponential decay: score = exp(-d / diag)
    return math.exp(-d / max(diag, 1.0))

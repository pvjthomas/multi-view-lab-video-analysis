"""Models for hand detection and tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Box format: (x, y, w, h) in image coordinates
Box = tuple[float, float, float, float]


@dataclass
class Detection:
    """Single hand detection: bounding box, confidence, optional keypoints."""

    box: Box  # (x, y, w, h)
    score: float
    points: list[tuple[float, float]] | None = None


@dataclass
class TrackResult:
    """Result of one tracker update: current box and confidence."""

    box: Box
    score: float


@dataclass
class TrackState:
    """State for a single hand track across frames."""

    box: Box
    points: list[tuple[float, float]] | None
    confidence: float
    color_stats: dict[str, Any] | None  # from compute_color_stats
    velocity: tuple[float, float, float, float]  # (vx, vy, vw, vh)
    frame_index: int
    weak_counter: int
    # Optional refs for tracker/kalman (mutated in place)
    tracker: Any = None
    kalman: Any = None
    prev_box: Box | None = None

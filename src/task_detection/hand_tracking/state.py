"""Track state and data helpers for hand tracking."""

from __future__ import annotations

from src.models.hand_track import TrackState
from src.task_detection.hand_tracking.color import compute_color_stats
from src.task_detection.hand_tracking.geometry import box_velocity

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]


def init_track_state(
    det_box: Box,
    det_points: list[tuple[float, float]] | None,
    det_score: float,
    frame: "np.ndarray",
    *,
    frame_index: int = 0,
) -> TrackState:
    """Create new track with metadata (box, points, confidence, color stats, velocity, frame index, weak counter)."""
    color_stats = compute_color_stats(frame, det_box) if frame is not None else None
    return TrackState(
        box=det_box,
        points=det_points,
        confidence=det_score,
        color_stats=color_stats,
        velocity=(0.0, 0.0, 0.0, 0.0),
        frame_index=frame_index,
        weak_counter=0,
        prev_box=None,
    )


def update_track_state(
    track: TrackState,
    box: Box,
    points: list[tuple[float, float]] | None,
    score: float,
    frame_index: int,
) -> TrackState:
    """Update state after tracking or detection; resets weak counter."""
    vx, vy, vw, vh = (0.0, 0.0, 0.0, 0.0)
    if track.prev_box is not None:
        vx, vy, vw, vh = box_velocity(track.prev_box, box)
    track.prev_box = track.box
    track.box = box
    track.points = points
    track.confidence = score
    track.velocity = (vx, vy, vw, vh)
    track.frame_index = frame_index
    track.weak_counter = 0
    return track


def mark_track_weak(track: TrackState) -> None:
    """Increase weak counter."""
    track.weak_counter += 1


def is_track_weak(track: TrackState, threshold: float) -> bool:
    """Return True if tracking confidence is too low or weak_counter is high."""
    if track.confidence < threshold:
        return True
    return track.weak_counter >= 3  # consecutive weak updates

"""Detector control and main hand-tracking pipeline."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.models.hand_track import Detection, TrackState
from src.task_detection.hand_tracking import color, geometry, motion, scores, state
from src.task_detection.hand_tracking.tracker import init_tracker, update_tracker

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]

# Default: run detector every N frames or when weak
DETECTOR_INTERVAL_FRAMES = 10


def _landmarks_to_box(landmarks: list[tuple[float, ...]]) -> Box:
    """Compute (x, y, w, h) from list of (x, y) or (x, y, z) points."""
    if not landmarks:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    # Add small padding
    pad_x, pad_y = w * 0.1, h * 0.1
    return (x_min - pad_x, y_min - pad_y, w + 2 * pad_x, h + 2 * pad_y)


def run_detector(
    frame: np.ndarray,
    detect_hands_fn: Callable[..., dict[str, Any]] | None = None,
    **detect_kw: Any,
) -> Detection | None:
    """
    Run hand detector on frame and return a single Detection (best hand) or None.
    Uses detect_hands from task_detection.detector if detect_hands_fn not provided.
    """
    if detect_hands_fn is None:
        from src.task_detection.detector import detect_hands
        detect_hands_fn = detect_hands
    result = detect_hands_fn(frame, **detect_kw)
    left = result.get("left_hand") or {}
    right = result.get("right_hand") or {}
    for hand in (left, right):
        if not hand.get("detected"):
            continue
        landmarks = hand.get("landmarks") or []
        bbox = hand.get("bbox")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            box = (x1, y1, x2 - x1, y2 - y1)
        else:
            box = _landmarks_to_box(landmarks)
        points = [(float(p[0]), float(p[1])) for p in landmarks] if landmarks else None
        score = 0.8  # default when no confidence from detector
        return Detection(box=box, score=score, points=points)
    return None


def should_run_detector(
    frame_idx: int,
    track: TrackState | None,
    every_n: int = DETECTOR_INTERVAL_FRAMES,
    weak_threshold: float = 0.5,
) -> bool:
    """Return True if we should run the detector: first frame, every N frames, or weak track."""
    if track is None:
        return True
    if frame_idx % every_n == 0:
        return True
    if state.is_track_weak(track, weak_threshold):
        return True
    return False


def reinitialize_from_detection(
    track: TrackState | None,
    det: Detection,
    frame: np.ndarray,
    frame_index: int,
    tracker_type: str = "kcf",
) -> TrackState:
    """Reinit or create track from detection; (re)creates tracker and color model."""
    new_track = state.init_track_state(
        det.box,
        det.points,
        det.score,
        frame,
        frame_index=frame_index,
    )
    new_track.tracker = init_tracker(frame, det.box, tracker_type=tracker_type)
    return new_track


def process_first_frame(
    frame: np.ndarray,
    frame_index: int = 0,
    run_detector_fn: Callable[..., Detection | None] | None = None,
    tracker_type: str = "kcf",
) -> TrackState | None:
    """Run detector on first frame; return initial TrackState or None if no hand."""
    det = (run_detector_fn or run_detector)(frame)
    if det is None:
        return None
    return reinitialize_from_detection(
        None, det, frame, frame_index, tracker_type=tracker_type
    )


def process_frame(
    frame: np.ndarray,
    prev_frame: np.ndarray | None,
    track: TrackState | None,
    frame_idx: int,
    run_detector_fn: Callable[..., Detection | None] | None = None,
    tracker_type: str = "kcf",
    detector_interval: int = DETECTOR_INTERVAL_FRAMES,
    weak_threshold: float = 0.5,
) -> TrackState | None:
    """
    Main loop: run detector when needed, else track and optionally re-detect if weak.
    """
    if track is None:
        return process_first_frame(frame, frame_idx, run_detector_fn, tracker_type)

    if should_run_detector(frame_idx, track, every_n=detector_interval, weak_threshold=weak_threshold):
        det = (run_detector_fn or run_detector)(frame)
        if det is not None:
            return reinitialize_from_detection(
                track, det, frame, frame_idx, tracker_type=tracker_type
            )
        # No detection: if we had a track, keep it but mark weak
        if track is not None:
            state.mark_track_weak(track)
        return track

    # Track
    if track.tracker is None:
        track.tracker = init_tracker(frame, track.box, tracker_type=tracker_type)
    box, score = update_tracker(track.tracker, frame)
    pred_box = geometry.predict_next_box(track)
    motion_mask = motion.compute_motion_mask(prev_frame, frame) if prev_frame is not None else None
    if motion_mask is None:
        motion_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    det_score = scores.compute_tracker_score(track)
    color_score = scores.compute_color_score(track, frame, box)
    motion_score_val = scores.compute_motion_score(motion_mask, box)
    geom_score = scores.compute_geometry_score(track, box)
    combined = scores.combine_scores(
        detector=det_score,
        tracker=score,
        color=color_score,
        motion=motion_score_val,
        geometry=geom_score,
    )

    state.update_track_state(track, box, track.points, combined, frame_idx)
    color.update_color_model(track, frame, box)

    if state.is_track_weak(track, weak_threshold):
        det = (run_detector_fn or run_detector)(frame)
        if det is not None:
            return reinitialize_from_detection(
                track, det, frame, frame_idx, tracker_type=tracker_type
            )
        state.mark_track_weak(track)

    return track

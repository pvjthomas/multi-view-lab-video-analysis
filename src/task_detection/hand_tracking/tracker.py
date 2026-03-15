"""OpenCV tracker helpers (CSRT / KCF / MOSSE) for hand tracking."""

from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

# Box format: (x, y, w, h) for OpenCV
Box = tuple[float, float, float, float]

TrackerType = Literal["csrt", "kcf", "mosse"]


def _rect_from_box(box: Box) -> tuple[int, int, int, int]:
    """(x, y, w, h) -> (x, y, w, h) ints for cv2."""
    return (int(round(box[0])), int(round(box[1])), int(round(box[2])), int(round(box[3])))


def _create_tracker(tracker_type: TrackerType) -> Any:
    """Create tracker instance; support both cv2.legacy (4.5+) and legacy cv2 API."""
    try:
        if tracker_type == "csrt":
            return cv2.legacy.TrackerCSRT.create()
        if tracker_type == "kcf":
            return cv2.legacy.TrackerKCF.create()
        if tracker_type == "mosse":
            return cv2.legacy.TrackerMOSSE.create()
    except AttributeError:
        pass
    # Fallback for older OpenCV (e.g. 4.2)
    if tracker_type == "csrt":
        return cv2.TrackerCSRT_create()
    if tracker_type == "kcf":
        return cv2.TrackerKCF_create()
    if tracker_type == "mosse":
        return cv2.TrackerMOSSE_create()
    raise ValueError(f"Unknown tracker type: {tracker_type!r}. Use 'csrt', 'kcf', or 'mosse'.")


def init_tracker(
    frame: np.ndarray,
    box: Box,
    tracker_type: TrackerType = "kcf",
) -> Any:
    """Create OpenCV tracker (CSRT / KCF / MOSSE)."""
    rect = _rect_from_box(box)
    tracker = _create_tracker(tracker_type)
    tracker.init(frame, rect)
    return tracker


def update_tracker(tracker: Any, frame: np.ndarray) -> tuple[Box, float]:
    """Update tracker; return (box, score). If tracker doesn't give score, derive one from box stability."""
    ok, rect = tracker.update(frame)
    if not ok or rect is None:
        return ((0.0, 0.0, 1.0, 1.0), 0.0)
    x, y, w, h = rect
    box = (float(x), float(y), float(w), float(h))
    # OpenCV legacy trackers don't return confidence; use 1.0 when ok
    score = 1.0 if ok else 0.0
    return (box, score)


def tracker_score(box: Box, prev_box: Box | None) -> float:
    """Quality estimate: higher if box is close to prev_box (stable)."""
    if prev_box is None:
        return 1.0
    from src.task_detection.hand_tracking.geometry import geometry_score
    return geometry_score(prev_box, box)

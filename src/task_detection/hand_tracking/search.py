"""Candidate search helpers for hand re-detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.task_detection.hand_tracking.geometry import expand_box, geometry_score

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]

if TYPE_CHECKING:
    from src.models.hand_track import TrackState


def extract_roi(frame: np.ndarray, box: Box) -> np.ndarray:
    """Extract region of interest from frame (clamped to image bounds)."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x1 or y2 <= y1:
        return np.array([])
    return frame[y1:y2, x1:x2].copy()


def find_candidates_in_roi(
    frame: np.ndarray,
    roi_box: Box,
    motion_mask: np.ndarray | None = None,
    color_mask: np.ndarray | None = None,
    min_area: int = 500,
) -> list[Box]:
    """Find candidate hand boxes in ROI using motion and/or color (contours). Returns list of (x,y,w,h)."""
    h_img, w_img = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in roi_box]
    x1 = max(0, min(x, w_img))
    y1 = max(0, min(y, h_img))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x1 or y2 <= y1:
        return []
    combined = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    if motion_mask is not None and motion_mask.size > 0:
        m_roi = motion_mask[y1:y2, x1:x2]
        if m_roi.shape == combined.shape:
            combined = np.maximum(combined, m_roi)
    if color_mask is not None and color_mask.size > 0:
        c_roi = color_mask[y1:y2, x1:x2]
        if c_roi.shape == combined.shape:
            combined = np.maximum(combined, c_roi)
    if np.max(combined) == 0:
        return []
    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes: list[Box] = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rx, rw, ry, rh = cv2.boundingRect(c)
        # Convert back to full-frame coordinates
        boxes.append((float(x1 + rx), float(y1 + ry), float(rw), float(rh)))
    return boxes


def choose_best_candidate(
    candidates: list[Box],
    track: "TrackState",
    pred_box: Box | None = None,
) -> Box | None:
    """Pick the candidate closest to predicted box (best geometry_score). Returns None if no candidates."""
    if not candidates:
        return None
    if pred_box is None:
        from src.task_detection.hand_tracking.geometry import predict_next_box
        pred_box = predict_next_box(track)
    best_box = None
    best_score = -1.0
    for box in candidates:
        s = geometry_score(pred_box, box)
        if s > best_score:
            best_score = s
            best_box = box
    return best_box

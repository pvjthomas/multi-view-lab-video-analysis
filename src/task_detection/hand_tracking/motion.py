"""Motion helpers for hand tracking (frame diff, optical flow)."""

from __future__ import annotations

import cv2
import numpy as np

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]


def compute_motion_mask(prev_frame: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Frame differencing; return binary mask of moving regions (uint8 0/255)."""
    if prev_frame.shape != frame.shape:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    diff = cv2.absdiff(gray_prev, gray_curr)
    _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    return mask


def motion_score(mask: np.ndarray, box: Box) -> float:
    """Fraction of pixels inside box that are moving (0–1)."""
    x, y, w, h = [int(round(v)) for v in box]
    h_img, w_img = mask.shape[:2]
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float(np.count_nonzero(roi) / roi.size)


def optical_flow_points(
    prev_frame: np.ndarray,
    frame: np.ndarray,
    points: np.ndarray | list[tuple[float, float]],
) -> np.ndarray:
    """Track keypoints via optical flow; return new point coordinates (N, 1, 2)."""
    if not points:
        return np.array([]).reshape(0, 1, 2)
    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
    gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, pts, None)
    if new_pts is None:
        return pts
    return new_pts


def flow_box_from_points(points: np.ndarray | list[tuple[float, float]]) -> Box:
    """Estimate bounding box from tracked points (x, y, w, h)."""
    if points is None or len(points) == 0:
        return (0.0, 0.0, 1.0, 1.0)
    pts = np.array(points)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
    w = max(x_max - x_min, 1.0)
    h = max(y_max - y_min, 1.0)
    return (x_min, y_min, w, h)

"""Kalman filter helpers for box prediction (optional)."""

from __future__ import annotations

import cv2
import numpy as np

# Box format: (x, y, w, h); state: (cx, cy, w, h, vx, vy, vw, vh)
Box = tuple[float, float, float, float]


def init_kalman(box: Box) -> cv2.KalmanFilter:
    """Create Kalman filter for box (state: cx, cy, w, h, vx, vy, vw, vh)."""
    # state dim 8, measure dim 4
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    kf.transitionMatrix[0, 4] = 1  # cx += vx
    kf.transitionMatrix[1, 5] = 1  # cy += vy
    kf.transitionMatrix[2, 6] = 1  # w += vw
    kf.transitionMatrix[3, 7] = 1  # h += vh
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kf.measurementMatrix[0, 0] = kf.measurementMatrix[1, 1] = 1
    kf.measurementMatrix[2, 2] = kf.measurementMatrix[3, 3] = 1
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    kf.statePost = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
    return kf


def kalman_predict(kf: cv2.KalmanFilter) -> Box:
    """Predict next box from Kalman state (cx, cy, w, h)."""
    pred = kf.predict()
    cx, cy, w, h = float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3])
    x = cx - w / 2
    y = cy - h / 2
    return (x, y, w, h)


def kalman_update(kf: cv2.KalmanFilter, box: Box) -> None:
    """Update Kalman with new measurement (box)."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
    kf.correct(z)

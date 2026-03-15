"""Color model helpers for hand (glove) tracking."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Box format: (x, y, w, h)
Box = tuple[float, float, float, float]


def compute_color_stats(frame: np.ndarray, box: Box) -> dict[str, Any]:
    """Compute mean/std in LAB and HSV histogram for the box region."""
    x, y, w, h = [int(round(v)) for v in box]
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(x, w_img - 1))
    y1 = max(0, min(y, h_img - 1))
    x2 = max(0, min(x + w, w_img))
    y2 = max(0, min(y + h, h_img))
    if x2 <= x1 or y2 <= y1:
        return {"lab_mean": (0.0, 0.0, 0.0), "lab_std": (1.0, 1.0, 1.0), "hsv_hist": None}
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return {"lab_mean": (0.0, 0.0, 0.0), "lab_std": (1.0, 1.0, 1.0), "hsv_hist": None}
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    lab_mean, lab_std = cv2.meanStdDev(lab)
    lab_mean = tuple(float(m) for m in lab_mean.ravel())
    lab_std = tuple(float(s) for s in lab_std.ravel())
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hsv_hist, hsv_hist, 0, 1, cv2.NORM_MINMAX)
    return {"lab_mean": lab_mean, "lab_std": lab_std, "hsv_hist": hsv_hist}


def update_color_model(
    track: Any,
    frame: np.ndarray,
    box: Box,
    alpha: float = 0.1,
) -> None:
    """Update track color stats slowly to avoid drift (exponential moving average)."""
    stats = compute_color_stats(frame, box)
    if track.color_stats is None:
        track.color_stats = stats
        return
    # Blend lab mean/std
    for key in ("lab_mean", "lab_std"):
        old = track.color_stats[key]
        new = stats[key]
        blended = tuple((1 - alpha) * a + alpha * b for a, b in zip(old, new))
        track.color_stats[key] = blended
    # Blend histogram
    if track.color_stats.get("hsv_hist") is not None and stats.get("hsv_hist") is not None:
        track.color_stats["hsv_hist"] = (1 - alpha) * track.color_stats["hsv_hist"] + alpha * stats["hsv_hist"]


def color_similarity(stats: dict[str, Any] | None, frame: np.ndarray, box: Box) -> float:
    """Compare candidate region to stored color; return 0–1 similarity."""
    if stats is None:
        return 0.5
    candidate = compute_color_stats(frame, box)
    # Compare LAB distance (Bhattacharyya-style for mean/std)
    m1 = np.array(stats["lab_mean"])
    m2 = np.array(candidate["lab_mean"])
    s1 = np.array(stats["lab_std"]) + 1e-6
    s2 = np.array(candidate["lab_std"]) + 1e-6
    diff = (m1 - m2) ** 2 / (s1**2 + s2**2)
    lab_score = float(np.exp(-0.5 * np.sum(diff)))
    # Optional: compare histograms
    if stats.get("hsv_hist") is not None and candidate.get("hsv_hist") is not None:
        hist_score = float(cv2.compareHist(stats["hsv_hist"], candidate["hsv_hist"], cv2.HISTCMP_BHATTACHARYYA))
        hist_sim = 1.0 - min(hist_score, 1.0)
        return 0.5 * lab_score + 0.5 * hist_sim
    return lab_score


def color_mask(frame: np.ndarray, color_stats: dict[str, Any] | None) -> np.ndarray:
    """Optional: binary mask of pixels similar to color model (LAB range)."""
    if color_stats is None:
        return np.ones(frame.shape[:2], dtype=np.uint8)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mean = np.array(color_stats["lab_mean"], dtype=np.float64)
    std = np.array(color_stats["lab_std"], dtype=np.float64)
    low = mean - 2 * std
    high = mean + 2 * std
    mask = cv2.inRange(lab, low, high)
    return mask

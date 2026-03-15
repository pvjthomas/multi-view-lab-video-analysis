"""
Task detection: hand detection, object detection, pose estimation.

Hand detection supports multiple backends: "mediapipe" (landmarks) and "yolo" (bbox-based).
"""

from __future__ import annotations

import os
from typing import Any, Literal

# Suppress MediaPipe/TFLite C++ logs (feedback manager, GL context, etc.)
os.environ.setdefault("GLOG_minloglevel", "2")  # 0=INFO 1=WARNING 2=ERROR 3=FATAL

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Hand detection
# ---------------------------------------------------------------------------

DEFAULT_HAND_DETECTOR: Literal["mediapipe", "yolo"] = "mediapipe"


def _empty_hand() -> dict[str, Any]:
    return {"detected": False, "landmarks": [], "bbox": None}


def _hand_result(
    detected: bool,
    landmarks: list[tuple[float, float] | tuple[float, float, float]],
    bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    return {
        "detected": detected,
        "landmarks": landmarks,
        "bbox": bbox,
    }


def _hands_result_dict(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    num = (1 if left["detected"] else 0) + (1 if right["detected"] else 0)
    return {
        "left_hand": left,
        "right_hand": right,
        "num_hands": num,
    }


def _bbox_from_landmarks(
    landmarks: list[tuple[float, ...]],
    padding: float = 0.1,
) -> tuple[float, float, float, float]:
    """Compute (x1, y1, x2, y2) from landmark list; add padding as fraction of size."""
    if not landmarks:
        return (0.0, 0.0, 1.0, 1.0)
    xs = [p[0] for p in landmarks]
    ys = [p[1] for p in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    pad_x = w * padding
    pad_y = h * padding
    return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


# ----- MediaPipe backend -----

def _detect_hands_mediapipe(frame: np.ndarray) -> dict[str, Any]:
    try:
        import mediapipe as mp
    except ImportError as e:
        raise ImportError(
            "MediaPipe backend requires: pip install mediapipe"
        ) from e

    h, w = frame.shape[:2]
    rgb = frame[:, :, ::-1].copy()  # BGR -> RGB

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    results = hands.process(rgb)
    hands.close()

    left = _empty_hand()
    right = _empty_hand()

    if not results.multi_hand_landmarks:
        return _hands_result_dict(left, right)

    for hand_landmarks, handedness in zip(
        results.multi_hand_landmarks,
        results.multi_handedness or [],
    ):
        label = (handedness.classification[0].label or "Unknown").lower()
        is_left = "left" in label
        landmarks = [
            (lm.x * w, lm.y * h, getattr(lm, "z", 0.0))
            for lm in hand_landmarks.landmark
        ]
        bbox = _bbox_from_landmarks(landmarks)
        hand_dict = _hand_result(detected=True, landmarks=landmarks, bbox=bbox)
        if is_left:
            left = hand_dict
        else:
            right = hand_dict

    return _hands_result_dict(left, right)


# ----- YOLO backend -----

def _detect_hands_yolo(
    frame: np.ndarray,
    model_path: str | None = None,
    hand_class_names: set[str] | None = None,
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(
            "YOLO backend requires: pip install ultralytics"
        ) from e

    if model_path is None:
        raise ValueError(
            "YOLO hand detector requires a model path (e.g. a hand-detection YOLO model). "
            "Use detect_hands(..., detector='yolo', model_path='path/to/hand_model.pt')"
        )

    model = YOLO(model_path)
    h, w = frame.shape[:2]
    center_x = w / 2.0

    # Default: common class names for hand models
    if hand_class_names is None:
        hand_class_names = {"hand", "hands", "left_hand", "right_hand"}

    results = model(frame, conf=conf_threshold, verbose=False)
    left = _empty_hand()
    right = _empty_hand()

    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = (r.names or {}).get(cls_id, "").lower()
            if name not in hand_class_names:
                continue
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
            bbox = (x1, y1, x2, y2)
            # Bbox corners as landmark-like points for API consistency
            landmarks = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
                ((x1 + x2) / 2, (y1 + y2) / 2),
            ]
            hand_dict = _hand_result(detected=True, landmarks=landmarks, bbox=bbox)
            box_center_x = (x1 + x2) / 2
            if box_center_x < center_x:
                if not left["detected"]:
                    left = hand_dict
            else:
                if not right["detected"]:
                    right = hand_dict

    return _hands_result_dict(left, right)


def detect_hands(
    frame: np.ndarray,
    detector: Literal["mediapipe", "yolo"] = DEFAULT_HAND_DETECTOR,
    *,
    model_path: str | None = None,
    hand_class_names: set[str] | None = None,
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Detect hand positions in a single frame using MediaPipe or YOLO.

    Args:
        frame: BGR image as numpy array (H, W, 3).
        detector: Backend to use: "mediapipe" (landmarks) or "yolo" (bbox-based).
        model_path: For detector="yolo", path to a hand-detection YOLO model (.pt).
            Required when using YOLO.
        hand_class_names: For YOLO, set of class names treated as hands
            (default: {"hand", "hands", "left_hand", "right_hand"}).
        conf_threshold: For YOLO, minimum confidence (0–1).

    Returns:
        Dict with:
          - "left_hand": {"detected": bool, "landmarks": list, "bbox": optional}
          - "right_hand": same
          - "num_hands": int

        MediaPipe returns 21 landmarks per hand (x, y, z in pixel coords).
        YOLO returns bbox and 5 landmark-like points (corners + center).
    """
    if detector == "mediapipe":
        return _detect_hands_mediapipe(frame)
    if detector == "yolo":
        return _detect_hands_yolo(
            frame,
            model_path=model_path,
            hand_class_names=hand_class_names,
            conf_threshold=conf_threshold,
        )
    raise ValueError(f"Unknown detector: {detector!r}. Use 'mediapipe' or 'yolo'.")


def track_hand_motion(
    frames: list[np.ndarray],
    detector: Literal["mediapipe", "yolo"] = DEFAULT_HAND_DETECTOR,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """
    Run hand detection on each frame. Returns one result dict per frame
    (same structure as detect_hands).
    """
    return [detect_hands(f, detector=detector, **kwargs) for f in frames]


def draw_hand_tracks(frame: np.ndarray, hand_data: dict[str, Any]) -> np.ndarray:
    """
    Draw hand keypoints, bounding boxes, confidence, and velocity on a frame.
    hand_data: left_hand, right_hand, num_hands; each hand may have bbox, confidence, velocity.
    Always draws bbox (from hand['bbox'] or computed from landmarks if missing).
    """
    out = frame.copy()
    colors = {"left_hand": (0, 165, 255), "right_hand": (255, 165, 0)}  # BGR: orange, blue
    h_img, w_img = out.shape[:2]

    for hand_key, color in colors.items():
        hand = hand_data.get(hand_key, _empty_hand())
        if not hand.get("detected"):
            continue
        label = "L" if hand_key == "left_hand" else "R"
        bbox = hand.get("bbox")
        landmarks = hand.get("landmarks") or []
        if bbox is None and landmarks:
            bbox = _bbox_from_landmarks(landmarks)
        if bbox is not None and len(bbox) >= 4:
            # bbox is (x1, y1, x2, y2)
            x1, y1 = max(0, int(round(bbox[0]))), max(0, int(round(bbox[1])))
            x2, y2 = min(w_img, int(round(bbox[2]))), min(h_img, int(round(bbox[3])))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            # Label and confidence
            conf = hand.get("confidence")
            if conf is not None:
                text = f"{label} {conf:.2f}"
            else:
                text = label
            cv2.putText(
                out, text, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )
            # Velocity below box
            vel = hand.get("velocity")
            if vel is not None and len(vel) >= 2:
                vx, vy = vel[0], vel[1]
                cv2.putText(
                    out, f"v: {vx:.0f},{vy:.0f}", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
                )
        for pt in landmarks:
            x, y = float(pt[0]), float(pt[1])
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < w_img and 0 <= cy < h_img:
                cv2.circle(out, (cx, cy), 3, color, -1)
    return out


# ---------------------------------------------------------------------------
# Object detection (stub for required_functions API)
# ---------------------------------------------------------------------------

def detect_objects(frame: np.ndarray) -> list[dict[str, Any]]:
    """
    Detect relevant lab objects (pipette, tube, flask, etc.) in a frame.
    Currently returns an empty list; implement with YOLO/Detectron2 and a lab-object model.
    """
    return []


# ---------------------------------------------------------------------------
# Pose estimation (stub for required_functions API)
# ---------------------------------------------------------------------------

def estimate_pose(frame: np.ndarray) -> dict[str, Any]:
    """
    Estimate body or upper-body pose in a frame.
    Currently returns a minimal structure; implement with MediaPipe Pose or similar.
    """
    return {
        "landmarks": [],
        "detected": False,
        "num_landmarks": 0,
    }

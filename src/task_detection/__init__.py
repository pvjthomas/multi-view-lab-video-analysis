"""
Task detection: hand detection, object detection, pose, and task inference.
"""

def __getattr__(name: str):
    """Lazy import detector so hand_tracking can be used without cv2."""
    if name in ("detect_hands", "detect_objects", "draw_hand_tracks", "estimate_pose", "track_hand_motion"):
        from src.task_detection.detector import (
            detect_hands,
            detect_objects,
            draw_hand_tracks,
            estimate_pose,
            track_hand_motion,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "detect_hands",
    "detect_objects",
    "draw_hand_tracks",
    "estimate_pose",
    "track_hand_motion",
]

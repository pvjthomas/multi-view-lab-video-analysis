"""
Video loading and inspection utilities for biomedical lab video annotation.
"""

from __future__ import annotations

import cv2
import numpy as np


def load_video(video_path: str) -> cv2.VideoCapture:
    """
    Open a video file and return a video capture object.

    Args:
        video_path: Path to the video file.

    Returns:
        cv2.VideoCapture instance. Call .release() when done.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    return cap


def get_video_info(video_path: str) -> dict:
    """
    Return metadata for a video file.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: frame_width, frame_height, fps, frame_count, duration_sec.
        duration_sec is derived from frame_count / fps when available.
    """
    cap = load_video(video_path)
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps and fps > 0:
            duration_sec = frame_count / fps
        else:
            duration_sec = 0.0

        return {
            "frame_width": frame_width,
            "frame_height": frame_height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_sec": duration_sec,
        }
    finally:
        cap.release()


def sample_frames(
    video_path: str,
    every_n_seconds: float,
) -> list[np.ndarray]:
    """
    Extract frames at regular time intervals for preview or annotation.

    Args:
        video_path: Path to the video file.
        every_n_seconds: Interval in seconds between sampled frames.

    Returns:
        List of frames as BGR numpy arrays (H, W, 3).
    """
    if every_n_seconds <= 0:
        raise ValueError("every_n_seconds must be positive")

    info = get_video_info(video_path)
    duration_sec = info["duration_sec"]
    fps = info["fps"]

    if duration_sec <= 0 or not fps or fps <= 0:
        return []

    timestamps_sec = np.arange(0.0, duration_sec, every_n_seconds)
    frames: list[np.ndarray] = []

    for t in timestamps_sec:
        frame = read_frame_at_time(video_path, t)
        if frame is not None and frame.size > 0:
            frames.append(frame)

    return frames


def read_frame_at_time(video_path: str, timestamp_sec: float) -> np.ndarray | None:
    """
    Read a single frame from a specific timestamp.

    Args:
        video_path: Path to the video file.
        timestamp_sec: Time in seconds from the start of the video.

    Returns:
        Frame as BGR numpy array (H, W, 3), or None if read failed.
    """
    if timestamp_sec < 0:
        raise ValueError("timestamp_sec must be non-negative")

    cap = load_video(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_sec * 1000.0)
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
        return None
    finally:
        cap.release()


def play_video(
    video_path: str,
    start_time_sec: float = 0.0,
    end_time_sec: float | None = None,
) -> None:
    """
    Play the video in a window from start_time_sec until end_time_sec (or end of file if None).
    Blocks until the user closes the window or playback finishes.

    Controls: Space = pause/resume, Q or Escape = quit.

    Args:
        video_path: Path to the video file.
        start_time_sec: Start playback at this time in seconds.
        end_time_sec: Stop playback at this time in seconds; None means play to end.
    """
    if start_time_sec < 0:
        raise ValueError("start_time_sec must be non-negative")
    if end_time_sec is not None and end_time_sec <= start_time_sec:
        raise ValueError("end_time_sec must be None or greater than start_time_sec")

    cap = load_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (frame_count / fps) if fps and fps > 0 else 0.0
        if not fps or fps <= 0:
            fps = 25.0
        frame_delay_ms = max(1, int(1000.0 / fps))

        end_at_sec = duration_sec if end_time_sec is None else min(end_time_sec, duration_sec)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000.0)

        window_name = "Video Playback"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_pos_sec >= end_at_sec:
                    break
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(frame_delay_ms if not paused else 100)
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()


def play_video_with_annotations(
    video_path: str,
    annotations: list[dict],
    start_time_sec: float = 0.0,
    end_time_sec: float | None = None,
) -> None:
    """
    Same as play_video, but draw the current segment's label on each frame.
    Annotation entries should have start_time, end_time, and label (e.g. from create_annotation_record).

    Args:
        video_path: Path to the video file.
        annotations: List of annotation dicts with start_time, end_time, label.
        start_time_sec: Start playback at this time in seconds.
        end_time_sec: Stop playback at this time in seconds; None means play to end.
    """
    if start_time_sec < 0:
        raise ValueError("start_time_sec must be non-negative")
    if end_time_sec is not None and end_time_sec <= start_time_sec:
        raise ValueError("end_time_sec must be None or greater than start_time_sec")

    cap = load_video(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (frame_count / fps) if fps and fps > 0 else 0.0
        if not fps or fps <= 0:
            fps = 25.0
        frame_delay_ms = max(1, int(1000.0 / fps))

        end_at_sec = duration_sec if end_time_sec is None else min(end_time_sec, duration_sec)
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time_sec * 1000.0)

        window_name = "Video Playback (with annotations)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        paused = False

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                current_pos_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_pos_sec >= end_at_sec:
                    break

                label = None
                for ann in annotations:
                    s = ann.get("start_time", 0)
                    e = ann.get("end_time", float("inf"))
                    if s <= current_pos_sec < e:
                        label = ann.get("label", "")
                        break
                if label:
                    cv2.putText(
                        frame,
                        label,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow(window_name, frame)

            key = cv2.waitKey(frame_delay_ms if not paused else 100)
            if key == -1:
                continue
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
    finally:
        cap.release()
        cv2.destroyAllWindows()

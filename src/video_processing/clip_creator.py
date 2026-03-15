"""
Clip extraction and segmenting for biomedical lab video annotation.
"""

from __future__ import annotations

from moviepy import VideoFileClip


def extract_clip(video_path: str, start_time: float, end_time: float, output_path: str) -> None:
    """
    Save a subclip from the original video.

    Args:
        video_path: Path to the source video file.
        start_time: Start time in seconds (inclusive).
        end_time: End time in seconds (inclusive).
        output_path: Path where the clip will be saved.

    Raises:
        ValueError: If start_time < 0, end_time <= start_time, or video cannot be opened.
    """
    if start_time < 0:
        raise ValueError("start_time must be non-negative")
    if end_time <= start_time:
        raise ValueError("end_time must be greater than start_time")

    clip = VideoFileClip(video_path)
    try:
        subclip = clip.subclipped(start_time, end_time)
        subclip.write_videofile(output_path, logger=None)
    finally:
        clip.close()

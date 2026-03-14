"""
Run video loading and inspection (required_functions.txt §1: load_video, get_video_info).
Usage: python scripts/run_video_loading.py [video_path]
"""

from __future__ import annotations

import os
import sys

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.video_utils import load_video, get_video_info

DEFAULT_VIDEO = "data/raw/Cell-Culture-Video-Step-by-Step-Guide-to_Media_CMRKKl9XSDU_001_1080p.mp4"


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, DEFAULT_VIDEO)

    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    print("Video loading and inspection")
    print("=" * 50)
    print(f"Path: {video_path}\n")

    # load_video
    print("load_video(video_path) -> cv2.VideoCapture")
    cap = load_video(video_path)
    print(f"  Opened: {cap.isOpened()}")
    cap.release()
    print()

    # get_video_info
    print("get_video_info(video_path) -> dict")
    info = get_video_info(video_path)
    print(f"  frame width:   {info['frame_width']}")
    print(f"  frame height: {info['frame_height']}")
    print(f"  fps:          {info['fps']}")
    print(f"  frame count:   {info['frame_count']}")
    print(f"  duration:     {info['duration_sec']:.2f} s")
    print()
    print("Done.")


if __name__ == "__main__":
    main()

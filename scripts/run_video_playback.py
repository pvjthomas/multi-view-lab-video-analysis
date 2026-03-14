"""
Play a video in a window (required_functions.txt §1a: play_video).
Usage: python scripts/run_video_playback.py [video_path]
Controls: Space = pause, Q or Escape = quit.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.video_utils import play_video

DEFAULT_VIDEO = "data/raw/Cell-Culture-Video-Step-by-Step-Guide-to_Media_CMRKKl9XSDU_001_1080p.mp4"

if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, DEFAULT_VIDEO)
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)
    play_video(video_path)

"""
Extract a subclip from a video.
Usage: python scripts/run_extract_clip.py [video_path] [start_time] [stop_time]
Defaults: Cell-Culture-Video... in data/raw/, start 29s, stop 32s
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.video_processing.clip_creator import extract_clip

DEFAULT_VIDEO = "data/raw/Cell-Culture-Video-Step-by-Step-Guide-to_Media_CMRKKl9XSDU_001_1080p.mp4"
tstart = 29.0
tstop = 32.0

def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_video = os.path.join(root, DEFAULT_VIDEO)

    parser = argparse.ArgumentParser(description="Extract a subclip from a video")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=default_video,
        help=f"Path to the video file (default: {DEFAULT_VIDEO})",
    )
    parser.add_argument(
        "start_time",
        nargs="?",
        type=float,
        default=tstart,
        help="Start time in seconds (default: 29)",
    )
    parser.add_argument(
        "stop_time",
        nargs="?",
        type=float,
        default=tstop,
        help="Stop time in seconds (default: 32)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Output directory for clips (default: clips/ in project root)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(root, "clips")
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{base}_{args.start_time:.1f}s_{args.stop_time:.1f}s_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)

    extract_clip(args.video_path, args.start_time, args.stop_time, output_path)
    print(f"Saved clip: {output_path}")


if __name__ == "__main__":
    main()

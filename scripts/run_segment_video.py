"""
Run procedural segmentation on a lab video.

1. Sample frames from the video (e.g. 1 per second) and save to a temp dir.
2. Run segment_video() to get segments and frame-difference series.
3. Write segments to JSON and optionally diff_series to CSV.

Usage:
  python scripts/run_segment_video.py [video_path] [--frames-dir DIR] [--threshold 25] [--min-duration 20] [--interval 1.0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.video_utils import sample_frames_to_dir
from src.video_processing.segmenter import segment_video


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_video = os.path.join(root, "data", "raw", "VID_20260315_005732_370_141.MP4")

    parser = argparse.ArgumentParser(description="Segment lab video into procedural steps")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=default_video,
        help="Path to the video file",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory for extracted frames (default: temp dir, deleted after)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Frame-difference threshold for new segment (default: 25)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
        help="Minimum segment duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample one frame every N seconds (default: 1.0)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep frames dir after run (only if --frames-dir is set)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for segments.json, diff_series.csv and diff_series.png (default: same dir as video)",
    )
    parser.add_argument(
        "--no-diff-plot",
        action="store_true",
        help="Do not generate diff_series.png plot",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        print(f"Error: video not found: {args.video_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.video_path))
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.video_path))[0]
    segments_path = os.path.join(output_dir, f"{base}_segments.json")
    diff_path = os.path.join(output_dir, f"{base}_diff_series.csv")
    diff_plot_path = None if args.no_diff_plot else os.path.join(output_dir, f"{base}_diff_series.png")

    use_temp = args.frames_dir is None
    frames_dir = args.frames_dir or tempfile.mkdtemp(prefix="labflow_frames_")

    try:
        print(f"[run_segment_video] Extracting frames every {args.interval}s to {frames_dir}")
        frames = sample_frames_to_dir(
            args.video_path,
            frames_dir,
            every_n_seconds=args.interval,
        )
        if not frames:
            print("No frames extracted. Check video path and duration.")
            sys.exit(1)
        print(f"[run_segment_video] Extracted {len(frames)} frames")

        segments, diff_series = segment_video(
            frames,
            diff_threshold=args.threshold,
            min_segment_duration=args.min_duration,
            diff_plot_path=diff_plot_path,
        )

        # Save segments as JSON (serialize Segment dataclass)
        segments_data = [
            {
                "segment_id": s.segment_id,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "representative_frame": s.representative_frame,
            }
            for s in segments
        ]
        with open(segments_path, "w") as f:
            json.dump(segments_data, f, indent=2)
        print(f"Wrote segments: {segments_path}")

        # Save diff_series as CSV for plotting
        with open(diff_path, "w") as f:
            f.write("timestamp_sec,mean_abs_diff\n")
            for t, d in diff_series:
                f.write(f"{t},{d}\n")
        print(f"Wrote diff series: {diff_path}")

    finally:
        if use_temp and not args.keep_frames:
            import shutil
            if os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

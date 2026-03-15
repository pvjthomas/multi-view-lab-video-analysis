"""
Run full analysis on a lab video at 1 fps: mean diff (frame difference) and hand detection.

1. Sample frames at 1 fps and save to a temp (or given) directory.
2. Compute frame-to-frame mean diff and segment boundaries (segment_video).
3. Run hand detection on each frame (detect_hands).
4. Write segments.json, diff_series.csv, diff_series.png, frame_analysis.csv.
5. When hand detection runs (default), also write a video with hand boxes drawn
   ({base}_hands_annotated.mp4). Use --no-hands to skip hand detection and the annotated video.

Usage:
  python scripts/run_video_analysis.py [video_path] [-o OUTPUT_DIR] [--no-diff-plot] [--no-hands]
"""

from __future__ import annotations

# Suppress MediaPipe/TFLite C++ logs before any detector import
import os
os.environ.setdefault("GLOG_minloglevel", "2")

import argparse
import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.task_detection.detector import detect_hands, draw_hand_tracks
from src.utils.video_utils import get_video_info, sample_frames_to_dir
from src.video_processing.segmenter import segment_video


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_video = os.path.join(root, "data", "raw", "VID_20260315_005732_370_141.MP4")

    parser = argparse.ArgumentParser(
        description="Analyze video at 1 fps: mean diff + hand detection"
    )
    parser.add_argument(
        "video_path",
        nargs="?",
        default=default_video,
        help="Path to the video file",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for outputs (default: same dir as video)",
    )
    parser.add_argument(
        "--frames-dir",
        default=None,
        help="Directory for extracted frames (default: temp dir, deleted after)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sample one frame every N seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-diff-plot",
        action="store_true",
        help="Do not generate diff_series.png",
    )
    parser.add_argument(
        "--no-hands",
        action="store_true",
        help="Skip hand detection (only mean diff + segments)",
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep frames dir after run (only if --frames-dir is set)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=25.0,
        help="Frame-difference threshold for segments (default: 25)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=20.0,
        help="Minimum segment duration in seconds (default: 20)",
    )
    parser.add_argument(
        "--detector",
        choices=("mediapipe", "yolo"),
        default="mediapipe",
        help="Hand detector backend (default: mediapipe)",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to YOLO hand model .pt (required if --detector yolo)",
    )
    args = parser.parse_args()

    if args.detector == "yolo" and not args.yolo_model:
        print("Error: --detector yolo requires --yolo-model path/to/hand_model.pt")
        sys.exit(1)

    if not os.path.isfile(args.video_path):
        print(f"Error: video not found: {args.video_path}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.video_path))
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.video_path))[0]
    segments_path = os.path.join(output_dir, f"{base}_segments.json")
    diff_csv_path = os.path.join(output_dir, f"{base}_diff_series.csv")
    diff_plot_path = None if args.no_diff_plot else os.path.join(
        output_dir, f"{base}_diff_series.png"
    )
    analysis_path = os.path.join(output_dir, f"{base}_frame_analysis.csv")
    annotated_video_path = os.path.join(output_dir, f"{base}_hands_annotated.mp4")

    use_temp = args.frames_dir is None
    frames_dir = args.frames_dir or tempfile.mkdtemp(prefix="labflow_frames_")

    try:
        print(f"[run_video_analysis] Extracting frames every {args.interval}s to {frames_dir}")
        frames = sample_frames_to_dir(
            args.video_path,
            frames_dir,
            every_n_seconds=args.interval,
        )
        if not frames:
            print("No frames extracted. Check video path and duration.")
            sys.exit(1)
        print(f"[run_video_analysis] Extracted {len(frames)} frames")

        # Mean diff + segments (and diff plot by default)
        segments, diff_series = segment_video(
            frames,
            diff_threshold=args.threshold,
            min_segment_duration=args.min_duration,
            diff_plot_path=diff_plot_path,
        )

        # Build timestamp -> mean_abs_diff for frame_analysis
        ts_to_diff = {t: d for t, d in diff_series}

        # Save segments
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

        # Save diff_series CSV
        with open(diff_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_sec", "mean_abs_diff"])
            for t, d in diff_series:
                w.writerow([t, d])
        print(f"Wrote diff series: {diff_csv_path}")

        if diff_plot_path:
            print(f"Wrote diff plot: {diff_plot_path}")

        # Hand detection per frame and combined frame_analysis.csv
        hand_kw: dict = {}
        if args.detector == "yolo":
            hand_kw["model_path"] = args.yolo_model

        rows = []
        hand_results: list[dict] = []  # full hand_data per frame for annotated video
        for i, (t_sec, frame_path) in enumerate(frames):
            mean_diff = ts_to_diff.get(t_sec, 0.0)
            row = {"timestamp_sec": t_sec, "mean_abs_diff": mean_diff}

            if not args.no_hands:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    hands = detect_hands(frame, detector=args.detector, **hand_kw)
                    hand_results.append(hands)
                    row["num_hands"] = hands["num_hands"]
                    row["left_hand_detected"] = hands["left_hand"]["detected"]
                    row["right_hand_detected"] = hands["right_hand"]["detected"]
                else:
                    hand_results.append({"left_hand": {"detected": False}, "right_hand": {"detected": False}, "num_hands": 0})
                    row["num_hands"] = 0
                    row["left_hand_detected"] = False
                    row["right_hand_detected"] = False
                if (i + 1) % 30 == 0 or i == 0:
                    print(f"  Hand detection: {i + 1}/{len(frames)} frames")
            rows.append(row)

        # Write frame_analysis.csv
        if rows:
            fieldnames = list(rows[0].keys())
            with open(analysis_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote frame analysis: {analysis_path}")

        # Output video with hand boxes drawn (when hand detection was run)
        if not args.no_hands and hand_results and frames:
            info = get_video_info(args.video_path)
            w = int(info["frame_width"])
            h = int(info["frame_height"])
            out_fps = max(1.0, 1.0 / args.interval)  # 1 fps when interval=1
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_video_path, fourcc, out_fps, (w, h))
            for (t_sec, frame_path), hand_data in zip(frames, hand_results):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frame = cv2.resize(frame, (w, h))
                    annotated = draw_hand_tracks(frame, hand_data)
                    writer.write(annotated)
            writer.release()
            print(f"Wrote hand-annotated video: {annotated_video_path}")

    finally:
        if use_temp and not args.keep_frames:
            import shutil
            if os.path.isdir(frames_dir):
                shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

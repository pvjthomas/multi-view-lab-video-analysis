"""
Video segmentation module for LabFlow AI.
Detects procedural boundaries using frame-difference scoring.

Tuned defaults for cell culture lab videos:
  diff_threshold=25   — ignores minor camera jitter, catches real scene changes
  min_segment=20      — enforces ~20s minimum, giving 8–15 steps for a 5-min video
"""

from __future__ import annotations

import cv2
import numpy as np

from src.models import Segment


def plot_diff_series(
    diff_series: list[tuple[float, float]],
    output_path: str,
) -> None:
    """
    Plot mean absolute frame difference over time and save as PNG.

    Args:
        diff_series: List of (timestamp_sec, mean_abs_diff) per frame (e.g. from segment_video).
        output_path: Path for the output PNG file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not diff_series:
        return
    times = [t for t, _ in diff_series]
    diffs = [d for _, d in diff_series]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, diffs, color="#2563eb", linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean abs diff")
    ax.set_title("Frame difference over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def compute_frame_difference(frame_a_path: str, frame_b_path: str) -> float:
    """
    Compute mean absolute pixel difference between two frames (grayscale).
    Returns a score 0–255; higher = more visual change.
    """
    img_a = cv2.imread(frame_a_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(frame_b_path, cv2.IMREAD_GRAYSCALE)

    if img_a is None or img_b is None:
        return 0.0

    size = (320, 180)
    img_a = cv2.resize(img_a, size)
    img_b = cv2.resize(img_b, size)

    diff = cv2.absdiff(img_a, img_b)
    return float(np.mean(diff))


def _merge_short_segments(
    segments: list[Segment],
    min_duration: float,
    frames: list[tuple[float, str]],
) -> list[Segment]:
    """
    Absorb any segments shorter than min_duration into the previous segment.
    Reassigns representative frame to the midpoint of the merged span.
    """
    if len(segments) <= 1:
        return segments

    merged = [segments[0]]
    for seg in segments[1:]:
        if seg.duration < min_duration:
            # Absorb into previous
            prev = merged[-1]
            prev.end_time = seg.end_time
            # Update representative frame to mid of merged span
            mid_time = (prev.start_time + prev.end_time) / 2
            closest = min(frames, key=lambda f: abs(f[0] - mid_time))
            prev.representative_frame = closest[1]
        else:
            merged.append(seg)

    # Re-number
    for i, seg in enumerate(merged):
        seg.segment_id = i + 1

    return merged


def segment_video(
    frames: list[tuple[float, str]],
    diff_threshold: float = 25.0,
    min_segment_duration: float = 20.0,
    diff_plot_path: str | None = "diff_series.png",
) -> tuple[list[Segment], list[tuple[float, float]]]:
    """
    Split frames into segments based on visual change magnitude.

    Args:
        frames: List of (timestamp, path) from frame extractor (e.g. extract_frames_to_dir).
        diff_threshold: Mean pixel diff score above which a new segment starts.
                        25 works well for stable BSC/lab camera setups.
        min_segment_duration: Minimum seconds per segment (default 20s).
                              Raises this to get fewer, larger procedural chunks.
        diff_plot_path: If set, plot mean diff over time and save as PNG (default: "diff_series.png").
                        Pass None to skip generating the plot.

    Returns:
        Tuple of (segments, diff_series). segments: list of Segment objects.
        diff_series: list of (timestamp_s, mean_abs_diff) per frame, for CSV/plot.
    """
    if not frames:
        return [], []

    print(
        f"[segmenter] Segmenting {len(frames)} frames "
        f"(threshold={diff_threshold}, min_duration={min_segment_duration}s)"
    )

    # Compute frame-to-frame differences
    diffs: list[float] = [0.0]
    for i in range(1, len(frames)):
        score = compute_frame_difference(frames[i - 1][1], frames[i][1])
        diffs.append(score)

    # Detect boundary frames where diff exceeds threshold,
    # enforcing min_segment_duration gap between boundaries
    segment_boundaries = [0]
    for i, score in enumerate(diffs):
        if score >= diff_threshold:
            last_boundary_time = frames[segment_boundaries[-1]][0]
            current_time = frames[i][0]
            if current_time - last_boundary_time >= min_segment_duration:
                segment_boundaries.append(i)

    segment_boundaries.append(len(frames))

    # Build Segment objects
    segments: list[Segment] = []
    for seg_idx in range(len(segment_boundaries) - 1):
        start_idx = segment_boundaries[seg_idx]
        end_idx = segment_boundaries[seg_idx + 1] - 1

        start_time = frames[start_idx][0]
        end_time = frames[min(end_idx, len(frames) - 1)][0]

        mid_idx = (start_idx + end_idx) // 2
        representative_frame = frames[min(mid_idx, len(frames) - 1)][1]

        segments.append(
            Segment(
                segment_id=seg_idx + 1,
                start_time=start_time,
                end_time=end_time,
                representative_frame=representative_frame,
            )
        )

    # Merge any remaining short orphan segments into the previous one
    segments = _merge_short_segments(segments, min_segment_duration, frames)

    # Build diff_series: (timestamp_s, mean_abs_diff) per frame for export/plot
    diff_series = [(frames[i][0], diffs[i]) for i in range(len(frames))]

    if diff_plot_path:
        plot_diff_series(diff_series, diff_plot_path)
        print(f"[segmenter] Wrote diff plot: {diff_plot_path}")

    avg_duration = sum(s.duration for s in segments) / len(segments) if segments else 0
    print(f"[segmenter] Found {len(segments)} segments (avg {avg_duration:.0f}s each)")
    return segments, diff_series

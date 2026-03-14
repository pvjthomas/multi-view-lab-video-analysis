"""
Tests for src.utils.video_utils (video loading and inspection).
Run with: python -m pytest tests/test_video_utils.py -v
     or: python tests/test_video_utils.py

Requires: pip install opencv-python numpy
"""

import os
import sys
import unittest

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.utils.video_utils import (
        load_video,
        get_video_info,
        sample_frames,
        read_frame_at_time,
    )
    VIDEO_UTILS_AVAILABLE = True
except ImportError as e:
    VIDEO_UTILS_AVAILABLE = False
    load_video = get_video_info = sample_frames = read_frame_at_time = None
    _import_error = e


# Path to sample video (relative to project root)
SAMPLE_VIDEO = "data/raw/Cell-Culture-Video-Step-by-Step-Guide-to_Media_CMRKKl9XSDU_001_1080p.mp4"


def sample_video_path():
    """Absolute path to sample video if it exists."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, SAMPLE_VIDEO)


@unittest.skipUnless(VIDEO_UTILS_AVAILABLE, "opencv-python/numpy not installed")
@unittest.skipUnless(os.path.exists(sample_video_path()), "Sample video not found")
class TestVideoUtilsWithFile(unittest.TestCase):
    """Tests that require the sample video file."""

    def setUp(self):
        self.video_path = sample_video_path()

    def test_load_video_opens_and_releases(self):
        cap = load_video(self.video_path)
        self.assertTrue(cap.isOpened())
        cap.release()

    def test_get_video_info_returns_expected_keys(self):
        info = get_video_info(self.video_path)
        for key in ("frame_width", "frame_height", "fps", "frame_count", "duration_sec"):
            self.assertIn(key, info, f"Missing key: {key}")

    def test_get_video_info_has_sensible_values(self):
        info = get_video_info(self.video_path)
        self.assertGreater(info["frame_width"], 0)
        self.assertGreater(info["frame_height"], 0)
        self.assertGreater(info["fps"], 0)
        self.assertGreaterEqual(info["frame_count"], 0)
        self.assertGreaterEqual(info["duration_sec"], 0)
        if info["fps"] > 0 and info["frame_count"] > 0:
            expected_duration = info["frame_count"] / info["fps"]
            self.assertAlmostEqual(info["duration_sec"], expected_duration, places=1)

    def test_sample_frames_returns_list_of_arrays(self):
        frames = sample_frames(self.video_path, every_n_seconds=10.0)
        self.assertIsInstance(frames, list)
        for f in frames:
            self.assertIsNotNone(f)
            self.assertEqual(f.ndim, 3)
            self.assertEqual(f.shape[2], 3)  # BGR

    def test_sample_frames_interval(self):
        info = get_video_info(self.video_path)
        duration = info["duration_sec"]
        if duration < 5:
            self.skipTest("Video too short for interval test")
        frames = sample_frames(self.video_path, every_n_seconds=5.0)
        # Should have roughly duration/5 frames (at least 1)
        self.assertGreaterEqual(len(frames), 1)
        self.assertLessEqual(len(frames), int(duration / 5.0) + 2)

    def test_read_frame_at_time_returns_frame(self):
        frame = read_frame_at_time(self.video_path, 5.0)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)

    def test_read_frame_at_time_at_zero(self):
        frame = read_frame_at_time(self.video_path, 0.0)
        self.assertIsNotNone(frame)
        self.assertGreater(frame.size, 0)


@unittest.skipUnless(VIDEO_UTILS_AVAILABLE, "opencv-python/numpy not installed")
class TestVideoUtilsValidation(unittest.TestCase):
    """Tests for argument validation (no video file needed)."""

    def test_load_video_nonexistent_raises(self):
        with self.assertRaises((ValueError, OSError)):
            load_video("/nonexistent/path/video.mp4")

    def test_sample_frames_invalid_interval_raises(self):
        # Use a path that might not exist; we only check argument validation
        with self.assertRaises(ValueError):
            sample_frames("/tmp/any.mp4", every_n_seconds=0)
        with self.assertRaises(ValueError):
            sample_frames("/tmp/any.mp4", every_n_seconds=-1.0)

    def test_read_frame_at_time_negative_raises(self):
        with self.assertRaises(ValueError):
            read_frame_at_time("/tmp/any.mp4", -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""
Tests for src.task_detection.detector (hand detection, object detection, pose).
Run with: python -m pytest tests/test_detector.py -v
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.task_detection.detector import (
        detect_hands,
        detect_objects,
        estimate_pose,
        track_hand_motion,
    )
    DETECTOR_AVAILABLE = True
except ImportError:
    DETECTOR_AVAILABLE = False
    detect_hands = detect_objects = estimate_pose = track_hand_motion = None


@unittest.skipUnless(DETECTOR_AVAILABLE, "task_detection module not available")
class TestDetectorStructure(unittest.TestCase):
    """Test that detector functions return expected structure (no ML deps for basic tests)."""

    def test_detect_hands_returns_dict(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detect_hands(frame)
        self.assertIsInstance(result, dict)
        self.assertIn("left_hand", result)
        self.assertIn("right_hand", result)
        self.assertIn("num_hands", result)
        self.assertIn("landmarks", result["left_hand"])
        self.assertIn("detected", result["left_hand"])

    def test_track_hand_motion_returns_list(self):
        frames = [
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.ones((240, 320, 3), dtype=np.uint8) * 128,
        ]
        result = track_hand_motion(frames)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], dict)

    def test_detect_objects_returns_list(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = detect_objects(frame)
        self.assertIsInstance(result, list)
        for det in result:
            self.assertIn("label", det)
            self.assertIn("bbox", det)
            self.assertIn("confidence", det)

    def test_estimate_pose_returns_dict(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = estimate_pose(frame)
        self.assertIsInstance(result, dict)
        self.assertIn("landmarks", result)
        self.assertIn("detected", result)
        self.assertIn("num_landmarks", result)

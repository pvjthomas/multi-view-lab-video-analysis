"""Tests for hand_tracking package (geometry, state, scores; full pipeline requires cv2)."""

import unittest

from src.models import TrackState

try:
    import cv2  # noqa: F401
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class TestGeometry(unittest.TestCase):
    """Test geometry helpers with minimal deps (no cv2)."""

    def test_box_center(self):
        from src.task_detection.hand_tracking.geometry import box_center
        self.assertEqual(box_center((0, 0, 10, 20)), (5.0, 10.0))

    def test_box_velocity(self):
        from src.task_detection.hand_tracking.geometry import box_velocity
        v = box_velocity((0, 0, 10, 10), (5, 5, 10, 10))
        self.assertEqual(v, (5.0, 5.0, 0.0, 0.0))

    def test_expand_box(self):
        from src.task_detection.hand_tracking.geometry import expand_box
        # (0,0,10,10) -> center (5,5), scale 1.5 -> w=15 h=15 -> (5-7.5, 5-7.5, 15, 15)
        out = expand_box((0, 0, 10, 10), 1.5)
        self.assertAlmostEqual(out[2], 15.0)
        self.assertAlmostEqual(out[3], 15.0)

    def test_box_distance(self):
        from src.task_detection.hand_tracking.geometry import box_distance
        d = box_distance((0, 0, 2, 2), (10, 0, 2, 2))
        self.assertAlmostEqual(d, 10.0)

    def test_geometry_score(self):
        from src.task_detection.hand_tracking.geometry import geometry_score
        s = geometry_score((0, 0, 10, 10), (0, 0, 10, 10))
        self.assertAlmostEqual(s, 1.0)
        s2 = geometry_score((0, 0, 10, 10), (100, 100, 10, 10))
        self.assertLess(s2, 0.5)


@unittest.skipUnless(CV2_AVAILABLE, "cv2 required for scores (pulls motion/color)")
class TestScores(unittest.TestCase):
    """Test score helpers."""

    def test_combine_scores(self):
        from src.task_detection.hand_tracking.scores import combine_scores
        s = combine_scores(0.5, 0.5, 0.5, 0.5, 0.5)
        self.assertAlmostEqual(s, 0.5)
        s2 = combine_scores(1.0, 1.0, 0.0, 0.0, 0.0)
        self.assertGreater(s2, 0.0)
        self.assertLessEqual(s2, 1.0)


@unittest.skipUnless(CV2_AVAILABLE, "cv2 required for pipeline imports")
class TestPipelineLogic(unittest.TestCase):
    """Test should_run_detector logic."""

    def test_should_run_detector_no_track(self):
        from src.task_detection.hand_tracking.pipeline import should_run_detector
        self.assertTrue(should_run_detector(0, None))
        self.assertTrue(should_run_detector(100, None))

    def test_should_run_detector_every_n(self):
        from src.task_detection.hand_tracking.pipeline import should_run_detector
        # Minimal track (no cv2)
        track = TrackState(
            box=(10, 10, 20, 20),
            points=None,
            confidence=0.9,
            color_stats=None,
            velocity=(0, 0, 0, 0),
            frame_index=0,
            weak_counter=0,
        )
        self.assertTrue(should_run_detector(10, track, every_n=10))
        self.assertTrue(should_run_detector(20, track, every_n=10))
        self.assertFalse(should_run_detector(11, track, every_n=10))


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for HandTracker, Landmark Extraction, and HUD Visualization
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform
"""

import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.settings import VisionConfig, AppConfig, DEFAULT_CONFIG
from vision.hand_tracker import (
    HandTracker,
    HandLandmarkData,
    HandTrackingResult,
    ModelNotFoundError,
    HandTrackerError,
)
from ui.overlay import draw_hand_landmarks, draw_camera_hud


class TestHandTracker(unittest.TestCase):

    def test_01_hand_tracker_defaults(self):
        """Verify HandTracker initializes with default VisionConfig and model path."""
        tracker = HandTracker()
        self.assertEqual(tracker.config.max_num_hands, 2)
        self.assertEqual(tracker.config.hand_min_detection_confidence, 0.7)
        self.assertEqual(tracker.config.hand_model_path, "models/hand_landmarker.task")
        self.assertFalse(tracker.is_initialized)

    def test_02_missing_model_file_raises_error(self):
        """Verify ModelNotFoundError is raised when the model file does not exist."""
        tracker = HandTracker(model_path="non_existent/path/to/model.task")
        with self.assertRaises(ModelNotFoundError) as ctx:
            tracker.start()
        self.assertIn("Hand Landmarker model file not found", str(ctx.exception))

    def test_03_hand_tracker_start_and_close_real_model(self):
        """Verify HandTracker starts and closes cleanly using the local models/hand_landmarker.task."""
        tracker = HandTracker()
        self.assertTrue(tracker.start())
        self.assertTrue(tracker.is_initialized)
        tracker.close()
        self.assertFalse(tracker.is_initialized)

    def test_04_hand_tracker_context_manager(self):
        """Verify HandTracker functions properly as a Python context manager."""
        with HandTracker() as tracker:
            self.assertTrue(tracker.is_initialized)
        self.assertFalse(tracker.is_initialized)

    def test_05_process_empty_frame(self):
        """Verify processing an empty or None frame returns empty result without crashing."""
        tracker = HandTracker()
        tracker.start()
        res_none = tracker.process(None)
        self.assertEqual(res_none.hand_count, 0)
        self.assertEqual(len(res_none.hands), 0)

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = tracker.process(empty_frame)
        self.assertEqual(res_empty.hand_count, 0)
        tracker.close()

    def test_06_process_synthetic_frame(self):
        """Verify processing a synthetic blank image executes inference cleanly."""
        tracker = HandTracker()
        tracker.start()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        res = tracker.process(dummy_frame)
        self.assertIsInstance(res, HandTrackingResult)
        self.assertEqual(res.hand_count, 0)  # Blank frame has 0 hands
        self.assertGreater(res.timestamp_ms, 0)
        tracker.close()

    def test_07_hand_landmark_data_structure(self):
        """Verify HandLandmarkData and HandTrackingResult data fields."""
        norm_lms = [(0.5, 0.5, 0.0)] * 21
        px_lms = [(320, 240)] * 21
        hand = HandLandmarkData(
            landmarks=norm_lms,
            landmarks_px=px_lms,
            handedness="Right",
            score=0.98,
        )
        self.assertEqual(len(hand.landmarks), 21)
        self.assertEqual(len(hand.landmarks_px), 21)
        self.assertEqual(hand.handedness, "Right")
        self.assertAlmostEqual(hand.score, 0.98)

        res = HandTrackingResult(hands=[hand], hand_count=1, timestamp_ms=12345)
        self.assertEqual(res.hand_count, 1)
        self.assertEqual(res.timestamp_ms, 12345)

    def test_08_draw_hand_landmarks_rendering(self):
        """Verify draw_hand_landmarks overlays landmarks without altering frame shape."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        norm_lms = [(0.5, 0.5, 0.0)] * 21
        # Spread pixel coordinates across the frame
        px_lms = [(100 + i * 10, 200 + i * 5) for i in range(21)]
        hand = HandLandmarkData(
            landmarks=norm_lms,
            landmarks_px=px_lms,
            handedness="Left",
            score=0.95,
        )
        result = HandTrackingResult(hands=[hand], hand_count=1, timestamp_ms=100)

        rendered = draw_hand_landmarks(frame, result)
        self.assertEqual(rendered.shape, (720, 1280, 3))
        # Frame should now contain drawn lines and circles
        self.assertGreater(rendered.sum(), 0)

    def test_09_draw_camera_hud_with_hands_info(self):
        """Verify draw_camera_hud displays hand tracking summary."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = draw_camera_hud(
            frame,
            fps=30.0,
            status="Active",
            resolution=(1280, 720),
            hand_count=2,
            hands_info="Left 95%, Right 98%",
        )
        self.assertEqual(rendered.shape, (720, 1280, 3))
        self.assertGreater(rendered.sum(), 0)

    @patch("app.cv2.destroyAllWindows")
    @patch("app.cv2.getWindowProperty", return_value=1)
    @patch("app.cv2.namedWindow")
    @patch("app.cv2.imshow")
    @patch("app.cv2.waitKey", side_effect=[ord("q"), 0])
    @patch("app.Camera")
    @patch("app.HandTracker")
    def test_10_app_run_with_hand_tracker(
        self,
        mock_tracker_cls,
        mock_camera_cls,
        mock_wait_key,
        mock_imshow,
        mock_named_win,
        mock_get_win_prop,
        mock_destroy,
    ):
        """Verify app.run integrates Camera and HandTracker, terminating cleanly on 'Q'."""
        import app

        mock_cam = MagicMock()
        mock_cam.start.return_value = True
        mock_cam.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_cam.resolution = (1280, 720)
        mock_camera_cls.return_value = mock_cam

        mock_tracker = MagicMock()
        mock_tracker.start.return_value = True
        mock_tracker.process.return_value = HandTrackingResult(hands=[], hand_count=0)
        mock_tracker_cls.return_value = mock_tracker

        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        mock_cam.start.assert_called_once()
        mock_tracker.start.assert_called_once()
        mock_tracker.process.assert_called()
        mock_cam.release.assert_called_once()
        mock_tracker.close.assert_called_once()
        mock_destroy.assert_called_once()

    def test_11_handedness_mapping_mirrored_and_unmirrored(self):
        """Verify handedness is accurately mapped in both mirrored and unmirrored modes."""
        class MockCategory:
            def __init__(self, name, score):
                self.category_name = name
                self.score = score

        class MockLandmark:
            x, y, z = 0.5, 0.5, 0.0

        mock_result = MagicMock()
        mock_result.hand_landmarks = [
            [MockLandmark()] * 21,
            [MockLandmark()] * 21,
        ]
        # MediaPipe raw model outputs: Hand 0 is 'Left', Hand 1 is 'Right'
        mock_result.handedness = [
            [MockCategory("Left", 0.96)],
            [MockCategory("Right", 0.91)],
        ]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. Mirrored mode (is_mirrored=True) -> Compensates for selfie flip
        tracker_mirrored = HandTracker(is_mirrored=True)
        tracker_mirrored._is_initialized = True
        tracker_mirrored._landmarker = MagicMock()
        tracker_mirrored._landmarker.detect_for_video.return_value = mock_result

        res_mirrored = tracker_mirrored.process(dummy_frame)
        self.assertEqual(res_mirrored.hand_count, 2)
        # Raw 'Left' should map to physical 'Right'
        self.assertEqual(res_mirrored.hands[0].handedness, "Right")
        self.assertAlmostEqual(res_mirrored.hands[0].score, 0.96)
        # Raw 'Right' should map to physical 'Left'
        self.assertEqual(res_mirrored.hands[1].handedness, "Left")
        self.assertAlmostEqual(res_mirrored.hands[1].score, 0.91)

        # 2. Unmirrored mode (is_mirrored=False) -> Retains raw classification
        tracker_unmirrored = HandTracker(is_mirrored=False)
        tracker_unmirrored._is_initialized = True
        tracker_unmirrored._landmarker = MagicMock()
        tracker_unmirrored._landmarker.detect_for_video.return_value = mock_result

        res_unmirrored = tracker_unmirrored.process(dummy_frame)
        self.assertEqual(res_unmirrored.hand_count, 2)
        self.assertEqual(res_unmirrored.hands[0].handedness, "Left")
        self.assertAlmostEqual(res_unmirrored.hands[0].score, 0.96)
        self.assertEqual(res_unmirrored.hands[1].handedness, "Right")
        self.assertAlmostEqual(res_unmirrored.hands[1].score, 0.91)


if __name__ == "__main__":
    unittest.main()

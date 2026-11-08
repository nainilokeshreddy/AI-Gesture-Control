"""
Environment & Scaffolding Sanity Test
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Verifies:
1. Python version >= 3.12.
2. OpenCV, MediaPipe, PyAutoGUI, and NumPy installations and initializations.
3. MediaPipe Hands and FaceMesh model instantiation.
4. PyAutoGUI screen dimension detection.
5. Configuration and GestureType enum specification completeness (all 18 controls).
6. Smoothing filters (EMA and 1-Euro) and Cooldown/Debounce utilities.
"""

import sys
import unittest
import numpy as np


class TestEnvironmentAndScaffolding(unittest.TestCase):

    def test_01_python_version(self):
        """Verify Python 3.12+ runtime."""
        major, minor = sys.version_info[:2]
        self.assertEqual(major, 3, f"Expected Python 3, got Python {major}")
        self.assertGreaterEqual(minor, 12, f"Expected Python >= 3.12, got Python 3.{minor}")

    def test_02_dependency_imports(self):
        """Verify core computer vision and control libraries import cleanly."""
        import cv2
        import mediapipe as mp
        import pyautogui

        self.assertTrue(hasattr(cv2, "__version__"))
        self.assertTrue(hasattr(mp, "__version__"))
        self.assertTrue(hasattr(pyautogui, "__version__"))

    def test_03_mediapipe_tasks_api(self):
        """Verify MediaPipe 1.0.1 Tasks API and HandLandmarker pipeline initialize properly."""
        from vision.hand_tracker import HandTracker

        tracker = HandTracker()
        self.assertTrue(tracker.start())
        self.assertTrue(tracker.is_initialized)
        tracker.close()
        self.assertFalse(tracker.is_initialized)

    def test_04_pyautogui_screen_resolution(self):
        """Verify PyAutoGUI can inspect macOS screen geometry."""
        import pyautogui

        screen_w, screen_h = pyautogui.size()
        self.assertGreater(screen_w, 0)
        self.assertGreater(screen_h, 0)

    def test_05_18_control_specification_enums(self):
        """Verify all 18 specified gestures exist in GestureType."""
        from config.settings import GestureType

        expected_gestures = [
            # Hand (1-8)
            "INDEX_CURSOR_MOVE",
            "THUMB_INDEX_PINCH_LEFT_CLICK",
            "TWO_FINGER_PINCH_RIGHT_CLICK",
            "FIST_TO_OPEN_PALM_PLAY_PAUSE",
            "SWIPE_RIGHT_NEXT",
            "SWIPE_LEFT_PREVIOUS",
            "STATIONARY_PALM_MODE_TOGGLE",
            "TWO_FINGERS_VERTICAL_ZOOM",
            # Head (9-10)
            "HEAD_LEFT",
            "HEAD_RIGHT",
            # Eyes (13-17)
            "IRIS_LEFT_DWELL_PREV",
            "IRIS_RIGHT_DWELL_NEXT",
            "IRIS_UP_DWELL_MENU",
            "IRIS_DOWN_DWELL_BACK",
            "DOUBLE_BLINK_CONFIRM",
            # Multimodal (18)
            "GAZE_DOUBLE_BLINK_SELECT",
        ]
        enum_names = [g.name for g in GestureType]
        self.assertEqual(len(enum_names), 16, "GestureType must define exactly 16 controls (nod up/down removed)")
        for g_name in expected_gestures:
            self.assertIn(g_name, enum_names, f"Missing gesture specification: {g_name}")

    def test_06_smoothing_filters(self):
        """Verify EMA and 1-Euro filters operate on 2D coordinates without drift."""
        from utils.smoothing import ExponentialMovingAverage, OneEuroFilter

        ema = ExponentialMovingAverage(alpha=0.5)
        p1 = ema.filter(np.array([100.0, 100.0]))
        self.assertTrue(np.allclose(p1, [100.0, 100.0]))
        p2 = ema.filter(np.array([200.0, 200.0]))
        self.assertTrue(np.allclose(p2, [150.0, 150.0]))

        oe = OneEuroFilter()
        out = oe.filter(np.array([50.0, 50.0]))
        self.assertEqual(len(out), 2)

    def test_07_cooldown_and_debouncer(self):
        """Verify cooldown manager and debouncing state machines."""
        import time
        from utils.cooldown import CooldownManager, Debouncer
        from config.settings import GestureType

        cm = CooldownManager(default_cooldown=0.1)
        self.assertTrue(cm.is_ready(GestureType.THUMB_INDEX_PINCH_LEFT_CLICK))
        cm.trigger(GestureType.THUMB_INDEX_PINCH_LEFT_CLICK)
        self.assertFalse(cm.is_ready(GestureType.THUMB_INDEX_PINCH_LEFT_CLICK))
        time.sleep(0.12)
        self.assertTrue(cm.is_ready(GestureType.THUMB_INDEX_PINCH_LEFT_CLICK))

        debouncer = Debouncer(required_duration_sec=0.05)
        self.assertFalse(debouncer.update(condition_met=False))
        self.assertFalse(debouncer.update(condition_met=True))
        time.sleep(0.06)
        self.assertTrue(debouncer.update(condition_met=True))
        # After firing once, should return False until reset
        self.assertFalse(debouncer.update(condition_met=True))


if __name__ == "__main__":
    unittest.main()

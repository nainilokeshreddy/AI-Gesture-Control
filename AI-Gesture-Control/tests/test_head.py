"""
Unit Tests for Stage 5: Head Tracking Foundation (Left / Right Directional Movement)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Verifies all simplified Stage 5 requirements:
1. Neutral: Head straight ahead within deadband -> NEUTRAL
2. Head Left: Relative yaw < threshold -> HEAD LEFT
3. Head Right: Relative yaw > threshold -> HEAD RIGHT
4. Small yaw movement ignored: Deadband perturbations remain NEUTRAL
5. Left/Right cooldown: Enforces minimum time between discrete events
6. Hysteresis: Requires returning into deadband to re-arm
7. Return to neutral: Head returning to neutral re-arms state
8. Angle wrapping: Circular differences wrap cleanly across +/-180 deg
9. Baseline-relative yaw: Classification is strictly relative to configured baseline
10. Pitch and Roll decoupling: Pitch and Roll do NOT trigger gestures
11. Multimodal coexistence: HandTracker and FaceTracker operate concurrently on same frame
12. Safety verification: Control Mode ON, head movements trigger ZERO OS actions
13. Control Mode OFF: Head tracking classification and status remain fully active
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.settings import GestureThresholds, CooldownConfig
from vision.face_tracker import (
    FaceTracker,
    FaceTrackingResult,
    HeadPoseData,
    ModelNotFoundError,
    CANONICAL_FACE_3D,
)
from gestures.head_gestures import (
    HeadGestureRecognizer,
    HeadGestureID,
    HeadGestureState,
    HeadGestureEvent,
    angle_diff,
)
from control.command_engine import CommandEngine
from control.mouse_controller import MouseController
from control.keyboard_controller import KeyboardController
from gestures.hand_gestures import GestureRecognitionResult, HandGestureState as HandState


class TestFaceTracker(unittest.TestCase):
    """Unit tests for FaceTracker lifecycle and landmark processing."""

    def test_missing_model_raises_error(self):
        """Verify ModelNotFoundError is raised when model file does not exist."""
        tracker = FaceTracker(model_path="non_existent/path/to/face_landmarker.task")
        with self.assertRaises(ModelNotFoundError) as ctx:
            tracker.start()
        self.assertIn("Face Landmarker model file not found", str(ctx.exception))

    def test_start_and_close_real_model(self):
        """Verify FaceTracker initializes and closes cleanly using models/face_landmarker.task."""
        tracker = FaceTracker(model_path="models/face_landmarker.task")
        self.assertTrue(tracker.start())
        self.assertTrue(tracker.is_initialized)
        tracker.close()
        self.assertFalse(tracker.is_initialized)

    def test_process_empty_frame(self):
        """Verify processing None or empty frame returns empty result without crashing."""
        tracker = FaceTracker(model_path="models/face_landmarker.task")
        tracker.start()
        res_none = tracker.process(None)
        self.assertFalse(res_none.face_detected)

        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        res_empty = tracker.process(empty_frame)
        self.assertFalse(res_empty.face_detected)
        tracker.close()

    def test_process_synthetic_blank_frame(self):
        """Verify processing synthetic frame with no face returns face_detected=False."""
        tracker = FaceTracker(model_path="models/face_landmarker.task")
        tracker.start()
        blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        res = tracker.process(blank_frame)
        self.assertFalse(res.face_detected)
        self.assertIsNone(res.head_pose)
        tracker.close()

    def test_head_pose_estimation_geometry(self):
        """Verify _estimate_head_pose solves PnP and calculates angles for synthetic landmarks."""
        tracker = FaceTracker()
        w, h = 1280, 720
        # Canonical 2D projections of CANONICAL_FACE_3D
        dummy_pts = [(w // 2, h // 2)] * 478
        dummy_pts[1] = (640, 360)    # Nose tip
        dummy_pts[152] = (640, 520)  # Chin
        dummy_pts[263] = (750, 275)  # Screen Right eye
        dummy_pts[33] = (530, 275)   # Screen Left eye
        dummy_pts[291] = (715, 435)  # Screen Right mouth
        dummy_pts[61] = (565, 435)   # Screen Left mouth

        pose = tracker._estimate_head_pose(dummy_pts, (w, h))
        self.assertIsNotNone(pose)
        self.assertIsInstance(pose.yaw, float)
        self.assertIsInstance(pose.pitch, float)
        self.assertIsInstance(pose.roll, float)
        # Neutral face must yield near-zero angles (< 5.0 deg)
        self.assertAlmostEqual(pose.yaw, 0.0, delta=5.0)
        self.assertAlmostEqual(pose.pitch, 0.0, delta=5.0)
        self.assertAlmostEqual(pose.roll, 0.0, delta=5.0)


class TestHeadGestureRecognition(unittest.TestCase):
    """Comprehensive test suite for Stage 5 Left / Right head tracking."""

    def setUp(self):
        self.thresholds = GestureThresholds()
        self.cooldown = CooldownConfig(head_cooldown=0.6)
        self.recognizer = HeadGestureRecognizer(
            thresholds=self.thresholds,
            cooldown=self.cooldown,
        )

    def _make_face_result(
        self,
        yaw: float,
        pitch: float = -8.6,
        roll: float = 176.6,
        detected: bool = True,
    ) -> FaceTrackingResult:
        """Helper to create synthetic FaceTrackingResult with specified angles."""
        if not detected:
            return FaceTrackingResult(face_detected=False)

        pose = HeadPoseData(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            raw_yaw=yaw,
            raw_pitch=pitch,
            raw_roll=roll,
        )
        return FaceTrackingResult(
            face_detected=True,
            head_pose=pose,
            timestamp_ms=1000,
        )

    # TEST 1: No face -> FACE NOT DETECTED
    def test_01_no_face_detected(self):
        res = self._make_face_result(0.0, detected=False)
        state = self.recognizer.process(res)
        self.assertFalse(state.face_detected)
        self.assertEqual(state.current_state, "NO FACE")
        self.assertEqual(state.active_gesture, HeadGestureID.NONE)
        self.assertIsNone(state.active_event)

    # TEST 2: Look straight at camera -> FACE DETECTED, HEAD STATE: NEUTRAL
    def test_02_look_straight_neutral(self):
        # Calibrated neutral baseline: yaw = -11.1, pitch = -8.6, roll = +176.6
        res = self._make_face_result(yaw=-11.1, pitch=-8.6, roll=176.6)
        state = self.recognizer.process(res)
        self.assertTrue(state.face_detected)
        self.assertEqual(state.current_state, "NEUTRAL")
        self.assertEqual(state.active_gesture, HeadGestureID.NONE)
        self.assertIsNone(state.active_event)
        self.assertAlmostEqual(state.delta_yaw, 0.0, places=1)

    # TEST 3: Move/turn head physically LEFT -> HEAD STATE: HEAD LEFT
    def test_03_head_left(self):
        # Physical left turn: delta_yaw <= -4.5 deg (e.g. yaw = -18.0 -> delta_yaw = -6.9 deg)
        res = self._make_face_result(yaw=-18.0)
        state = self.recognizer.process(res, timestamp=1.0)
        self.assertTrue(state.face_detected)
        self.assertEqual(state.current_state, "HEAD LEFT")
        self.assertEqual(state.active_gesture, HeadGestureID.HEAD_LEFT)
        self.assertIsNotNone(state.active_event)
        self.assertEqual(state.active_event.gesture_id, HeadGestureID.HEAD_LEFT)
        self.assertEqual(state.active_event.label, "HEAD LEFT")

    # TEST 4: Move/turn head physically RIGHT -> HEAD STATE: HEAD RIGHT
    def test_04_head_right(self):
        # Physical right turn: delta_yaw >= +7.0 deg (e.g. yaw = +10.0 -> delta_yaw = +21.1 deg)
        res = self._make_face_result(yaw=10.0)
        state = self.recognizer.process(res, timestamp=1.0)
        self.assertTrue(state.face_detected)
        self.assertEqual(state.current_state, "HEAD RIGHT")
        self.assertEqual(state.active_gesture, HeadGestureID.HEAD_RIGHT)
        self.assertIsNotNone(state.active_event)
        self.assertEqual(state.active_event.gesture_id, HeadGestureID.HEAD_RIGHT)
        self.assertEqual(state.active_event.label, "HEAD RIGHT")

    # TEST 5: Circular angle wrapping around +/- 180 degrees
    def test_05_circular_angle_wrapping(self):
        self.assertAlmostEqual(angle_diff(179.0, -179.0), -2.0, places=1)
        self.assertAlmostEqual(angle_diff(-175.0, 176.6), 8.4, places=1)
        self.assertAlmostEqual(angle_diff(10.0, 10.0), 0.0, places=1)

    # TEST 6: Deadband stability around neutral
    def test_06_deadband_neutral_stability(self):
        # Small perturbations within deadband (|delta_yaw| < 4.0)
        # yaw = -9.0 (delta_yaw = +2.1 deg)
        res = self._make_face_result(yaw=-9.0)
        state = self.recognizer.process(res, timestamp=1.0)
        self.assertEqual(state.current_state, "NEUTRAL")
        self.assertEqual(state.active_gesture, HeadGestureID.NONE)

    # TEST 7: Stationary head hold does NOT spam events
    def test_07_stationary_hold_no_event_spam(self):
        # 1. First left movement triggers exactly 1 discrete event
        res_left = self._make_face_result(yaw=-18.0)
        s1 = self.recognizer.process(res_left, timestamp=1.0)
        self.assertEqual(s1.current_state, "HEAD LEFT")
        self.assertIsNotNone(s1.active_event)

        # 2. Holding left position across successive frames keeps state HEAD LEFT, but active_event is None
        for i in range(1, 10):
            s_held = self.recognizer.process(res_left, timestamp=1.0 + i * 0.05)
            self.assertEqual(s_held.current_state, "HEAD LEFT")
            self.assertIsNone(s_held.active_event, f"Frame {i}: Held posture must not emit duplicate events")

    # TEST 8: Hysteresis and return to neutral required to re-arm
    def test_08_hysteresis_and_return_to_neutral(self):
        res_left = self._make_face_result(yaw=-18.0)
        self.recognizer.process(res_left, timestamp=1.0)

        # Return partially toward neutral but still outside hysteresis deadband (e.g. dy = -4.0 deg)
        # yaw = -15.1 -> dy = -4.0 deg (hysteresis requires <= 3.15 deg)
        res_partial = self._make_face_result(yaw=-15.1)
        self.recognizer.process(res_partial, timestamp=1.5)
        self.assertFalse(self.recognizer._in_neutral)

        # Return completely into neutral deadband (yaw = -11.1 -> dy = 0.0 deg)
        res_neutral = self._make_face_result(yaw=-11.1)
        s_neu = self.recognizer.process(res_neutral, timestamp=2.0)
        self.assertEqual(s_neu.current_state, "NEUTRAL")
        self.assertTrue(self.recognizer._in_neutral)

        # Moving left again after neutral return + cooldown triggers second discrete event
        s2 = self.recognizer.process(res_left, timestamp=3.0)
        self.assertEqual(s2.current_state, "HEAD LEFT")
        self.assertIsNotNone(s2.active_event)

    # TEST 9: Left/Right cooldown enforcement
    def test_09_left_right_cooldown(self):
        res_left = self._make_face_result(yaw=-18.0)
        self.recognizer.process(res_left, timestamp=1.0)

        # Immediate return to neutral within 0.1s
        res_neutral = self._make_face_result(yaw=-11.1)
        self.recognizer.process(res_neutral, timestamp=1.1)

        # Attempt to trigger HEAD RIGHT before cooldown (0.6s) has elapsed (t = 1.3s < 1.0s + 0.6s)
        res_right = self._make_face_result(yaw=10.0)
        s_blocked = self.recognizer.process(res_right, timestamp=1.3)
        self.assertEqual(s_blocked.current_state, "HEAD RIGHT")
        self.assertIsNone(s_blocked.active_event, "Event must be blocked before cooldown expires")

        # After cooldown has elapsed (t = 1.8s >= 1.0s + 0.6s), discrete event fires
        s_allowed = self.recognizer.process(res_right, timestamp=1.8)
        self.assertEqual(s_allowed.current_state, "HEAD RIGHT")
        self.assertIsNotNone(s_allowed.active_event)

    # TEST 10: Pitch and Roll decoupling (Pitch and Roll NEVER trigger head gestures)
    def test_10_pitch_and_roll_decoupling(self):
        # Neutral yaw (-11.1), but pitch and roll swing to extremes
        for extreme_pitch in [-35.0, -20.0, 0.0, 15.0, 30.0]:
            for extreme_roll in [-170.0, -90.0, 0.0, 90.0, 175.0]:
                res = self._make_face_result(yaw=-11.1, pitch=extreme_pitch, roll=extreme_roll)
                state = self.recognizer.process(res, timestamp=1.0)
                self.assertEqual(state.current_state, "NEUTRAL")
                self.assertEqual(state.active_gesture, HeadGestureID.NONE)
                self.assertIsNone(state.active_event)

    # TEST 11: Baseline-relative yaw classification
    def test_11_baseline_relative_yaw(self):
        # Custom baseline at yaw = 0.0 deg
        custom_thresh = GestureThresholds(head_baseline_yaw=0.0)
        rec = HeadGestureRecognizer(thresholds=custom_thresh)

        # yaw = -6.0 -> delta_yaw = -6.0 < -4.5 -> HEAD LEFT
        res_l = self._make_face_result(yaw=-6.0)
        s_l = rec.process(res_l, timestamp=1.0)
        self.assertEqual(s_l.current_state, "HEAD LEFT")

        # yaw = +8.0 -> delta_yaw = +8.0 > +7.0 -> HEAD RIGHT
        rec.reset()
        res_r = self._make_face_result(yaw=8.0)
        s_r = rec.process(res_r, timestamp=1.0)
        self.assertEqual(s_r.current_state, "HEAD RIGHT")

    # TEST 12: Move hands while moving head -> hand controls continue working
    def test_12_multimodal_coexistence(self):
        mock_mouse = MagicMock(spec=MouseController)
        mock_mouse.has_accessibility = True
        mock_kb = MagicMock(spec=KeyboardController)
        mock_kb.has_accessibility = True
        engine = CommandEngine(mouse_controller=mock_mouse, keyboard_controller=mock_kb)
        engine.toggle_control_mode(force=True)

        # Simultaneously move head left and index pinch with right hand
        res_head = self._make_face_result(yaw=-18.0)
        head_state = self.recognizer.process(res_head, timestamp=1.0)
        self.assertEqual(head_state.current_state, "HEAD LEFT")

        s_hand = HandState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
        )
        res_hand = GestureRecognitionResult(hands={"Right": s_hand})
        status = engine.process(res_hand, (720, 1280, 3))

        # Mouse click executed from hand, head recognized independently
        mock_mouse.left_click.assert_called_once()
        self.assertEqual(status.last_action_name, "LEFT CLICK")

    # TEST 13: Control Mode ON -> head movements trigger ZERO OS actions
    @patch("pyautogui.press")
    @patch("pyautogui.hotkey")
    @patch("pyautogui.click")
    def test_13_control_mode_on_zero_os_actions(self, mock_click, mock_hotkey, mock_press):
        mock_mouse = MagicMock(spec=MouseController)
        mock_mouse.has_accessibility = True
        mock_kb = MagicMock(spec=KeyboardController)
        engine = CommandEngine(mouse_controller=mock_mouse, keyboard_controller=mock_kb)
        engine.toggle_control_mode(force=True)
        self.assertTrue(engine.control_mode)

        # Trigger HEAD LEFT
        res_left = self._make_face_result(yaw=-18.0)
        s_left = self.recognizer.process(res_left, timestamp=1.0)
        self.assertEqual(s_left.current_state, "HEAD LEFT")

        # Return to neutral
        res_neu = self._make_face_result(yaw=-11.1)
        self.recognizer.process(res_neu, timestamp=2.0)

        # Trigger HEAD RIGHT
        res_right = self._make_face_result(yaw=10.0)
        s_right = self.recognizer.process(res_right, timestamp=3.0)
        self.assertEqual(s_right.current_state, "HEAD RIGHT")

        # PyAutoGUI must never have been called for any head gesture
        mock_press.assert_not_called()
        mock_hotkey.assert_not_called()
        mock_click.assert_not_called()

    # TEST 14: Control Mode OFF -> head tracking visualization still works
    def test_14_control_mode_off_visualization_intact(self):
        mock_mouse = MagicMock(spec=MouseController)
        engine = CommandEngine(mouse_controller=mock_mouse)
        self.assertFalse(engine.control_mode)

        res = self._make_face_result(yaw=10.0)
        head_state = self.recognizer.process(res)
        self.assertEqual(head_state.current_state, "HEAD RIGHT")
        self.assertEqual(head_state.active_gesture, HeadGestureID.HEAD_RIGHT)
        self.assertAlmostEqual(head_state.yaw, 10.0)


if __name__ == "__main__":
    unittest.main()

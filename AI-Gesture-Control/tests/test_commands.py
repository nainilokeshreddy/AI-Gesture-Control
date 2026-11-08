"""
Unit Tests for Hand Identity, Pinch State Independence, and Left Click Dispatch (Tests 1 - 7)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Verifies all 7 test cases specified in the task:
- Test 1: Physical Right hand only, index pinch -> 1 hand, Right, INDEX PINCH, LEFT CLICK, no Left hand
- Test 2: Physical Left hand only, index pinch -> 1 hand, Left, INDEX PINCH, LEFT CLICK
- Test 3: Right hand only, moving fingers without pinch -> 1 hand, Right, Index cursor moves, no click
- Test 4: Right hand only, pinch and hold -> Exactly one click, no repeated clicks
- Test 5: Right hand pinch -> release -> pinch -> Exactly two separate clicks
- Test 6: Both hands visible, Right hand pinch only -> 2 hands, Right hand click executes, Left hand no click
- Test 7: Both hands visible, neither pinches -> 2 hands, no clicks
- Additional: Phantom duplicate suppression and handedness stabilization unit tests
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.settings import ControlConfig, SmoothingConfig
from control.mouse_controller import MouseController
from control.keyboard_controller import KeyboardController
from control.action_mapper import ActionMapper, SemanticAction, ActionRequest
from control.command_engine import CommandEngine, CommandEngineStatus
from gestures.hand_gestures import (
    GestureRecognitionResult,
    HandGestureState,
    HandGestureID,
    HandGestureEvent,
    GESTURE_TYPE_MAP,
)
from vision.hand_tracker import (
    HandTracker,
    HandLandmarkData,
    HandTrackingResult,
)


class TestHandIdentityAndLeftClick(unittest.TestCase):
    """
    Test suite for hand identity in left-click control, ensuring physical hand attribution
    remains completely independent and accurate without false left-hand confirmation.
    """

    def setUp(self):
        self.mock_mouse = MagicMock(spec=MouseController)
        self.mock_mouse.has_accessibility = True
        self.mock_mouse.move_cursor.return_value = True
        self.mock_mouse.left_click.return_value = True
        self.mock_mouse.right_click.return_value = True

        self.mock_kb = MagicMock(spec=KeyboardController)
        self.mock_kb.has_accessibility = True
        self.mock_kb.press.return_value = True
        self.mock_kb.hotkey.return_value = True

        self.engine = CommandEngine(
            mouse_controller=self.mock_mouse,
            keyboard_controller=self.mock_kb,
        )
        self.engine.toggle_control_mode(force=True)

    # TEST 1: Physical Right hand only, index pinch
    def test_01_right_hand_only_index_pinch(self):
        """
        Physical Right hand only, index pinch:
        Expected:
        - Detected hands: 1
        - Hand: Right
        - Gesture: INDEX PINCH
        - Action: LEFT CLICK
        - No Left hand displayed / active
        """
        s_right = HandGestureState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
            raw_index_pos=(640, 360),
        )
        res = GestureRecognitionResult(hands={"Right": s_right})

        status = self.engine.process(res, (720, 1280, 3))

        self.assertEqual(len(res.hands), 1)
        self.assertIn("Right", res.hands)
        self.assertNotIn("Left", res.hands)
        self.assertTrue(s_right.is_pinching_index)
        self.mock_mouse.left_click.assert_called_once()
        self.assertEqual(status.last_action_name, "LEFT CLICK")
        self.assertEqual(status.last_action_hand, "Right")

    # TEST 2: Physical Left hand only, index pinch
    def test_02_left_hand_only_index_pinch(self):
        """
        Physical Left hand only, index pinch:
        Expected:
        - Detected hands: 1
        - Hand: Left
        - Gesture: INDEX PINCH
        - Action: LEFT CLICK
        """
        s_left = HandGestureState(
            handedness="Left",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(320, 240),
            raw_index_pos=(320, 240),
        )
        res = GestureRecognitionResult(hands={"Left": s_left})

        status = self.engine.process(res, (720, 1280, 3))

        self.assertEqual(len(res.hands), 1)
        self.assertIn("Left", res.hands)
        self.assertNotIn("Right", res.hands)
        self.assertTrue(s_left.is_pinching_index)
        self.mock_mouse.left_click.assert_called_once()
        self.assertEqual(status.last_action_name, "LEFT CLICK")
        self.assertEqual(status.last_action_hand, "Left")

    # TEST 3: Right hand only, moving fingers without pinch
    def test_03_right_hand_only_moving_without_pinch(self):
        """
        Right hand only, moving fingers without pinch:
        Expected:
        - Detected hands: 1
        - Hand: Right
        - Index cursor moves
        - No click
        """
        s_right = HandGestureState(
            handedness="Right",
            is_index_pointing=True,
            is_pinching_index=False,
            is_pinching_middle=False,
            smoothed_index_pos=(700, 400),
            raw_index_pos=(700, 400),
        )
        res = GestureRecognitionResult(hands={"Right": s_right})

        status = self.engine.process(res, (720, 1280, 3))

        self.assertEqual(len(res.hands), 1)
        self.assertIn("Right", res.hands)
        self.assertFalse(s_right.is_pinching_index)
        self.mock_mouse.move_cursor.assert_called_once_with(700, 400, 1280, 720)
        self.mock_mouse.left_click.assert_not_called()
        self.mock_mouse.right_click.assert_not_called()
        self.assertNotEqual(status.last_action_name, "LEFT CLICK")

    # TEST 4: Right hand only, pinch and hold
    def test_04_right_hand_pinch_and_hold(self):
        """
        Right hand only, pinch and hold:
        Expected:
        - Exactly one click
        - No repeated clicks
        """
        s_right = HandGestureState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
            raw_index_pos=(640, 360),
        )
        res = GestureRecognitionResult(hands={"Right": s_right})

        # Frame 1: Initial pinch
        status1 = self.engine.process(res, (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 1)
        self.assertEqual(status1.last_action_hand, "Right")

        # Frames 2-6: Held pinch
        for _ in range(5):
            status = self.engine.process(res, (720, 1280, 3))
            self.assertEqual(self.mock_mouse.left_click.call_count, 1)

    # TEST 5: Right hand pinch -> release -> pinch
    def test_05_right_hand_pinch_release_pinch(self):
        """
        Right hand pinch -> release -> pinch:
        Expected:
        - Exactly two separate clicks
        """
        s_pinch = HandGestureState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
            raw_index_pos=(640, 360),
        )
        s_release = HandGestureState(
            handedness="Right",
            is_pinching_index=False,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
            raw_index_pos=(640, 360),
        )

        # 1. First pinch
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 1)

        # 2. Release
        self.engine.process(GestureRecognitionResult(hands={"Right": s_release}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 1)

        # 3. Second pinch
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 2)

    # TEST 6: Both hands visible, Right hand pinch only
    def test_06_both_hands_visible_right_hand_pinch_only(self):
        """
        Both hands visible, Right hand pinch only:
        Expected:
        - Detected hands: 2
        - Right hand click executes
        - Left hand does NOT trigger a click
        - Action attribution clearly indicates Right hand
        """
        s_right = HandGestureState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(800, 400),
            raw_index_pos=(800, 400),
        )
        s_left = HandGestureState(
            handedness="Left",
            is_pinching_index=False,
            is_index_pointing=False,
            smoothed_index_pos=(200, 400),
            raw_index_pos=(200, 400),
        )
        res = GestureRecognitionResult(hands={"Right": s_right, "Left": s_left})

        status = self.engine.process(res, (720, 1280, 3))

        self.assertEqual(len(res.hands), 2)
        self.mock_mouse.left_click.assert_called_once()
        self.assertEqual(status.last_action_name, "LEFT CLICK")
        self.assertEqual(status.last_action_hand, "Right")

    # TEST 7: Both hands visible, neither pinches
    def test_07_both_hands_visible_neither_pinches(self):
        """
        Both hands visible, neither pinches:
        Expected:
        - Detected hands: 2
        - No clicks
        """
        s_right = HandGestureState(
            handedness="Right",
            is_pinching_index=False,
            is_pinching_middle=False,
            is_index_pointing=True,
            smoothed_index_pos=(800, 400),
            raw_index_pos=(800, 400),
        )
        s_left = HandGestureState(
            handedness="Left",
            is_pinching_index=False,
            is_pinching_middle=False,
            is_index_pointing=False,
            smoothed_index_pos=(200, 400),
            raw_index_pos=(200, 400),
        )
        res = GestureRecognitionResult(hands={"Right": s_right, "Left": s_left})

        status = self.engine.process(res, (720, 1280, 3))

        self.assertEqual(len(res.hands), 2)
        self.mock_mouse.left_click.assert_not_called()
        self.mock_mouse.right_click.assert_not_called()
        self.assertNotEqual(status.last_action_name, "LEFT CLICK")

    # TEST 8: Stale hand cleanup
    def test_08_stale_hand_state_cleanup(self):
        """
        Verify that when a pinching hand disappears from frame, its state is cleaned up,
        so reappearing without pinch does not cause an unexpected click or latch.
        """
        s_pinch = HandGestureState(
            handedness="Right",
            is_pinching_index=True,
            is_index_pointing=True,
            smoothed_index_pos=(640, 360),
            raw_index_pos=(640, 360),
        )
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 1)

        # Hand leaves frame
        self.engine.process(GestureRecognitionResult(hands={}), (720, 1280, 3))

        # Hand returns still pinching -> should be treated as fresh pinch transition
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.left_click.call_count, 2)


class TestPhantomAndStabilization(unittest.TestCase):
    """
    Unit tests specifically verifying phantom duplicate suppression and handedness stabilization.
    """

    def test_phantom_duplicate_filtered(self):
        """
        When two candidate hands have overlapping palm/wrist locations,
        the lower-confidence candidate is filtered out as a phantom duplicate.
        """
        tracker = HandTracker()

        # Build two candidates: One high confidence, one lower confidence almost co-located
        lms_main = [(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
        px_main = [(int(x * 640), int(y * 480)) for x, y, _ in lms_main]
        hand_real = HandLandmarkData(
            landmarks=lms_main,
            landmarks_px=px_main,
            handedness="Right",
            score=0.95,
        )

        # Phantom candidate located 20 pixels away
        lms_phantom = [(0.5 + i * 0.01 + 0.01, 0.5 + i * 0.01 + 0.01, 0.0) for i in range(21)]
        px_phantom = [(int(x * 640), int(y * 480)) for x, y, _ in lms_phantom]
        hand_phantom = HandLandmarkData(
            landmarks=lms_phantom,
            landmarks_px=px_phantom,
            handedness="Left",
            score=0.72,
        )

        filtered = tracker._filter_phantom_duplicates([hand_real, hand_phantom])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].handedness, "Right")
        self.assertAlmostEqual(filtered[0].score, 0.95)

    def test_two_distinct_hands_not_filtered(self):
        """
        When two candidate hands are spaced far apart (e.g. left and right hands),
        neither is filtered.
        """
        tracker = HandTracker()

        lms_r = [(0.75 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
        px_r = [(int(x * 640), int(y * 480)) for x, y, _ in lms_r]
        hand_r = HandLandmarkData(
            landmarks=lms_r,
            landmarks_px=px_r,
            handedness="Right",
            score=0.92,
        )

        lms_l = [(0.25 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
        px_l = [(int(x * 640), int(y * 480)) for x, y, _ in lms_l]
        hand_l = HandLandmarkData(
            landmarks=lms_l,
            landmarks_px=px_l,
            handedness="Left",
            score=0.90,
        )

        filtered = tracker._filter_phantom_duplicates([hand_r, hand_l])
        self.assertEqual(len(filtered), 2)

    def test_handedness_stabilization_prevents_single_frame_flip(self):
        """
        A single-frame handedness flip on a tracked spatial hand is smoothed out.
        """
        tracker = HandTracker()

        lms = [(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
        px = [(int(x * 640), int(y * 480)) for x, y, _ in lms]

        # Feed 3 consistent frames of Right hand
        for i in range(3):
            hand_frame = HandLandmarkData(landmarks=lms, landmarks_px=px, handedness="Right", score=0.9)
            res = tracker._stabilize_handedness([hand_frame], current_ms=1000 + i * 33)
            self.assertEqual(res[0].handedness, "Right")

        # Frame 4: Sudden single-frame flip to Left with lower confidence at the same location
        hand_frame_flip = HandLandmarkData(landmarks=lms, landmarks_px=px, handedness="Left", score=0.7)
        res_flip = tracker._stabilize_handedness([hand_frame_flip], current_ms=1100)
        self.assertEqual(res_flip[0].handedness, "Right")  # Stabilized!


if __name__ == "__main__":
    unittest.main()

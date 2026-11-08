"""
Unit Tests for Stage 4B Part 2: Complete Hand Controls (Controls 1 through 8)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Verifies all 12 test requirements:
TEST 1: Middle pinch once -> exactly one right click
TEST 2: Hold middle pinch -> no repeated right clicks
TEST 3: Release then middle pinch again -> one additional right click
TEST 4: Fist -> open palm -> exactly one play/pause action (press 'space')
TEST 5: Swipe right -> exactly one NEXT action (hotkey 'shift', 'n')
TEST 6: Swipe left -> exactly one PREVIOUS action (hotkey 'shift', 'p')
TEST 7: Open palm held ~2s while OFF -> CONTROL MODE ON
TEST 8: Open palm held ~2s while ON -> CONTROL MODE OFF
TEST 9: Two-finger upward movement -> ZOOM IN action (hotkey 'command', '+')
TEST 10: Two-finger downward movement -> ZOOM OUT action (hotkey 'command', '-')
TEST 11: CONTROL MODE OFF -> none of the computer actions execute
TEST 12: ESC -> immediate control disable (stop all control)
"""

import unittest
from unittest.mock import MagicMock, patch

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


class TestMouseController(unittest.TestCase):
    """Test suite for MouseController coordinate mapping, click, and right click handling."""

    def setUp(self):
        self.config = ControlConfig(cursor_margin_percent=0.15)
        self.smoothing = SmoothingConfig()
        self.controller = MouseController(
            config=self.config,
            smoothing=self.smoothing,
            screen_size=(1000, 800),
        )

    def test_map_camera_to_screen_center(self):
        """Center of camera frame maps to center of screen."""
        cam_w, cam_h = 1000, 1000
        sx, sy = self.controller.map_camera_to_screen(500, 500, cam_w, cam_h)
        self.assertAlmostEqual(sx, 500, delta=20)
        self.assertAlmostEqual(sy, 400, delta=20)

    def test_map_camera_to_screen_corners_deadzone(self):
        """Points at or beyond the 15% margins reach the screen edges."""
        cam_w, cam_h = 1000, 1000
        sx_tl, sy_tl = self.controller.map_camera_to_screen(150, 150, cam_w, cam_h)
        self.assertLessEqual(sx_tl, 50)
        self.assertLessEqual(sy_tl, 50)

        self.controller.reset_smoother()

        sx_br, sy_br = self.controller.map_camera_to_screen(850, 850, cam_w, cam_h)
        self.assertGreaterEqual(sx_br, 950)
        self.assertGreaterEqual(sy_br, 750)

    @patch("pyautogui.moveTo")
    def test_move_cursor_calls_pyautogui(self, mock_move):
        """move_cursor invokes pyautogui.moveTo with target coordinates."""
        self.controller.has_accessibility = True
        success = self.controller.move_cursor(500, 500, 1000, 1000)
        self.assertTrue(success)
        mock_move.assert_called_once()

    @patch("pyautogui.moveTo")
    def test_move_cursor_skips_identical_coordinates(self, mock_move):
        """Identical successive screen coordinates skip redundant pyautogui.moveTo call."""
        self.controller.has_accessibility = True
        self.controller.move_cursor(500, 500, 1000, 1000)
        self.assertEqual(mock_move.call_count, 1)

        # Call again with identical coordinates
        success = self.controller.move_cursor(500, 500, 1000, 1000)
        self.assertTrue(success)
        # Should still be 1 (skipped redundant call)
        self.assertEqual(mock_move.call_count, 1)

        # Call with different coordinates
        success2 = self.controller.move_cursor(200, 200, 1000, 1000)
        self.assertTrue(success2)
        # Should now be 2
        self.assertEqual(mock_move.call_count, 2)

    @patch("pyautogui.click")
    def test_left_click_calls_pyautogui(self, mock_click):
        """left_click invokes pyautogui.click."""
        self.controller.has_accessibility = True
        success = self.controller.left_click()
        self.assertTrue(success)
        mock_click.assert_called_once_with(button="left")

    @patch("pyautogui.rightClick")
    def test_right_click_calls_pyautogui(self, mock_rclick):
        """right_click invokes pyautogui.rightClick."""
        self.controller.has_accessibility = True
        success = self.controller.right_click()
        self.assertTrue(success)
        mock_rclick.assert_called_once()

    @patch("pyautogui.rightClick")
    def test_right_click_cooldown(self, mock_rclick):
        """Rapid successive right clicks are debounced."""
        self.controller.has_accessibility = True
        first = self.controller.right_click()
        self.assertTrue(first)

        second = self.controller.right_click()
        self.assertFalse(second)
        self.assertEqual(mock_rclick.call_count, 1)


class TestKeyboardController(unittest.TestCase):
    """Test suite for KeyboardController key press and hotkey dispatch."""

    def setUp(self):
        self.controller = KeyboardController()
        self.controller.has_accessibility = True

    @patch("pyautogui.press")
    def test_press_space(self, mock_press):
        """press invokes pyautogui.press."""
        success = self.controller.press("space", cooldown_sec=0.2)
        self.assertTrue(success)
        mock_press.assert_called_once_with("space")

    @patch("pyautogui.hotkey")
    def test_hotkey(self, mock_hotkey):
        """hotkey invokes pyautogui.hotkey with keys."""
        success = self.controller.hotkey("shift", "n", cooldown_sec=0.2)
        self.assertTrue(success)
        mock_hotkey.assert_called_once_with("shift", "n")

    @patch("pyautogui.press")
    def test_press_cooldown(self, mock_press):
        """Rapid successive key presses within cooldown are debounced."""
        first = self.controller.press("space", cooldown_sec=0.5)
        self.assertTrue(first)
        second = self.controller.press("space", cooldown_sec=0.5)
        self.assertFalse(second)
        self.assertEqual(mock_press.call_count, 1)


class TestActionMapper(unittest.TestCase):
    """Test suite for ActionMapper transition and state translation."""

    def setUp(self):
        self.mapper = ActionMapper()

    def test_index_cursor_move_action(self):
        """Index pointing generates MOVE_CURSOR action."""
        state = HandGestureState(
            handedness="Right",
            is_index_pointing=True,
            is_pinching_index=False,
            is_pinching_middle=False,
            smoothed_index_pos=(640, 360),
        )
        res = GestureRecognitionResult(hands={"Right": state})
        actions = self.mapper.map_gestures(res)

        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].action, SemanticAction.MOVE_CURSOR)

    def test_middle_pinch_transition(self):
        """
        Verify Middle Pinch:
        Frame 1: NO PINCH -> 0 clicks
        Frame 2: PINCH -> 1 RIGHT_CLICK
        Frame 3: PINCH (held) -> 0 clicks
        Frame 4: NO PINCH (release) -> 0 clicks
        Frame 5: PINCH (re-pinch) -> 1 RIGHT_CLICK
        """
        # Frame 1: Not pinching
        s1 = HandGestureState(handedness="Right", is_pinching_middle=False)
        a1 = self.mapper.map_gestures(GestureRecognitionResult(hands={"Right": s1}))
        self.assertEqual(len([a for a in a1 if a.action == SemanticAction.RIGHT_CLICK]), 0)

        # Frame 2: Middle Pinch DOWN!
        s2 = HandGestureState(handedness="Right", is_pinching_middle=True)
        a2 = self.mapper.map_gestures(GestureRecognitionResult(hands={"Right": s2}))
        rclicks_2 = [a for a in a2 if a.action == SemanticAction.RIGHT_CLICK]
        self.assertEqual(len(rclicks_2), 1)

        # Frame 3: Held
        s3 = HandGestureState(handedness="Right", is_pinching_middle=True)
        a3 = self.mapper.map_gestures(GestureRecognitionResult(hands={"Right": s3}))
        rclicks_3 = [a for a in a3 if a.action == SemanticAction.RIGHT_CLICK]
        self.assertEqual(len(rclicks_3), 0)

        # Frame 4: Release
        s4 = HandGestureState(handedness="Right", is_pinching_middle=False)
        a4 = self.mapper.map_gestures(GestureRecognitionResult(hands={"Right": s4}))
        rclicks_4 = [a for a in a4 if a.action == SemanticAction.RIGHT_CLICK]
        self.assertEqual(len(rclicks_4), 0)

        # Frame 5: Re-pinch!
        s5 = HandGestureState(handedness="Right", is_pinching_middle=True)
        a5 = self.mapper.map_gestures(GestureRecognitionResult(hands={"Right": s5}))
        rclicks_5 = [a for a in a5 if a.action == SemanticAction.RIGHT_CLICK]
        self.assertEqual(len(rclicks_5), 1)


class TestStage4BPart2Controls(unittest.TestCase):
    """
    Comprehensive tests validating all 12 specific test cases for Stage 4B Part 2.
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

    # TEST 1: Middle pinch once -> exactly one right click
    def test_01_middle_pinch_once_triggers_one_right_click(self):
        self.engine.toggle_control_mode(force=True)

        # Frame 1: not pinching
        s1 = HandGestureState(handedness="Right", is_pinching_middle=False)
        self.engine.process(GestureRecognitionResult(hands={"Right": s1}), (720, 1280, 3))

        # Frame 2: middle pinch down
        s2 = HandGestureState(handedness="Right", is_pinching_middle=True)
        status = self.engine.process(GestureRecognitionResult(hands={"Right": s2}), (720, 1280, 3))

        self.mock_mouse.right_click.assert_called_once()
        self.assertEqual(status.last_action_name, "RIGHT CLICK")

    # TEST 2: Hold middle pinch -> no repeated right clicks
    def test_02_hold_middle_pinch_no_repeated_right_clicks(self):
        self.engine.toggle_control_mode(force=True)

        s_pinched = HandGestureState(handedness="Right", is_pinching_middle=True)
        # First contact
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinched}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.right_click.call_count, 1)

        # Held for 5 subsequent frames
        for _ in range(5):
            self.engine.process(GestureRecognitionResult(hands={"Right": s_pinched}), (720, 1280, 3))

        self.assertEqual(self.mock_mouse.right_click.call_count, 1)

    # TEST 3: Release then middle pinch again -> one additional right click
    def test_03_release_and_middle_pinch_again(self):
        self.engine.toggle_control_mode(force=True)

        s_pinched = HandGestureState(handedness="Right", is_pinching_middle=True)
        s_released = HandGestureState(handedness="Right", is_pinching_middle=False)

        # Pinch 1
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinched}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.right_click.call_count, 1)

        # Release
        self.engine.process(GestureRecognitionResult(hands={"Right": s_released}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.right_click.call_count, 1)

        # Pinch 2
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinched}), (720, 1280, 3))
        self.assertEqual(self.mock_mouse.right_click.call_count, 2)

    # TEST 4: Fist -> open palm -> exactly one play/pause action (press 'space')
    def test_04_fist_to_open_palm_play_pause(self):
        self.engine.toggle_control_mode(force=True)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_FIST_OPEN,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_FIST_OPEN],
            handedness="Right",
            label="FIST -> OPEN PALM DETECTED",
            timestamp=1.0,
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.mock_kb.press.assert_called_once_with("space", cooldown_sec=0.8)
        self.assertEqual(status.last_action_name, "PLAY/PAUSE")

    # TEST 5: Swipe right -> exactly one NEXT action (hotkey 'shift', 'n')
    def test_05_swipe_right_next(self):
        self.engine.toggle_control_mode(force=True)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_SWIPE_RIGHT,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_SWIPE_RIGHT],
            handedness="Right",
            label="SWIPE RIGHT",
            timestamp=1.0,
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.mock_kb.hotkey.assert_called_once_with("shift", "n", cooldown_sec=0.6)
        self.assertEqual(status.last_action_name, "NEXT")

    # TEST 6: Swipe left -> exactly one PREVIOUS action (hotkey 'shift', 'p')
    def test_06_swipe_left_previous(self):
        self.engine.toggle_control_mode(force=True)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_SWIPE_LEFT,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_SWIPE_LEFT],
            handedness="Right",
            label="SWIPE LEFT",
            timestamp=1.0,
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.mock_kb.hotkey.assert_called_once_with("shift", "p", cooldown_sec=0.6)
        self.assertEqual(status.last_action_name, "PREVIOUS")

    # TEST 7: Open palm held ~2s while OFF -> CONTROL MODE ON
    def test_07_open_palm_toggle_on_when_off(self):
        self.assertFalse(self.engine.control_mode)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_MODE_TOGGLE,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_MODE_TOGGLE],
            handedness="Right",
            label="MODE TOGGLE READY",
            timestamp=2.0,
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.assertTrue(status.control_mode)
        self.assertEqual(status.last_action_name, "MODE ON")

    # TEST 8: Open palm held ~2s while ON -> CONTROL MODE OFF
    def test_08_open_palm_toggle_off_when_on(self):
        self.engine.toggle_control_mode(force=True)
        self.assertTrue(self.engine.control_mode)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_MODE_TOGGLE,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_MODE_TOGGLE],
            handedness="Right",
            label="MODE TOGGLE READY",
            timestamp=2.0,
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.assertFalse(status.control_mode)
        self.assertEqual(status.last_action_name, "MODE OFF")

    # TEST 9: Two-finger upward movement -> ZOOM IN action (hotkey 'command', '+')
    def test_09_two_finger_up_zoom_in(self):
        self.engine.toggle_control_mode(force=True)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_VERTICAL_TWO_FINGER,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_VERTICAL_TWO_FINGER],
            handedness="Right",
            label="TWO-FINGER UP",
            timestamp=1.0,
            details={"direction": "UP"},
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.mock_kb.hotkey.assert_called_once_with("command", "+", cooldown_sec=0.5)
        self.assertEqual(status.last_action_name, "ZOOM IN")

    # TEST 10: Two-finger downward movement -> ZOOM OUT action (hotkey 'command', '-')
    def test_10_two_finger_down_zoom_out(self):
        self.engine.toggle_control_mode(force=True)

        ev = HandGestureEvent(
            gesture_id=HandGestureID.HAND_VERTICAL_TWO_FINGER,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_VERTICAL_TWO_FINGER],
            handedness="Right",
            label="TWO-FINGER DOWN",
            timestamp=1.0,
            details={"direction": "DOWN"},
        )
        res = GestureRecognitionResult(events=[ev])
        status = self.engine.process(res, (720, 1280, 3))

        self.mock_kb.hotkey.assert_called_once_with("command", "-", cooldown_sec=0.5)
        self.assertEqual(status.last_action_name, "ZOOM OUT")

    # TEST 11: CONTROL MODE OFF -> none of the computer actions execute
    def test_11_control_mode_off_blocks_all_actions(self):
        self.assertFalse(self.engine.control_mode)

        # 1. Middle Pinch
        s_pinch = HandGestureState(handedness="Right", is_pinching_middle=True)
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.mock_mouse.right_click.assert_not_called()

        # 2. Fist -> Open Palm
        ev_fist = HandGestureEvent(
            gesture_id=HandGestureID.HAND_FIST_OPEN,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_FIST_OPEN],
            handedness="Right",
            label="FIST -> OPEN PALM DETECTED",
            timestamp=1.0,
        )
        self.engine.process(GestureRecognitionResult(events=[ev_fist]), (720, 1280, 3))
        self.mock_kb.press.assert_not_called()

        # 3. Swipe Right
        ev_swipe = HandGestureEvent(
            gesture_id=HandGestureID.HAND_SWIPE_RIGHT,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_SWIPE_RIGHT],
            handedness="Right",
            label="SWIPE RIGHT",
            timestamp=1.0,
        )
        self.engine.process(GestureRecognitionResult(events=[ev_swipe]), (720, 1280, 3))
        self.mock_kb.hotkey.assert_not_called()

        # 4. Zoom
        ev_zoom = HandGestureEvent(
            gesture_id=HandGestureID.HAND_VERTICAL_TWO_FINGER,
            gesture_type=GESTURE_TYPE_MAP[HandGestureID.HAND_VERTICAL_TWO_FINGER],
            handedness="Right",
            label="TWO-FINGER UP",
            timestamp=1.0,
            details={"direction": "UP"},
        )
        self.engine.process(GestureRecognitionResult(events=[ev_zoom]), (720, 1280, 3))
        self.mock_kb.hotkey.assert_not_called()

    # TEST 12: ESC -> immediate control disable
    def test_12_esc_immediate_emergency_disable(self):
        self.engine.toggle_control_mode(force=True)
        self.assertTrue(self.engine.control_mode)

        self.engine.emergency_disable()
        self.assertFalse(self.engine.control_mode)
        self.assertEqual(self.engine.last_action_name, "DISABLED (ESC)")

        # Verify no actions execute after ESC
        s_pinch = HandGestureState(handedness="Right", is_pinching_middle=True)
        self.engine.process(GestureRecognitionResult(hands={"Right": s_pinch}), (720, 1280, 3))
        self.mock_mouse.right_click.assert_not_called()


if __name__ == "__main__":
    unittest.main()

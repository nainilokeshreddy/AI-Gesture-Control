"""
Unit Tests for Stage 4A - Hand Gesture Recognition Foundation
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Tests:
- Finger extension & curl classification
- Continuous index finger tracking & coordinate smoothing
- Quick pinch detection (index & middle) with hysteresis
- Fist-to-open palm edge-triggered transition
- Horizontal swipe detection (right & left) with dominance checks
- 2-second stationary open palm hold (mode toggle) with jitter reset
- Two-finger vertical movement (up & down)
- Multi-hand independent tracking (Left and Right)
- Cooldown and debounce management
"""

import unittest
from typing import List, Tuple

from config.settings import DEFAULT_CONFIG, GestureThresholds, CooldownConfig, SmoothingConfig
from gestures.hand_gestures import (
    HandGestureID,
    HandGestureEvent,
    HandGestureState,
    GestureRecognitionResult,
    HandGestureRecognizer,
    SingleHandTracker,
)
from vision.hand_tracker import HandLandmarkData, HandTrackingResult


def _make_21_landmarks(
    wrist: Tuple[int, int] = (640, 600),
    thumb: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (600, 560), (570, 520), (550, 480), (530, 440)
    ),
    index: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (610, 480), (600, 420), (590, 370), (580, 320)
    ),
    middle: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (640, 470), (640, 400), (640, 350), (640, 300)
    ),
    ring: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (670, 480), (675, 420), (680, 370), (685, 320)
    ),
    pinky: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (
        (700, 500), (710, 450), (720, 410), (730, 370)
    ),
) -> List[Tuple[int, int]]:
    """Generate a 21-point hand landmark list from joint tuples."""
    pts = [wrist]
    pts.extend(thumb)   # 1, 2, 3, 4
    pts.extend(index)   # 5, 6, 7, 8
    pts.extend(middle)  # 9, 10, 11, 12
    pts.extend(ring)    # 13, 14, 15, 16
    pts.extend(pinky)   # 17, 18, 19, 20
    return pts


def make_open_palm(cx: int = 640, cy: int = 500, handedness: str = "Right") -> HandLandmarkData:
    """Generate landmarks for an open palm centered at (cx, cy)."""
    wrist = (cx, cy + 120)
    thumb = ((cx - 40, cy + 60), (cx - 70, cy + 20), (cx - 90, cy - 20), (cx - 110, cy - 60))
    index = ((cx - 30, cy), (cx - 35, cy - 60), (cx - 40, cy - 110), (cx - 45, cy - 160))
    middle = ((cx, cy - 10), (cx, cy - 70), (cx, cy - 120), (cx, cy - 170))
    ring = ((cx + 30, cy), (cx + 35, cy - 60), (cx + 40, cy - 110), (cx + 45, cy - 160))
    pinky = ((cx + 60, cy + 20), (cx + 70, cy - 30), (cx + 80, cy - 70), (cx + 90, cy - 120))
    pts = _make_21_landmarks(wrist, thumb, index, middle, ring, pinky)
    return HandLandmarkData(landmarks_px=pts, handedness=handedness, score=0.95)


def make_fist(cx: int = 640, cy: int = 500, handedness: str = "Right") -> HandLandmarkData:
    """Generate landmarks for a closed fist (all fingers curled inward)."""
    wrist = (cx, cy + 120)
    # Curled thumb: tucked over MCPs
    thumb = ((cx - 30, cy + 60), (cx - 40, cy + 30), (cx - 20, cy + 10), (cx - 10, cy + 10))
    # Curled fingers: tips are closer to wrist/palm than PIPs
    index = ((cx - 30, cy), (cx - 35, cy - 50), (cx - 30, cy - 20), (cx - 25, cy + 10))
    middle = ((cx, cy - 10), (cx, cy - 60), (cx, cy - 30), (cx, cy + 5))
    ring = ((cx + 30, cy), (cx + 35, cy - 50), (cx + 30, cy - 20), (cx + 25, cy + 10))
    pinky = ((cx + 60, cy + 20), (cx + 65, cy - 30), (cx + 60, cy - 5), (cx + 55, cy + 15))
    pts = _make_21_landmarks(wrist, thumb, index, middle, ring, pinky)
    return HandLandmarkData(landmarks_px=pts, handedness=handedness, score=0.95)


def make_index_pointing(tip_x: int = 600, tip_y: int = 300, handedness: str = "Right") -> HandLandmarkData:
    """Generate landmarks for index pointing: index extended, others curled."""
    cx, cy = tip_x, tip_y + 160
    wrist = (cx, cy + 120)
    thumb = ((cx - 30, cy + 60), (cx - 40, cy + 30), (cx - 20, cy + 10), (cx - 10, cy + 10))
    # Index fully extended to (tip_x, tip_y)
    index = ((cx - 20, cy), (cx - 15, cy - 60), (cx - 10, cy - 110), (tip_x, tip_y))
    middle = ((cx, cy - 10), (cx, cy - 60), (cx, cy - 30), (cx, cy + 5))
    ring = ((cx + 30, cy), (cx + 35, cy - 50), (cx + 30, cy - 20), (cx + 25, cy + 10))
    pinky = ((cx + 60, cy + 20), (cx + 65, cy - 30), (cx + 60, cy - 5), (cx + 55, cy + 15))
    pts = _make_21_landmarks(wrist, thumb, index, middle, ring, pinky)
    return HandLandmarkData(landmarks_px=pts, handedness=handedness, score=0.95)


def make_two_finger_pointing(cx: int = 640, cy: int = 500, handedness: str = "Right") -> HandLandmarkData:
    """Generate landmarks for two-finger pointing: index & middle extended, ring & pinky curled."""
    wrist = (cx, cy + 120)
    thumb = ((cx - 30, cy + 60), (cx - 40, cy + 30), (cx - 20, cy + 10), (cx - 10, cy + 10))
    index = ((cx - 20, cy), (cx - 20, cy - 60), (cx - 20, cy - 110), (cx - 20, cy - 160))
    middle = ((cx + 20, cy - 10), (cx + 20, cy - 70), (cx + 20, cy - 120), (cx + 20, cy - 170))
    ring = ((cx + 40, cy), (cx + 45, cy - 50), (cx + 40, cy - 20), (cx + 35, cy + 10))
    pinky = ((cx + 65, cy + 20), (cx + 70, cy - 30), (cx + 65, cy - 5), (cx + 60, cy + 15))
    pts = _make_21_landmarks(wrist, thumb, index, middle, ring, pinky)
    return HandLandmarkData(landmarks_px=pts, handedness=handedness, score=0.95)


def make_pinch_data(
    pinch_type: str = "index",
    distance_px: float = 25.0,
    cx: int = 640,
    cy: int = 500,
    handedness: str = "Right",
) -> HandLandmarkData:
    """Generate landmarks where thumb tip is distance_px away from index or middle tip."""
    wrist = (cx, cy + 120)
    t_tip = (cx - int(distance_px / 2), cy - 40)
    thumb = ((cx - 30, cy + 60), (cx - 50, cy + 20), (cx - 40, cy - 10), t_tip)

    if pinch_type == "index":
        i_tip = (cx + int(distance_px / 2), cy - 40)
        index = ((cx - 20, cy), (cx - 15, cy - 20), (cx - 5, cy - 30), i_tip)
        # Middle extended away
        middle = ((cx + 30, cy - 10), (cx + 30, cy - 70), (cx + 30, cy - 120), (cx + 30, cy - 170))
    else:  # middle pinch
        # Index extended away
        index = ((cx - 30, cy), (cx - 35, cy - 60), (cx - 40, cy - 110), (cx - 45, cy - 160))
        m_tip = (cx + int(distance_px / 2), cy - 40)
        middle = ((cx + 10, cy - 10), (cx + 10, cy - 20), (cx + 5, cy - 30), m_tip)

    ring = ((cx + 60, cy), (cx + 65, cy - 50), (cx + 60, cy - 20), (cx + 55, cy + 10))
    pinky = ((cx + 85, cy + 20), (cx + 90, cy - 30), (cx + 85, cy - 5), (cx + 80, cy + 15))
    pts = _make_21_landmarks(wrist, thumb, index, middle, ring, pinky)
    return HandLandmarkData(landmarks_px=pts, handedness=handedness, score=0.95)


class TestHandGestureRecognition(unittest.TestCase):
    """Test suite for Stage 4A hand gesture recognition."""

    def setUp(self):
        self.recognizer = HandGestureRecognizer()

    def test_no_hand_detected(self):
        """Empty hand tracking input returns empty result and resets gracefully."""
        res = self.recognizer.process(HandTrackingResult())
        self.assertEqual(len(res.hands), 0)
        self.assertEqual(len(res.events), 0)

    def test_finger_classification_open_palm(self):
        """Verify Open Palm correctly classifies all fingers as extended."""
        hand = make_open_palm()
        tracker = SingleHandTracker("Right")
        ext, crl = tracker.classify_fingers(hand.landmarks_px)

        self.assertTrue(ext["index"])
        self.assertTrue(ext["middle"])
        self.assertTrue(ext["ring"])
        self.assertTrue(ext["pinky"])
        self.assertFalse(crl["index"])
        self.assertFalse(crl["middle"])

    def test_finger_classification_fist(self):
        """Verify Fist correctly classifies all 4 fingers as curled."""
        hand = make_fist()
        tracker = SingleHandTracker("Right")
        ext, crl = tracker.classify_fingers(hand.landmarks_px)

        self.assertTrue(crl["index"])
        self.assertTrue(crl["middle"])
        self.assertTrue(crl["ring"])
        self.assertTrue(crl["pinky"])
        self.assertFalse(ext["index"])
        self.assertFalse(ext["middle"])

    def test_finger_classification_pointing(self):
        """Verify index pointing classifies index as extended, middle/ring/pinky as curled."""
        hand = make_index_pointing()
        tracker = SingleHandTracker("Right")
        ext, crl = tracker.classify_fingers(hand.landmarks_px)

        self.assertTrue(ext["index"])
        self.assertTrue(crl["middle"])
        self.assertTrue(crl["ring"])
        self.assertTrue(crl["pinky"])

    def test_index_cursor_move_and_smoothing(self):
        """Verify index pointing updates smoothed position and active gesture."""
        hand = make_index_pointing(tip_x=500, tip_y=300)
        tracking_res = HandTrackingResult(hands=[hand], hand_count=1)

        res = self.recognizer.process(tracking_res, timestamp=100.0)
        self.assertIn("Right", res.hands)
        state = res.hands["Right"]

        self.assertEqual(state.active_gesture, HandGestureID.HAND_INDEX_MOVE)
        self.assertEqual(state.active_gesture_label, "INDEX MOVE")
        self.assertIsNotNone(state.smoothed_index_pos)
        self.assertEqual(state.raw_index_pos, (500, 300))

    def test_index_pinch_hysteresis_and_discrete_event(self):
        """
        Verify index pinch:
        1. Dist < 40px triggers continuous pinch state.
        2. Dist at 50px (between 40 and 55) stays pinching (hysteresis).
        3. Dist > 55px releases pinch and triggers discrete HAND_PINCH_INDEX event.
        """
        # 1. Close pinch: dist = 20px
        p_close = make_pinch_data("index", distance_px=20.0)
        res1 = self.recognizer.process(HandTrackingResult(hands=[p_close], hand_count=1), timestamp=1.0)
        state1 = res1.hands["Right"]
        self.assertTrue(state1.is_pinching_index)
        self.assertEqual(state1.active_gesture, HandGestureID.HAND_PINCH_INDEX)
        self.assertEqual(len(res1.events), 0)  # No discrete event on contact

        # 2. Hysteresis holding: dist = 50px (above 40px start, below 55px release)
        p_hold = make_pinch_data("index", distance_px=50.0)
        res2 = self.recognizer.process(HandTrackingResult(hands=[p_hold], hand_count=1), timestamp=1.1)
        state2 = res2.hands["Right"]
        self.assertTrue(state2.is_pinching_index)

        # 3. Release: dist = 65px (above 55px) within 0.5s window
        p_open = make_pinch_data("index", distance_px=65.0)
        res3 = self.recognizer.process(HandTrackingResult(hands=[p_open], hand_count=1), timestamp=1.25)
        state3 = res3.hands["Right"]
        self.assertFalse(state3.is_pinching_index)

        # Should emit exactly one discrete event
        self.assertEqual(len(res3.events), 1)
        self.assertEqual(res3.events[0].gesture_id, HandGestureID.HAND_PINCH_INDEX)
        self.assertEqual(res3.events[0].label, "INDEX PINCH")

    def test_middle_pinch_hysteresis_and_event(self):
        """Verify middle finger pinch triggers cleanly and is distinct from index pinch."""
        # 1. Middle pinch contact: dist = 25px
        p_close = make_pinch_data("middle", distance_px=25.0)
        res1 = self.recognizer.process(HandTrackingResult(hands=[p_close], hand_count=1), timestamp=2.0)
        self.assertTrue(res1.hands["Right"].is_pinching_middle)
        self.assertFalse(res1.hands["Right"].is_pinching_index)
        self.assertEqual(res1.hands["Right"].active_gesture, HandGestureID.HAND_PINCH_MIDDLE)

        # 2. Release middle pinch: dist = 70px (> 60px)
        p_release = make_pinch_data("middle", distance_px=70.0)
        res2 = self.recognizer.process(HandTrackingResult(hands=[p_release], hand_count=1), timestamp=2.2)
        self.assertEqual(len(res2.events), 1)
        self.assertEqual(res2.events[0].gesture_id, HandGestureID.HAND_PINCH_MIDDLE)
        self.assertEqual(res2.events[0].label, "MIDDLE PINCH")

    def test_fist_to_open_palm_transition(self):
        """
        Verify transition from Fist to Open Palm:
        - Fires exactly ONCE on transition.
        - Subsequent frames of Open Palm do NOT re-trigger.
        """
        fist = make_fist()
        open_palm = make_open_palm()

        # Step 1: Hand is in fist
        for t in [1.0, 1.1, 1.2]:
            res = self.recognizer.process(HandTrackingResult(hands=[fist], hand_count=1), timestamp=t)
            self.assertEqual(len(res.events), 0)

        # Step 2: Hand transitions to open palm at t=1.4s (0.2s after fist)
        res_transition = self.recognizer.process(HandTrackingResult(hands=[open_palm], hand_count=1), timestamp=1.4)
        self.assertEqual(len(res_transition.events), 1)
        self.assertEqual(res_transition.events[0].gesture_id, HandGestureID.HAND_FIST_OPEN)
        self.assertEqual(res_transition.events[0].label, "FIST -> OPEN PALM DETECTED")

        # Step 3: Next frame at t=1.5s (still open palm) -> NO duplicate event
        res_next = self.recognizer.process(HandTrackingResult(hands=[open_palm], hand_count=1), timestamp=1.5)
        self.assertEqual(len(res_next.events), 0)

    def test_fist_to_open_palm_timeout(self):
        """Transition taking longer than 0.8s does not fire event."""
        fist = make_fist()
        open_palm = make_open_palm()

        self.recognizer.process(HandTrackingResult(hands=[fist], hand_count=1), timestamp=1.0)
        # 1.2 seconds pass before opening palm
        res = self.recognizer.process(HandTrackingResult(hands=[open_palm], hand_count=1), timestamp=2.2)
        self.assertEqual(len(res.events), 0)

    def test_horizontal_swipe_right(self):
        """Verify fast horizontal rightward motion triggers SWIPE RIGHT."""
        hand1 = make_open_palm(cx=300, cy=400)
        hand2 = make_open_palm(cx=460, cy=405)  # dx = +160px, dy = +5px

        self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=1.0)
        res2 = self.recognizer.process(HandTrackingResult(hands=[hand2], hand_count=1), timestamp=1.2)

        self.assertEqual(len(res2.events), 1)
        self.assertEqual(res2.events[0].gesture_id, HandGestureID.HAND_SWIPE_RIGHT)
        self.assertEqual(res2.events[0].label, "SWIPE RIGHT")

    def test_horizontal_swipe_left(self):
        """Verify fast horizontal leftward motion triggers SWIPE LEFT."""
        hand1 = make_open_palm(cx=600, cy=400)
        hand2 = make_open_palm(cx=440, cy=395)  # dx = -160px, dy = -5px

        self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=1.0)
        res2 = self.recognizer.process(HandTrackingResult(hands=[hand2], hand_count=1), timestamp=1.2)

        self.assertEqual(len(res2.events), 1)
        self.assertEqual(res2.events[0].gesture_id, HandGestureID.HAND_SWIPE_LEFT)
        self.assertEqual(res2.events[0].label, "SWIPE LEFT")

    def test_swipe_vertical_dominant_rejected(self):
        """Diagonal movement where vertical motion dominates horizontal is rejected."""
        hand1 = make_open_palm(cx=400, cy=300)
        hand2 = make_open_palm(cx=540, cy=550)  # dx = 140px, dy = 250px (vertical dominant)

        self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=1.0)
        res2 = self.recognizer.process(HandTrackingResult(hands=[hand2], hand_count=1), timestamp=1.2)

        # Dominance check: |dx| > 1.5 * |dy| is false (140 is not > 1.5 * 250)
        self.assertEqual(len(res2.events), 0)

    def test_stationary_open_palm_mode_toggle(self):
        """
        Verify stationary open palm:
        - Shows progress climbing toward 1.0 (2.0s).
        - Triggers MODE TOGGLE READY at exactly 2.0s.
        - Does NOT re-trigger on subsequent frames while palm remains open.
        - Latch resets once hand leaves open palm state.
        """
        hand = make_open_palm(cx=600, cy=400)

        # Start hold at t=1.0
        res1 = self.recognizer.process(HandTrackingResult(hands=[hand], hand_count=1), timestamp=1.0)
        self.assertEqual(res1.hands["Right"].hold_progress, 0.0)
        self.assertEqual(len(res1.events), 0)

        # Midpoint at t=2.0 (1.0s held, 50% progress)
        res2 = self.recognizer.process(HandTrackingResult(hands=[hand], hand_count=1), timestamp=2.0)
        self.assertAlmostEqual(res2.hands["Right"].hold_progress, 0.5, places=1)
        self.assertEqual(len(res2.events), 0)

        # Completion at t=3.0 (2.0s held, 100% progress)
        res3 = self.recognizer.process(HandTrackingResult(hands=[hand], hand_count=1), timestamp=3.0)
        self.assertEqual(res3.hands["Right"].hold_progress, 1.0)
        self.assertEqual(len(res3.events), 1)
        self.assertEqual(res3.events[0].gesture_id, HandGestureID.HAND_MODE_TOGGLE)
        self.assertEqual(res3.events[0].label, "MODE TOGGLE READY")

        # Held past 2.0s at t=3.5 -> No duplicate trigger!
        res4 = self.recognizer.process(HandTrackingResult(hands=[hand], hand_count=1), timestamp=3.5)
        self.assertEqual(len(res4.events), 0)

        # Close into fist at t=4.0 -> Resets hold
        fist = make_fist(cx=600, cy=400)
        res5 = self.recognizer.process(HandTrackingResult(hands=[fist], hand_count=1), timestamp=4.0)
        self.assertEqual(res5.hands["Right"].hold_progress, 0.0)

    def test_stationary_open_palm_jitter_reset(self):
        """Moving hand more than palm_max_movement_px resets the stationary hold timer."""
        hand1 = make_open_palm(cx=600, cy=400)
        # Move 50px away (> 30px threshold)
        hand2 = make_open_palm(cx=650, cy=400)

        self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=1.0)
        res_mid = self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=2.0)
        self.assertAlmostEqual(res_mid.hands["Right"].hold_progress, 0.5, places=1)

        # Jitter frame resets progress
        res_jitter = self.recognizer.process(HandTrackingResult(hands=[hand2], hand_count=1), timestamp=2.1)
        self.assertEqual(res_jitter.hands["Right"].hold_progress, 0.0)

    def test_two_finger_vertical_movement(self):
        """Verify two-finger vertical motion triggers UP or DOWN event."""
        # Upward motion: y decreases (dy = -70px, dx = 0px)
        hand1 = make_two_finger_pointing(cx=600, cy=450)
        hand2 = make_two_finger_pointing(cx=600, cy=380)

        self.recognizer.process(HandTrackingResult(hands=[hand1], hand_count=1), timestamp=1.0)
        res_up = self.recognizer.process(HandTrackingResult(hands=[hand2], hand_count=1), timestamp=1.2)

        self.assertEqual(len(res_up.events), 1)
        self.assertEqual(res_up.events[0].gesture_id, HandGestureID.HAND_VERTICAL_TWO_FINGER)
        self.assertEqual(res_up.events[0].label, "TWO-FINGER UP")

        # Clear recognizer for downward test
        self.recognizer.reset()

        # Downward motion: y increases (dy = +70px, dx = 0px)
        hand3 = make_two_finger_pointing(cx=600, cy=350)
        hand4 = make_two_finger_pointing(cx=600, cy=420)

        self.recognizer.process(HandTrackingResult(hands=[hand3], hand_count=1), timestamp=2.0)
        res_down = self.recognizer.process(HandTrackingResult(hands=[hand4], hand_count=1), timestamp=2.2)

        self.assertEqual(len(res_down.events), 1)
        self.assertEqual(res_down.events[0].gesture_id, HandGestureID.HAND_VERTICAL_TWO_FINGER)
        self.assertEqual(res_down.events[0].label, "TWO-FINGER DOWN")

    def test_multi_hand_independence(self):
        """Verify Left and Right hands are tracked simultaneously without state collision."""
        left_hand = make_index_pointing(tip_x=200, tip_y=300, handedness="Left")
        right_hand = make_pinch_data("index", distance_px=20.0, cx=800, cy=500, handedness="Right")

        tracking_res = HandTrackingResult(hands=[left_hand, right_hand], hand_count=2)
        res = self.recognizer.process(tracking_res, timestamp=1.0)

        self.assertIn("Left", res.hands)
        self.assertIn("Right", res.hands)

        # Left hand is pointing
        self.assertEqual(res.hands["Left"].active_gesture, HandGestureID.HAND_INDEX_MOVE)
        self.assertEqual(res.hands["Left"].active_gesture_label, "INDEX MOVE")

        # Right hand is pinching
        self.assertEqual(res.hands["Right"].active_gesture, HandGestureID.HAND_PINCH_INDEX)
        self.assertEqual(res.hands["Right"].active_gesture_label, "INDEX PINCH")


if __name__ == "__main__":
    unittest.main()

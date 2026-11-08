"""
Action Mapper
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Translates low-level gesture states and transitions into semantic computer control actions:
- Index Movement -> MOVE_CURSOR
- Thumb + Index Pinch Down -> LEFT_CLICK
- Thumb + Middle Pinch Down -> RIGHT_CLICK
- Fist to Open Palm -> PLAY_PAUSE
- Swipe Right -> NEXT
- Swipe Left -> PREVIOUS
- 2-Second Stationary Open Palm -> TOGGLE_CONTROL_MODE
- Two-Finger Vertical Up / Down -> ZOOM_IN / ZOOM_OUT
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from gestures.hand_gestures import GestureRecognitionResult, HandGestureID, HandGestureState


class SemanticAction(Enum):
    """Semantic action types supported by the control layer."""
    NONE = auto()
    MOVE_CURSOR = auto()
    LEFT_CLICK = auto()
    RIGHT_CLICK = auto()
    PLAY_PAUSE = auto()
    NEXT = auto()
    PREVIOUS = auto()
    TOGGLE_CONTROL_MODE = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()


@dataclass
class ActionRequest:
    """Encapsulates an OS action to be dispatched by the CommandEngine."""
    action: SemanticAction
    cursor_pos: Optional[Tuple[int, int]] = None  # (cam_x, cam_y)
    handedness: str = "Unknown"
    details: Dict[str, Any] = field(default_factory=dict)


class ActionMapper:
    """
    Stateful translator that detects edge triggers (e.g. pinch contact down)
    and maps continuous gestures and discrete events into semantic action requests.
    """

    def __init__(self):
        # Per-hand previous pinch states to detect exact NO PINCH -> PINCH contact transitions
        self._prev_pinching_index: Dict[str, bool] = {}
        self._prev_pinching_middle: Dict[str, bool] = {}

    def reset(self) -> None:
        """Reset internal transition history."""
        self._prev_pinching_index.clear()
        self._prev_pinching_middle.clear()

    def map_gestures(
        self, gesture_result: Optional[GestureRecognitionResult]
    ) -> List[ActionRequest]:
        """
        Evaluate gesture results and emit corresponding semantic action requests.

        Args:
            gesture_result: Output from HandGestureRecognizer.process().

        Returns:
            List of ActionRequest objects to execute.
        """
        if not gesture_result:
            return []

        actions: List[ActionRequest] = []

        # 1. Process Discrete Gesture Events from the current frame
        for ev in gesture_result.events:
            # Control 7: Open Palm 2s Hold -> Mode Toggle
            if ev.gesture_id == HandGestureID.HAND_MODE_TOGGLE:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.TOGGLE_CONTROL_MODE,
                        handedness=ev.handedness,
                        details={"trigger": "HAND_MODE_TOGGLE"},
                    )
                )

            # Control 4: Fist -> Open Palm -> Play/Pause
            elif ev.gesture_id == HandGestureID.HAND_FIST_OPEN:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.PLAY_PAUSE,
                        handedness=ev.handedness,
                        details=ev.details,
                    )
                )

            # Control 5: Swipe Right -> Next
            elif ev.gesture_id == HandGestureID.HAND_SWIPE_RIGHT:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.NEXT,
                        handedness=ev.handedness,
                        details=ev.details,
                    )
                )

            # Control 6: Swipe Left -> Previous
            elif ev.gesture_id == HandGestureID.HAND_SWIPE_LEFT:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.PREVIOUS,
                        handedness=ev.handedness,
                        details=ev.details,
                    )
                )

            # Control 8: Two-Finger Vertical Movement -> Zoom
            elif ev.gesture_id == HandGestureID.HAND_VERTICAL_TWO_FINGER:
                direction = ev.details.get("direction", "UP").upper()
                zoom_action = SemanticAction.ZOOM_IN if direction == "UP" else SemanticAction.ZOOM_OUT
                actions.append(
                    ActionRequest(
                        action=zoom_action,
                        handedness=ev.handedness,
                        details=ev.details,
                    )
                )

        if not gesture_result.hands:
            self.reset()
            return actions

        # Clean up pinch tracking for hands that are no longer detected
        active_hands = set(gesture_result.hands.keys())
        for h in list(self._prev_pinching_index.keys()):
            if h not in active_hands:
                del self._prev_pinching_index[h]
        for h in list(self._prev_pinching_middle.keys()):
            if h not in active_hands:
                del self._prev_pinching_middle[h]

        # 2. Evaluate Click Gestures for EVERY detected hand independently
        for h_name, hand_state in gesture_result.hands.items():
            # --- A. Control 2: Thumb + Index Pinch Down -> Left Click ---
            is_index_pinch = hand_state.is_pinching_index
            was_index_pinch = self._prev_pinching_index.get(h_name, False)

            if not was_index_pinch and is_index_pinch:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.LEFT_CLICK,
                        cursor_pos=hand_state.smoothed_index_pos,
                        handedness=h_name,
                        details={"event": "INDEX_PINCH_DOWN", "hand": h_name},
                    )
                )
            self._prev_pinching_index[h_name] = is_index_pinch

            # --- B. Control 3: Thumb + Middle Pinch Down -> Right Click ---
            is_middle_pinch = hand_state.is_pinching_middle
            was_middle_pinch = self._prev_pinching_middle.get(h_name, False)

            if not was_middle_pinch and is_middle_pinch:
                actions.append(
                    ActionRequest(
                        action=SemanticAction.RIGHT_CLICK,
                        cursor_pos=hand_state.smoothed_index_pos,
                        handedness=h_name,
                        details={"event": "MIDDLE_PINCH_DOWN", "hand": h_name},
                    )
                )
            self._prev_pinching_middle[h_name] = is_middle_pinch

        # 3. Control 1: Index Movement -> Real Cursor (primary interacting hand)
        target_hand: Optional[HandGestureState] = None
        if "Right" in gesture_result.hands and gesture_result.hands["Right"].is_index_pointing:
            target_hand = gesture_result.hands["Right"]
        elif "Left" in gesture_result.hands and gesture_result.hands["Left"].is_index_pointing:
            target_hand = gesture_result.hands["Left"]
        elif "Right" in gesture_result.hands:
            target_hand = gesture_result.hands["Right"]
        elif "Left" in gesture_result.hands:
            target_hand = gesture_result.hands["Left"]
        elif gesture_result.hands:
            target_hand = next(iter(gesture_result.hands.values()))

        if target_hand:
            h_name = target_hand.handedness
            is_index_pinch = target_hand.is_pinching_index
            is_middle_pinch = target_hand.is_pinching_middle
            if (
                target_hand.is_index_pointing
                and not is_index_pinch
                and not is_middle_pinch
                and target_hand.smoothed_index_pos
            ):
                actions.append(
                    ActionRequest(
                        action=SemanticAction.MOVE_CURSOR,
                        cursor_pos=target_hand.smoothed_index_pos,
                        handedness=h_name,
                    )
                )

        return actions

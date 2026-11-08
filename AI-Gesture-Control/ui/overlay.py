"""
UI Overlay & HUD Rendering
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Renders visual status badges, FPS counter, hand skeletal landmarks/connections,
handedness labels, gesture recognition cards, smoothed cursor crosshairs,
mode toggle progress bars, Control Mode (OFF/ON) panel, and user guidance over camera frames.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from vision.hand_tracker import HandTrackingResult, HandLandmarkData
from vision.face_tracker import FaceTrackingResult, HeadPoseData
from gestures.hand_gestures import GestureRecognitionResult, HandGestureState, HandGestureID
from gestures.head_gestures import HeadGestureState as HeadStateData, HeadGestureID as HeadGestureIDEnum
from control.command_engine import CommandEngineStatus

# Standard 21 MediaPipe hand skeletal connections
HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 5), (9, 13), (13, 17), (5, 9), (0, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index finger
    (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
]

# Indices corresponding to finger tips
FINGERTIP_INDICES = {4, 8, 12, 16, 20}


def draw_translucent_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """
    Draw a translucent filled rectangle using fast in-place ROI blending.
    Avoids copying or blending the entire frame canvas.
    """
    fh, fw = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + w)
    y2 = min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    cv2.addWeighted(
        np.full_like(sub, color, dtype=np.uint8),
        alpha,
        sub,
        1.0 - alpha,
        0,
        sub,
    )


def draw_hand_landmarks(
    frame: np.ndarray,
    result: HandTrackingResult,
    gesture_result: Optional[GestureRecognitionResult] = None,
) -> np.ndarray:
    """
    Render hand landmarks, skeleton connections, pinch highlights, and handedness badges onto the frame.

    Args:
        frame: Mirrored camera frame (BGR format).
        result: HandTrackingResult containing detected hands and 21 landmarks each.
        gesture_result: Optional GestureRecognitionResult to emphasize active pinches or states.

    Returns:
        Frame with rendered skeletal connections, landmark joints, and handedness tags.
    """
    if not result or result.hand_count == 0:
        return frame

    for hand in result.hands:
        points = hand.landmarks_px
        if len(points) < 21:
            continue

        is_left = hand.handedness.lower() == "left"
        # Vibrant colors: Cyan/Azure for Right, Mint/Green for Left
        if is_left:
            conn_color = (130, 230, 120)   # Mint green (BGR)
            joint_color = (80, 180, 80)
            tag_color = (40, 140, 40)
        else:
            conn_color = (245, 180, 50)    # Sky blue / Cyan (BGR)
            joint_color = (200, 130, 20)
            tag_color = (180, 100, 10)

        # 1. Draw skeletal bones (connections)
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(frame, pt1, pt2, conn_color, 2, cv2.LINE_AA)

        # 2. Draw landmark joints
        for idx, pt in enumerate(points):
            if idx in FINGERTIP_INDICES:
                # Highlighted outer circle for fingertips
                cv2.circle(frame, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pt, 4, conn_color, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, pt, 4, joint_color, -1, cv2.LINE_AA)

        # 3. Highlight pinch contact if active
        if gesture_result and hand.handedness in gesture_result.hands:
            g_state = gesture_result.hands[hand.handedness]
            if g_state.is_pinching_index:
                # Draw glowing connection between Thumb (4) and Index (8)
                cv2.line(frame, points[4], points[8], (0, 240, 255), 4, cv2.LINE_AA)
                cv2.circle(frame, points[4], 8, (0, 240, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, points[8], 8, (0, 240, 255), 2, cv2.LINE_AA)
            elif g_state.is_pinching_middle:
                # Draw glowing connection between Thumb (4) and Middle (12)
                cv2.line(frame, points[4], points[12], (255, 100, 255), 4, cv2.LINE_AA)
                cv2.circle(frame, points[4], 8, (255, 100, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, points[12], 8, (255, 100, 255), 2, cv2.LINE_AA)

        # 4. Floating handedness badge near the wrist (landmark 0)
        wrist_pt = points[0]
        label = f"{hand.handedness} ({int(hand.score * 100)}%)" if hand.score > 0 else hand.handedness

        bx = max(10, min(wrist_pt[0] - 40, frame.shape[1] - 130))
        by = min(frame.shape[0] - 45, max(70, wrist_pt[1] + 28))

        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (bx - 4, by - lh - 4), (bx + lw + 4, by + 4), tag_color, -1)
        cv2.rectangle(frame, (bx - 4, by - lh - 4), (bx + lw + 4, by + 4), (255, 255, 255), 1)
        cv2.putText(
            frame,
            label,
            (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_gesture_hud(
    frame: np.ndarray,
    gesture_result: Optional[GestureRecognitionResult],
) -> np.ndarray:
    """
    Render Stage 4A Gesture Recognition HUD cards and pointing indicators.

    Args:
        frame: Mirrored camera frame (BGR format).
        gesture_result: GestureRecognitionResult containing per-hand states and events.

    Returns:
        Frame with gesture overlays rendered.
    """
    if not gesture_result or not gesture_result.hands:
        return frame

    h, w = frame.shape[:2]
    now = time.time()
    card_w = min(310, int(w * 0.44))
    card_h = 136

    for handedness, state in gesture_result.hands.items():
        is_left = handedness.lower() == "left"

        # Position Left hand card on left, Right hand card on right.
        # Positioned at y = 176 (below Head HUD y=68..164 and Master Control y=68..144) to prevent overlap.
        if is_left:
            card_x = 16
            card_y = 176
            header_color = (130, 230, 120)  # Mint Green
        else:
            card_x = w - card_w - 16
            card_y = 176
            header_color = (245, 180, 50)   # Sky Blue

        # Draw translucent card background
        draw_translucent_rect(frame, card_x, card_y, card_w, card_h, (22, 24, 30), 0.78)

        # Card border and header accent
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (70, 75, 85),
            1,
        )
        cv2.line(
            frame,
            (card_x, card_y + 26),
            (card_x + card_w, card_y + 26),
            header_color,
            2,
        )

        # 1. Header: Handedness Badge
        cv2.putText(
            frame,
            f"HAND: {handedness.upper()}",
            (card_x + 10, card_y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            header_color,
            2,
            cv2.LINE_AA,
        )

        # 2. Active Gesture
        active_str = state.active_gesture_label
        if state.active_gesture == HandGestureID.HAND_PINCH_INDEX:
            act_color = (0, 240, 255)     # Yellow / Gold
        elif state.active_gesture == HandGestureID.HAND_PINCH_MIDDLE:
            act_color = (255, 100, 255)   # Magenta
        elif state.active_gesture == HandGestureID.HAND_MODE_TOGGLE:
            act_color = (100, 255, 150)   # Bright green
        elif state.active_gesture == HandGestureID.HAND_INDEX_MOVE:
            act_color = (255, 220, 80)    # Cyan
        elif state.active_gesture == HandGestureID.HAND_VERTICAL_TWO_FINGER:
            act_color = (255, 180, 100)
        else:
            act_color = (180, 180, 180)   # Neutral gray

        cv2.putText(
            frame,
            f"Gesture: {active_str}",
            (card_x + 10, card_y + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            act_color,
            1,
            cv2.LINE_AA,
        )

        # 3. Smoothed Index Coordinates
        if state.smoothed_index_pos:
            sx, sy = state.smoothed_index_pos
            coord_str = f"Index: X={sx:4d}  Y={sy:4d}"
            coord_color = (240, 240, 240) if state.is_index_pointing else (140, 140, 140)
        else:
            coord_str = "Index: Not Detected"
            coord_color = (140, 140, 140)

        cv2.putText(
            frame,
            coord_str,
            (card_x + 10, card_y + 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            coord_color,
            1,
            cv2.LINE_AA,
        )

        # 4. Mode Toggle Progress Bar (2.0s hold)
        bar_x = card_x + 10
        bar_y = card_y + 78
        bar_w = card_w - 20
        bar_h = 10

        # Background track
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 48, 55), -1)

        # Fill track
        fill_w = int(bar_w * max(0.0, min(1.0, state.hold_progress)))
        if fill_w > 0:
            fill_col = (80, 220, 100) if state.hold_progress >= 1.0 else (0, 215, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 95, 105), 1)

        # Progress text
        prog_pct = int(state.hold_progress * 100)
        prog_text = f"Hold: {state.hold_duration:.1f}s / 2.0s ({prog_pct}%)"
        cv2.putText(
            frame,
            prog_text,
            (bar_x, bar_y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )

        # 5. Last Discrete Event Badge
        last_ev = state.last_event
        if last_ev and (now - last_ev.timestamp) < 2.0:
            ev_age = now - last_ev.timestamp
            badge_bg = (30, 80, 30) if "OPEN PALM" in last_ev.label or "MODE" in last_ev.label else (80, 50, 20)
            badge_text = f"EVENT: {last_ev.label} ({ev_age:.1f}s ago)"
            bw = card_w - 20
            by = card_y + 110
            cv2.rectangle(frame, (card_x + 10, by), (card_x + 10 + bw, by + 18), badge_bg, -1)
            cv2.rectangle(frame, (card_x + 10, by), (card_x + 10 + bw, by + 18), (120, 200, 120), 1)
            cv2.putText(
                frame,
                badge_text,
                (card_x + 14, by + 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            # Subtle placeholder
            cv2.putText(
                frame,
                "Event: Idle",
                (card_x + 12, card_y + 123),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (120, 120, 130),
                1,
                cv2.LINE_AA,
            )

        # 6. Draw smoothed pointer crosshair if index is pointing
        if state.is_index_pointing and state.smoothed_index_pos:
            cx, cy = state.smoothed_index_pos
            # Crosshair and circular halo
            cross_color = (0, 240, 255) if is_left else (255, 180, 50)
            cv2.circle(frame, (cx, cy), 14, cross_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.line(frame, (cx - 20, cy), (cx - 6, cy), cross_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx + 6, cy), (cx + 20, cy), cross_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - 20), (cx, cy - 6), cross_color, 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy + 6), (cx, cy + 20), cross_color, 1, cv2.LINE_AA)

            tag = f"{handedness[0]}: ({cx}, {cy})"
            cv2.putText(
                frame,
                tag,
                (cx + 18, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                cross_color,
                1,
                cv2.LINE_AA,
            )

    return frame


def draw_control_hud(
    frame: np.ndarray,
    control_status: Optional[CommandEngineStatus],
) -> np.ndarray:
    """
    Render Stage 4B Part 1 Control Mode status bar, action badges, and safety warnings.

    Displays:
    - CONTROL MODE: OFF / ON
    - Cursor Control: ACTIVE / INACTIVE
    - Last Action: NONE / LEFT CLICK
    - Warning banner when Control Mode is ON: "GESTURE CONTROL ACTIVE"
    - macOS Accessibility notice if permission is missing.
    """
    if not control_status:
        return frame

    h, w = frame.shape[:2]
    now = time.time()

    # 1. Master Control Panel in Top Center
    panel_w = 390
    panel_h = 76
    panel_x = (w - panel_w) // 2
    panel_y = 68

    draw_translucent_rect(frame, panel_x, panel_y, panel_w, panel_h, (18, 20, 26), 0.82)

    # Accent outline
    mode_color = (80, 255, 100) if control_status.control_mode else (80, 100, 240)
    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        mode_color if control_status.control_mode else (60, 65, 75),
        2 if control_status.control_mode else 1,
    )

    # Row 1: CONTROL MODE: OFF / ON
    mode_str = "CONTROL MODE: ON" if control_status.control_mode else "CONTROL MODE: OFF"
    cv2.putText(
        frame,
        mode_str,
        (panel_x + 14, panel_y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    # Row 2: Cursor Control: ACTIVE / INACTIVE
    cursor_str = "Cursor: ACTIVE" if control_status.is_cursor_active else "Cursor: INACTIVE"
    cursor_color = (0, 240, 255) if control_status.is_cursor_active else (150, 150, 160)
    cv2.putText(
        frame,
        cursor_str,
        (panel_x + 14, panel_y + 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        cursor_color,
        1,
        cv2.LINE_AA,
    )

    # Row 3: Last Action
    action_age = now - control_status.last_action_time
    hand_tag = f" [{control_status.last_action_hand} Hand]" if getattr(control_status, "last_action_hand", "") and control_status.last_action_name != "NONE" else ""
    action_str = f"Last Action: {control_status.last_action_name}{hand_tag}"
    action_color = (0, 240, 255) if action_age < 1.5 and control_status.last_action_name != "NONE" else (150, 150, 160)
    cv2.putText(
        frame,
        action_str,
        (panel_x + 14, panel_y + 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        action_color,
        1,
        cv2.LINE_AA,
    )

    # 2. Safety Warning Banner when CONTROL MODE is ON
    if control_status.control_mode:
        warn_text = "GESTURE CONTROL ACTIVE  |  Press ESC to Disable"
        (tw, th), _ = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx = (w - tw) // 2
        by = panel_y + panel_h + 24

        cv2.rectangle(frame, (bx - 12, by - th - 6), (bx + tw + 12, by + 6), (20, 30, 80), -1)
        cv2.rectangle(frame, (bx - 12, by - th - 6), (bx + tw + 12, by + 6), (50, 80, 255), 1)
        cv2.putText(
            frame,
            warn_text,
            (bx, by),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 220, 255),
            1,
            cv2.LINE_AA,
        )

    # 3. Accessibility Warning Banner if permission is missing
    if not control_status.has_accessibility:
        ax_text = "WARN: macOS Accessibility Required (System Settings -> Privacy & Security -> Accessibility)"
        (aw, ah), _ = cv2.getTextSize(ax_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        abx = (w - aw) // 2
        aby = h - 44

        cv2.rectangle(frame, (abx - 8, aby - ah - 4), (abx + aw + 8, aby + 4), (20, 40, 90), -1)
        cv2.rectangle(frame, (abx - 8, aby - ah - 4), (abx + aw + 8, aby + 4), (0, 180, 255), 1)
        cv2.putText(
            frame,
            ax_text,
            (abx, aby),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    return frame


def draw_camera_hud(
    frame: np.ndarray,
    fps: float,
    status: str = "Active",
    resolution: Optional[Tuple[int, int]] = None,
    hand_count: int = 0,
    hands_info: Optional[str] = None,
    title_text: str = "AI Gesture Control - Stage 4B Part 2",
    control_mode: bool = False,
) -> np.ndarray:
    """
    Render top and bottom HUD bars onto the camera frame.

    Args:
        frame: Mirrored camera frame (BGR format).
        fps: Current measured frames per second.
        status: Status string.
        resolution: (width, height) capture resolution tuple.
        hand_count: Number of hands currently detected.
        hands_info: Formatted string of detected hands.
        title_text: Title string for the top bar.
        control_mode: Current state of real computer control.

    Returns:
        Frame with HUD elements rendered.
    """
    h, w = frame.shape[:2]

    # 1. Top HUD Header Bar
    bar_height = 56
    draw_translucent_rect(frame, 0, 0, w, bar_height, (20, 20, 25), 0.72)

    # 2. Bottom Status Bar
    bottom_bar_height = 36
    draw_translucent_rect(frame, 0, h - bottom_bar_height, w, bottom_bar_height, (20, 20, 25), 0.72)

    # Accent divider lines
    cv2.line(frame, (0, bar_height), (w, bar_height), (70, 70, 80), 1)
    cv2.line(frame, (0, h - bottom_bar_height), (w, h - bottom_bar_height), (70, 70, 80), 1)

    # Title
    cv2.putText(
        frame,
        title_text,
        (18, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Hands Counter Badge in Header
    if hand_count > 0:
        hand_badge_text = f"Hands: {hand_count}" + (f" ({hands_info})" if hands_info else "")
        badge_color = (100, 230, 100)
    else:
        hand_badge_text = "Hands: 0"
        badge_color = (180, 180, 180)

    cv2.putText(
        frame,
        hand_badge_text,
        (max(440, int(w * 0.44)), 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        badge_color,
        2,
        cv2.LINE_AA,
    )

    # FPS Color Coding: Green >= 24, Yellow >= 15, Red < 15
    if fps >= 24.0:
        fps_color = (80, 220, 100)   # Green
    elif fps >= 15.0:
        fps_color = (50, 200, 255)   # Amber/Yellow
    else:
        fps_color = (60, 60, 230)    # Red

    fps_text = f"FPS: {fps:.1f}"
    fps_x = max(w - 180, 320)
    cv2.putText(
        frame,
        fps_text,
        (fps_x, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        fps_color,
        2,
        cv2.LINE_AA,
    )

    # Camera Status & Resolution (Left side of bottom bar)
    res_str = f" | {resolution[0]}x{resolution[1]}" if resolution else ""
    status_text = f"Status: {status}{res_str}"
    cv2.putText(
        frame,
        status_text,
        (18, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Safety Indicator (Center of bottom bar)
    if control_mode:
        safety_text = "[STAGE 4B: MOUSE CONTROL ACTIVE | ESC: EMERGENCY STOP]"
        safety_color = (80, 255, 100)
    else:
        safety_text = "[STAGE 4B: CONTROL MODE OFF | 2s Open Palm Hold to Enable]"
        safety_color = (100, 200, 255)

    cv2.putText(
        frame,
        safety_text,
        (max(240, int(w * 0.28)), h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        safety_color,
        1,
        cv2.LINE_AA,
    )

    # Quit prompt (Right side of bottom bar)
    quit_text = "ESC: Off | Q: Quit"
    quit_x = max(w - 180, 300)
    cv2.putText(
        frame,
        quit_text,
        (quit_x, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (0, 215, 255),
        1,
        cv2.LINE_AA,
    )

    return frame


def draw_debug_hands_hud(
    frame: np.ndarray,
    hand_result: Optional[HandTrackingResult],
    gesture_result: Optional[GestureRecognitionResult] = None,
) -> np.ndarray:
    """
    Render debug telemetry displaying detected hands count, handedness, confidence,
    index tip pixel coordinates, and pinch state for every detected hand.
    Positioned in the lower-left area above the bottom bar.
    """
    if not hand_result:
        return frame

    h, w = frame.shape[:2]
    hand_count = hand_result.hand_count
    box_w = 340
    box_h = 24 + max(1, hand_count) * 38
    box_x = 16
    box_y = h - 48 - box_h

    # Translucent card background
    draw_translucent_rect(frame, box_x, box_y, box_w, box_h, (18, 20, 26), 0.80)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (70, 75, 85), 1)

    # Header
    cv2.putText(
        frame,
        f"DEBUG: Detected hands: {hand_count}",
        (box_x + 8, box_y + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 240, 255),
        1,
        cv2.LINE_AA,
    )

    if hand_count == 0:
        cv2.putText(
            frame,
            "Hand: None",
            (box_x + 8, box_y + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
    else:
        for idx, hand in enumerate(hand_result.hands):
            entry_y = box_y + 20 + idx * 38
            cv2.line(frame, (box_x, entry_y), (box_x + box_w, entry_y), (45, 48, 55), 1)

            is_pinch = False
            if gesture_result and hand.handedness in gesture_result.hands:
                is_pinch = gesture_result.hands[hand.handedness].is_pinching_index

            tip_pos = hand.landmarks_px[8] if len(hand.landmarks_px) > 8 else (0, 0)
            score_pct = int(hand.score * 100)

            hand_col = (130, 230, 120) if hand.handedness.lower() == "left" else (245, 180, 50)
            pinch_str = "TRUE" if is_pinch else "FALSE"
            pinch_col = (0, 240, 255) if is_pinch else (160, 160, 160)

            cv2.putText(
                frame,
                f"Hand {idx}: {hand.handedness} ({score_pct}%)",
                (box_x + 8, entry_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                hand_col,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Tip: ({tip_pos[0]:4d}, {tip_pos[1]:4d}) | Pinch: {pinch_str}",
                (box_x + 8, entry_y + 31),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                pinch_col,
                1,
                cv2.LINE_AA,
            )

    return frame


# Key landmark indices for clean facial subset rendering
_FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109, 10
]
_LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
_RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
_NOSE_BRIDGE_INDICES = [168, 6, 197, 195, 5, 4, 1, 2]
_LIPS_OUTER_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61]


def draw_face_landmarks(
    frame: np.ndarray,
    face_result: Optional[FaceTrackingResult],
) -> np.ndarray:
    """
    Render a clean, lightweight facial structure subset (eyes, nose, lips, jawline)
    and project 3D head pose orientation axes from the nose tip.
    """
    if not face_result or not face_result.face_detected or face_result.face_data is None:
        return frame

    pts = face_result.face_data.landmarks_px
    if len(pts) < 468:
        return frame

    h, w = frame.shape[:2]

    # 1. Draw Facial Contours (Subtle cyan/blue lines)
    contour_color = (180, 160, 100)  # Subtle cyan-slate in BGR
    for idx_list in (_LEFT_EYE_INDICES, _RIGHT_EYE_INDICES, _NOSE_BRIDGE_INDICES, _LIPS_OUTER_INDICES):
        pts_arr = np.array([pts[i] for i in idx_list], dtype=np.int32)
        cv2.polylines(frame, [pts_arr], False, contour_color, 1, cv2.LINE_AA)

    # Draw face oval outline
    oval_pts = np.array([pts[i] for i in _FACE_OVAL_INDICES], dtype=np.int32)
    cv2.polylines(frame, [oval_pts], False, (120, 110, 80), 1, cv2.LINE_AA)

    # 2. Draw 3D Orientation Axes from the Nose Tip
    pose = face_result.head_pose
    if pose is not None and pose.rvec is not None and pose.tvec is not None:
        axis_len = 70.0
        # 3D points: Origin (nose tip), X (pitch/lateral), Y (yaw/vertical), Z (roll/optical)
        axis_3d = np.array([
            (0.0, 0.0, 0.0),
            (axis_len, 0.0, 0.0),
            (0.0, axis_len, 0.0),
            (0.0, 0.0, -axis_len),
        ], dtype=np.float64)

        focal_length = float(w)
        center = (float(w) / 2.0, float(h) / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        try:
            pts_2d, _ = cv2.projectPoints(axis_3d, pose.rvec, pose.tvec, camera_matrix, dist_coeffs)
            p_origin = tuple(map(int, pts_2d[0].ravel()))
            p_x = tuple(map(int, pts_2d[1].ravel()))
            p_y = tuple(map(int, pts_2d[2].ravel()))
            p_z = tuple(map(int, pts_2d[3].ravel()))

            # Red = X axis (lateral / pitch)
            cv2.line(frame, p_origin, p_x, (50, 50, 240), 2, cv2.LINE_AA)
            # Green = Y axis (vertical / yaw)
            cv2.line(frame, p_origin, p_y, (50, 230, 50), 2, cv2.LINE_AA)
            # Blue = Z axis (optical / pointing out)
            cv2.line(frame, p_origin, p_z, (240, 120, 50), 2, cv2.LINE_AA)
            cv2.circle(frame, p_origin, 3, (0, 240, 255), -1, cv2.LINE_AA)
        except Exception:
            pass

    return frame


def draw_head_hud(
    frame: np.ndarray,
    head_state: Optional[HeadStateData],
    face_result: Optional[FaceTrackingResult],
) -> np.ndarray:
    """
    Render Stage 5 Head Tracking HUD card displaying Face Detection, Yaw, Pitch, Roll,
    and discrete head gesture classification state. Positioned in the upper right.
    """
    h, w = frame.shape[:2]
    box_w = 320
    box_h = 114
    box_x = w - box_w - 18
    box_y = 68

    # Translucent card background
    draw_translucent_rect(frame, box_x, box_y, box_w, box_h, (18, 20, 26), 0.82)

    face_detected = face_result.face_detected if face_result else False
    border_color = (80, 200, 100) if face_detected else (60, 65, 75)
    cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), border_color, 1)

    # Header
    cv2.putText(
        frame,
        "HEAD TRACKING (STAGE 5)",
        (box_x + 12, box_y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (0, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 1: Face Detection Status
    if face_detected:
        face_str = "FACE: DETECTED"
        face_col = (80, 240, 100)
    else:
        face_str = "FACE: NOT DETECTED"
        face_col = (100, 100, 240)

    cv2.putText(
        frame,
        face_str,
        (box_x + box_w - 138, box_y + 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        face_col,
        1,
        cv2.LINE_AA,
    )

    cv2.line(frame, (box_x + 10, box_y + 26), (box_x + box_w - 10, box_y + 26), (50, 55, 65), 1)

    # Row 2: Pose Angles (Yaw, Pitch, Roll in degrees)
    if head_state and face_detected:
        yaw = head_state.yaw
        pitch = head_state.pitch
        roll = head_state.roll
        angles_str = f"YAW: {yaw:+5.1f}°  |  PITCH: {pitch:+5.1f}°  |  ROLL: {roll:+5.1f}°"
    else:
        angles_str = "YAW:  --.-°   |  PITCH:  --.-°   |  ROLL:  --.-°"

    cv2.putText(
        frame,
        angles_str,
        (box_x + 12, box_y + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (200, 205, 215),
        1,
        cv2.LINE_AA,
    )

    # Row 3: Relative Yaw Delta from Calibrated Baseline
    if head_state and face_detected:
        dy = head_state.delta_yaw
        rel_str = f"dYAW: {dy:+5.1f}° (Left: < -4.5°, Right: > +7.0°)"
    else:
        rel_str = "dYAW:  --.-°"

    cv2.putText(
        frame,
        rel_str,
        (box_x + 12, box_y + 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    # Row 4: Head State (NEUTRAL, HEAD LEFT, HEAD RIGHT)
    state_str = head_state.current_state if head_state else "NO FACE"
    if state_str == "NEUTRAL":
        state_col = (180, 180, 180)
    elif state_str in ("NO FACE", "UNKNOWN"):
        state_col = (120, 120, 130)
    else:
        state_col = (0, 240, 255)  # Bright cyan highlight for active gesture

    cv2.putText(
        frame,
        f"HEAD STATE: {state_str}",
        (box_x + 12, box_y + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        state_col,
        1 if state_str == "NEUTRAL" else 2,
        cv2.LINE_AA,
    )

    # Row 5: Stage 5 Safety Note
    cv2.putText(
        frame,
        "[Foundation Only - Zero OS Actions]",
        (box_x + 12, box_y + 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.33,
        (130, 130, 140),
        1,
        cv2.LINE_AA,
    )

    return frame

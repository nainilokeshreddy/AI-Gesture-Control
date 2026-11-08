"""
Application Entry Point - Stage 4B Part 2: Complete Hand Controls (Controls 1-8)
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Coordinates camera lifecycle, real-time MediaPipe HandLandmarker tracking,
hand gesture recognition (8 gestures), Control Engine (Cursor, Clicks, Play/Pause, Swipes, Zoom),
rich UI HUD overlays, emergency safety controls, and graceful application termination.

SAFETY CONTROLS:
- CONTROL MODE defaults to OFF on startup.
- ESC: Emergency disable (instantly switches control mode OFF and stops all computer control).
- Q: Exits the application cleanly.
"""

import sys
import time
from typing import Optional
import cv2

from config.settings import AppConfig, DEFAULT_CONFIG
from vision.camera import Camera, CameraInitError, CameraError
from vision.hand_tracker import HandTracker, HandTrackerError, ModelNotFoundError
from vision.face_tracker import FaceTracker, FaceTrackerError
from gestures.hand_gestures import HandGestureRecognizer
from gestures.head_gestures import HeadGestureRecognizer
from control.command_engine import CommandEngine
from utils.fps import FPSCounter
from utils.logger import setup_logger
from ui.overlay import (
    draw_camera_hud,
    draw_hand_landmarks,
    draw_gesture_hud,
    draw_control_hud,
    draw_debug_hands_hud,
    draw_face_landmarks,
    draw_head_hud,
)


def run(config: Optional[AppConfig] = None) -> int:
    """
    Execute the Multimodal Real Computer Control application (Hands + Head Tracking Foundation).

    Args:
        config: Optional AppConfig instance (defaults to DEFAULT_CONFIG).

    Returns:
        Exit code (0 for success, 1 for error).
    """
    logger = setup_logger("App")
    app_config = config if config is not None else DEFAULT_CONFIG
    window_title = app_config.camera.window_title

    logger.info("Initializing Stage 5 - Head Tracking Foundation (Left / Right) & Hand Controls...")
    camera = Camera(app_config.camera)
    hand_tracker = HandTracker(app_config.vision, is_mirrored=app_config.camera.mirror)
    face_tracker = FaceTracker(app_config.vision, smoothing_config=app_config.smoothing)
    gesture_recognizer = HandGestureRecognizer(
        thresholds=app_config.thresholds,
        cooldowns=app_config.cooldown,
        smoothing=app_config.smoothing,
    )
    head_recognizer = HeadGestureRecognizer(
        thresholds=app_config.thresholds,
        cooldown=app_config.cooldown,
    )
    command_engine = CommandEngine(
        control_config=app_config.control,
        smoothing_config=app_config.smoothing,
    )
    fps_counter = FPSCounter()

    try:
        # Start camera hardware stream
        camera.start()

        # Start MediaPipe local tracking pipelines
        hand_tracker.start()
        face_tracker.start()

        logger.info(f"Displaying stream in window: '{window_title}'")
        logger.info("Control Mode defaults to OFF. Hold Open Palm for 2s to enable.")
        logger.info("Press ESC for Emergency Disable (turns control mode OFF).")
        logger.info("Press 'Q' to quit.")

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        consecutive_read_failures = 0
        max_allowed_failures = 30

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_allowed_failures:
                    logger.error(
                        f"Failed to read {max_allowed_failures} consecutive frames. Exiting capture loop."
                    )
                    break
                cv2.waitKey(10)
                continue

            consecutive_read_failures = 0

            # Compute synchronized timestamp and RGB frame once for both MediaPipe pipelines
            current_ms = int(time.time() * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 1. Run local hand tracking (MediaPipe HandLandmarker)
            hand_result = hand_tracker.process(frame, timestamp_ms=current_ms, rgb_frame=rgb_frame)

            # 2. Run local face & head pose tracking (MediaPipe FaceLandmarker + solvePnP)
            face_result = face_tracker.process(frame, timestamp_ms=current_ms, rgb_frame=rgb_frame)

            # 3. Run gesture recognition pipelines
            gesture_result = gesture_recognizer.process(hand_result)
            head_state = head_recognizer.process(face_result)

            # 4. Dispatch to Command Engine (evaluates control mode, moves cursor, clicks)
            # IMPORTANT: Head movements perform ZERO OS actions.
            control_status = command_engine.process(gesture_result, frame.shape)

            # 5. Draw visual overlays on frame
            # 5a. Face landmarks and 3D orientation axes
            frame = draw_face_landmarks(frame, face_result)

            # 5b. 21 hand skeletal landmarks, pinch highlights, and handedness badges
            frame = draw_hand_landmarks(frame, hand_result, gesture_result)

            # 5c. Hand gesture recognition status cards, mode toggle progress, cursor crosshairs
            frame = draw_gesture_hud(frame, gesture_result)

            # 5d. Stage 4B Control Mode HUD (Mode, Cursor, Action, Warnings)
            frame = draw_control_hud(frame, control_status)

            # 5e. Stage 5 Head Tracking HUD (Face detection, Yaw/Pitch/Roll, Head State)
            frame = draw_head_hud(frame, head_state, face_result)

            # 5f. Hands debug HUD (Detected hands, Handedness, Confidence, Tip, Pinch)
            frame = draw_debug_hands_hud(frame, hand_result, gesture_result)

            # 6. Update FPS and draw top/bottom HUD overlay bars
            current_fps = fps_counter.tick()

            hands_info = ""
            if hand_result.hand_count > 0:
                hands_info = ", ".join(
                    f"{h.handedness} {int(h.score * 100)}%" if h.score > 0 else h.handedness
                    for h in hand_result.hands
                )

            display_frame = draw_camera_hud(
                frame=frame,
                fps=current_fps,
                status="Active",
                resolution=camera.resolution,
                hand_count=hand_result.hand_count,
                hands_info=hands_info,
                title_text="AI Gesture Control - Stage 5 Head Tracking",
                control_mode=control_status.control_mode,
            )

            # 7. Display in preview window
            cv2.imshow(window_title, display_frame)

            # 8. Handle user keyboard input
            key = cv2.waitKey(1) & 0xFF

            # Key 27 = ESC: Emergency safety disable (does NOT exit app; sets Control Mode to OFF)
            if key == 27:
                command_engine.emergency_disable()
                logger.warning("Emergency disable triggered via ESC key. Control mode is OFF.")

            # Key 'q' or 'Q': Quit application
            elif key in (ord("q"), ord("Q")):
                logger.info("Termination signal received (Q key pressed). Shutting down.")
                break

            # 9. Handle user closing window via OS close button (X)
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Window closed by user. Shutting down.")
                break

    except CameraInitError as e:
        logger.critical(f"Camera Initialization Failed: {e}")
        return 1
    except ModelNotFoundError as e:
        logger.critical(f"Model Not Found: {e}")
        return 1
    except (HandTrackerError, FaceTrackerError) as e:
        logger.critical(f"Vision Tracker Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received. Exiting cleanly.")
        return 0
    except Exception as e:
        logger.critical(f"Unexpected runtime error in application loop: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Releasing control engine, vision trackers, camera, and OpenCV windows...")
        try:
            command_engine.reset()
        except Exception as e:
            logger.warning(f"Error resetting command engine: {e}")

        try:
            gesture_recognizer.reset()
        except Exception as e:
            logger.warning(f"Error resetting gesture recognizer: {e}")

        try:
            head_recognizer.reset()
        except Exception as e:
            logger.warning(f"Error resetting head recognizer: {e}")

        try:
            face_tracker.close()
        except Exception as e:
            logger.warning(f"Error releasing face tracker: {e}")

        try:
            hand_tracker.close()
        except Exception as e:
            logger.warning(f"Error releasing hand tracker: {e}")

        try:
            camera.release()
        except Exception as e:
            logger.warning(f"Error releasing camera: {e}")

        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            logger.warning(f"Error destroying OpenCV windows: {e}")

        logger.info("Application shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(run())


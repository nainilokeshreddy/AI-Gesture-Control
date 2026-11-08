"""
Unit Tests for Camera Foundation, FPS Counter, and HUD Overlay
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from config.settings import CameraConfig
from vision.camera import Camera, CameraInitError, CameraError
from utils.fps import FPSCounter
from ui.overlay import draw_camera_hud


class TestCameraFoundation(unittest.TestCase):

    def test_01_camera_initialization_defaults(self):
        """Verify camera initializes with default CameraConfig."""
        cam = Camera()
        self.assertEqual(cam.config.device_index, 0)
        self.assertEqual(cam.config.frame_width, 1280)
        self.assertEqual(cam.config.frame_height, 720)
        self.assertTrue(cam.config.mirror)
        self.assertFalse(cam.is_opened)

    def test_02_camera_custom_config(self):
        """Verify camera honors custom configuration parameters."""
        config = CameraConfig(device_index=1, frame_width=1920, frame_height=1080, fps=60, mirror=False)
        cam = Camera(config)
        self.assertEqual(cam.config.device_index, 1)
        self.assertEqual(cam.config.frame_width, 1920)
        self.assertEqual(cam.config.frame_height, 1080)
        self.assertEqual(cam.config.fps, 60)
        self.assertFalse(cam.config.mirror)

    @patch("vision.camera.cv2.VideoCapture")
    def test_03_backend_selection_avfoundation_on_macos(self, mock_cv2_cap):
        """Verify cv2.CAP_AVFOUNDATION is used when running on macOS."""
        import cv2

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.side_effect = lambda prop: 1280.0 if prop == cv2.CAP_PROP_FRAME_WIDTH else (720.0 if prop == cv2.CAP_PROP_FRAME_HEIGHT else 30.0)
        mock_cv2_cap.return_value = mock_instance

        cam = Camera()
        with patch("vision.camera.sys.platform", "darwin"):
            cam.start()
            mock_cv2_cap.assert_called_once_with(0, cv2.CAP_AVFOUNDATION)
            self.assertTrue(cam.is_opened)
            self.assertEqual(cam.resolution, (1280, 720))
            cam.release()

    @patch("vision.camera.cv2.VideoCapture")
    def test_04_backend_selection_fallback_on_linux(self, mock_cv2_cap):
        """Verify fallback to cv2.CAP_ANY on non-macOS platforms."""
        import cv2

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.return_value = 0.0
        mock_cv2_cap.return_value = mock_instance

        cam = Camera()
        with patch("vision.camera.sys.platform", "linux"):
            cam.start()
            mock_cv2_cap.assert_called_once_with(0, cv2.CAP_ANY)
            cam.release()

    @patch("vision.camera.cv2.VideoCapture")
    def test_05_camera_init_failure_raises_error(self, mock_cv2_cap):
        """Verify CameraInitError is raised when isOpened returns False."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = False
        mock_cv2_cap.return_value = mock_instance

        cam = Camera()
        with self.assertRaises(CameraInitError) as ctx:
            cam.start()

        self.assertIn("Failed to open camera device", str(ctx.exception))
        self.assertFalse(cam.is_opened)

    @patch("vision.camera.cv2.VideoCapture")
    def test_06_frame_mirroring(self, mock_cv2_cap):
        """Verify horizontal flip is applied when mirror=True."""
        import cv2

        # Create an asymmetrical dummy frame
        test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        test_frame[:, :50] = 255  # Left half is white, right half is black

        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.return_value = 100.0
        mock_instance.read.return_value = (True, test_frame.copy())
        mock_cv2_cap.return_value = mock_instance

        # Test with mirror=True
        cam = Camera(CameraConfig(mirror=True))
        cam.start()
        success, mirrored_frame = cam.read()
        self.assertTrue(success)
        # For mirrored frame, right half should now be white (255)
        self.assertEqual(mirrored_frame[:, 50:].mean(), 255)
        self.assertEqual(mirrored_frame[:, :50].mean(), 0)

        # Test with raw=True (bypass mirror)
        mock_instance.read.return_value = (True, test_frame.copy())
        success, raw_frame = cam.read(raw=True)
        self.assertTrue(success)
        self.assertEqual(raw_frame[:, :50].mean(), 255)
        self.assertEqual(raw_frame[:, 50:].mean(), 0)

        cam.release()

    @patch("vision.camera.cv2.VideoCapture")
    def test_07_frame_read_failure_handling(self, mock_cv2_cap):
        """Verify clean handling when camera.read fails."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.return_value = 100.0
        mock_instance.read.return_value = (False, None)
        mock_cv2_cap.return_value = mock_instance

        cam = Camera()
        cam.start()
        success, frame = cam.read()
        self.assertFalse(success)
        self.assertIsNone(frame)
        cam.release()

    @patch("vision.camera.cv2.VideoCapture")
    def test_08_context_manager(self, mock_cv2_cap):
        """Verify Camera works correctly as a Python context manager."""
        mock_instance = MagicMock()
        mock_instance.isOpened.return_value = True
        mock_instance.get.return_value = 100.0
        mock_cv2_cap.return_value = mock_instance

        with Camera() as cam:
            self.assertTrue(cam.is_opened)

        mock_instance.release.assert_called_once()
        self.assertFalse(cam.is_opened)

    def test_09_fps_counter(self):
        """Verify FPSCounter records and calculates frame rate."""
        fps = FPSCounter(window_size=10)
        self.assertEqual(fps.fps, 0.0)

        # Simulate 10 frames with 0.033s delta (~30 FPS)
        with patch("time.perf_counter", side_effect=[1.0, 1.033, 1.066, 1.099, 1.132]):
            fps.tick()  # First tick records baseline time
            fps.tick()
            fps.tick()
            fps.tick()
            current_fps = fps.tick()
            self.assertGreater(current_fps, 28.0)
            self.assertLess(current_fps, 32.0)

        fps.reset()
        self.assertEqual(fps.fps, 0.0)

    def test_10_overlay_hud_rendering(self):
        """Verify draw_camera_hud renders properly onto a frame without modifying dimensions."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = draw_camera_hud(frame, fps=29.8, status="Active", resolution=(1280, 720))

        self.assertEqual(rendered.shape, (720, 1280, 3))
        # HUD overlays content, so frame should no longer be completely black
        self.assertGreater(rendered.sum(), 0)

    @patch("app.cv2.destroyAllWindows")
    @patch("app.cv2.getWindowProperty", return_value=1)
    @patch("app.cv2.namedWindow")
    @patch("app.cv2.imshow")
    @patch("app.cv2.waitKey", side_effect=[ord("q"), 0])
    @patch("app.Camera")
    def test_11_app_run_clean_exit_on_q(
        self, mock_camera_cls, mock_wait_key, mock_imshow, mock_named_win, mock_get_win_prop, mock_destroy
    ):
        """Verify app.run exits cleanly with code 0 on pressing 'q'."""
        import app

        mock_cam = MagicMock()
        mock_cam.start.return_value = True
        mock_cam.read.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
        mock_cam.resolution = (1280, 720)
        mock_camera_cls.return_value = mock_cam

        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        mock_cam.release.assert_called_once()
        mock_destroy.assert_called_once()

    @patch("app.Camera")
    def test_12_app_run_camera_init_failure(self, mock_camera_cls):
        """Verify app.run handles CameraInitError gracefully and returns exit code 1."""
        import app

        mock_cam = MagicMock()
        mock_cam.start.side_effect = CameraInitError("Hardware unavailable")
        mock_camera_cls.return_value = mock_cam

        exit_code = app.run()
        self.assertEqual(exit_code, 1)
        mock_cam.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""
Head Pose Calibration & Diagnostic Tool
AI-Powered Multimodal Gesture Recognition and Hands-Free Computer Control Platform

Measures and records exact head pose orientation values across the 3 calibration movements:
1. NEUTRAL (3 seconds)
2. LEFT (2 seconds)
3. RIGHT (2 seconds)

For each movement, tracks and reports:
- Raw RQDecomp Euler angles (Yaw, Pitch, Roll)
- Normalized & Oriented Euler angles (Yaw, Pitch, Roll)
- EMA Smoothed Euler angles (Yaw, Pitch, Roll)
- Classifier Detected State & Gesture ID

Does NOT modify any existing configuration, thresholds, or hand controls.
Saves comprehensive results to calibration_results.json and prints a formatted diagnostic report.
"""

import os
import sys
import time
import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

from config.settings import DEFAULT_CONFIG
from vision.camera import Camera, CameraInitError
from vision.face_tracker import FaceTracker, FaceTrackingResult, HeadPoseData
from gestures.head_gestures import HeadGestureRecognizer, HeadGestureState, HeadGestureID


@dataclass
class PoseSample:
    """Telemetry data collected for a single video frame."""
    timestamp: float
    rq_yaw: float
    rq_pitch: float
    rq_roll: float
    norm_yaw: float
    norm_pitch: float
    norm_roll: float
    smooth_yaw: float
    smooth_pitch: float
    smooth_roll: float
    detected_state: str
    active_gesture: str


@dataclass
class MovementResult:
    """Aggregated samples and statistics for one calibration movement."""
    movement_name: str
    target_duration_sec: float
    actual_duration_sec: float
    sample_count: int
    samples: List[PoseSample] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class HeadCalibrator:
    """
    Guides the user through sequential head calibration movements,
    recording synchronized pose data directly from FaceTracker and HeadGestureRecognizer.
    """

    MOVEMENTS = [
        {"name": "NEUTRAL", "duration": 3.0, "prompt": "Look straight at the camera with neutral relaxed posture"},
        {"name": "LEFT", "duration": 2.0, "prompt": "Turn / Tilt your head to the LEFT"},
        {"name": "RIGHT", "duration": 2.0, "prompt": "Turn / Tilt your head to the RIGHT"},
    ]

    PREP_COUNTDOWN_SEC = 3.0

    def __init__(self):
        self.camera = Camera(DEFAULT_CONFIG.camera)
        self.face_tracker = FaceTracker(DEFAULT_CONFIG.vision, smoothing_config=DEFAULT_CONFIG.smoothing)
        self.head_recognizer = HeadGestureRecognizer(
            thresholds=DEFAULT_CONFIG.thresholds,
            cooldown=DEFAULT_CONFIG.cooldown,
        )
        self.results: Dict[str, MovementResult] = {}

    @staticmethod
    def _compute_stats(samples: List[PoseSample]) -> Dict[str, Any]:
        """Compute statistical summary (mean, std, min, max, median) for sample metrics."""
        if not samples:
            return {}

        metrics = {
            "rq_yaw": [s.rq_yaw for s in samples],
            "rq_pitch": [s.rq_pitch for s in samples],
            "rq_roll": [s.rq_roll for s in samples],
            "norm_yaw": [s.norm_yaw for s in samples],
            "norm_pitch": [s.norm_pitch for s in samples],
            "norm_roll": [s.norm_roll for s in samples],
            "smooth_yaw": [s.smooth_yaw for s in samples],
            "smooth_pitch": [s.smooth_pitch for s in samples],
            "smooth_roll": [s.smooth_roll for s in samples],
        }

        stats = {}
        for key, vals in metrics.items():
            arr = np.array(vals, dtype=np.float64)
            stats[key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
            }

        # State frequency distribution
        states: Dict[str, int] = {}
        for s in samples:
            states[s.detected_state] = states.get(s.detected_state, 0) + 1

        total = len(samples)
        state_pct = {k: f"{(v / total) * 100:.1f}% ({v}/{total})" for k, v in states.items()}
        stats["state_distribution"] = state_pct

        return stats

    def run(self) -> int:
        """Run the guided calibration session."""
        print("=" * 70)
        print("      AI GESTURE CONTROL: HEAD CALIBRATION & DIAGNOSTIC MODE")
        print("=" * 70)
        print("This tool will guide you through 5 movements to record camera pose angles:\n")
        for i, m in enumerate(self.MOVEMENTS, 1):
            print(f"  {i}. {m['name']} ({m['duration']:.0f}s): {m['prompt']}")
        print("\nControls:")
        print("  - Press SPACE to pause/resume or skip preparation countdown.")
        print("  - Press 'R' to restart the current movement.")
        print("  - Press ESC or 'Q' to abort.\n")
        print("Starting camera and MediaPipe FaceLandmarker...")

        try:
            self.camera.start()
            self.face_tracker.start()
        except CameraInitError as e:
            print(f"\n[ERROR] Failed to start camera: {e}")
            print("Please run this script directly in your macOS terminal outside sandbox:")
            print("  .venv/bin/python calibrate_head.py")
            return 1
        except Exception as e:
            print(f"\n[ERROR] Initialization error: {e}")
            return 1

        window_name = "Head Pose Calibration & Diagnostic"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            for mov_idx, mov in enumerate(self.MOVEMENTS):
                mov_name = mov["name"]
                duration = mov["duration"]
                prompt = mov["prompt"]

                print(f"\n>>> [{mov_idx + 1}/{len(self.MOVEMENTS)}] PREPARE FOR: {mov_name}")
                print(f"    Instruction: {prompt}")

                # 1. Preparation countdown
                prep_start = time.time()
                while True:
                    success, frame = self.camera.read()
                    if not success or frame is None:
                        continue

                    now = time.time()
                    elapsed_prep = now - prep_start
                    remaining_prep = max(0.0, self.PREP_COUNTDOWN_SEC - elapsed_prep)

                    # Process face for preview
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_res = self.face_tracker.process(frame, rgb_frame=rgb_frame)
                    head_state = self.head_recognizer.process(face_res)

                    # Render preparation overlay
                    display_frame = self._render_prep_overlay(
                        frame.copy(),
                        mov_name=mov_name,
                        mov_idx=mov_idx + 1,
                        total_movs=len(self.MOVEMENTS),
                        prompt=prompt,
                        countdown=remaining_prep,
                        head_state=head_state,
                        face_res=face_res,
                    )
                    cv2.imshow(window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        print("\nCalibration aborted by user.")
                        return 0
                    elif key == ord(' '):
                        break  # Skip prep countdown

                    if remaining_prep <= 0.0:
                        break

                # 2. Recording phase
                print(f"    * RECORDING {mov_name} for {duration:.1f}s...")
                samples: List[PoseSample] = []
                rec_start = time.time()

                while True:
                    success, frame = self.camera.read()
                    if not success or frame is None:
                        continue

                    now = time.time()
                    elapsed_rec = now - rec_start
                    remaining_rec = max(0.0, duration - elapsed_rec)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    t_now_ms = int(now * 1000)
                    face_res = self.face_tracker.process(frame, timestamp_ms=t_now_ms, rgb_frame=rgb_frame)
                    head_state = self.head_recognizer.process(face_res)

                    # Collect sample if face is detected
                    if face_res.face_detected and face_res.head_pose is not None:
                        pose = face_res.head_pose
                        rmat, _ = cv2.Rodrigues(pose.rvec)
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                        sample = PoseSample(
                            timestamp=round(elapsed_rec, 3),
                            rq_yaw=round(float(angles[1]), 2),
                            rq_pitch=round(float(angles[0]), 2),
                            rq_roll=round(float(angles[2]), 2),
                            norm_yaw=round(pose.raw_yaw, 2),
                            norm_pitch=round(pose.raw_pitch, 2),
                            norm_roll=round(pose.raw_roll, 2),
                            smooth_yaw=round(pose.yaw, 2),
                            smooth_pitch=round(pose.pitch, 2),
                            smooth_roll=round(pose.roll, 2),
                            detected_state=head_state.current_state,
                            active_gesture=head_state.active_gesture.value,
                        )
                        samples.append(sample)

                    # Render recording overlay
                    display_frame = self._render_rec_overlay(
                        frame.copy(),
                        mov_name=mov_name,
                        mov_idx=mov_idx + 1,
                        total_movs=len(self.MOVEMENTS),
                        prompt=prompt,
                        progress=min(1.0, elapsed_rec / duration),
                        remaining=remaining_rec,
                        sample_count=len(samples),
                        head_state=head_state,
                        face_res=face_res,
                    )
                    cv2.imshow(window_name, display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q'), ord('Q')):
                        print("\nCalibration aborted by user.")
                        return 0

                    if elapsed_rec >= duration:
                        break

                print(f"    [COMPLETED] Recorded {len(samples)} frames for {mov_name}.")
                stats = self._compute_stats(samples)
                self.results[mov_name] = MovementResult(
                    movement_name=mov_name,
                    target_duration_sec=duration,
                    actual_duration_sec=round(time.time() - rec_start, 2),
                    sample_count=len(samples),
                    samples=samples,
                    stats=stats,
                )

            # Generate final reports
            self._print_final_report()
            self._save_results()

        finally:
            self.camera.release()
            self.face_tracker.close()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        return 0

    def _render_prep_overlay(
        self,
        frame: np.ndarray,
        mov_name: str,
        mov_idx: int,
        total_movs: int,
        prompt: str,
        countdown: float,
        head_state: HeadGestureState,
        face_res: FaceTrackingResult,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (20, 20, 26), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 100), (w, 100), (0, 200, 255), 2)

        title = f"PREPARE FOR [{mov_idx}/{total_movs}]: {mov_name}"
        cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 240, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, prompt, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Starting in: {countdown:.1f}s (or press SPACE)", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 120), 1, cv2.LINE_AA)

        # Center big countdown
        cd_str = f"{int(math.ceil(countdown))}"
        (cw, ch), _ = cv2.getTextSize(cd_str, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)
        cv2.putText(frame, cd_str, ((w - cw) // 2, h // 2 + ch // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 240, 255), 4, cv2.LINE_AA)

        self._render_telemetry_box(frame, head_state, face_res)
        return frame

    def _render_rec_overlay(
        self,
        frame: np.ndarray,
        mov_name: str,
        mov_idx: int,
        total_movs: int,
        prompt: str,
        progress: float,
        remaining: float,
        sample_count: int,
        head_state: HeadGestureState,
        face_res: FaceTrackingResult,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        # Top banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 26), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.line(frame, (0, 110), (w, 110), (50, 255, 80), 2)

        title = f"RECORDING [{mov_idx}/{total_movs}]: {mov_name}"
        cv2.putText(frame, title, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (50, 255, 80), 2, cv2.LINE_AA)
        cv2.putText(frame, prompt, (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        # Progress bar
        bar_x = 20
        bar_y = 78
        bar_w = w - 40
        bar_h = 16
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, progress)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (50, 220, 80), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 85, 95), 1)

        info_text = f"Remaining: {remaining:.1f}s | Samples: {sample_count} frames"
        cv2.putText(frame, info_text, (20, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

        self._render_telemetry_box(frame, head_state, face_res)
        return frame

    def _render_telemetry_box(
        self,
        frame: np.ndarray,
        head_state: HeadGestureState,
        face_res: FaceTrackingResult,
    ) -> None:
        h, w = frame.shape[:2]
        box_w = 420
        box_h = 170
        box_x = w - box_w - 20
        box_y = h - box_h - 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 18, 22), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (70, 75, 85), 1)

        cv2.putText(frame, "REAL-TIME POSE TELEMETRY", (box_x + 12, box_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1, cv2.LINE_AA)
        cv2.line(frame, (box_x + 10, box_y + 30), (box_x + box_w - 10, box_y + 30), (50, 55, 65), 1)

        if face_res.face_detected and face_res.head_pose is not None:
            pose = face_res.head_pose
            rmat, _ = cv2.Rodrigues(pose.rvec)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

            rq_str = f"RQ Raw:     Y={angles[1]:+5.1f} | P={angles[0]:+5.1f} | R={angles[2]:+5.1f}"
            norm_str = f"Normalized: Y={pose.raw_yaw:+5.1f} | P={pose.raw_pitch:+5.1f} | R={pose.raw_roll:+5.1f}"
            smooth_str = f"Smoothed:   Y={pose.yaw:+5.1f} | P={pose.pitch:+5.1f} | R={pose.roll:+5.1f}"
            state_str = f"Classifier State:  {head_state.current_state}"
            face_status = "FACE: DETECTED"
            face_col = (80, 240, 100)
        else:
            rq_str = "RQ Raw:     Y=  --.- | P=  --.- | R=  --.-"
            norm_str = "Normalized: Y=  --.- | P=  --.- | R=  --.-"
            smooth_str = "Smoothed:   Y=  --.- | P=  --.- | R=  --.-"
            state_str = "Classifier State:  NO FACE"
            face_status = "FACE: NOT DETECTED"
            face_col = (100, 100, 240)

        cv2.putText(frame, face_status, (box_x + 12, box_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, face_col, 1, cv2.LINE_AA)
        cv2.putText(frame, rq_str, (box_x + 12, box_y + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, norm_str, (box_x + 12, box_y + 102), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 215, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, smooth_str, (box_x + 12, box_y + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, state_str, (box_x + 12, box_y + 154), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    def _print_final_report(self) -> None:
        print("\n" + "=" * 80)
        print("                 HEAD POSE CALIBRATION DIAGNOSTIC REPORT")
        print("=" * 80)
        print(f"{'MOVEMENT':<16} | {'METRIC':<13} | {'MEAN':>7} | {'MEDIAN':>7} | {'MIN':>7} | {'MAX':>7} | {'STD':>6}")
        print("-" * 80)

        for name, res in self.results.items():
            st = res.stats
            if not st:
                print(f"{name:<16} | NO VALID SAMPLES RECORDED")
                continue

            orig_name = name
            for metric_label, key in [
                ("Yaw (norm)", "norm_yaw"),
                ("Pitch (norm)", "norm_pitch"),
                ("Roll (norm)", "norm_roll"),
                ("Yaw (smooth)", "smooth_yaw"),
                ("Pitch (smooth)", "smooth_pitch"),
                ("Roll (smooth)", "smooth_roll"),
                ("Yaw (RQ raw)", "rq_yaw"),
                ("Pitch (RQ raw)", "rq_pitch"),
                ("Roll (RQ raw)", "rq_roll"),
            ]:
                s = st.get(key, {})
                print(
                    f"{orig_name:<16} | {metric_label:<13} | {s.get('mean', 0.0):+7.1f}° | "
                    f"{s.get('median', 0.0):+7.1f}° | {s.get('min', 0.0):+7.1f}° | "
                    f"{s.get('max', 0.0):+7.1f}° | {s.get('std', 0.0):6.1f}°"
                )
                orig_name = ""

            # Print state distribution
            dist = st.get("state_distribution", {})
            dist_str = ", ".join([f"{k}: {v}" for k, v in dist.items()])
            print(f"{'':<16} | {'Detected State':<13} : {dist_str}")
            print("-" * 80)

    def _save_results(self) -> None:
        out_json = "calibration_results.json"
        out_txt = "calibration_report.txt"

        # Serialize to JSON
        serializable = {}
        for name, res in self.results.items():
            serializable[name] = {
                "movement_name": res.movement_name,
                "target_duration_sec": res.target_duration_sec,
                "actual_duration_sec": res.actual_duration_sec,
                "sample_count": res.sample_count,
                "stats": res.stats,
                "samples": [asdict(s) for s in res.samples],
            }

        with open(out_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n[SAVED] Full sample dataset saved to: {os.path.abspath(out_json)}")

        # Write text report
        with open(out_txt, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("                 HEAD POSE CALIBRATION DIAGNOSTIC REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'MOVEMENT':<16} | {'METRIC':<13} | {'MEAN':>7} | {'MEDIAN':>7} | {'MIN':>7} | {'MAX':>7} | {'STD':>6}\n")
            f.write("-" * 80 + "\n")
            for name, res in self.results.items():
                st = res.stats
                if not st:
                    continue
                orig_name = name
                for metric_label, key in [
                    ("Yaw (norm)", "norm_yaw"),
                    ("Pitch (norm)", "norm_pitch"),
                    ("Roll (norm)", "norm_roll"),
                    ("Yaw (smooth)", "smooth_yaw"),
                    ("Pitch (smooth)", "smooth_pitch"),
                    ("Roll (smooth)", "smooth_roll"),
                    ("Yaw (RQ raw)", "rq_yaw"),
                    ("Pitch (RQ raw)", "rq_pitch"),
                    ("Roll (RQ raw)", "rq_roll"),
                ]:
                    s = st.get(key, {})
                    f.write(
                        f"{orig_name:<16} | {metric_label:<13} | {s.get('mean', 0.0):+7.1f}° | "
                        f"{s.get('median', 0.0):+7.1f}° | {s.get('min', 0.0):+7.1f}° | "
                        f"{s.get('max', 0.0):+7.1f}° | {s.get('std', 0.0):6.1f}°\n"
                    )
                    orig_name = ""
                dist = st.get("state_distribution", {})
                dist_str = ", ".join([f"{k}: {v}" for k, v in dist.items()])
                f.write(f"{'':<16} | {'Detected State':<13} : {dist_str}\n")
                f.write("-" * 80 + "\n")

        print(f"[SAVED] Text report summary saved to: {os.path.abspath(out_txt)}")


def main():
    calibrator = HeadCalibrator()
    return calibrator.run()


if __name__ == "__main__":
    sys.exit(main())

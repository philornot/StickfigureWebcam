"""
Camera thread for GUI application.

This module handles camera capture and MediaPipe processing
in a separate thread to keep the GUI responsive.
"""

import time

import cv2
import mediapipe as mp
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

import config


class CameraThread(QThread):
    """
    Thread for camera capture and processing.

    This thread continuously captures frames, runs MediaPipe detection,
    and emits results to the main GUI thread via Qt signals.
    """

    frame_ready = pyqtSignal(np.ndarray, object, object, bool, bool)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    def __init__(self, live_config):
        """
        Initialize the camera thread.

        Args:
            live_config: LiveConfig instance for accessing current settings.
        """
        super().__init__()
        self.live_config = live_config
        self.running = False
        self.cap = None
        self.pose = None
        self.face_mesh = None
        self.eyes_closed_frame_counter = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0

    def run(self):
        """Main camera processing loop."""
        try:
            self._initialize_camera()
            # Note: MediaPipe models are NOT initialized here - they will be
            # initialized lazily when the first frame is captured

            self.running = True
            print("[CameraThread] Started successfully")

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print("[CameraThread] Failed to read frame")
                    time.sleep(0.01)
                    continue

                # Lazy initialization: Initialize MediaPipe on first frame
                if self.pose is None or self.face_mesh is None:
                    print("[CameraThread] Initializing MediaPipe models...")
                    self._initialize_mediapipe()
                    print("[CameraThread] MediaPipe models ready")

                # Mirror frame
                frame = cv2.flip(frame, 1)

                # Process frame
                self._process_frame(frame)

                # Update FPS
                self._update_fps()

        except Exception as e:
            print(f"[CameraThread] Error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self._cleanup()

    def _initialize_camera(self):
        """Initialize camera with configured settings."""
        self.cap = cv2.VideoCapture(config.CAMERA_ID)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        print(f"[CameraThread] Camera initialized: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")

    def _initialize_mediapipe(self):
        """
        Initialize MediaPipe models with current config.

        This is called lazily when the first frame is captured to avoid
        blocking the camera initialization and speed up startup.
        """
        mp_pose = mp.solutions.pose
        mp_face_mesh = mp.solutions.face_mesh

        cfg = self.live_config.get_snapshot()

        self.pose = mp_pose.Pose(
            min_detection_confidence=cfg['pose_min_detection_confidence'],
            min_tracking_confidence=cfg['pose_min_tracking_confidence'],
            model_complexity=cfg['pose_model_complexity']
        )

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=config.FACE_MESH_MAX_FACES,
            min_detection_confidence=config.FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.FACE_MESH_MIN_TRACKING_CONFIDENCE
        )

    def _process_frame(self, frame):
        """Process single frame through detection pipeline."""
        # Skip processing if models aren't ready yet
        if self.pose is None or self.face_mesh is None:
            return

        # Resize for faster processing
        processing_frame = cv2.resize(
            frame,
            (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT)
        )

        frame_rgb = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

        # Run detections
        pose_results = self.pose.process(frame_rgb)
        face_results = self.face_mesh.process(frame_rgb)

        mouth_open = False
        eyes_closed = False

        # Get current thresholds
        cfg = self.live_config.get_snapshot()

        # Process face landmarks
        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark

            # Detect mouth opening with current threshold
            mouth_open = self._detect_mouth_opening(
                landmarks,
                cfg['mouth_open_threshold_ratio']
            )

            # Detect closed eyes with current threshold
            eyes_closed = self._detect_eyes_closed(
                landmarks,
                cfg['eyes_closed_ratio_threshold'],
                cfg['eyes_closed_consecutive_frames']
            )
        else:
            self.eyes_closed_frame_counter = 0

        # Emit frame with detection results
        self.frame_ready.emit(
            frame,
            pose_results,
            face_results,
            mouth_open,
            eyes_closed
        )

    def _detect_mouth_opening(self, landmarks, threshold_ratio):
        """Detect if mouth is open using current threshold."""
        try:
            upper_y = landmarks[config.MOUTH_LANDMARKS['upper_lip_top']].y * config.PROCESSING_HEIGHT
            lower_y = landmarks[config.MOUTH_LANDMARKS['lower_lip_bottom']].y * config.PROCESSING_HEIGHT
            forehead_y = landmarks[config.MOUTH_LANDMARKS['forehead']].y * config.PROCESSING_HEIGHT
            chin_y = landmarks[config.MOUTH_LANDMARKS['chin']].y * config.PROCESSING_HEIGHT

            mouth_distance = abs(lower_y - upper_y)
            face_height = abs(chin_y - forehead_y)

            if face_height == 0:
                return False

            threshold = face_height * threshold_ratio
            return mouth_distance > threshold
        except:
            return False

    def _detect_eyes_closed(self, landmarks, ratio_threshold, consecutive_frames):
        """Detect if eyes are closed using current threshold."""
        try:
            def get_coords(idx):
                lm = landmarks[idx]
                return int(lm.x * config.PROCESSING_WIDTH), int(lm.y * config.PROCESSING_HEIGHT)

            left_eye_top = get_coords(config.LEFT_EYE_TOP)
            left_eye_bottom = get_coords(config.LEFT_EYE_BOTTOM)
            right_eye_top = get_coords(config.RIGHT_EYE_TOP)
            right_eye_bottom = get_coords(config.RIGHT_EYE_BOTTOM)
            forehead = get_coords(config.MOUTH_LANDMARKS['forehead'])
            chin = get_coords(config.MOUTH_LANDMARKS['chin'])

            left_eye_dist = np.sqrt((left_eye_top[0] - left_eye_bottom[0]) ** 2 +
                                    (left_eye_top[1] - left_eye_bottom[1]) ** 2)
            right_eye_dist = np.sqrt((right_eye_top[0] - right_eye_bottom[0]) ** 2 +
                                     (right_eye_top[1] - right_eye_bottom[1]) ** 2)

            avg_eye_dist = (left_eye_dist + right_eye_dist) * 0.5
            face_height = np.sqrt((forehead[0] - chin[0]) ** 2 + (forehead[1] - chin[1]) ** 2)

            if face_height == 0:
                return False

            ear_ratio = avg_eye_dist / face_height

            if ear_ratio < ratio_threshold:
                self.eyes_closed_frame_counter += 1
            else:
                self.eyes_closed_frame_counter = 0

            return self.eyes_closed_frame_counter >= consecutive_frames

        except:
            return False

    def _update_fps(self):
        """Calculate and emit FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            elapsed = time.time() - self.fps_start_time
            if elapsed > 0:
                self.current_fps = self.fps_counter / elapsed
                self.fps_updated.emit(self.current_fps)
            self.fps_counter = 0
            self.fps_start_time = time.time()

    def stop(self):
        """Stop the camera thread."""
        print("[CameraThread] Stopping...")
        self.running = False
        self.wait(2000)  # Wait up to 2 seconds

    def _cleanup(self):
        """Clean up resources."""
        if self.pose:
            self.pose.close()
        if self.face_mesh:
            self.face_mesh.close()
        if self.cap:
            self.cap.release()
        print("[CameraThread] Cleaned up")
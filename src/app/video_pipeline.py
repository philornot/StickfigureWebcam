#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Video processing pipeline handling camera input and MediaPipe processing."""

from typing import Any, Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.drawing.pose_analyzer import PoseAnalyzer
from src.drawing.stick_figure_renderer import StickFigureRenderer
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class VideoPipeline:
    """Video processing pipeline.

    Handles:
    - Camera capture
    - MediaPipe pose/face detection
    - Stick figure rendering
    - Virtual camera output
    """

    def __init__(
        self,
        camera_config: Dict[str, Any],
        processing_config: Dict[str, Any],
        logger: Optional[CustomLogger] = None,
    ):
        """Initialize the video pipeline.

        Args:
            camera_config: Camera configuration dict
            processing_config: Processing configuration dict
            logger: Optional custom logger
        """
        self.camera_config = camera_config
        self.processing_config = processing_config
        self.logger = logger or CustomLogger()

        # Components
        self.camera = None
        self.virtual_camera = None
        self.face_mesh = None
        self.hands = None
        self.renderer = None
        self.pose_analyzer = None

        # Performance monitoring
        self.performance = PerformanceMonitor("VideoPipeline")

        # MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        self.logger.info("Pipeline", "Video pipeline created")

    def initialize(self) -> bool:
        """Initialize all pipeline components.

        Returns:
            True if initialization successful
        """
        try:
            # Initialize camera
            if not self._init_camera():
                return False

            # Initialize MediaPipe
            if not self._init_mediapipe():
                return False

            # Initialize renderer
            if not self._init_renderer():
                return False

            # Initialize pose analyzer
            self.pose_analyzer = PoseAnalyzer(logger=self.logger)

            # Initialize virtual camera (optional)
            self._init_virtual_camera()

            self.logger.info("Pipeline", "Pipeline initialized successfully")
            return True

        except Exception as e:
            self.logger.error("Pipeline", f"Initialization failed: {str(e)}")
            return False

    def _init_camera(self) -> bool:
        """Initialize physical camera."""
        try:
            camera_id = self.camera_config.get("id", 0)
            width = self.camera_config.get("width", 640)
            height = self.camera_config.get("height", 480)
            fps = self.camera_config.get("fps", 30)

            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

            if not self.camera.isOpened():
                self.logger.error("Pipeline", f"Cannot open camera {camera_id}")
                return False

            self.logger.info("Pipeline", f"Camera initialized: {width}x{height}@{fps}")
            return True

        except Exception as e:
            self.logger.error("Pipeline", f"Camera init failed: {str(e)}")
            return False

    def _init_mediapipe(self) -> bool:
        """Initialize MediaPipe detectors."""
        try:
            # Face mesh for facial features and landmarks
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # Hands for hand/arm tracking
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            # ADD THIS: Pose for body tracking
            mp_pose = mp.solutions.pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.logger.info("Pipeline", "MediaPipe initialized")
            return True

        except Exception as e:
            self.logger.error("Pipeline", f"MediaPipe init failed: {str(e)}")
            return False

    def _init_renderer(self) -> bool:
        """Initialize stick figure renderer."""
        try:
            width = self.camera_config.get("width", 640)
            height = self.camera_config.get("height", 480)

            self.renderer = StickFigureRenderer(
                canvas_width=width,
                canvas_height=height,
                line_thickness=self.processing_config.get("line_thickness", 3),
                head_radius_factor=self.processing_config.get("head_radius_factor", 0.075),
                bg_color=tuple(self.processing_config.get("bg_color", [255, 255, 255])),
                figure_color=tuple(self.processing_config.get("figure_color", [0, 0, 0])),
                smooth_factor=self.processing_config.get("smooth_factor", 0.3),
                logger=self.logger,
            )

            self.logger.info("Pipeline", "Renderer initialized")
            return True

        except Exception as e:
            self.logger.error("Pipeline", f"Renderer init failed: {str(e)}")
            return False

    def _init_virtual_camera(self):
        """Initialize virtual camera (non-critical)."""
        try:
            import pyvirtualcam

            width = self.camera_config.get("width", 640)
            height = self.camera_config.get("height", 480)
            fps = self.camera_config.get("fps", 30)

            self.virtual_camera = pyvirtualcam.Camera(width=width, height=height, fps=fps)

            self.logger.info("Pipeline", f"Virtual camera: {self.virtual_camera.device}")

        except Exception as e:
            self.logger.warning("Pipeline", f"Virtual camera unavailable: {str(e)}")
            self.virtual_camera = None

    def process_frame(self) -> Optional[Dict[str, Any]]:
        """Process a single frame through the pipeline.

        Returns:
            Dictionary containing:
                - original_frame: Raw camera frame
                - processed_frame: Stick figure output
                - face_data: Detected face/hand data
                - upper_body_data: Upper body skeleton data
                - fps: Current FPS
        """
        self.performance.start_timer()

        try:
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                return None

            # Flip if configured
            if self.camera_config.get("flip_horizontal", True):
                frame = cv2.flip(frame, 1)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect face and hands
            face_results = self.face_mesh.process(rgb_frame)
            hands_results = self.hands.process(rgb_frame)
            pose_results = self.pose.process(rgb_frame)

            # Process detections
            face_data = self._process_detections(face_results, hands_results)

            # Analyze upper body from POSE (not face)
            upper_body_data = None
            if pose_results and pose_results.pose_landmarks:
                try:
                    upper_body_data = self._extract_upper_body_from_pose(
                        pose_results.pose_landmarks, frame.shape[1], frame.shape[0]
                    )
                except Exception as e:
                    self.logger.debug("Pipeline", f"Error extracting upper body: {str(e)}")
                    upper_body_data = None

            # Render stick figure
            stick_figure = self.renderer.render(face_data)

            # Send to virtual camera
            if self.virtual_camera:
                self._send_to_virtual_camera(stick_figure)

            self.performance.stop_timer()

            return {
                "original_frame": frame,
                "processed_frame": stick_figure,
                "face_data": face_data,
                "upper_body_data": upper_body_data,
                "fps": self.performance.get_current_fps(),
            }

        except Exception as e:
            self.logger.error("Pipeline", f"Frame processing failed: {str(e)}")
            return None

    def _process_detections(self, face_results, hands_results) -> Dict[str, Any]:
        """Process MediaPipe detection results.

        Args:
            face_results: Face mesh results
            hands_results: Hand detection results

        Returns:
            Processed face and hand data
        """
        face_data = {"has_face": False, "landmarks": None}

        # Process face landmarks
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z, 1.0) for lm in face_landmarks.landmark]
            face_data = {
                "has_face": True,
                "landmarks": landmarks,
                "expressions": self._analyze_expressions(landmarks),
            }

        # Process hand landmarks
        if hands_results.multi_hand_landmarks:
            face_data["hands_data"] = self._process_hands(hands_results.multi_hand_landmarks)

        return face_data

    def _analyze_expressions(self, landmarks: list) -> Dict[str, float]:
        """Analyze facial expressions from landmarks.

        Args:
            landmarks: Face mesh landmarks

        Returns:
            Expression values (0.0-1.0)
        """
        try:
            # Mouth open detection (simplified)
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]
            mouth_height = abs(lower_lip[1] - upper_lip[1])
            mouth_open = min(1.0, mouth_height * 20)

            # Smile detection (simplified)
            left_corner = landmarks[61]
            right_corner = landmarks[291]
            center_mouth = landmarks[13]
            corner_height = (left_corner[1] + right_corner[1]) / 2
            smile = max(0.0, min(1.0, (center_mouth[1] - corner_height) * 10 + 0.5))

            return {
                "mouth_open": mouth_open,
                "smile": smile,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
            }

        except Exception:
            return {"mouth_open": 0.0, "smile": 0.5, "left_eye_open": 1.0, "right_eye_open": 1.0}

    def _process_hands(self, multi_hand_landmarks) -> Dict[str, Any]:
        """Process hand landmarks.

        Args:
            multi_hand_landmarks: List of hand landmarks

        Returns:
            Processed hand data with shoulder/elbow estimation
        """
        hands_data = {"left_hand": None, "right_hand": None}

        for hand_landmarks in multi_hand_landmarks:
            # Get wrist position (landmark 0)
            wrist = hand_landmarks.landmark[0]
            wrist_pos = (wrist.x, wrist.y, wrist.z, 1.0)

            # Simple left/right detection based on x position
            is_left = wrist.x < 0.5

            # Estimate elbow position (simplified - above and towards center from wrist)
            elbow_x = wrist.x + (0.5 - wrist.x) * 0.4
            elbow_y = max(0.1, wrist.y - 0.2)  # Above wrist, not too high
            elbow_pos = (elbow_x, elbow_y, 0.0, 0.8)

            # Estimate shoulder position (more towards center and higher)
            shoulder_x = wrist.x + (0.5 - wrist.x) * 0.7
            shoulder_y = max(0.05, elbow_y - 0.15)  # Above elbow
            shoulder_pos = (shoulder_x, shoulder_y, 0.0, 0.7)

            hand_data = {
                "wrist": wrist_pos,
                "elbow": elbow_pos,
                "shoulder": shoulder_pos,
                "is_left": is_left,
            }

            if is_left:
                hands_data["left_hand"] = hand_data
            else:
                hands_data["right_hand"] = hand_data

        return hands_data

    def _send_to_virtual_camera(self, frame: np.ndarray):
        """Send frame to virtual camera.

        Args:
            frame: BGR frame to send
        """
        try:
            if self.virtual_camera:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.virtual_camera.send(rgb_frame)
        except Exception as e:
            # Non-critical error, just log
            self.logger.debug("Pipeline", f"Virtual camera send failed: {str(e)}")

    def _create_upper_body_from_hands(
        self, hands_data: Dict[str, Any], frame_width: int, frame_height: int
    ) -> Optional[Dict[str, Any]]:
        """Create upper body data from hand detection.

        Args:
            hands_data: Hand detection data
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Upper body data dictionary or None
        """
        if not hands_data:
            return None

        left_hand = hands_data.get("left_hand")
        right_hand = hands_data.get("right_hand")

        if not left_hand and not right_hand:
            return None

        result = {
            "has_shoulders": False,
            "has_arms": False,
            "shoulder_positions": (None, None),
            "elbow_positions": (None, None),
            "wrist_positions": (None, None),
        }

        # Process left hand
        if left_hand:
            result["has_arms"] = True

            # Convert normalized coords to pixel coords
            if left_hand.get("shoulder"):
                sx, sy = (
                    left_hand["shoulder"][0] * frame_width,
                    left_hand["shoulder"][1] * frame_height,
                )
                result["shoulder_positions"] = ((int(sx), int(sy)), result["shoulder_positions"][1])
                result["has_shoulders"] = True

            if left_hand.get("elbow"):
                ex, ey = left_hand["elbow"][0] * frame_width, left_hand["elbow"][1] * frame_height
                result["elbow_positions"] = ((int(ex), int(ey)), result["elbow_positions"][1])

            if left_hand.get("wrist"):
                wx, wy = left_hand["wrist"][0] * frame_width, left_hand["wrist"][1] * frame_height
                result["wrist_positions"] = ((int(wx), int(wy)), result["wrist_positions"][1])

        # Process right hand
        if right_hand:
            result["has_arms"] = True

            if right_hand.get("shoulder"):
                sx, sy = (
                    right_hand["shoulder"][0] * frame_width,
                    right_hand["shoulder"][1] * frame_height,
                )
                result["shoulder_positions"] = (result["shoulder_positions"][0], (int(sx), int(sy)))
                result["has_shoulders"] = True

            if right_hand.get("elbow"):
                ex, ey = right_hand["elbow"][0] * frame_width, right_hand["elbow"][1] * frame_height
                result["elbow_positions"] = (result["elbow_positions"][0], (int(ex), int(ey)))

            if right_hand.get("wrist"):
                wx, wy = right_hand["wrist"][0] * frame_width, right_hand["wrist"][1] * frame_height
                result["wrist_positions"] = (result["wrist_positions"][0], (int(wx), int(wy)))

        return result

    def _extract_upper_body_from_pose(
        self, pose_landmarks, frame_width: int, frame_height: int
    ) -> Dict[str, Any]:
        """Extract upper body data from MediaPipe Pose.

        Args:
            pose_landmarks: Pose landmarks from MediaPipe
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Upper body data dictionary
        """
        # MediaPipe Pose landmark indices
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16

        landmarks = pose_landmarks.landmark

        result = {
            "has_shoulders": False,
            "has_arms": False,
            "shoulder_positions": (None, None),
            "elbow_positions": (None, None),
            "wrist_positions": (None, None),
        }

        # Check visibility threshold
        visibility_threshold = 0.5

        # Extract shoulders
        left_shoulder = landmarks[LEFT_SHOULDER]
        right_shoulder = landmarks[RIGHT_SHOULDER]

        if (
            left_shoulder.visibility > visibility_threshold
            and right_shoulder.visibility > visibility_threshold
        ):
            result["has_shoulders"] = True
            result["shoulder_positions"] = (
                (int(left_shoulder.x * frame_width), int(left_shoulder.y * frame_height)),
                (int(right_shoulder.x * frame_width), int(right_shoulder.y * frame_height)),
            )

        # Extract elbows
        left_elbow = landmarks[LEFT_ELBOW]
        right_elbow = landmarks[RIGHT_ELBOW]

        left_elbow_pos = None
        right_elbow_pos = None

        if left_elbow.visibility > visibility_threshold:
            left_elbow_pos = (int(left_elbow.x * frame_width), int(left_elbow.y * frame_height))
            result["has_arms"] = True

        if right_elbow.visibility > visibility_threshold:
            right_elbow_pos = (int(right_elbow.x * frame_width), int(right_elbow.y * frame_height))
            result["has_arms"] = True

        result["elbow_positions"] = (left_elbow_pos, right_elbow_pos)

        # Extract wrists
        left_wrist = landmarks[LEFT_WRIST]
        right_wrist = landmarks[RIGHT_WRIST]

        left_wrist_pos = None
        right_wrist_pos = None

        if left_wrist.visibility > visibility_threshold:
            left_wrist_pos = (int(left_wrist.x * frame_width), int(left_wrist.y * frame_height))

        if right_wrist.visibility > visibility_threshold:
            right_wrist_pos = (int(right_wrist.x * frame_width), int(right_wrist.y * frame_height))

        result["wrist_positions"] = (left_wrist_pos, right_wrist_pos)

        return result

    def shutdown(self):
        """Shutdown the pipeline and release resources."""
        self.logger.info("Pipeline", "Shutting down pipeline")

        if self.camera:
            self.camera.release()

        if self.face_mesh:
            self.face_mesh.close()

        if self.hands:
            self.hands.close()

        if self.pose:
            self.pose.close()

        if self.virtual_camera:
            self.virtual_camera.close()

        self.logger.info("Pipeline", "Pipeline shutdown complete")

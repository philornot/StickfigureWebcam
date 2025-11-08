#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class PoseDetector:
    """
    Class for detecting body pose using MediaPipe Pose.
    Detects 33 body landmarks and provides their coordinates and visualization.
    """

    # Important landmark indices
    NOSE = 0

    # Face points
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10

    # Torso points
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    # Hip and leg points
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    # Define connections between points for visualization
    POSE_CONNECTIONS = [
        # Head
        (NOSE, LEFT_EYE),
        (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR),
        (RIGHT_EYE, RIGHT_EAR),
        # Torso
        (NOSE, LEFT_SHOULDER),
        (NOSE, RIGHT_SHOULDER),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW),
        (RIGHT_SHOULDER, RIGHT_ELBOW),
        (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP),
        (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        # Legs
        (LEFT_HIP, LEFT_KNEE),
        (RIGHT_HIP, RIGHT_KNEE),
        (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_KNEE, RIGHT_ANKLE),
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
        smooth_landmarks: bool = True,
        enable_segmentation: bool = False,
        logger: Optional[CustomLogger] = None,
    ):
        """
        Initializes pose detector.

        Args:
            min_detection_confidence (float): Minimum pose detection confidence (0.0-1.0)
            min_tracking_confidence (float): Minimum pose tracking confidence (0.0-1.0)
            model_complexity (int): Model complexity (0, 1, 2) - higher values mean
                                   greater accuracy but slower performance
            smooth_landmarks (bool): Whether to smooth landmark movements
            enable_segmentation (bool): Whether to enable human segmentation from background
            logger (CustomLogger, optional): Logger for recording messages
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("PoseDetector")

        # Detection parameters
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation

        # Detection statistics
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_score = 0.0

        # Initialize MediaPipe Pose
        self.logger.debug(
            "PoseDetector",
            f"Initializing MediaPipe Pose (model_complexity={model_complexity})",
            log_type="POSE",
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_pose = mp.solutions.pose

        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.logger.info("PoseDetector", "MediaPipe Pose initialized successfully", log_type="POSE")

    def detect_pose(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """
        Detects pose in image.

        Args:
            image (np.ndarray): Input image in BGR format (OpenCV)

        Returns:
            Tuple[bool, Dict[str, Any]]:
                - bool: True if pose detected, False otherwise
                - Dict: Dictionary containing information about detected pose
        """
        self.performance.start_timer()
        self.frame_count += 1

        # MediaPipe requires RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Pose detection
        try:
            results = self.pose.process(image_rgb)

            # Prepare results
            pose_data = {
                "landmarks": None,
                "world_landmarks": None,
                "segmentation_mask": None,
                "detection_score": 0.0,
                "has_pose": False,
                "frame_height": image.shape[0],
                "frame_width": image.shape[1],
            }

            # Check if pose was detected
            if results.pose_landmarks:
                # Pose detected
                self.detection_count += 1
                pose_data["has_pose"] = True

                # Convert landmarks to list of tuples (x, y, z, visibility)
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.append((landmark.x, landmark.y, landmark.z, landmark.visibility))

                pose_data["landmarks"] = landmarks

                # Also store 3D world landmarks
                if results.pose_world_landmarks:
                    world_landmarks = []
                    for landmark in results.pose_world_landmarks.landmark:
                        world_landmarks.append(
                            (landmark.x, landmark.y, landmark.z, landmark.visibility)
                        )

                    pose_data["world_landmarks"] = world_landmarks

                # Extract segmentation mask if available
                if self.enable_segmentation and results.segmentation_mask is not None:
                    pose_data["segmentation_mask"] = results.segmentation_mask

                # Estimate detection confidence (average point visibility)
                visibilities = [landmark[3] for landmark in landmarks]
                pose_data["detection_score"] = sum(visibilities) / len(visibilities)
                self.last_detection_score = pose_data["detection_score"]

                # Log every 30 frames
                if self.frame_count % 30 == 0:
                    self.logger.debug(
                        "PoseDetector",
                        f"Detected pose with confidence {self.last_detection_score:.2f}",
                        log_type="POSE",
                    )

                # Performance monitoring
                self.performance.stop_timer()
                execution_time = self.performance.get_last_execution_time() * 1000  # ms

                # Performance statistics every 100 detections
                if self.detection_count % 100 == 0:
                    self.logger.info(
                        "PoseDetector",
                        f"Detected {self.detection_count} poses, "
                        f"detection ratio: {self.detection_count / self.frame_count:.2f}",
                        log_type="POSE",
                    )
                    self.logger.performance_metrics(0, execution_time, "PoseDetector")

                return True, pose_data
            else:
                # No pose detection
                self.performance.stop_timer()

                # Log lack of detection every 50 frames without detection
                if self.frame_count % 50 == 0 and self.detection_count == 0:
                    self.logger.warning(
                        "PoseDetector", "No pose detected in last 50 frames", log_type="POSE"
                    )
                elif self.frame_count % 100 == 0:
                    self.logger.debug(
                        "PoseDetector",
                        f"No pose detection, detection ratio: {self.detection_count / self.frame_count:.2f}",
                        log_type="POSE",
                    )

                return False, pose_data

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "PoseDetector",
                f"Error during pose detection: {str(e)}",
                log_type="POSE",
                error={"error": str(e)},
            )
            return False, {"error": str(e), "has_pose": False}

    def draw_pose_on_image(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float, float, float]],
        draw_connections: bool = True,
        keypoint_radius: int = 5,
        keypoint_color: Tuple[int, int, int] = (0, 255, 0),  # BGR: green
        connection_color: Tuple[int, int, int] = (255, 255, 0),  # BGR: turquoise
        connection_thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws detected pose on image.

        Args:
            image (np.ndarray): Input image (BGR)
            landmarks (List[Tuple[float, float, float, float]]): List of points (x, y, z, visibility)
            draw_connections (bool): Whether to draw connections between points
            keypoint_radius (int): Point radius
            keypoint_color (Tuple[int, int, int]): Point color (BGR)
            connection_color (Tuple[int, int, int]): Connection color (BGR)
            connection_thickness (int): Connection line thickness

        Returns:
            np.ndarray: Image with drawn pose
        """
        if landmarks is None or len(landmarks) == 0:
            return image

        # Create image copy to avoid modifying original
        img_copy = image.copy()

        h, w, _ = img_copy.shape

        # Draw points (keypoints)
        for i, landmark in enumerate(landmarks):
            x, y, _, visibility = landmark

            # Only points with good visibility
            if visibility > 0.5:
                # Convert relative coordinates to absolute
                cx, cy = int(x * w), int(y * h)

                # Draw point
                cv2.circle(img_copy, (cx, cy), keypoint_radius, keypoint_color, -1)

        # Draw connections
        if draw_connections:
            for connection in self.POSE_CONNECTIONS:
                start_idx, end_idx = connection

                # Check if points exist and are visible
                if (
                    len(landmarks) > start_idx
                    and len(landmarks) > end_idx
                    and landmarks[start_idx][3] > 0.5
                    and landmarks[end_idx][3] > 0.5
                ):
                    start_point = (
                        int(landmarks[start_idx][0] * w),
                        int(landmarks[start_idx][1] * h),
                    )
                    end_point = (int(landmarks[end_idx][0] * w), int(landmarks[end_idx][1] * h))

                    cv2.line(
                        img_copy, start_point, end_point, connection_color, connection_thickness
                    )

        return img_copy

    def draw_pose_with_mediapipe(self, image: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Draws pose on image using built-in MediaPipe functions.

        Args:
            image (np.ndarray): Input image (BGR)
            landmarks: pose_landmarks object from MediaPipe

        Returns:
            np.ndarray: Image with drawn pose
        """
        if landmarks is None:
            return image

        # Convert list of tuples to MediaPipe format if needed
        if isinstance(landmarks, list):
            mp_landmarks = self._convert_to_mp_landmarks(landmarks, image.shape[1], image.shape[0])
        else:
            mp_landmarks = landmarks

        # Create image copy
        img_copy = image.copy()

        # Draw pose using MediaPipe
        self.mp_drawing.draw_landmarks(
            img_copy,
            mp_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        return img_copy

    def _convert_to_mp_landmarks(
        self, landmarks: List[Tuple[float, float, float, float]], img_width: int, img_height: int
    ) -> Any:
        """
        Converts list of points to MediaPipe format.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of points (x, y, z, visibility)
            img_width (int): Image width
            img_height (int): Image height

        Returns:
            mp_pose.PoseLandmarkList: Points in MediaPipe format
        """
        landmark_list = self.mp_pose.PoseLandmarkList()

        for x, y, z, visibility in landmarks:
            landmark = (
                self.mp_pose.PoseLandmark()
            )  # todo: Warning:(359, 50) Parameter(s) unfilledPossible callees:EnumMeta.__call__(cls: Type[_EnumMemberT], value, names: None = None)EnumMeta.__call__(cls: EnumMeta, value: str, names: str | Iterable[str] | Iterable[Iterable[str]] | Mapping[str, Any], *, module: str | None = None, qualname: str | None = None, type: Any | None = None, start: int = 1)
            landmark.x = x
            landmark.y = y
            landmark.z = z
            landmark.visibility = visibility
            landmark_list.landmark.append(landmark)

        return landmark_list

    def get_landmark_position(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        landmark_id: int,
        img_width: int,
        img_height: int,
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Returns position of specific landmark.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks
            landmark_id (int): Landmark index (e.g., PoseDetector.NOSE)
            img_width (int): Image width
            img_height (int): Image height

        Returns:
            Optional[Tuple[int, int, float, float]]:
                (x, y, z, visibility) or None if landmark doesn't exist
        """
        if landmarks is None or landmark_id >= len(landmarks):
            return None

        x, y, z, visibility = landmarks[landmark_id]
        return (int(x * img_width), int(y * img_height), z, visibility)

    def calculate_angle(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        point1: int,
        point2: int,
        point3: int,
    ) -> float:
        """
        Calculates angle between three points.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks
            point1 (int): First point index
            point2 (int): Second point index (angle vertex)
            point3 (int): Third point index

        Returns:
            float: Angle in degrees (0-180)
        """
        if (
            landmarks is None
            or point1 >= len(landmarks)
            or point2 >= len(landmarks)
            or point3 >= len(landmarks)
        ):
            return 0.0

        # Get point coordinates
        x1, y1, _, _ = landmarks[point1]
        x2, y2, _, _ = landmarks[point2]
        x3, y3, _, _ = landmarks[point3]

        # Calculate angle
        angle_radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle_degrees = np.abs(angle_radians * 180.0 / np.pi)

        # Ensure angle is in range 0-180
        if angle_degrees > 180.0:
            angle_degrees = 360.0 - angle_degrees

        return angle_degrees

    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Returns detection statistics.

        Returns:
            Dict[str, Any]: Detection statistics
        """
        detection_ratio = self.detection_count / max(1, self.frame_count)

        return {
            "total_frames": self.frame_count,
            "detection_count": self.detection_count,
            "detection_ratio": detection_ratio,
            "last_detection_score": self.last_detection_score,
            "model_complexity": self.model_complexity,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }

    def reset_stats(self) -> None:
        """
        Resets detection statistics.
        """
        self.frame_count = 0
        self.detection_count = 0
        self.last_detection_score = 0.0

    def close(self) -> None:
        """
        Releases resources used by detector.
        """
        try:
            if hasattr(self, "pose") and self.pose is not None:
                self.pose.close()
                self.pose = None  # Set to None after closing
                self.logger.debug("PoseDetector", "Pose detector closed", log_type="POSE")
        except Exception as e:
            # Don't raise exception, only log error
            self.logger.warning(
                "PoseDetector", f"Error closing pose detector: {str(e)}", log_type="POSE"
            )

    def __del__(self):
        """
        Class destructor ensuring resources are released.
        """
        try:
            # Call close() only if pose is not None
            if hasattr(self, "pose") and self.pose is not None:
                self.close()
        except:
            # Ignore errors in destructor - no point in logging them,
            # as logger may already be unavailable
            pass

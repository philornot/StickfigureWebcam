#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Face mesh detection module using MediaPipe FaceMesh.

This module provides face detection and landmark extraction capabilities
using Google's MediaPipe FaceMesh solution.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class FaceMeshDetector:
    """Detects faces and facial landmarks using MediaPipe FaceMesh.

    Provides face detection with 468 facial landmarks and expression analysis.
    Properly manages MediaPipe resources to prevent memory leaks.
    """

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        logger: Optional[CustomLogger] = None,
    ):
        """Initialize face mesh detector.

        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
            logger: Optional logger instance
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("FaceMeshDetector")

        # Detection parameters
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Statistics
        self.frame_count = 0
        self.detection_count = 0

        # Initialize MediaPipe FaceMesh
        self.mp_face_mesh = mp.solutions.face_mesh

        # CRITICAL: Create FaceMesh instance that will be properly managed
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.logger.info(
            "FaceMeshDetector",
            f"FaceMesh initialized (max_faces={max_num_faces}, refine={refine_landmarks})",
            log_type="POSE",
        )

    def detect_face(self, image: np.ndarray) -> Tuple[bool, Dict[str, Any]]:
        """Detect face and extract landmarks from image.

        Args:
            image: Input image in BGR format (OpenCV)

        Returns:
            Tuple containing:
                - bool: True if face detected
                - dict: Face data including landmarks and expressions
        """
        self.performance.start_timer()
        self.frame_count += 1

        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        face_data = {
            "has_face": False,
            "landmarks": None,
            "expressions": None,
            "frame_height": image.shape[0],
            "frame_width": image.shape[1],
        }

        try:
            # Process image
            results = self.face_mesh.process(image_rgb)

            # Check if face was detected
            if results.multi_face_landmarks:
                self.detection_count += 1
                face_data["has_face"] = True

                # Extract first face landmarks
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [
                    (lm.x, lm.y, lm.z, 1.0)  # Visibility always 1.0 for FaceMesh
                    for lm in face_landmarks.landmark
                ]

                face_data["landmarks"] = landmarks

                # Analyze facial expressions
                face_data["expressions"] = self._analyze_expressions(landmarks)

                # Log periodically
                if self.frame_count % 100 == 0:
                    self.logger.debug(
                        "FaceMeshDetector",
                        f"Detected face (count: {self.detection_count}/{self.frame_count})",
                        log_type="POSE",
                    )

            self.performance.stop_timer()
            return face_data["has_face"], face_data

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "FaceMeshDetector", f"Error during face detection: {str(e)}", log_type="POSE"
            )
            return False, face_data

    def _analyze_expressions(
        self, landmarks: List[Tuple[float, float, float, float]]
    ) -> Dict[str, float]:
        """Analyze facial expressions from landmarks.

        Args:
            landmarks: List of facial landmarks

        Returns:
            Dictionary with expression values (0.0-1.0)
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

        except Exception as e:
            self.logger.warning(
                "FaceMeshDetector", f"Error analyzing expressions: {str(e)}", log_type="POSE"
            )
            return {
                "mouth_open": 0.0,
                "smile": 0.5,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
            }

    def draw_landmarks_on_image(
        self,
        image: np.ndarray,
        landmarks: List[Tuple[float, float, float, float]],
        draw_contours: bool = True,
    ) -> np.ndarray:
        """Draw facial landmarks on image.

        Args:
            image: Input image (BGR)
            landmarks: List of landmarks (x, y, z, visibility)
            draw_contours: Whether to draw face contours

        Returns:
            Image with drawn landmarks
        """
        if landmarks is None or len(landmarks) == 0:
            return image

        img_copy = image.copy()
        h, w, _ = img_copy.shape

        # Draw landmarks as small circles
        for i, landmark in enumerate(landmarks):
            x, y, _, _ = landmark
            cx, cy = int(x * w), int(y * h)

            # Different colors for different face regions
            if i < 10:  # Face oval
                color = (0, 255, 0)  # Green
            elif 10 <= i < 68:  # Eyebrows and eyes
                color = (255, 0, 0)  # Blue
            elif 68 <= i < 200:  # Nose and mouth
                color = (0, 0, 255)  # Red
            else:  # Rest
                color = (255, 255, 0)  # Cyan

            cv2.circle(img_copy, (cx, cy), 1, color, -1)

        # Draw face contours if requested
        if draw_contours:
            # Face oval
            face_oval_indices = [
                10,
                338,
                297,
                332,
                284,
                251,
                389,
                356,
                454,
                323,
                361,
                288,
                397,
                365,
                379,
                378,
                400,
                377,
                152,
                148,
                176,
                149,
                150,
                136,
                172,
                58,
                132,
                93,
                234,
                127,
                162,
                21,
                54,
                103,
                67,
                109,
            ]
            for i in range(len(face_oval_indices) - 1):
                pt1_idx = face_oval_indices[i]
                pt2_idx = face_oval_indices[i + 1]
                if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                    pt1 = (int(landmarks[pt1_idx][0] * w), int(landmarks[pt1_idx][1] * h))
                    pt2 = (int(landmarks[pt2_idx][0] * w), int(landmarks[pt2_idx][1] * h))
                    cv2.line(img_copy, pt1, pt2, (0, 255, 0), 1)

        return img_copy

    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection statistics.

        Returns:
            Dictionary with detection statistics
        """
        detection_ratio = self.detection_count / max(1, self.frame_count)

        return {
            "total_frames": self.frame_count,
            "detection_count": self.detection_count,
            "detection_ratio": detection_ratio,
            "max_num_faces": self.max_num_faces,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
        }

    def reset_stats(self) -> None:
        """Reset detection statistics."""
        self.frame_count = 0
        self.detection_count = 0
        self.logger.debug("FaceMeshDetector", "Statistics reset", log_type="POSE")

    def close(self) -> None:
        """Release MediaPipe resources.

        CRITICAL: This must be called to prevent memory leaks!
        """
        try:
            if hasattr(self, "face_mesh") and self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None
                self.logger.info("FaceMeshDetector", "FaceMesh closed", log_type="POSE")
        except Exception as e:
            self.logger.warning(
                "FaceMeshDetector", f"Error closing FaceMesh: {str(e)}", log_type="POSE"
            )

    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            if hasattr(self, "face_mesh") and self.face_mesh is not None:
                self.close()
        except:
            # Ignore errors in destructor
            pass

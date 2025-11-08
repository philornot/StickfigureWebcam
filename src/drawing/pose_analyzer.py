#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/pose_analyzer.py

from typing import Any, Dict, List, Optional, Tuple

from src.utils.custom_logger import CustomLogger


class PoseAnalyzer:
    """
    Simplified class for analyzing human poses based on key landmarks.
    Focuses on the upper body (torso).
    """

    # MediaPipe Pose/FaceMesh landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self, sitting_threshold: float = 0.3, logger: Optional[CustomLogger] = None):
        """
        Initialize the pose analyzer.

        Args:
            sitting_threshold: Threshold for shoulder position ratio
                              (kept for compatibility)
            logger: Logger for recording messages
        """
        self.sitting_threshold = sitting_threshold
        self.logger = logger or CustomLogger()
        self.logger.debug("PoseAnalyzer", "Pose analyzer initialized", log_type="POSE")

    def analyze_upper_body(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
    ) -> Dict[str, Any]:
        """
        Analyzes the upper body (torso) based on key landmarks.

        Args:
            landmarks: List of points (x, y, z, visibility)
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Dictionary with upper body position information
        """
        result = {
            "has_shoulders": False,
            "has_arms": False,
            "shoulder_positions": None,
            "elbow_positions": None,
            "wrist_positions": None,
            "shoulder_width_ratio": 0.0,
            "arms_extended": False,
            "left_arm_angle": 0.0,
            "right_arm_angle": 0.0,
        }

        try:
            if landmarks is None or len(landmarks) < 17:  # Need at least points up to wrists
                return result

            # Check shoulder visibility
            left_shoulder_visible = self._is_point_visible(landmarks[self.LEFT_SHOULDER])
            right_shoulder_visible = self._is_point_visible(landmarks[self.RIGHT_SHOULDER])
            result["has_shoulders"] = left_shoulder_visible and right_shoulder_visible

            # Shoulder positions (pixel coordinates)
            if result["has_shoulders"]:
                left_shoulder_pos = self._to_pixel_coords(
                    landmarks[self.LEFT_SHOULDER], frame_width, frame_height
                )
                right_shoulder_pos = self._to_pixel_coords(
                    landmarks[self.RIGHT_SHOULDER], frame_width, frame_height
                )
                result["shoulder_positions"] = (left_shoulder_pos, right_shoulder_pos)

                # Calculate shoulder width as proportion of frame width
                shoulder_width = abs(right_shoulder_pos[0] - left_shoulder_pos[0])
                result["shoulder_width_ratio"] = shoulder_width / frame_width

            # Check elbow visibility
            left_elbow_visible = self._is_point_visible(landmarks[self.LEFT_ELBOW])
            right_elbow_visible = self._is_point_visible(landmarks[self.RIGHT_ELBOW])
            left_wrist_visible = self._is_point_visible(landmarks[self.LEFT_WRIST])
            right_wrist_visible = self._is_point_visible(landmarks[self.RIGHT_WRIST])

            # Flag indicating if arms are detected
            result["has_arms"] = left_elbow_visible or right_elbow_visible

            # Elbow positions (pixel coordinates)
            if left_elbow_visible or right_elbow_visible:
                elbow_positions = []
                if left_elbow_visible:
                    elbow_positions.append(
                        self._to_pixel_coords(landmarks[self.LEFT_ELBOW], frame_width, frame_height)
                    )
                else:
                    elbow_positions.append(None)

                if right_elbow_visible:
                    elbow_positions.append(
                        self._to_pixel_coords(
                            landmarks[self.RIGHT_ELBOW], frame_width, frame_height
                        )
                    )
                else:
                    elbow_positions.append(None)

                result["elbow_positions"] = tuple(elbow_positions)

            # Wrist positions (pixel coordinates)
            if left_wrist_visible or right_wrist_visible:
                wrist_positions = []
                if left_wrist_visible:
                    wrist_positions.append(
                        self._to_pixel_coords(landmarks[self.LEFT_WRIST], frame_width, frame_height)
                    )
                else:
                    wrist_positions.append(None)

                if right_wrist_visible:
                    wrist_positions.append(
                        self._to_pixel_coords(
                            landmarks[self.RIGHT_WRIST], frame_width, frame_height
                        )
                    )
                else:
                    wrist_positions.append(None)

                result["wrist_positions"] = tuple(wrist_positions)

            # Check if arms are extended (e.g., raised hands)
            # Only if we have both shoulders and elbows
            if (
                result["has_shoulders"]
                and result["has_arms"]
                and result["shoulder_positions"] is not None
                and result["elbow_positions"] is not None
            ):

                left_extended = False
                right_extended = False

                # Left arm
                if left_elbow_visible and left_shoulder_visible:
                    left_shoulder = landmarks[self.LEFT_SHOULDER]
                    left_elbow = landmarks[self.LEFT_ELBOW]

                    # Angle between arm and vertical
                    left_angle = self._calculate_vertical_angle(left_shoulder, left_elbow)
                    result["left_arm_angle"] = left_angle

                    # If angle is greater than 45 degrees, arm is extended
                    left_extended = left_angle > 45

                # Right arm
                if right_elbow_visible and right_shoulder_visible:
                    right_shoulder = landmarks[self.RIGHT_SHOULDER]
                    right_elbow = landmarks[self.RIGHT_ELBOW]

                    # Angle between arm and vertical
                    right_angle = self._calculate_vertical_angle(right_shoulder, right_elbow)
                    result["right_arm_angle"] = right_angle

                    # If angle is greater than 45 degrees, arm is extended
                    right_extended = right_angle > 45

                result["arms_extended"] = left_extended or right_extended

            if self.logger and result["has_shoulders"]:
                self.logger.debug(
                    "PoseAnalyzer",
                    f"Upper body detected: shoulders={result['has_shoulders']}, "
                    f"arms={result['has_arms']}, extended={result['arms_extended']}",
                    log_type="POSE",
                )

        except Exception as e:
            if self.logger:
                self.logger.error(
                    "PoseAnalyzer",
                    f"Error during upper body analysis: {str(e)}",
                    log_type="POSE",
                    error={"error": str(e)},
                )

        return result

    def _is_point_visible(
        self, point: Tuple[float, float, float, float], threshold: float = 0.5
    ) -> bool:
        """
        Checks if a point is sufficiently visible.

        Args:
            point: Point (x, y, z, visibility)
            threshold: Visibility threshold (0.0-1.0)

        Returns:
            True if point visibility is above threshold
        """
        return point[3] >= threshold

    def _to_pixel_coords(
        self, point: Tuple[float, float, float, float], frame_width: int, frame_height: int
    ) -> Tuple[int, int]:
        """
        Converts normalized point coordinates (0.0-1.0) to pixel coordinates.

        Args:
            point: Point (x, y, z, visibility)
            frame_width: Frame width
            frame_height: Frame height

        Returns:
            Pixel coordinates (x, y)
        """
        return (int(point[0] * frame_width), int(point[1] * frame_height))

    def _calculate_vertical_angle(
        self, point1: Tuple[float, float, float, float], point2: Tuple[float, float, float, float]
    ) -> float:
        """
        Calculates the angle between the line connecting two points and the vertical.

        Args:
            point1: First point (x, y, z, visibility)
            point2: Second point (x, y, z, visibility)

        Returns:
            Angle in degrees (0-90)
        """
        import math

        # Points are already normalized (0.0-1.0)
        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        # Vector from point 1 to point 2
        vector_x = x2 - x1
        vector_y = y2 - y1

        # If vector is zero, return 0
        if vector_x == 0 and vector_y == 0:
            return 0.0

        # Vertical vector is (0, 1)
        # Angle between vectors: cos(θ) = (u·v) / (|u|·|v|)
        # Where u·v is dot product, and |u| and |v| are vector lengths

        # Dot product
        dot_product = vector_y  # dot product with vector (0, 1) is simply vector_y

        # Vector lengths
        vector_length = math.sqrt(vector_x**2 + vector_y**2)
        vertical_length = 1.0  # length of vector (0, 1)

        # Cosine of angle
        cos_angle = dot_product / (vector_length * vertical_length)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to range [-1, 1]

        # Angle in radians, then in degrees
        angle_rad = math.acos(cos_angle)
        angle_deg = angle_rad * 180 / math.pi

        return angle_deg

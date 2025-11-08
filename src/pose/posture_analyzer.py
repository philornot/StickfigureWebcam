#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.pose.pose_detector import PoseDetector
from src.utils.custom_logger import CustomLogger
from src.utils.performance import PerformanceMonitor


class PostureAnalyzer:
    """
    Class for analyzing body posture (e.g., whether a person is sitting or standing).
    Uses landmarks detected by PoseDetector to determine user position.
    """

    def __init__(
        self,
        standing_hip_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
        smoothing_factor: float = 0.7,
        temporal_smoothing: int = 5,
        partial_visibility_bias: float = 0.8,  # Sitting preference for partial visibility
        logger: Optional[CustomLogger] = None,
    ):
        """
        Initializes posture analyzer.

        Args:
            standing_hip_threshold (float): Hip height threshold for standing position
                                         (as proportion of image height from top)
            confidence_threshold (float): Minimum confidence level for points (0.0-1.0)
            smoothing_factor (float): Detection smoothing factor (0.0-1.0)
                                     (higher values = slower changes)
            temporal_smoothing (int): Number of frames used for temporal smoothing
            partial_visibility_bias (float): Sitting position preference for partial visibility
                                           (0.0-1.0, higher values = stronger preference)
            logger (CustomLogger, optional): Logger for recording messages
        """
        self.logger = logger or CustomLogger()
        self.performance = PerformanceMonitor("PostureAnalyzer")

        # Detection parameters
        self.standing_hip_threshold = standing_hip_threshold
        self.confidence_threshold = confidence_threshold
        self.smoothing_factor = smoothing_factor
        self.temporal_smoothing = temporal_smoothing
        self.partial_visibility_bias = partial_visibility_bias

        # Detection state
        self.is_sitting = None  # Initially unknown, need several frames
        self.sitting_probability = 0.5  # Start with 50% confidence
        self.history_buffer = []  # Detection history for temporal smoothing
        self.consecutive_frames = 0  # Number of consecutive frames with same detection
        self.last_visible_keypoints_count = 0  # Number of visible points in last frame

        # Constants - landmark indices from PoseDetector
        self.LEFT_SHOULDER = PoseDetector.LEFT_SHOULDER
        self.RIGHT_SHOULDER = PoseDetector.RIGHT_SHOULDER
        self.LEFT_HIP = PoseDetector.LEFT_HIP
        self.RIGHT_HIP = PoseDetector.RIGHT_HIP
        self.LEFT_KNEE = PoseDetector.LEFT_KNEE
        self.RIGHT_KNEE = PoseDetector.RIGHT_KNEE
        self.LEFT_ANKLE = PoseDetector.LEFT_ANKLE
        self.RIGHT_ANKLE = PoseDetector.RIGHT_ANKLE

        # Define basic body parts for visibility analysis
        self.UPPER_BODY_POINTS = [
            PoseDetector.NOSE,
            PoseDetector.LEFT_EYE,
            PoseDetector.RIGHT_EYE,
            PoseDetector.LEFT_SHOULDER,
            PoseDetector.RIGHT_SHOULDER,
            PoseDetector.LEFT_ELBOW,
            PoseDetector.RIGHT_ELBOW,
        ]

        self.LOWER_BODY_POINTS = [
            PoseDetector.LEFT_HIP,
            PoseDetector.RIGHT_HIP,
            PoseDetector.LEFT_KNEE,
            PoseDetector.RIGHT_KNEE,
            PoseDetector.LEFT_ANKLE,
            PoseDetector.RIGHT_ANKLE,
        ]

        self.logger.info(
            "PostureAnalyzer",
            f"Posture analyzer initialized (standing threshold: {standing_hip_threshold})",
            log_type="POSE",
        )

    def analyze_posture(
        self,
        landmarks: List[Tuple[float, float, float, float]],
        frame_height: int,
        frame_width: int,
    ) -> Dict[str, Any]:
        """
        Analyzes user posture and determines if sitting or standing.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks
            frame_height (int): Frame height
            frame_width (int): Frame width

        Returns:
            Dict[str, Any]: Dictionary with posture information
        """
        self.performance.start_timer()

        result = {
            "is_sitting": None,
            "sitting_probability": 0.0,
            "confidence": 0.0,
            "posture": "unknown",
            "visible_keypoints": 0,
            "visibility_type": "unknown",
        }

        if landmarks is None or len(landmarks) < 33:  # MediaPipe Pose has 33 points
            self.logger.debug(
                "PostureAnalyzer", "Not enough points for posture analysis", log_type="POSE"
            )
            return result

        try:
            # Check visibility of key points
            visible_keypoints = self._count_visible_keypoints(landmarks)
            self.last_visible_keypoints_count = visible_keypoints
            result["visible_keypoints"] = visible_keypoints

            # Too few visible points - but still try to analyze, with reduced confidence
            if visible_keypoints < 10:  # Lowered threshold from 15 to 10
                self.logger.debug(
                    "PostureAnalyzer",
                    f"Too few visible points ({visible_keypoints}/33), "
                    "but attempting analysis with limited confidence",
                    log_type="POSE",
                )

            # Analyze visibility type (full body, upper body, etc.)
            visibility_type = self._analyze_visibility_type(landmarks)
            result["visibility_type"] = visibility_type

            # Calculate sitting probability based on several heuristics
            hip_score = self._analyze_hip_position(landmarks, frame_height)
            leg_score = self._analyze_leg_visibility(landmarks)
            torso_score = self._analyze_torso_proportion(landmarks)
            visibility_score = self._analyze_partial_visibility(landmarks, visibility_type)

            # Weight individual results based on visibility type
            if visibility_type == "full_body":
                # If full body visible, use standard weights
                sitting_probability = (
                    0.5 * hip_score
                    + 0.25 * leg_score
                    + 0.20 * torso_score
                    + 0.05 * visibility_score
                )
            elif visibility_type == "upper_body":
                # If only upper body visible, give more weight to visibility analysis
                sitting_probability = (
                    0.3 * hip_score + 0.4 * leg_score + 0.1 * torso_score + 0.2 * visibility_score
                )
            else:  # partial_visibility or unknown
                # With very limited visibility, strongly prefer sitting (typical video conference scenario)
                sitting_probability = (
                    0.1 * hip_score + 0.2 * leg_score + 0.1 * torso_score + 0.6 * visibility_score
                )

            # Limit to range 0.0-1.0
            sitting_probability = max(0.0, min(1.0, sitting_probability))

            # IMPORTANT FIX: For very limited visibility (torso_only_detection)
            # enforce high sitting probability
            if visible_keypoints <= 12 and visibility_type in ["partial_visibility", "unknown"]:
                sitting_probability = max(
                    sitting_probability, 0.85
                )  # Enforce high sitting probability
                self.logger.debug(
                    "PostureAnalyzer",
                    f"Very limited visibility: increasing sitting probability to {sitting_probability:.2f}",
                    log_type="POSE",
                )

            # Apply exponential smoothing for current detection
            if self.sitting_probability is not None:
                sitting_probability = (
                    self.smoothing_factor * self.sitting_probability
                    + (1 - self.smoothing_factor) * sitting_probability
                )

            # Add to history for temporal smoothing
            self.history_buffer.append(sitting_probability)
            if len(self.history_buffer) > self.temporal_smoothing:
                self.history_buffer.pop(0)

            # Calculate smoothed value
            smoothed_probability = sum(self.history_buffer) / len(self.history_buffer)
            self.sitting_probability = smoothed_probability

            # Determine state based on smoothed value
            is_sitting = smoothed_probability >= 0.5

            # IMPORTANT FIX: For torso_only_detection, immediately set
            # self.is_sitting to True, without waiting for history
            if visible_keypoints <= 12 and visibility_type in ["partial_visibility", "unknown"]:
                is_sitting = True  # Enforce sitting
                self.is_sitting = True  # Directly set value
                self.consecutive_frames = 5  # Eliminate value "flickering"

            # Standard handling for other cases
            elif self.is_sitting is None:
                # For first detection, set state according to calculated probability
                self.is_sitting = is_sitting
                self.consecutive_frames = 1
            elif self.is_sitting == is_sitting:
                self.consecutive_frames += 1
            else:
                # State change only after threshold
                threshold_frames = (
                    5 if is_sitting else 10
                )  # Require more frames to change to standing
                if self.consecutive_frames >= threshold_frames:
                    self.is_sitting = is_sitting

                    # Log state change
                    self.logger.info(
                        "PostureAnalyzer",
                        f"Posture change: {'sitting' if is_sitting else 'standing'} "
                        f"(confidence: {smoothed_probability:.2f}, visibility type: {visibility_type})",
                        log_type="POSE",
                    )

                self.consecutive_frames = 1

            # Fill result
            result["is_sitting"] = self.is_sitting
            result["sitting_probability"] = smoothed_probability
            result["confidence"] = self._calculate_confidence(landmarks)
            result["posture"] = "sitting" if self.is_sitting else "standing"

            # Additional information for debugging
            result["debug"] = {
                "hip_score": hip_score,
                "leg_score": leg_score,
                "torso_score": torso_score,
                "visibility_score": visibility_score,
                "consecutive_frames": self.consecutive_frames,
                "visibility_type": visibility_type,
            }

            self.performance.stop_timer()
            return result

        except Exception as e:
            self.performance.stop_timer()
            self.logger.error(
                "PostureAnalyzer",
                f"Error during posture analysis: {str(e)}",
                log_type="POSE",
                error={"error": str(e)},
            )
            return result

    def _analyze_visibility_type(self, landmarks: List[Tuple[float, float, float, float]]) -> str:
        """
        Analyzes body visibility type and categorizes it.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks

        Returns:
            str: Visibility type: "full_body", "upper_body", "partial_visibility" or "unknown"
        """
        # Count visible points in upper and lower body
        upper_visible = sum(
            1
            for point_id in self.UPPER_BODY_POINTS
            if landmarks[point_id][3] > self.confidence_threshold
        )

        lower_visible = sum(
            1
            for point_id in self.LOWER_BODY_POINTS
            if landmarks[point_id][3] > self.confidence_threshold
        )

        upper_ratio = upper_visible / len(self.UPPER_BODY_POINTS)
        lower_ratio = lower_visible / len(self.LOWER_BODY_POINTS)

        # Visibility type categorization
        if upper_ratio > 0.7 and lower_ratio > 0.7:
            return "full_body"  # Full body visible
        elif upper_ratio > 0.7 and lower_ratio < 0.3:
            return "upper_body"  # Mainly upper body visible
        elif upper_ratio > 0.3:  # Changed from 0.5 to 0.3 for greater sensitivity
            return "partial_visibility"  # Partial visibility, but mainly upper body
        else:
            return "unknown"  # Hard to determine

    def _analyze_partial_visibility(
        self, landmarks: List[Tuple[float, float, float, float]], visibility_type: str
    ) -> float:
        """
        Analyzes partial visibility and returns sitting probability based on
        the assumption that during video conferences the user is most often sitting.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks
            visibility_type (str): Determined visibility type

        Returns:
            float: Score 0.0-1.0, where higher values indicate greater
                  sitting probability
        """
        # If full body visible, don't apply additional assumption
        if visibility_type == "full_body":
            return 0.5  # Neutral score

        # If only upper body visible, user is very likely sitting
        elif visibility_type == "upper_body":
            return 0.9  # High sitting probability

        # With partial visibility, also assume sitting with high confidence
        elif visibility_type == "partial_visibility":
            return 0.85  # Even higher probability

        # With unknown type, give very high preference value
        else:
            return 0.9  # Extreme sitting preference for unknown case

    def _analyze_hip_position(
        self, landmarks: List[Tuple[float, float, float, float]], frame_height: int
    ) -> float:
        """
        Analyzes hip position relative to image height.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks
            frame_height (int): Frame height

        Returns:
            float: Score 0.0-1.0, where higher values indicate greater sitting probability
        """
        # Hip position
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]

        # Check visibility
        if left_hip[3] < self.confidence_threshold and right_hip[3] < self.confidence_threshold:
            # If hips not visible, check if shoulders are visible
            if (
                landmarks[self.LEFT_SHOULDER][3] > self.confidence_threshold
                or landmarks[self.RIGHT_SHOULDER][3] > self.confidence_threshold
            ):
                # Hips not visible, but shoulders visible - typical scenario for sitting position
                return 0.85  # Increased value indicating sitting
            return 0.7  # Slightly increased neutral value

        # Use hip with better visibility
        hip_y = left_hip[1] if left_hip[3] > right_hip[3] else right_hip[1]

        # Compare with threshold
        if hip_y < self.standing_hip_threshold:
            # Hips are higher than threshold - probably standing
            # The higher the hips, the lower the value
            return max(0, 1 - (self.standing_hip_threshold - hip_y) / self.standing_hip_threshold)
        else:
            # Hips are lower than threshold - probably sitting
            # The lower the hips, the higher the value
            return min(
                1,
                (hip_y - self.standing_hip_threshold) / (1 - self.standing_hip_threshold) * 0.8
                + 0.2,
            )

    def _analyze_leg_visibility(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        """
        Analyzes leg visibility - when sitting, legs are often not visible.
        Improved score scale to increase sensitivity to lack of leg visibility.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks

        Returns:
            float: Score 0.0-1.0, where higher values indicate greater sitting probability
        """
        # Check ankle and knee visibility
        left_ankle_visible = landmarks[self.LEFT_ANKLE][3] > self.confidence_threshold
        right_ankle_visible = landmarks[self.RIGHT_ANKLE][3] > self.confidence_threshold
        left_knee_visible = landmarks[self.LEFT_KNEE][3] > self.confidence_threshold
        right_knee_visible = landmarks[self.RIGHT_KNEE][3] > self.confidence_threshold

        # Count visible leg parts
        visible_parts = sum(
            [left_ankle_visible, right_ankle_visible, left_knee_visible, right_knee_visible]
        )

        # The fewer visible leg parts, the greater the sitting probability
        # Improved value scale with stronger sitting preference
        if visible_parts == 0:
            return 0.95  # Even higher sitting probability when no legs visible
        elif visible_parts == 1:
            return 0.85  # Increased value
        elif visible_parts == 2:
            return 0.5  # Neutral
        elif visible_parts == 3:
            return 0.2  # Decreased value
        else:  # visible_parts == 4
            return (
                0.05  # Even lower sitting probability when all leg parts visible
            )

    def _analyze_torso_proportion(
        self, landmarks: List[Tuple[float, float, float, float]]
    ) -> float:
        """
        Analyzes torso proportions (hip-shoulder distance vs hip-knee distance).

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks

        Returns:
            float: Score 0.0-1.0, where higher values indicate greater sitting probability
        """
        # Check point visibility
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]
        left_hip = landmarks[self.LEFT_HIP]
        right_hip = landmarks[self.RIGHT_HIP]
        left_knee = landmarks[self.LEFT_KNEE]
        right_knee = landmarks[self.RIGHT_KNEE]

        # Check if we have enough points for analysis
        left_side_visible = (
            left_shoulder[3] > self.confidence_threshold
            and left_hip[3] > self.confidence_threshold
            and left_knee[3] > self.confidence_threshold
        )

        right_side_visible = (
            right_shoulder[3] > self.confidence_threshold
            and right_hip[3] > self.confidence_threshold
            and right_knee[3] > self.confidence_threshold
        )

        if not left_side_visible and not right_side_visible:
            # If no full visibility on either side
            # Check if we have at least visible shoulders
            shoulders_visible = (
                left_shoulder[3] > self.confidence_threshold
                or right_shoulder[3] > self.confidence_threshold
            )

            if shoulders_visible:
                # If only upper body visible, increase sitting probability
                return 0.8  # Prefer sitting when only upper body visible

            return 0.7  # Slightly prefer sitting as safer assumption

        # Choose side with better visibility
        if left_side_visible and (
            not right_side_visible
            or (left_shoulder[3] + left_hip[3] + left_knee[3])
            > (right_shoulder[3] + right_hip[3] + right_knee[3])
        ):
            shoulder = left_shoulder
            hip = left_hip
            knee = left_knee
        else:
            shoulder = right_shoulder
            hip = right_hip
            knee = right_knee

        # Calculate lengths
        torso_length = np.sqrt((shoulder[0] - hip[0]) ** 2 + (shoulder[1] - hip[1]) ** 2)
        upper_leg_length = np.sqrt((hip[0] - knee[0]) ** 2 + (hip[1] - knee[1]) ** 2)

        # If any length is close to zero, return neutral score
        if torso_length < 0.01 or upper_leg_length < 0.01:
            return 0.5

        # Calculate torso to upper leg length ratio
        ratio = torso_length / upper_leg_length

        # Proportion interpretation:
        # - In standing position, torso to leg ratio is typically smaller (approx. 0.8-1.2)
        # - In sitting position, ratio may be larger (approx. 1.5-2.5)

        if ratio < 0.8:
            return 0.2  # Very likely standing
        elif ratio < 1.2:
            return 0.3  # Probably standing
        elif ratio < 1.5:
            return 0.6  # Slight indication of sitting
        elif ratio < 2.5:
            return 0.8  # Probably sitting
        else:
            return 0.7  # Very large ratio may indicate detection error

    def _count_visible_keypoints(self, landmarks: List[Tuple[float, float, float, float]]) -> int:
        """
        Counts number of visible landmarks.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks

        Returns:
            int: Number of visible points
        """
        visible_count = 0
        for _, _, _, visibility in landmarks:
            if visibility > self.confidence_threshold:
                visible_count += 1

        return visible_count

    def _calculate_confidence(self, landmarks: List[Tuple[float, float, float, float]]) -> float:
        """
        Calculates posture detection confidence based on key point visibility.

        Args:
            landmarks (List[Tuple[float, float, float, float]]): List of landmarks

        Returns:
            float: Detection confidence (0.0-1.0)
        """
        # Upper body points are most important for detection in video conference mode
        upper_body_points = [
            self.LEFT_SHOULDER,
            self.RIGHT_SHOULDER,
            PoseDetector.NOSE,
            PoseDetector.LEFT_EYE,
            PoseDetector.RIGHT_EYE,
        ]

        # Lower body points
        lower_body_points = [
            self.LEFT_HIP,
            self.RIGHT_HIP,
            self.LEFT_KNEE,
            self.RIGHT_KNEE,
            self.LEFT_ANKLE,
            self.RIGHT_ANKLE,
        ]

        # Calculate average visibility for upper and lower body
        upper_visibility = 0.0
        for point_id in upper_body_points:
            if point_id < len(landmarks):
                upper_visibility += landmarks[point_id][3]
        upper_visibility /= len(upper_body_points)

        lower_visibility = 0.0
        for point_id in lower_body_points:
            if point_id < len(landmarks):
                lower_visibility += landmarks[point_id][3]
        lower_visibility /= len(lower_body_points)

        # Calculate confidence based on visibility
        # Give greater weight to upper body (80%) than lower body (20%)
        visibility_confidence = upper_visibility * 0.8 + lower_visibility * 0.2

        # Also consider number of consecutive detection frames
        temporal_factor = min(1.0, self.consecutive_frames / 10.0)

        # Combine both factors
        confidence = visibility_confidence * 0.7 + temporal_factor * 0.3

        return confidence

    def reset(self) -> None:
        """
        Resets analyzer state.
        """
        self.is_sitting = None
        self.sitting_probability = 0.5
        self.history_buffer = []
        self.consecutive_frames = 0
        self.last_visible_keypoints_count = 0

        self.logger.debug(
            "PostureAnalyzer", "Posture analyzer state reset", log_type="POSE"
        )

    def get_current_posture(self) -> Dict[str, Any]:
        """
        Returns current posture state.

        Returns:
            Dict[str, Any]: Dictionary with current posture information
        """
        return {
            "is_sitting": self.is_sitting,
            "posture": (
                "sitting"
                if self.is_sitting
                else ("standing" if self.is_sitting is not None else "unknown")
            ),
            "sitting_probability": self.sitting_probability,
            "consecutive_frames": self.consecutive_frames,
            "visible_keypoints": self.last_visible_keypoints_count,
        }

    def update_thresholds(
        self,
        standing_hip_threshold: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        smoothing_factor: Optional[float] = None,
        partial_visibility_bias: Optional[float] = None,
    ) -> None:
        """
        Updates detection thresholds.

        Args:
            standing_hip_threshold (Optional[float]): New hip height threshold
            confidence_threshold (Optional[float]): New confidence threshold
            smoothing_factor (Optional[float]): New smoothing factor
            partial_visibility_bias (Optional[float]): New sitting position preference
                                                      for partial visibility
        """
        if standing_hip_threshold is not None:
            self.standing_hip_threshold = standing_hip_threshold

        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        if smoothing_factor is not None:
            self.smoothing_factor = smoothing_factor

        if partial_visibility_bias is not None:
            self.partial_visibility_bias = partial_visibility_bias

        self.logger.info(
            "PostureAnalyzer",
            f"Updated analyzer parameters (hip_threshold={self.standing_hip_threshold}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"smoothing_factor={self.smoothing_factor})",
            log_type="POSE",
        )

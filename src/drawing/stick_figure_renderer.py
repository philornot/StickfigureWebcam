#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger
from .face_renderer import SimpleFaceRenderer
from .pose_analyzer import PoseAnalyzer


class StickFigureRenderer:
    """
    Class for rendering stick figures with focus on the upper body (torso).
    Provides smooth arm animations with more precise shoulder detection and rendering
    instead of just hands.
    """

    # MediaPipe FaceMesh and Pose landmark indices
    NOSE = 0
    LEFT_EYE = 2
    RIGHT_EYE = 5
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(
        self,
        canvas_width: int = 640,
        canvas_height: int = 480,
        line_thickness: int = 4,  # Increased line thickness
        head_radius_factor: float = 0.12,  # Increased head radius
        bg_color: Tuple[int, int, int] = (255, 255, 255),  # White background
        figure_color: Tuple[int, int, int] = (0, 0, 0),  # Black stick figure
        chair_color: Tuple[int, int, int] = (150, 75, 0),
        # Chair color (unused, but kept for compatibility)
        smooth_factor: float = 0.3,  # Motion smoothing factor
        smoothing_history: int = 3,  # Number of frames for smoothing history
        logger: Optional[CustomLogger] = None,
    ):
        """
        Initializes the stick figure renderer.

        Args:
            canvas_width: Drawing area width
            canvas_height: Drawing area height
            line_thickness: Stick figure line thickness
            head_radius_factor: Head radius as fraction of height
            bg_color: Background color (BGR)
            figure_color: Stick figure color (BGR)
            chair_color: Chair color (BGR) - unused, kept for compatibility
            smooth_factor: Motion smoothing factor (0.0-1.0)
            smoothing_history: Number of frames for smoothing history
            logger: Logger for recording messages
        """
        # Logger initialization
        self.logger = logger or CustomLogger()

        # Rendering parameters
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.line_thickness = max(2, line_thickness)
        self.head_radius_factor = head_radius_factor
        self.bg_color = bg_color
        self.figure_color = figure_color
        self.chair_color = chair_color  # Kept for compatibility
        self.smooth_factor = smooth_factor
        self.smoothing_history = smoothing_history

        # Calculate head radius
        self.head_radius = int(head_radius_factor * canvas_height)

        # Initialize face renderer
        self.face_renderer = SimpleFaceRenderer(
            feature_color=figure_color, smooth_factor=smooth_factor, logger=self.logger
        )

        # Initialize pose analyzer for upper body analysis
        self.pose_analyzer = PoseAnalyzer(logger=self.logger)

        # Mood state
        self.mood = "happy"  # Default mood

        # Head position - now higher (1/4 instead of 1/3 of height)
        self.head_center = (canvas_width // 2, canvas_height // 4)

        # Landmark detection history - for smoothing
        self.landmark_history: List = []

        # Frame counter - for animation
        self.frame_count = 0
        self.animation_start_time = time.time()

        # Last detected shoulder and arm positions
        self.last_left_shoulder = None
        self.last_right_shoulder = None
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None

        # Visibility tracking flags
        self.left_arm_visible = False
        self.right_arm_visible = False
        self.left_arm_visibility_time = 0
        self.right_arm_visibility_time = 0

        # Arm animation parameters
        self.arms_animation_speed = 0.8  # Slowed animation speed
        self.arms_animation_range = 15  # Increased movement range for idle animation

        # Torso proportion parameters
        self.torso_length_factor = 1.8  # Torso length as head radius multiplier
        self.shoulder_width_factor = 3.2  # Shoulder width as head radius multiplier

        self.logger.info(
            "StickFigureRenderer",
            f"Stick figure renderer initialized ({canvas_width}x{canvas_height}) "
            f"with larger torso (head_radius={self.head_radius})",
            log_type="DRAWING",
        )

    def render(self, face_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Renders stick figure focusing on the upper body (torso).

        If face or hand landmark data is available, uses it for
        face and arm animations. Otherwise animates the figure in default mode.

        Args:
            face_data: Face and hand data from detector

        Returns:
            Image with drawn stick figure
        """
        self.frame_count += 1

        try:
            # Create empty image
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Update arm positions based on data (if available)
            self._update_arm_positions(face_data)

            # Draw stick figure - only upper body
            self._draw_upper_body(canvas)

            # Draw face
            self.face_renderer.draw_face(
                canvas, self.head_center, self.head_radius, self.mood, face_data
            )

            # Log every 300 frames
            if self.frame_count % 300 == 0:
                self.logger.debug(
                    "StickFigureRenderer",
                    f"Rendered {self.frame_count} frames. Shoulder visibility: "
                    f"L={self.left_arm_visible}, R={self.right_arm_visible}",
                    log_type="DRAWING",
                )

            return canvas

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Error during stick figure rendering: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)},
            )
            # Return empty white image in case of error
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

    def _update_arm_positions(self, face_data: Optional[Dict[str, Any]]) -> None:
        """
        Updates shoulder and arm positions based on face_data or idle animation.

        Expanded version that includes shoulder detection.

        Args:
            face_data: Face and hand detector data
        """
        try:
            current_time = time.time()

            # Head center - reference point for the entire figure
            head_x, head_y = self.head_center

            # Shoulder positions - dependent on head center
            shoulder_y = head_y + self.head_radius + int(self.head_radius * 0.3)
            shoulder_width = int(self.head_radius * self.shoulder_width_factor)
            left_shoulder_x = head_x - shoulder_width // 2
            right_shoulder_x = head_x + shoulder_width // 2

            # Default shoulder positions
            default_left_shoulder = (left_shoulder_x, shoulder_y)
            default_right_shoulder = (right_shoulder_x, shoulder_y)

            # Check if we have mediapipe data
            have_landmarks = False
            landmarks = None
            upper_body_data = None

            if face_data and "landmarks" in face_data and face_data["landmarks"]:
                landmarks = face_data["landmarks"]
                have_landmarks = True

                # Analyze upper body if we have landmark data
                if len(landmarks) >= 17:  # Need points up to wrists
                    upper_body_data = self.pose_analyzer.analyze_upper_body(
                        landmarks, self.canvas_width, self.canvas_height
                    )

            # Idle animation parameters for arms
            animation_time = current_time - self.animation_start_time
            idle_animation_factor = (
                math.sin(animation_time * self.arms_animation_speed) * 0.5 + 0.5
            )  # Value 0-1

            # Neutral elbow positions (when no detection)
            neutral_elbow_offset_y = int(self.head_radius * 0.8)
            neutral_elbow_offset_x = int(self.head_radius * 0.7)

            # Neutral wrist positions
            neutral_wrist_offset_y = int(self.head_radius * 0.8)
            neutral_wrist_offset_x = int(self.head_radius * 0.5)

            # Add subtle animation to neutral position
            idle_animation_amount = self.arms_animation_range * idle_animation_factor

            # Check hands_data (backward compatibility)
            hands_data = None
            if face_data and "hands_data" in face_data:
                hands_data = face_data["hands_data"]

            # ===== SHOULDER POSITION UPDATE =====
            left_shoulder_detected = False
            right_shoulder_detected = False

            # Check if we have shoulder data from upper body analysis
            if upper_body_data and upper_body_data["has_shoulders"]:
                shoulders = upper_body_data["shoulder_positions"]
                if shoulders and shoulders[0]:  # Left shoulder
                    left_shoulder_detected = True
                    canvas_x, canvas_y = shoulders[0]

                    # Smooth transition to detected position
                    if self.last_left_shoulder is None:
                        self.last_left_shoulder = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_left_shoulder[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_left_shoulder[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_left_shoulder = (new_x, new_y)

                if shoulders and shoulders[1]:  # Right shoulder
                    right_shoulder_detected = True
                    canvas_x, canvas_y = shoulders[1]

                    # Smooth transition to detected position
                    if self.last_right_shoulder is None:
                        self.last_right_shoulder = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_right_shoulder[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_right_shoulder[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_right_shoulder = (new_x, new_y)

            # If left shoulder not detected, use default position
            if not left_shoulder_detected:
                if self.last_left_shoulder is None:
                    self.last_left_shoulder = default_left_shoulder
                else:
                    # Smooth transition to default position
                    new_x = int(
                        self.last_left_shoulder[0] * self.smooth_factor
                        + default_left_shoulder[0] * (1 - self.smooth_factor)
                    )
                    new_y = int(
                        self.last_left_shoulder[1] * self.smooth_factor
                        + default_left_shoulder[1] * (1 - self.smooth_factor)
                    )
                    self.last_left_shoulder = (new_x, new_y)

            # If right shoulder not detected, use default position
            if not right_shoulder_detected:
                if self.last_right_shoulder is None:
                    self.last_right_shoulder = default_right_shoulder
                else:
                    # Smooth transition to default position
                    new_x = int(
                        self.last_right_shoulder[0] * self.smooth_factor
                        + default_right_shoulder[0] * (1 - self.smooth_factor)
                    )
                    new_y = int(
                        self.last_right_shoulder[1] * self.smooth_factor
                        + default_right_shoulder[1] * (1 - self.smooth_factor)
                    )
                    self.last_right_shoulder = (new_x, new_y)

            # ===== ELBOW POSITION UPDATE =====
            left_elbow_detected = False
            right_elbow_detected = False

            # First check upper body analysis data
            if (
                upper_body_data
                and upper_body_data["has_arms"]
                and upper_body_data["elbow_positions"]
            ):
                elbows = upper_body_data["elbow_positions"]

                # Left elbow
                if elbows[0]:
                    left_elbow_detected = True
                    canvas_x, canvas_y = elbows[0]

                    # If first time elbow detected - record time
                    if not self.left_arm_visible:
                        self.left_arm_visibility_time = current_time
                        self.left_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Left arm detected - starting tracking",
                            log_type="DRAWING",
                        )

                    # Smooth transition to detected position
                    if self.last_left_elbow is None:
                        self.last_left_elbow = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_left_elbow[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_left_elbow[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_left_elbow = (new_x, new_y)

                # Right elbow
                if elbows[1]:
                    right_elbow_detected = True
                    canvas_x, canvas_y = elbows[1]

                    # If first time elbow detected - record time
                    if not self.right_arm_visible:
                        self.right_arm_visibility_time = current_time
                        self.right_arm_visible = True
                        self.logger.debug(
                            "StickFigureRenderer",
                            "Right arm detected - starting tracking",
                            log_type="DRAWING",
                        )

                    # Smooth transition to detected position
                    if self.last_right_elbow is None:
                        self.last_right_elbow = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_right_elbow[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_right_elbow[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_right_elbow = (new_x, new_y)

            # Alternatively, check hands_data (backward compatibility)
            elif (
                hands_data
                and not left_elbow_detected
                and "left_hand" in hands_data
                and hands_data["left_hand"]
            ):
                if "elbow" in hands_data["left_hand"] and hands_data["left_hand"]["elbow"]:
                    left_elbow_detected = True
                    elbow_data = hands_data["left_hand"]["elbow"]
                    canvas_x = int(
                        elbow_data[0] * self.canvas_width if elbow_data[0] <= 1.0 else elbow_data[0]
                    )
                    canvas_y = int(
                        elbow_data[1] * self.canvas_height
                        if elbow_data[1] <= 1.0
                        else elbow_data[1]
                    )

                    if self.last_left_elbow is None:
                        self.last_left_elbow = (canvas_x, canvas_y)
                    else:
                        new_x = int(
                            self.last_left_elbow[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_left_elbow[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_left_elbow = (new_x, new_y)

            if (
                hands_data
                and not right_elbow_detected
                and "right_hand" in hands_data
                and hands_data["right_hand"]
            ):
                if "elbow" in hands_data["right_hand"] and hands_data["right_hand"]["elbow"]:
                    right_elbow_detected = True
                    elbow_data = hands_data["right_hand"]["elbow"]
                    canvas_x = int(
                        elbow_data[0] * self.canvas_width if elbow_data[0] <= 1.0 else elbow_data[0]
                    )
                    canvas_y = int(
                        elbow_data[1] * self.canvas_height
                        if elbow_data[1] <= 1.0
                        else elbow_data[1]
                    )

                    if self.last_right_elbow is None:
                        self.last_right_elbow = (canvas_x, canvas_y)
                    else:
                        new_x = int(
                            self.last_right_elbow[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_right_elbow[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_right_elbow = (new_x, new_y)

            # If left elbow not detected
            if not left_elbow_detected:
                # If previously visible, mark as invisible and record time
                if self.left_arm_visible:
                    self.left_arm_visible = False
                    self.left_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Lost left arm tracking - returning to default animation",
                        log_type="DRAWING",
                    )

                # Time since visibility lost
                time_since_lost = current_time - self.left_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Full animation after 0.5s

                # Neutral left elbow position with animation
                left_shoulder_pos = self.last_left_shoulder or default_left_shoulder
                neutral_left_elbow_x = left_shoulder_pos[0] - neutral_elbow_offset_x
                neutral_left_elbow_y = (
                    left_shoulder_pos[1] + neutral_elbow_offset_y + int(idle_animation_amount)
                )

                # If we have saved last positions, smoothly transition to neutral position
                if self.last_left_elbow is not None:
                    # Linear interpolation between last detected position and neutral position
                    new_x = int(
                        self.last_left_elbow[0] * (1 - transition_factor)
                        + neutral_left_elbow_x * transition_factor
                    )
                    new_y = int(
                        self.last_left_elbow[1] * (1 - transition_factor)
                        + neutral_left_elbow_y * transition_factor
                    )
                    self.last_left_elbow = (new_x, new_y)
                else:
                    # If no history, use neutral position directly
                    self.last_left_elbow = (neutral_left_elbow_x, neutral_left_elbow_y)

            # If right elbow not detected
            if not right_elbow_detected:
                # If previously visible, mark as invisible and record time
                if self.right_arm_visible:
                    self.right_arm_visible = False
                    self.right_arm_visibility_time = current_time
                    self.logger.debug(
                        "StickFigureRenderer",
                        "Lost right arm tracking - returning to default animation",
                        log_type="DRAWING",
                    )

                # Time since visibility lost
                time_since_lost = current_time - self.right_arm_visibility_time
                transition_factor = min(1.0, time_since_lost * 2.0)  # Full animation after 0.5s

                # Neutral right elbow position with animation
                right_shoulder_pos = self.last_right_shoulder or default_right_shoulder
                neutral_right_elbow_x = right_shoulder_pos[0] + neutral_elbow_offset_x
                neutral_right_elbow_y = (
                    right_shoulder_pos[1] + neutral_elbow_offset_y + int(idle_animation_amount)
                )

                # If we have saved last positions, smoothly transition to neutral position
                if self.last_right_elbow is not None:
                    # Linear interpolation between last detected position and neutral position
                    new_x = int(
                        self.last_right_elbow[0] * (1 - transition_factor)
                        + neutral_right_elbow_x * transition_factor
                    )
                    new_y = int(
                        self.last_right_elbow[1] * (1 - transition_factor)
                        + neutral_right_elbow_y * transition_factor
                    )
                    self.last_right_elbow = (new_x, new_y)
                else:
                    # If no history, use neutral position directly
                    self.last_right_elbow = (neutral_right_elbow_x, neutral_right_elbow_y)

            # ===== WRIST POSITION UPDATE =====
            # This can be handled more simply as wrists are less critical
            left_wrist_detected = False
            right_wrist_detected = False

            # First check upper body analysis data
            if upper_body_data and upper_body_data["wrist_positions"]:
                wrists = upper_body_data["wrist_positions"]

                # Left wrist
                if wrists[0]:
                    left_wrist_detected = True
                    canvas_x, canvas_y = wrists[0]

                    # Smooth transition to detected position
                    if self.last_left_wrist is None:
                        self.last_left_wrist = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_left_wrist[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_left_wrist[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_left_wrist = (new_x, new_y)

                # Right wrist
                if wrists[1]:
                    right_wrist_detected = True
                    canvas_x, canvas_y = wrists[1]

                    # Smooth transition to detected position
                    if self.last_right_wrist is None:
                        self.last_right_wrist = (canvas_x, canvas_y)
                    else:
                        # Motion smoothing
                        new_x = int(
                            self.last_right_wrist[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_right_wrist[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_right_wrist = (new_x, new_y)

            # Alternatively, check hands_data (backward compatibility)
            elif (
                hands_data
                and not left_wrist_detected
                and "left_hand" in hands_data
                and hands_data["left_hand"]
            ):
                if "wrist" in hands_data["left_hand"] and hands_data["left_hand"]["wrist"]:
                    left_wrist_detected = True
                    wrist_data = hands_data["left_hand"]["wrist"]
                    canvas_x = int(
                        wrist_data[0] * self.canvas_width if wrist_data[0] <= 1.0 else wrist_data[0]
                    )
                    canvas_y = int(
                        wrist_data[1] * self.canvas_height
                        if wrist_data[1] <= 1.0
                        else wrist_data[1]
                    )

                    if self.last_left_wrist is None:
                        self.last_left_wrist = (canvas_x, canvas_y)
                    else:
                        new_x = int(
                            self.last_left_wrist[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_left_wrist[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_left_wrist = (new_x, new_y)

            if (
                hands_data
                and not right_wrist_detected
                and "right_hand" in hands_data
                and hands_data["right_hand"]
            ):
                if "wrist" in hands_data["right_hand"] and hands_data["right_hand"]["wrist"]:
                    right_wrist_detected = True
                    wrist_data = hands_data["right_hand"]["wrist"]
                    canvas_x = int(
                        wrist_data[0] * self.canvas_width if wrist_data[0] <= 1.0 else wrist_data[0]
                    )
                    canvas_y = int(
                        wrist_data[1] * self.canvas_height
                        if wrist_data[1] <= 1.0
                        else wrist_data[1]
                    )

                    if self.last_right_wrist is None:
                        self.last_right_wrist = (canvas_x, canvas_y)
                    else:
                        new_x = int(
                            self.last_right_wrist[0] * self.smooth_factor
                            + canvas_x * (1 - self.smooth_factor)
                        )
                        new_y = int(
                            self.last_right_wrist[1] * self.smooth_factor
                            + canvas_y * (1 - self.smooth_factor)
                        )
                        self.last_right_wrist = (new_x, new_y)

            # If left wrist not detected, calculate its position based on elbow
            if not left_wrist_detected and self.last_left_elbow is not None:
                left_elbow_pos = self.last_left_elbow

                # Wrist position below elbow - continuation of shoulder-elbow direction
                elbow_to_wrist_vector_x = -neutral_wrist_offset_x
                elbow_to_wrist_vector_y = neutral_wrist_offset_y - int(idle_animation_amount)

                # Calculate wrist position
                left_wrist_x = left_elbow_pos[0] + elbow_to_wrist_vector_x
                left_wrist_y = left_elbow_pos[1] + elbow_to_wrist_vector_y

                # Update wrist position with smoothing
                if self.last_left_wrist is None:
                    self.last_left_wrist = (left_wrist_x, left_wrist_y)
                else:
                    new_x = int(
                        self.last_left_wrist[0] * self.smooth_factor
                        + left_wrist_x * (1 - self.smooth_factor)
                    )
                    new_y = int(
                        self.last_left_wrist[1] * self.smooth_factor
                        + left_wrist_y * (1 - self.smooth_factor)
                    )
                    self.last_left_wrist = (new_x, new_y)

            # If right wrist not detected, calculate its position based on elbow
            if not right_wrist_detected and self.last_right_elbow is not None:
                right_elbow_pos = self.last_right_elbow

                # Wrist position below elbow - continuation of shoulder-elbow direction
                elbow_to_wrist_vector_x = neutral_wrist_offset_x
                elbow_to_wrist_vector_y = neutral_wrist_offset_y - int(idle_animation_amount)

                # Calculate wrist position
                right_wrist_x = right_elbow_pos[0] + elbow_to_wrist_vector_x
                right_wrist_y = right_elbow_pos[1] + elbow_to_wrist_vector_y

                # Update wrist position with smoothing
                if self.last_right_wrist is None:
                    self.last_right_wrist = (right_wrist_x, right_wrist_y)
                else:
                    new_x = int(
                        self.last_right_wrist[0] * self.smooth_factor
                        + right_wrist_x * (1 - self.smooth_factor)
                    )
                    new_y = int(
                        self.last_right_wrist[1] * self.smooth_factor
                        + right_wrist_y * (1 - self.smooth_factor)
                    )
                    self.last_right_wrist = (new_x, new_y)

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Error during arm position update: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)},
            )

    def _draw_upper_body(self, canvas: np.ndarray) -> None:
        """
        Draws the upper body (torso) of the stick figure.

        This method focuses exclusively on drawing the head, torso and shoulders,
        omitting legs and other lower body elements.

        Args:
            canvas: Drawing canvas
        """
        try:
            # 1. Head - circle
            cv2.circle(
                canvas, self.head_center, self.head_radius, self.figure_color, self.line_thickness
            )

            # 2. Calculate torso and shoulder positions relative to head
            # Head center
            head_x, head_y = self.head_center

            # Chest - below head (shorter than in previous version)
            torso_top_y = head_y + self.head_radius + 5
            torso_length = int(self.head_radius * self.torso_length_factor)
            torso_bottom_y = torso_top_y + torso_length

            # 3. Draw torso (vertical line from head down) - shorter torso
            cv2.line(
                canvas,
                (head_x, torso_top_y),
                (head_x, torso_bottom_y),
                self.figure_color,
                self.line_thickness,
            )

            # 4. Draw shoulder line if we have detected shoulder positions
            if self.last_left_shoulder and self.last_right_shoulder:
                cv2.line(
                    canvas,
                    self.last_left_shoulder,
                    self.last_right_shoulder,
                    self.figure_color,
                    self.line_thickness,
                )
            else:
                # Default shoulder line drawing if no detected positions
                shoulder_y = torso_top_y + int(self.head_radius * 0.3)
                shoulder_width = int(self.head_radius * self.shoulder_width_factor)
                left_shoulder_x = head_x - shoulder_width // 2
                right_shoulder_x = head_x + shoulder_width // 2

                cv2.line(
                    canvas,
                    (left_shoulder_x, shoulder_y),
                    (right_shoulder_x, shoulder_y),
                    self.figure_color,
                    self.line_thickness,
                )

                # Update default shoulder positions
                self.last_left_shoulder = (left_shoulder_x, shoulder_y)
                self.last_right_shoulder = (right_shoulder_x, shoulder_y)

            # 5. Draw arms (shoulders -> elbows -> wrists)
            # Left arm - from shoulder to elbow
            if self.last_left_shoulder and self.last_left_elbow:
                cv2.line(
                    canvas,
                    self.last_left_shoulder,
                    self.last_left_elbow,
                    self.figure_color,
                    self.line_thickness,
                )

                # Left forearm - from elbow to wrist
                if self.last_left_wrist:
                    cv2.line(
                        canvas,
                        self.last_left_elbow,
                        self.last_left_wrist,
                        self.figure_color,
                        self.line_thickness,
                    )

            # Right arm - from shoulder to elbow
            if self.last_right_shoulder and self.last_right_elbow:
                cv2.line(
                    canvas,
                    self.last_right_shoulder,
                    self.last_right_elbow,
                    self.figure_color,
                    self.line_thickness,
                )

                # Right forearm - from elbow to wrist
                if self.last_right_wrist:
                    cv2.line(
                        canvas,
                        self.last_right_elbow,
                        self.last_right_wrist,
                        self.figure_color,
                        self.line_thickness,
                    )

        except Exception as e:
            self.logger.error(
                "StickFigureRenderer",
                f"Error during torso drawing: {str(e)}",
                log_type="DRAWING",
                error={"error": str(e)},
            )

    def set_colors(
        self,
        bg_color: Optional[Tuple[int, int, int]] = None,
        figure_color: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Updates colors used for drawing.

        Args:
            bg_color: New background color (BGR)
            figure_color: New stick figure color (BGR)
        """
        if bg_color is not None:
            self.bg_color = bg_color
            self.logger.debug(
                "StickFigureRenderer", f"Changed background color to {bg_color}", log_type="DRAWING"
            )

        if figure_color is not None:
            self.figure_color = figure_color
            # Update color in face renderer
            self.face_renderer.feature_color = figure_color
            self.logger.debug(
                "StickFigureRenderer",
                f"Changed figure color to {figure_color}",
                log_type="DRAWING",
            )

    def set_mood(self, mood: str) -> None:
        """
        Sets the stick figure mood which affects facial expression.

        Args:
            mood: Mood: "happy", "sad", "neutral", "surprised", "wink"
        """
        valid_moods = ["happy", "sad", "neutral", "surprised", "wink"]
        if mood in valid_moods:
            self.mood = mood
            self.logger.info("StickFigureRenderer", f"Changed mood to: {mood}", log_type="DRAWING")
        else:
            self.logger.warning(
                "StickFigureRenderer",
                f"Invalid mood: {mood}. Allowed values: {valid_moods}",
                log_type="DRAWING",
            )

    def set_line_thickness(self, thickness: int) -> None:
        """
        Updates line thickness.

        Args:
            thickness: New line thickness
        """
        self.line_thickness = max(1, thickness)
        self.logger.debug(
            "StickFigureRenderer",
            f"Changed line thickness to {self.line_thickness}",
            log_type="DRAWING",
        )

    def set_smoothing(self, smooth_factor: float, history_length: int) -> None:
        """
        Updates smoothing parameters.

        Args:
            smooth_factor: New smoothing factor (0.0-1.0)
            history_length: New smoothing history length
        """
        self.smooth_factor = max(0.0, min(1.0, smooth_factor))
        self.smoothing_history = max(1, history_length)
        self.logger.debug(
            "StickFigureRenderer",
            f"Updated smoothing parameters: factor={self.smooth_factor}, "
            f"history length={self.smoothing_history}",
            log_type="DRAWING",
        )

    def resize(self, width: int, height: int) -> None:
        """
        Resizes the canvas.

        Args:
            width: New width
            height: New height
        """
        self.canvas_width = width
        self.canvas_height = height

        # Update head radius
        self.head_radius = int(self.head_radius_factor * height)

        # Update head position to screen center and higher (1/4 of height)
        self.head_center = (width // 2, height // 4)

        # Reset history - because we're changing scale
        self.landmark_history = []
        self.last_left_shoulder = None
        self.last_right_shoulder = None
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None

        self.logger.info(
            "StickFigureRenderer",
            f"Changed canvas size to {width}x{height}",
            log_type="DRAWING",
        )

    def reset(self) -> None:
        """
        Resets the renderer's internal state.
        """
        self.mood = "happy"  # Restore default mood
        self.landmark_history = []
        self.last_left_shoulder = None
        self.last_right_shoulder = None
        self.last_left_elbow = None
        self.last_right_elbow = None
        self.last_left_wrist = None
        self.last_right_wrist = None
        self.left_arm_visible = False
        self.right_arm_visible = False
        self.animation_start_time = time.time()
        self.face_renderer.reset()

        self.logger.info("StickFigureRenderer", "Reset renderer state", log_type="DRAWING")

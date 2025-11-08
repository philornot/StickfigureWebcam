#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/drawing/face_renderer.py

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class SimpleFaceRenderer:
    """
    Simplified face renderer for stick figures with basic facial expressions.
    Provides smooth transitions between facial expressions and increased sensitivity to expression changes.
    """

    def __init__(
        self,
        feature_color: Tuple[int, int, int] = (0, 0, 0),  # Black
        smooth_factor: float = 0.3,
        logger: Optional[CustomLogger] = None,
    ):
        """
        Initializes face renderer.

        Args:
            feature_color (Tuple[int, int, int]): Color of facial features (BGR)
            smooth_factor (float): Movement smoothing factor (0.0-1.0)
            logger (Optional[CustomLogger]): Logger for recording messages
        """
        # Initialize logger
        self.logger = logger or CustomLogger()

        # Rendering parameters
        self.feature_color = feature_color
        self.smooth_factor = smooth_factor

        # Buffer for smoothing facial expressions
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.5,  # 0.5 is neutral smile
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0,
        }

        # Add hysteresis to avoid switching between smile and sad
        self.smile_histeresis = 0.05  # Reduced hysteresis margin for greater sensitivity

        # Frame counter for logging
        self.frame_count = 0

        # Neutral expression range (reduced for greater sensitivity)
        self.neutral_lower = 0.45
        self.neutral_upper = 0.55

        # Add filter for averaging last N facial expressions for greater smoothness
        self.expressions_history: List[Dict[str, float]] = []
        self.expressions_history_size = 5  # Number of frames to average

        self.logger.info(
            "SimpleFaceRenderer", "Face renderer initialized", log_type="DRAWING"
        )

    def draw_face(
        self,
        canvas: np.ndarray,
        head_center: Tuple[int, int],
        head_radius: int,
        mood: str = "neutral",
        face_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Draws face on stick figure.

        Args:
            canvas (np.ndarray): Canvas for drawing
            head_center (Tuple[int, int]): Head center coordinates (x, y)
            head_radius (int): Head radius
            mood (str): Mood to express ("happy", "sad", "neutral", "surprised", "wink")
            face_data (Optional[Dict[str, Any]]): Face data from face detector (if available)
        """
        self.frame_count += 1

        try:
            # Head center
            cx, cy = head_center

            # If we have face data, use it for animation
            if face_data and "expressions" in face_data:
                expressions = face_data["expressions"]

                # Check if values are reasonable
                for key, value in expressions.items():
                    if not (0.0 <= value <= 1.0):
                        # If value is outside 0-1 range, use last valid value
                        expressions[key] = self.last_expressions.get(key, 0.5)

                # Add current expressions to history
                self.expressions_history.append(expressions)
                if len(self.expressions_history) > self.expressions_history_size:
                    self.expressions_history.pop(0)

                # Average expressions from history for greater smoothness
                avg_expressions = self._average_expressions()

                # Then smooth facial expressions
                smoothed_expressions = self._smooth_expressions(avg_expressions)

                # Then apply hysteresis for smile to avoid switching
                smile_value = smoothed_expressions["smile"]

                # Update mood value based on smile with hysteresis
                current_mood = self._determine_mood_from_smile(smile_value)

                # If there's no forced mood (when mood is not one of predefined),
                # use mood determined from facial expressions
                if mood not in ["happy", "sad", "neutral", "surprised", "wink"]:
                    mood = current_mood

                # Draw face considering facial expressions
                self._draw_face_with_expressions(
                    canvas, head_center, head_radius, smoothed_expressions, mood
                )
            else:
                # Otherwise draw simple face based on mood
                self._draw_mood_face(canvas, head_center, head_radius, mood)

            # Add logging every 500 frames
            if self.frame_count % 500 == 0:
                self.logger.debug(
                    "SimpleFaceRenderer",
                    f"Rendered {self.frame_count} face frames, current mood: {mood}",
                    log_type="DRAWING",
                )

        except Exception as e:
            self.logger.error(
                "SimpleFaceRenderer", f"Error drawing face: {str(e)}", log_type="DRAWING"
            )

    def _average_expressions(self) -> Dict[str, float]:
        """
        Averages facial expressions from history.

        Returns:
            Dict[str, float]: Averaged values
        """
        if not self.expressions_history:
            return self.last_expressions

        # Initialize dictionary with keys of all expressions
        avg_expressions = {key: 0.0 for key in self.last_expressions.keys()}

        # Sum all values
        for expressions in self.expressions_history:
            for key in avg_expressions.keys():
                if key in expressions:
                    avg_expressions[key] += expressions[key]

        # Divide by number of samples
        history_size = len(self.expressions_history)
        for key in avg_expressions.keys():
            avg_expressions[key] /= history_size

        return avg_expressions

    def _determine_mood_from_smile(self, smile_value: float) -> str:
        """
        Determines mood based on smile value with hysteresis.

        Args:
            smile_value (float): Smile value (0.0-1.0)

        Returns:
            str: Mood ("happy", "sad", "neutral")
        """
        # Current mood based on last smile value
        current_mood = "neutral"

        if self.last_expressions["smile"] > self.neutral_upper + self.smile_histeresis:
            current_mood = "happy"
        elif self.last_expressions["smile"] < self.neutral_lower - self.smile_histeresis:
            current_mood = "sad"

        # Check if smile value changed significantly
        if smile_value > self.neutral_upper + self.smile_histeresis:
            # Clear smile
            current_mood = "happy"
        elif smile_value < self.neutral_lower - self.smile_histeresis:
            # Clear sadness
            current_mood = "sad"
        elif self.neutral_lower <= smile_value <= self.neutral_upper:
            # Neutral expression
            current_mood = "neutral"

        return current_mood

    def _smooth_expressions(self, expressions: Dict[str, float]) -> Dict[str, float]:
        """
        Smooths facial expressions between frames.

        Args:
            expressions (Dict[str, float]): New facial expression values

        Returns:
            Dict[str, float]: Smoothed values
        """
        smoothed = {}

        for key, value in expressions.items():
            if key in self.last_expressions:
                # Smooth facial expressions using smooth_factor
                smoothed[key] = self.last_expressions[key] * self.smooth_factor + value * (
                    1 - self.smooth_factor
                )
            else:
                smoothed[key] = value

        # Save for next frame
        self.last_expressions = smoothed

        return smoothed

    def _draw_face_with_expressions(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        expressions: Dict[str, float],
        mood: str,
    ) -> None:
        """
        Draws face based on facial expression values.

        Args:
            canvas (np.ndarray): Canvas for drawing
            center (Tuple[int, int]): Head center (x, y)
            radius (int): Head radius
            expressions (Dict[str, float]): Facial expression values
            mood (str): Current facial mood ("happy", "sad", "neutral")
        """
        # Head center
        cx, cy = center

        # Get facial expression values
        mouth_open = expressions.get("mouth_open", 0.0)
        smile = expressions.get("smile", 0.5)  # Default smile
        left_eye_open = expressions.get("left_eye_open", 1.0)
        right_eye_open = expressions.get("right_eye_open", 1.0)

        # Draw eyes
        eye_offset_x = int(radius * 0.3)
        eye_offset_y = int(radius * 0.2)
        eye_size = max(2, int(radius * 0.1))

        # Left eye
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        if left_eye_open > 0.3:
            cv2.circle(canvas, left_eye_pos, eye_size, self.feature_color, -1)
        else:
            cv2.line(
                canvas,
                (left_eye_pos[0] - eye_size, left_eye_pos[1]),
                (left_eye_pos[0] + eye_size, left_eye_pos[1]),
                self.feature_color,
                2,
            )

        # Right eye
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        if right_eye_open > 0.3:
            cv2.circle(canvas, right_eye_pos, eye_size, self.feature_color, -1)
        else:
            cv2.line(
                canvas,
                (right_eye_pos[0] - eye_size, right_eye_pos[1]),
                (right_eye_pos[0] + eye_size, right_eye_pos[1]),
                self.feature_color,
                2,
            )

        # Draw mouth
        mouth_y = cy + int(radius * 0.2)
        mouth_width = int(radius * 0.6)
        mouth_height = int(radius * 0.3)

        # Log every 100 frames
        if self.frame_count % 100 == 0:
            self.logger.debug(
                "SimpleFaceRenderer",
                f"Facial expressions: smile={smile:.2f}, mouth_open={mouth_open:.2f}, mood={mood}",
                log_type="DRAWING",
            )

        if mouth_open > 0.2:
            # Open mouth - ellipse
            mouth_open_height = int(mouth_height * min(2.5, 1.0 + mouth_open * 2))
            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_open_height),
                0,  # angle
                0,
                360,  # full ellipse
                self.feature_color,
                2,  # thickness
            )
        else:
            # Closed mouth - different types depending on mood
            if mood == "happy":
                # Smile - downward arc
                cv2.ellipse(
                    canvas,
                    (cx, mouth_y),
                    (mouth_width, mouth_height),
                    0,  # angle
                    0,
                    180,  # downward arc
                    self.feature_color,
                    2,  # thickness
                )
            elif mood == "sad":
                # Sadness - upward arc
                cv2.ellipse(
                    canvas,
                    (cx, mouth_y + mouth_height // 2),
                    (mouth_width, mouth_height),
                    0,  # angle
                    180,
                    360,  # upward arc
                    self.feature_color,
                    2,  # thickness
                )
            else:  # Neutral
                # Neutral - straight line
                cv2.line(
                    canvas,
                    (cx - mouth_width // 2, mouth_y),
                    (cx + mouth_width // 2, mouth_y),
                    self.feature_color,
                    2,  # thickness
                )

    def _draw_mood_face(
        self, canvas: np.ndarray, center: Tuple[int, int], radius: int, mood: str
    ) -> None:
        """
        Draws face with specified mood.

        Args:
            canvas (np.ndarray): Canvas for drawing
            center (Tuple[int, int]): Head center (x, y)
            radius (int): Head radius
            mood (str): Mood ("happy", "sad", "neutral", "surprised", "wink")
        """
        # Head center
        cx, cy = center

        # Draw eyes
        eye_offset_x = int(radius * 0.3)
        eye_offset_y = int(radius * 0.2)
        eye_size = max(2, int(radius * 0.1))

        # Left eye - exception for wink
        left_eye_pos = (cx - eye_offset_x, cy - eye_offset_y)
        if mood == "wink":
            cv2.line(
                canvas,
                (left_eye_pos[0] - eye_size, left_eye_pos[1]),
                (left_eye_pos[0] + eye_size, left_eye_pos[1]),
                self.feature_color,
                2,
            )
        else:
            cv2.circle(canvas, left_eye_pos, eye_size, self.feature_color, -1)

        # Right eye
        right_eye_pos = (cx + eye_offset_x, cy - eye_offset_y)
        cv2.circle(canvas, right_eye_pos, eye_size, self.feature_color, -1)

        # Draw mouth depending on mood
        mouth_y = cy + int(radius * 0.2)
        mouth_width = int(radius * 0.6)
        mouth_height = int(radius * 0.3)

        if mood == "happy" or mood == "wink":
            # Smile
            cv2.ellipse(
                canvas,
                (cx, mouth_y),
                (mouth_width, mouth_height),
                0,  # angle
                0,
                180,  # downward arc
                self.feature_color,
                2,  # thickness
            )
        elif mood == "sad":
            # Sadness - inverted arc
            cv2.ellipse(
                canvas,
                (cx, mouth_y + mouth_height // 2),
                (mouth_width, mouth_height),
                0,  # angle
                180,
                360,  # upward arc
                self.feature_color,
                2,  # thickness
            )
        elif mood == "surprised":
            # Surprise - open mouth (circle)
            cv2.circle(
                canvas, (cx, mouth_y), int(mouth_width * 0.4), self.feature_color, 2  # thickness
            )
        else:  # neutral
            # Neutral - straight line
            cv2.line(
                canvas,
                (cx - mouth_width // 2, mouth_y),
                (cx + mouth_width // 2, mouth_y),
                self.feature_color,
                2,  # thickness
            )

    def reset(self) -> None:
        """
        Resets internal renderer state.
        """
        self.last_expressions = {
            "mouth_open": 0.0,
            "smile": 0.5,  # Neutral value
            "left_eye_open": 1.0,
            "right_eye_open": 1.0,
            "eyebrow_raised": 0.0,
            "surprise": 0.0,
        }
        self.expressions_history = []
        self.frame_count = 0

        self.logger.debug(
            "SimpleFaceRenderer", "Face renderer state reset", log_type="DRAWING"
        )

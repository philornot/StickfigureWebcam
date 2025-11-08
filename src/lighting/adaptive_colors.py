#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/lighting/adaptive_colors.py
"""
Lighting analysis and adaptive color adjustment module.
Ensures that background and stick figure colors adapt to environmental lighting conditions
while maintaining consistent contrast levels.
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.custom_logger import CustomLogger


class AdaptiveLightingManager:
    """
    Class managing adaptive color adjustment to lighting conditions.
    """

    def __init__(
        self,
        adaptation_speed: float = 0.02,  # Very slow adaptation (0.0-1.0)
        smoothing_window: int = 30,  # Smoothing window (number of frames)
        min_brightness: int = 20,  # Minimum background brightness (0-255)
        max_brightness: int = 250,  # Maximum background brightness (0-255)
        min_contrast: float = 0.4,  # Minimum contrast (0.0-1.0)
        sampling_interval: int = 5,  # How often to sample frames (CPU saving)
        logger: Optional[CustomLogger] = None,
    ):
        """
        Initializes adaptive lighting manager.

        Args:
            adaptation_speed (float): Color adaptation speed (0.0-1.0)
            smoothing_window (int): Number of frames used for smoothing changes
            min_brightness (int): Minimum background brightness (0-255)
            max_brightness (int): Maximum background brightness (0-255)
            min_contrast (float): Minimum contrast between background and contours (0.0-1.0)
            sampling_interval (int): How often to analyze brightness (CPU saving)
            logger (CustomLogger, optional): Logger for recording messages
        """
        self.logger = logger or CustomLogger()

        # Adaptation parameters
        self.adaptation_speed = max(0.001, min(0.1, adaptation_speed))  # Limit range
        self.smoothing_window = smoothing_window
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.sampling_interval = max(1, sampling_interval)

        # Internal state
        self.brightness_history: List[float] = []
        self.current_frame_index = 0
        self.current_bg_color = (255, 255, 255)  # Default white background (BGR)
        self.current_figure_color = (0, 0, 0)  # Default black contour (BGR)
        self.target_bg_brightness = 255
        self.last_update_time = time.time()

        self.logger.info(
            "AdaptiveLighting",
            f"Initializing adaptive lighting manager (speed={adaptation_speed}, "
            f"window={smoothing_window}, min_brightness={min_brightness})",
            log_type="LIGHTING",
        )

    def analyze_frame(self, frame: np.ndarray) -> float:
        """
        Analyzes video frame brightness.

        Args:
            frame (np.ndarray): Input frame (BGR)

        Returns:
            float: Average frame brightness (0.0-1.0)
        """
        # Increment frame counter
        self.current_frame_index += 1

        # Analyze only every X-th frame for CPU saving
        if self.current_frame_index % self.sampling_interval != 0:
            # Return last known value if available
            if self.brightness_history:
                return self.brightness_history[-1]
            return 0.5  # Default value

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate average brightness (normalized to 0.0-1.0)
            mean_brightness = np.mean(gray) / 255.0

            # Add to history
            self.brightness_history.append(mean_brightness)

            # Limit history size
            if len(self.brightness_history) > self.smoothing_window:
                self.brightness_history.pop(0)

            # Smoothed value (average from history)
            smoothed_brightness = np.mean(self.brightness_history)

            return smoothed_brightness

        except Exception as e:
            self.logger.error(
                "AdaptiveLighting",
                f"Error during brightness analysis: {str(e)}",
                log_type="LIGHTING",
            )
            # Return last known value, or default value
            if self.brightness_history:
                return self.brightness_history[-1]
            return 0.5

    def update_colors(
        self, frame_brightness: float
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Updates background and contour colors based on environmental brightness.
        Maintains constant contrast between background and contour.

        Args:
            frame_brightness (float): Environmental brightness (0.0-1.0)

        Returns:
            Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
                - background color (BGR)
                - contour color (BGR)
        """
        # Calculate target background brightness based on environmental brightness
        # Invert scale - brighter environment means brighter background
        target_bg_value = self.min_brightness + frame_brightness * (
            self.max_brightness - self.min_brightness
        )

        # Smooth adaptation to target value
        current_time = time.time()
        elapsed_time = current_time - self.last_update_time
        adaptation_factor = min(1.0, elapsed_time * self.adaptation_speed * 10)

        # Update target brightness considering adaptation factor
        self.target_bg_brightness = (
            self.target_bg_brightness * (1 - adaptation_factor)
            + target_bg_value * adaptation_factor
        )

        # Limit to range
        self.target_bg_brightness = max(
            self.min_brightness, min(self.max_brightness, self.target_bg_brightness)
        )

        # Convert to BGR color
        bg_color_value = int(self.target_bg_brightness)
        bg_color = (bg_color_value, bg_color_value, bg_color_value)  # Grayscale shade (BGR)

        # Calculate contour color - inverse of background, but maintaining contrast
        # Brighter background means darker contour and vice versa
        contour_value = self._calculate_contrasting_value(bg_color_value)
        figure_color = (contour_value, contour_value, contour_value)  # Grayscale shade (BGR)

        # Update state
        self.current_bg_color = bg_color
        self.current_figure_color = figure_color
        self.last_update_time = current_time

        return bg_color, figure_color

    def _calculate_contrasting_value(self, bg_value: int) -> int:
        """
        Calculates contour color value contrasting with background.

        Args:
            bg_value (int): Background brightness value (0-255)

        Returns:
            int: Contour brightness value (0-255)
        """
        # Midpoint of brightness scale
        mid_point = 127.5

        # Calculate relative background brightness (-1.0 to 1.0 where 0 is scale midpoint)
        relative_brightness = (bg_value - mid_point) / mid_point

        # Invert and scale, maintaining minimum contrast
        contrast_factor = max(self.min_contrast, abs(relative_brightness) + self.min_contrast)

        if bg_value > mid_point:
            # For bright background - dark contour
            contour_value = max(0, int(bg_value * (1.0 - contrast_factor)))
        else:
            # For dark background - bright contour
            contour_value = min(255, int(bg_value * (1.0 + contrast_factor)))

        return contour_value

    def get_current_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Returns currently used colors.

        Returns:
            Dict[str, Tuple[int, int, int]]: Dictionary with colors
        """
        return {
            "bg_color": self.current_bg_color,
            "figure_color": self.current_figure_color,
            "brightness_level": self.target_bg_brightness / 255.0,
        }

    def reset(self) -> None:
        """
        Resets manager state.
        """
        self.brightness_history = []
        self.current_frame_index = 0
        self.current_bg_color = (255, 255, 255)
        self.current_figure_color = (0, 0, 0)
        self.target_bg_brightness = 255
        self.last_update_time = time.time()

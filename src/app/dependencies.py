#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dependency injection container for the application.

This module provides a simple dependency injection container that manages
the creation and lifecycle of application components.
"""

from typing import Any, Dict, Optional

import mediapipe as mp

from src.drawing.face_renderer import SimpleFaceRenderer
from src.drawing.pose_analyzer import PoseAnalyzer
from src.drawing.stick_figure_renderer import StickFigureRenderer
from src.utils.custom_logger import CustomLogger


class DependencyContainer:
    """Simple dependency injection container.

    Manages creation and caching of application dependencies with
    proper initialization order and configuration injection.

    Example:
        >>> container = DependencyContainer(config, logger)
        >>> renderer = container.get_renderer()
        >>> analyzer = container.get_pose_analyzer()
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[CustomLogger] = None):
        """Initialize the dependency container.

        Args:
            config: Application configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or CustomLogger()

        # Cache for singleton instances
        self._cache: Dict[str, Any] = {}

        self.logger.debug("DI", "Dependency container initialized")

    def get_logger(self) -> CustomLogger:
        """Get logger instance.

        Returns:
            Logger instance
        """
        return self.logger

    def get_pose_analyzer(self) -> PoseAnalyzer:
        """Get or create PoseAnalyzer instance.

        Returns:
            PoseAnalyzer instance
        """
        if "pose_analyzer" not in self._cache:
            self._cache["pose_analyzer"] = PoseAnalyzer(
                sitting_threshold=self.config.get("posture_analyzer.standing_hip_threshold", 0.7),
                logger=self.logger,
            )
            self.logger.debug("DI", "Created PoseAnalyzer")

        return self._cache["pose_analyzer"]

    def get_face_renderer(self) -> SimpleFaceRenderer:
        """Get or create SimpleFaceRenderer instance.

        Returns:
            SimpleFaceRenderer instance
        """
        if "face_renderer" not in self._cache:
            figure_color = tuple(self.config.get("stick_figure.figure_color", [0, 0, 0]))
            smooth_factor = self.config.get("stick_figure.smooth_factor", 0.3)

            self._cache["face_renderer"] = SimpleFaceRenderer(
                feature_color=figure_color, smooth_factor=smooth_factor, logger=self.logger
            )
            self.logger.debug("DI", "Created SimpleFaceRenderer")

        return self._cache["face_renderer"]

    def get_renderer(self) -> StickFigureRenderer:
        """Get or create StickFigureRenderer instance with dependencies.

        Returns:
            StickFigureRenderer instance
        """
        if "renderer" not in self._cache:
            # Get dependencies
            pose_analyzer = self.get_pose_analyzer()
            face_renderer = self.get_face_renderer()

            # Get configuration
            width = self.config.get("camera.width", 640)
            height = self.config.get("camera.height", 480)
            line_thickness = self.config.get("stick_figure.line_thickness", 3)
            head_radius_factor = self.config.get("stick_figure.head_radius_factor", 0.075)
            bg_color = tuple(self.config.get("stick_figure.bg_color", [255, 255, 255]))
            figure_color = tuple(self.config.get("stick_figure.figure_color", [0, 0, 0]))
            smooth_factor = self.config.get("stick_figure.smooth_factor", 0.3)

            # Create renderer with injected dependencies
            self._cache["renderer"] = StickFigureRenderer(
                canvas_width=width,
                canvas_height=height,
                line_thickness=line_thickness,
                head_radius_factor=head_radius_factor,
                bg_color=bg_color,
                figure_color=figure_color,
                smooth_factor=smooth_factor,
                logger=self.logger,
            )

            # Manually inject dependencies (since we're refactoring)
            self._cache["renderer"].pose_analyzer = pose_analyzer
            self._cache["renderer"].face_renderer = face_renderer

            self.logger.debug("DI", "Created StickFigureRenderer with dependencies")

        return self._cache["renderer"]

    def get_face_mesh(self) -> Any:
        """Get or create MediaPipe FaceMesh instance.

        Returns:
            FaceMesh instance
        """
        if "face_mesh" not in self._cache:
            mp_face_mesh = mp.solutions.face_mesh

            self._cache["face_mesh"] = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.config.get(
                    "pose_detection.min_detection_confidence", 0.5
                ),
                min_tracking_confidence=self.config.get(
                    "pose_detection.min_tracking_confidence", 0.5
                ),
            )
            self.logger.debug("DI", "Created MediaPipe FaceMesh")

        return self._cache["face_mesh"]

    def get_hands(self) -> Any:
        """Get or create MediaPipe Hands instance.

        Returns:
            Hands instance
        """
        if "hands" not in self._cache:
            mp_hands = mp.solutions.hands

            self._cache["hands"] = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.logger.debug("DI", "Created MediaPipe Hands")

        return self._cache["hands"]

    def cleanup(self):
        """Cleanup all cached instances that need cleanup."""
        self.logger.debug("DI", "Cleaning up dependencies")

        # Close MediaPipe instances
        if "face_mesh" in self._cache:
            try:
                self._cache["face_mesh"].close()
            except Exception as e:
                self.logger.warning("DI", f"Failed to close face_mesh: {e}")

        if "hands" in self._cache:
            try:
                self._cache["hands"].close()
            except Exception as e:
                self.logger.warning("DI", f"Failed to close hands: {e}")

        self._cache.clear()
        self.logger.debug("DI", "Cleanup complete")


def create_container(
    config: Dict[str, Any], logger: Optional[CustomLogger] = None
) -> DependencyContainer:
    """Factory function to create a configured dependency container.

    Args:
        config: Application configuration
        logger: Optional logger instance

    Returns:
        Configured DependencyContainer

    Example:
        >>> config = load_config()
        >>> container = create_container(config)
        >>> renderer = container.get_renderer()
    """
    return DependencyContainer(config, logger)

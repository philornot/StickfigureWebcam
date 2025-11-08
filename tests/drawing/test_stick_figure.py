# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/drawing/test_stick_figure.py
"""
Unit tests for stick figure renderer (StickFigureRenderer).
"""

import unittest

import numpy as np

from src.drawing.pose_analyzer import PoseAnalyzer
from src.drawing.stick_figure_renderer import StickFigureRenderer


class TestStickFigureRenderer(unittest.TestCase):
    """
    Tests for StickFigureRenderer class that draws stick figures based on detected points.
    """

    def setUp(self):
        """Initialization before each test."""
        # Create renderer
        self.renderer = StickFigureRenderer(
            canvas_width=640,
            canvas_height=480,
            line_thickness=3,
            head_radius_factor=0.12,  # Increased head size
            bg_color=(255, 255, 255),
            figure_color=(0, 0, 0),
            chair_color=(150, 75, 0),
            smooth_factor=0.3,
            smoothing_history=3,
        )

    def test_initialization(self):
        """Test stick figure renderer initialization."""
        # Check if initialization completed successfully
        self.assertEqual(self.renderer.canvas_width, 640)
        self.assertEqual(self.renderer.canvas_height, 480)
        self.assertEqual(self.renderer.line_thickness, 3)
        self.assertEqual(self.renderer.bg_color, (255, 255, 255))
        self.assertEqual(self.renderer.figure_color, (0, 0, 0))
        self.assertEqual(self.renderer.chair_color, (150, 75, 0))
        self.assertEqual(self.renderer.smooth_factor, 0.3)
        self.assertEqual(self.renderer.smoothing_history, 3)

        # Check head radius calculation
        self.assertEqual(self.renderer.head_radius, int(0.12 * 480))

        # Check if we have instances of pose analyzer and face renderer
        self.assertIsInstance(self.renderer.pose_analyzer, PoseAnalyzer)
        self.assertIsNotNone(self.renderer.face_renderer)

    def test_render_no_landmarks(self):
        """Test rendering without landmarks."""
        # Call render without points
        result = self.renderer.render(None)

        # Check dimensions and color of resulting image
        self.assertEqual(result.shape, (480, 640, 3))
        # Check if image is white (background)
        self.assertTrue(np.all(result == 255))  # 255 is value for white color

    def create_mock_landmarks(self):
        """Creates artificial landmark data for tests."""
        # Create 15 points with good visibility
        landmarks = [(0, 0, 0, 0.9)] * 15

        # Set basic points for test
        # Head
        landmarks[self.renderer.NOSE] = (0.5, 0.2, 0, 0.9)
        landmarks[self.renderer.LEFT_EYE] = (0.45, 0.18, 0, 0.9)
        landmarks[self.renderer.RIGHT_EYE] = (0.55, 0.18, 0, 0.9)

        # Shoulders
        landmarks[self.renderer.LEFT_SHOULDER] = (0.4, 0.3, 0, 0.9)
        landmarks[self.renderer.RIGHT_SHOULDER] = (0.6, 0.3, 0, 0.9)

        # Elbows
        landmarks[self.renderer.LEFT_ELBOW] = (0.3, 0.4, 0, 0.9)
        landmarks[self.renderer.RIGHT_ELBOW] = (0.7, 0.4, 0, 0.9)

        # Wrists
        landmarks[self.renderer.LEFT_WRIST] = (0.25, 0.5, 0, 0.9)
        landmarks[self.renderer.RIGHT_WRIST] = (0.75, 0.5, 0, 0.9)

        return landmarks

    def test_render_upper_body(self):
        """Test rendering stick figure with upper body."""
        # Create artificial landmarks
        landmarks = self.create_mock_landmarks()

        # Prepare face data with hand information
        face_data = {
            "has_face": True,
            "landmarks": landmarks,
            "expressions": {
                "mouth_open": 0.2,
                "smile": 0.6,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
            },
            "hands_data": {
                "left_hand": {
                    "wrist": landmarks[self.renderer.LEFT_WRIST],
                    "elbow": landmarks[self.renderer.LEFT_ELBOW],
                    "is_left": True,
                },
                "right_hand": {
                    "wrist": landmarks[self.renderer.RIGHT_WRIST],
                    "elbow": landmarks[self.renderer.RIGHT_ELBOW],
                    "is_left": False,
                },
            },
        }

        # Render stick figure with upper body
        result = self.renderer.render(face_data)

        # Check resulting image dimensions
        self.assertEqual(result.shape, (480, 640, 3))

        # Check if image is not completely white (something was drawn)
        self.assertFalse(np.all(result == 255))

        # Some points on image should have stick figure color (black)
        # For example in head area
        nose_x, nose_y = int(landmarks[self.renderer.NOSE][0] * 640), int(
            landmarks[self.renderer.NOSE][1] * 480
        )
        # Check area around nose - should have pixels of color different than background
        head_area = result[
            max(0, nose_y - 10) : min(480, nose_y + 10), max(0, nose_x - 10) : min(640, nose_x + 10)
        ]
        self.assertFalse(np.all(head_area == 255))

    def test_set_colors(self):
        """Test changing renderer colors."""
        # Initial colors
        original_bg_color = self.renderer.bg_color
        original_figure_color = self.renderer.figure_color

        # New colors
        new_bg_color = (240, 240, 240)
        new_figure_color = (50, 50, 50)

        # Set new colors
        self.renderer.set_colors(bg_color=new_bg_color, figure_color=new_figure_color)

        # Check if colors were changed
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, new_figure_color)

        # Check partial update - only figure color
        newer_figure_color = (100, 100, 100)
        self.renderer.set_colors(figure_color=newer_figure_color)

        # Check if only figure color was changed
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, newer_figure_color)

    def test_set_line_thickness(self):
        """Test changing line thickness."""
        # Initial thickness
        original_thickness = self.renderer.line_thickness

        # New thickness
        new_thickness = 5

        # Set new thickness
        self.renderer.set_line_thickness(new_thickness)

        # Check if thickness was changed
        self.assertEqual(self.renderer.line_thickness, new_thickness)

        # Check if too small value gets corrected
        self.renderer.set_line_thickness(0)

        # Thickness should be at least 1
        self.assertEqual(self.renderer.line_thickness, 1)

    def test_set_smoothing(self):
        """Test changing smoothing parameters."""
        # Initial values
        original_smooth_factor = self.renderer.smooth_factor
        original_history_length = self.renderer.smoothing_history

        # New values
        new_smooth_factor = 0.5
        new_history_length = 5

        # Set new values
        self.renderer.set_smoothing(new_smooth_factor, new_history_length)

        # Check if values were changed
        self.assertEqual(self.renderer.smooth_factor, new_smooth_factor)
        self.assertEqual(self.renderer.smoothing_history, new_history_length)

    def test_set_mood(self):
        """Test changing mood."""
        # Initial mood
        original_mood = self.renderer.mood

        # New mood
        new_mood = "sad"

        # Set new mood
        self.renderer.set_mood(new_mood)

        # Check if mood was changed
        self.assertEqual(self.renderer.mood, new_mood)

        # Check if invalid mood gets ignored
        self.renderer.set_mood("invalid")

        # Mood should not change
        self.assertEqual(self.renderer.mood, new_mood)

    def test_resize(self):
        """Test changing canvas size."""
        # Initial dimensions
        original_width = self.renderer.canvas_width
        original_height = self.renderer.canvas_height
        original_head_radius = self.renderer.head_radius

        # New dimensions
        new_width = 800
        new_height = 600

        # Resize
        self.renderer.resize(new_width, new_height)

        # Check if dimensions were changed
        self.assertEqual(self.renderer.canvas_width, new_width)
        self.assertEqual(self.renderer.canvas_height, new_height)

        # Head radius should be recalculated
        expected_head_radius = int(self.renderer.head_radius_factor * new_height)
        self.assertEqual(self.renderer.head_radius, expected_head_radius)

        # Smoothing history should be reset
        self.assertEqual(len(self.renderer.landmark_history), 0)

    def test_reset(self):
        """Test resetting renderer's internal state."""
        # First add something to smoothing history
        landmarks = self.create_mock_landmarks()
        face_data = {"landmarks": landmarks, "has_face": True}
        self.renderer.render(face_data)
        self.renderer.render(face_data)

        # Check if history was populated
        self.assertTrue(hasattr(self.renderer, "last_left_elbow"))
        self.assertTrue(hasattr(self.renderer, "last_right_elbow"))

        # Reset state
        self.renderer.reset()

        # Check if history was cleared
        self.assertIsNone(self.renderer.last_left_elbow)
        self.assertIsNone(self.renderer.last_right_elbow)

        # Check if mood returned to default
        self.assertEqual(self.renderer.mood, "happy")


class TestPoseAnalyzer(unittest.TestCase):
    """
    Tests for PoseAnalyzer class that analyzes human pose.
    """

    def setUp(self):
        """Initialization before each test."""
        self.analyzer = PoseAnalyzer()

    def test_initialization(self):
        """Test pose analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.sitting_threshold, 0.3)


if __name__ == "__main__":
    unittest.main()

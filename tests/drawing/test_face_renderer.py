#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for face renderer."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.drawing.face_renderer import SimpleFaceRenderer


class TestSimpleFaceRenderer(unittest.TestCase):
    """Tests for SimpleFaceRenderer class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()
        self.renderer = SimpleFaceRenderer(
            feature_color=(0, 0, 0), smooth_factor=0.3, logger=self.mock_logger
        )

    def test_initialization(self):
        """Test renderer initialization."""
        self.assertEqual(self.renderer.feature_color, (0, 0, 0))
        self.assertEqual(self.renderer.smooth_factor, 0.3)
        self.assertEqual(self.renderer.last_expressions["smile"], 0.5)
        self.assertEqual(len(self.renderer.expressions_history), 0)

    def test_draw_face_without_data(self):
        """Test drawing face without facial data."""
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        head_center = (320, 100)
        head_radius = 50

        # Should draw simple mood-based face
        self.renderer.draw_face(canvas, head_center, head_radius, mood="happy")

        # Check if something was drawn (canvas changed)
        self.assertFalse(np.all(canvas == 255))

    def test_draw_face_with_expressions(self):
        """Test drawing face with expression data."""
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        head_center = (320, 100)
        head_radius = 50

        face_data = {
            "expressions": {
                "smile": 0.8,
                "mouth_open": 0.2,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0,
            }
        }

        self.renderer.draw_face(canvas, head_center, head_radius, face_data=face_data)

        # Check if expressions were smoothed
        self.assertIsNotNone(self.renderer.last_expressions["smile"])
        self.assertGreater(len(self.renderer.expressions_history), 0)

    def test_smooth_expressions(self):
        """Test expression smoothing."""
        new_expressions = {"smile": 0.8, "mouth_open": 0.3}

        smoothed = self.renderer._smooth_expressions(new_expressions)

        # First call should be close to input
        self.assertAlmostEqual(smoothed["smile"], 0.8, delta=0.3)

        # Second call should smooth further
        smoothed2 = self.renderer._smooth_expressions(new_expressions)
        self.assertNotEqual(smoothed["smile"], smoothed2["smile"])

    def test_determine_mood_from_smile(self):
        """Test mood determination from smile value."""
        # Happy
        mood = self.renderer._determine_mood_from_smile(0.7)
        self.assertEqual(mood, "happy")

        # Sad
        mood = self.renderer._determine_mood_from_smile(0.3)
        self.assertEqual(mood, "sad")

        # Neutral
        mood = self.renderer._determine_mood_from_smile(0.5)
        self.assertEqual(mood, "neutral")

    def test_average_expressions(self):
        """Test averaging expressions from history."""
        # Add multiple expression frames
        for i in range(5):
            self.renderer.expressions_history.append({"smile": 0.5 + i * 0.1, "mouth_open": 0.2})

        avg = self.renderer._average_expressions()

        # Should be close to middle value
        self.assertAlmostEqual(avg["smile"], 0.7, delta=0.1)
        self.assertAlmostEqual(avg["mouth_open"], 0.2, delta=0.01)

    def test_draw_mood_face_variations(self):
        """Test drawing different mood faces."""
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        head_center = (320, 100)
        head_radius = 50

        moods = ["happy", "sad", "neutral", "surprised", "wink"]

        for mood in moods:
            canvas_copy = canvas.copy()
            self.renderer._draw_mood_face(canvas_copy, head_center, head_radius, mood)
            # Each mood should draw something different
            self.assertFalse(np.all(canvas_copy == 255))

    def test_reset(self):
        """Test resetting renderer state."""
        # Add some data
        self.renderer.last_expressions["smile"] = 0.8
        self.renderer.expressions_history.append({"smile": 0.7})
        self.renderer.frame_count = 100

        self.renderer.reset()

        # Check if reset
        self.assertEqual(self.renderer.last_expressions["smile"], 0.5)
        self.assertEqual(len(self.renderer.expressions_history), 0)
        self.assertEqual(self.renderer.frame_count, 0)


if __name__ == "__main__":
    unittest.main()

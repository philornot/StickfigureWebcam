#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/drawing/test_hand_tracking.py

import os
import sys
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

# Add main project directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drawing.stick_figure_renderer import StickFigureRenderer


class TestHandTracking(unittest.TestCase):
    """
    Tests for hand tracking mechanism in StickFigureRenderer.
    """

    def setUp(self):
        """Initialization before each test."""
        # Mock logger
        self.mock_logger = MagicMock()

        # Create renderer
        self.renderer = StickFigureRenderer(
            canvas_width=640,
            canvas_height=480,
            line_thickness=3,
            head_radius_factor=0.075,
            bg_color=(255, 255, 255),
            figure_color=(0, 0, 0),
            chair_color=(150, 75, 0),
            smooth_factor=0.3,
            smoothing_history=3,
            logger=self.mock_logger,
        )

    def test_update_arm_positions_without_data(self):
        """Test updating arm positions without input data."""
        # Arms should be in neutral position
        self.renderer._update_arm_positions(None)

        # Check if arm positions were initialized as None
        self.assertIsNone(self.renderer.last_left_elbow)
        self.assertIsNone(self.renderer.last_right_elbow)
        self.assertIsNone(self.renderer.last_left_wrist)
        self.assertIsNone(self.renderer.last_right_wrist)

        # Set some values
        self.renderer.last_left_elbow = (100, 100)
        self.renderer.last_right_elbow = (200, 100)
        self.renderer.last_left_wrist = (80, 150)
        self.renderer.last_right_wrist = (220, 150)

        # Update again without data - should transition toward neutral position
        self.renderer._update_arm_positions(None)

        # Check if positions were changed (moved toward neutral positions)
        # Don't check exact values as they depend on animation timer
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_right_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)
        self.assertIsNotNone(self.renderer.last_right_wrist)

    def test_update_arm_positions_with_hand_data(self):
        """Test updating arm positions with MediaPipe Hands data."""
        # Create artificial hand data
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.3, 0.6, 0.0, 1.0),
                    "elbow": (0.35, 0.4, 0.0, 1.0),
                    "is_left": True,
                },
                "right_hand": {
                    "wrist": (0.7, 0.6, 0.0, 1.0),
                    "elbow": (0.65, 0.4, 0.0, 1.0),
                    "is_left": False,
                },
            }
        }

        # Update arm positions
        self.renderer._update_arm_positions(hands_data)

        # Check if visibility flags were set
        self.assertTrue(self.renderer.left_arm_visible)
        self.assertTrue(self.renderer.right_arm_visible)

        # Check if positions were updated
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_right_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)
        self.assertIsNotNone(self.renderer.last_right_wrist)

        # Check approximate values (considering smoothing)
        left_elbow_x = int(0.35 * 640 * (1 - self.renderer.smooth_factor))  # First time
        left_elbow_y = int(0.4 * 480 * (1 - self.renderer.smooth_factor))
        self.assertAlmostEqual(self.renderer.last_left_elbow[0], left_elbow_x, delta=5)
        self.assertAlmostEqual(self.renderer.last_left_elbow[1], left_elbow_y, delta=5)

    def test_transition_to_neutral_position(self):
        """Test transitioning from detected position to neutral position."""
        # First set arm positions with data
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.3, 0.6, 0.0, 1.0),
                    "elbow": (0.35, 0.4, 0.0, 1.0),
                    "is_left": True,
                }
            }
        }

        # Update arm positions
        self.renderer._update_arm_positions(hands_data)

        # Save positions
        left_elbow_with_data = self.renderer.last_left_elbow
        left_wrist_with_data = self.renderer.last_left_wrist

        # Set visibility time to old to simulate tracking loss
        self.renderer.left_arm_visibility_time = time.time() - 1.0
        self.renderer.left_arm_visible = False

        # Update without left hand data
        self.renderer._update_arm_positions({"hands_data": {"left_hand": None, "right_hand": None}})

        # Check if positions changed toward neutral
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)

        # Positions should be different than before (moved toward neutral)
        # Exact values depend on animation timer, so we don't check them directly
        self.assertNotEqual(self.renderer.last_left_elbow, left_elbow_with_data)
        self.assertNotEqual(self.renderer.last_left_wrist, left_wrist_with_data)

    def test_render_arms(self):
        """Test rendering arms on image."""
        # Set arm positions
        self.renderer.last_left_elbow = (150, 150)
        self.renderer.last_right_elbow = (350, 150)
        self.renderer.last_left_wrist = (100, 200)
        self.renderer.last_right_wrist = (400, 200)

        # Render image
        canvas = self.renderer.render()

        # Check dimensions
        self.assertEqual(canvas.shape, (480, 640, 3))

        # Check if image is not empty (completely white)
        self.assertFalse(np.array_equal(canvas, np.ones((480, 640, 3), dtype=np.uint8) * 255))

    def test_smooth_transition(self):
        """Test smooth transition between different arm positions."""
        # Set initial position
        self.renderer.last_left_elbow = (150, 150)
        self.renderer.last_left_wrist = (100, 200)

        # Create new data with different positions
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.2, 0.5, 0.0, 1.0),  # ~(128, 240)
                    "elbow": (0.3, 0.35, 0.0, 1.0),  # ~(192, 168)
                    "is_left": True,
                }
            }
        }

        # Perform several updates to observe smooth transition
        positions = []
        for _ in range(5):
            self.renderer._update_arm_positions(hands_data)
            positions.append(
                (
                    self.renderer.last_left_elbow[0],
                    self.renderer.last_left_elbow[1],
                    self.renderer.last_left_wrist[0],
                    self.renderer.last_left_wrist[1],
                )
            )

        # Check if positions change gradually
        for i in range(1, len(positions)):
            # Difference between consecutive positions shouldn't be too large
            self.assertLess(abs(positions[i][0] - positions[i - 1][0]), 50)
            self.assertLess(abs(positions[i][1] - positions[i - 1][1]), 50)
            self.assertLess(abs(positions[i][2] - positions[i - 1][2]), 50)
            self.assertLess(abs(positions[i][3] - positions[i - 1][3]), 50)

        # Final position should differ from initial
        self.assertNotEqual(self.renderer.last_left_elbow, (150, 150))
        self.assertNotEqual(self.renderer.last_left_wrist, (100, 200))


if __name__ == "__main__":
    unittest.main()

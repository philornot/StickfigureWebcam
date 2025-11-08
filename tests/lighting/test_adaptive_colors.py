#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for adaptive lighting module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.lighting.adaptive_colors import AdaptiveLightingManager


class TestAdaptiveLightingManager(unittest.TestCase):
    """Tests for AdaptiveLightingManager class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()
        self.manager = AdaptiveLightingManager(
            adaptation_speed=0.02,
            smoothing_window=30,
            min_brightness=20,
            max_brightness=250,
            min_contrast=0.4,
            sampling_interval=5,
            logger=self.mock_logger,
        )

    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.adaptation_speed, 0.02)
        self.assertEqual(self.manager.smoothing_window, 30)
        self.assertEqual(self.manager.min_brightness, 20)
        self.assertEqual(self.manager.max_brightness, 250)
        self.assertEqual(len(self.manager.brightness_history), 0)

    def test_analyze_frame_bright(self):
        """Test analyzing bright frame."""
        # Create bright frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200

        brightness = self.manager.analyze_frame(frame)

        self.assertGreater(brightness, 0.5)
        self.assertLessEqual(brightness, 1.0)

    def test_analyze_frame_dark(self):
        """Test analyzing dark frame."""
        # Create dark frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50

        brightness = self.manager.analyze_frame(frame)

        self.assertLess(brightness, 0.5)
        self.assertGreaterEqual(brightness, 0.0)

    def test_sampling_interval(self):
        """Test that sampling interval is respected."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # First frame should be analyzed
        brightness1 = self.manager.analyze_frame(frame)

        # Next few frames should use cached value
        for i in range(4):
            brightness = self.manager.analyze_frame(frame)
            self.assertEqual(brightness, brightness1)

    def test_update_colors_bright_environment(self):
        """Test color update for bright environment."""
        # Simulate bright environment
        frame_brightness = 0.8

        bg_color, figure_color = self.manager.update_colors(frame_brightness)

        # Background should be bright
        self.assertGreater(bg_color[0], 150)
        # Figure should be dark for contrast
        self.assertLess(figure_color[0], 100)

    def test_update_colors_dark_environment(self):
        """Test color update for dark environment."""
        # Simulate dark environment
        frame_brightness = 0.2

        bg_color, figure_color = self.manager.update_colors(frame_brightness)

        # Background should be dark
        self.assertLess(bg_color[0], 100)

    def test_calculate_contrasting_value(self):
        """Test calculating contrasting color value."""
        # Bright background should give dark figure
        figure_value = self.manager._calculate_contrasting_value(200)
        self.assertLess(figure_value, 100)

        # Dark background should give bright figure
        figure_value = self.manager._calculate_contrasting_value(50)
        self.assertGreater(figure_value, 100)

    def test_get_current_colors(self):
        """Test getting current color values."""
        colors = self.manager.get_current_colors()

        self.assertIn("bg_color", colors)
        self.assertIn("figure_color", colors)
        self.assertIn("brightness_level", colors)

    def test_reset(self):
        """Test resetting manager state."""
        # Add some data
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.manager.analyze_frame(frame)
        self.manager.update_colors(0.5)

        self.manager.reset()

        # Check if reset
        self.assertEqual(len(self.manager.brightness_history), 0)
        self.assertEqual(self.manager.current_bg_color, (255, 255, 255))
        self.assertEqual(self.manager.current_figure_color, (0, 0, 0))

    def test_brightness_smoothing(self):
        """Test brightness value smoothing."""
        # Add multiple frames with different brightness
        for i in range(10):
            brightness_val = 100 + i * 10
            frame = np.ones((480, 640, 3), dtype=np.uint8) * brightness_val

            # Skip frames according to sampling interval
            for _ in range(5):
                self.manager.analyze_frame(frame)

        # History should contain smoothed values
        self.assertGreater(len(self.manager.brightness_history), 0)


if __name__ == "__main__":
    unittest.main()

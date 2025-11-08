#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for drawing utilities."""

import unittest

from src.drawing.utils import (
    calculate_visibility,
    get_midpoint,
    is_point_visible,
    normalize_coordinates,
    smooth_points,
)


class TestDrawingUtils(unittest.TestCase):
    """Tests for drawing utility functions."""

    def test_get_midpoint_valid(self):
        """Test getting midpoint between two points."""
        p1 = (100, 100)
        p2 = (200, 200)

        midpoint = get_midpoint(p1, p2)

        self.assertEqual(midpoint, (150, 150))

    def test_get_midpoint_none(self):
        """Test getting midpoint with None input."""
        result = get_midpoint(None, (100, 100))
        self.assertIsNone(result)

        result = get_midpoint((100, 100), None)
        self.assertIsNone(result)

    def test_normalize_coordinates(self):
        """Test normalizing coordinates to pixel values."""
        x, y = normalize_coordinates(0.5, 0.5, 640, 480)

        self.assertEqual(x, 320)
        self.assertEqual(y, 240)

    def test_normalize_coordinates_edges(self):
        """Test normalizing at edges."""
        x, y = normalize_coordinates(0.0, 0.0, 640, 480)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

        x, y = normalize_coordinates(1.0, 1.0, 640, 480)
        self.assertEqual(x, 640)
        self.assertEqual(y, 480)

    def test_calculate_visibility(self):
        """Test calculating point visibility."""
        landmarks = [(0.5, 0.5, 0.0, 0.9), (0.3, 0.3, 0.0, 0.5), (0.1, 0.1, 0.0, 0.1)]

        self.assertAlmostEqual(calculate_visibility(landmarks, 0), 0.9)
        self.assertAlmostEqual(calculate_visibility(landmarks, 1), 0.5)
        self.assertAlmostEqual(calculate_visibility(landmarks, 2), 0.1)

    def test_calculate_visibility_invalid_index(self):
        """Test calculating visibility with invalid index."""
        landmarks = [(0.5, 0.5, 0.0, 0.9)]

        result = calculate_visibility(landmarks, 10)

        self.assertEqual(result, 0.0)

    def test_is_point_visible(self):
        """Test checking if point is visible."""
        landmarks = [(0.5, 0.5, 0.0, 0.9), (0.3, 0.3, 0.0, 0.3)]

        self.assertTrue(is_point_visible(landmarks, 0, threshold=0.5))
        self.assertFalse(is_point_visible(landmarks, 1, threshold=0.5))

    def test_smooth_points_no_history(self):
        """Test smoothing with no history."""
        current = [(0.5, 0.5, 0.0, 0.9)]
        history = []

        result = smooth_points(current, history, 5, 0.5)

        self.assertEqual(result, current)

    def test_smooth_points_with_history(self):
        """Test smoothing with history."""
        current = [(0.6, 0.6, 0.0, 0.9)]
        history = [[(0.5, 0.5, 0.0, 0.9)], [(0.5, 0.5, 0.0, 0.9)]]

        result = smooth_points(current, history, 5, 0.5)

        # Result should be between current and history
        self.assertGreater(result[0][0], 0.5)
        self.assertLess(result[0][0], 0.6)

    def test_smooth_points_different_lengths(self):
        """Test smoothing with different point counts."""
        current = [(0.6, 0.6, 0.0, 0.9), (0.7, 0.7, 0.0, 0.8)]
        history = [[(0.5, 0.5, 0.0, 0.9)]]  # Shorter history

        result = smooth_points(current, history, 5, 0.5)

        # Should handle gracefully
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()

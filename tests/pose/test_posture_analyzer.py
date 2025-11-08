#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for posture analyzer (PostureAnalyzer).
"""

import unittest
from unittest.mock import MagicMock

from src.pose.posture_analyzer import PostureAnalyzer


class TestPostureAnalyzer(unittest.TestCase):
    """
    Tests for PostureAnalyzer class that analyzes posture and determines if user is sitting or standing.
    """

    def setUp(self):
        """Initialization before each test."""
        # Mock logger
        self.mock_logger = MagicMock()

        # Create posture analyzer with mock logger
        self.analyzer = PostureAnalyzer(
            standing_hip_threshold=0.7,
            confidence_threshold=0.6,
            smoothing_factor=0.7,
            temporal_smoothing=3,
            partial_visibility_bias=0.8,
            logger=self.mock_logger,
        )

    def test_analyze_posture_no_landmarks(self):
        """Test posture analysis when no landmarks are available."""
        result = self.analyzer.analyze_posture(None, 480, 640)

        # Check if result contains expected keys
        self.assertIsNone(result["is_sitting"])
        self.assertEqual(result["posture"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["visible_keypoints"], 0)

    def test_analyze_posture_too_few_landmarks(self):
        """Test posture analysis when there are too few landmarks."""
        # Create only 10 points instead of required 33
        landmarks = [(0, 0, 0, 0.9)] * 10

        result = self.analyzer.analyze_posture(landmarks, 480, 640)

        self.assertIsNone(result["is_sitting"])
        self.assertEqual(result["posture"], "unknown")

    def test_analyze_posture_sitting(self):
        """Test posture analysis in sitting position."""
        # Create artificial landmark data for sitting posture
        # Format: (x, y, z, visibility)
        landmarks = [(0, 0, 0, 0.9)] * 33  # First initialize all points

        # Now overwrite specific points that are important for detection
        # Place hips low (y value > 0.7)
        landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.8, 0, 0.9)
        landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.8, 0, 0.9)

        # Place shoulders higher
        landmarks[self.analyzer.LEFT_SHOULDER] = (0.35, 0.5, 0, 0.9)
        landmarks[self.analyzer.RIGHT_SHOULDER] = (0.65, 0.5, 0, 0.9)

        # Place knees low but with low visibility (hidden under desk)
        landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.9, 0, 0.3)
        landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.9, 0, 0.3)

        # Ankles are invisible (hidden under desk)
        landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.95, 0, 0.1)
        landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.95, 0, 0.1)

        # Simulate several frames of same position to get stable detection
        for _ in range(5):
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # After several frames we should have stable sitting detection
        self.assertTrue(result["is_sitting"])
        self.assertEqual(result["posture"], "sitting")
        self.assertGreaterEqual(result["sitting_probability"], 0.6)

    def test_analyze_posture_standing(self):
        """Test posture analysis in standing position."""
        # Create artificial landmark data for standing posture
        landmarks = [(0, 0, 0, 0.9)] * 33

        # Place hips high (y value < 0.5) to ensure standing detection
        landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.4, 0, 0.9)
        landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.4, 0, 0.9)

        # Place shoulders even higher
        landmarks[self.analyzer.LEFT_SHOULDER] = (0.35, 0.2, 0, 0.9)
        landmarks[self.analyzer.RIGHT_SHOULDER] = (0.65, 0.2, 0, 0.9)

        # Knees are visible
        landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.6, 0, 0.9)
        landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.6, 0, 0.9)

        # Ankles are visible
        landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.8, 0, 0.9)
        landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.8, 0, 0.9)

        # Simulate many frames of same position to get stable detection
        for _ in range(15):  # More frames to overcome sitting preference
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # After many frames we should have stable standing detection
        self.assertFalse(result["is_sitting"])
        self.assertEqual(result["posture"], "standing")
        self.assertLessEqual(result["sitting_probability"], 0.4)

    def test_temporal_smoothing(self):
        """Test temporal smoothing of detection."""
        # First simulate 5 frames of sitting person
        sitting_landmarks = [(0, 0, 0, 0.9)] * 33
        sitting_landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.8, 0, 0.9)
        sitting_landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.8, 0, 0.9)
        sitting_landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.9, 0, 0.3)
        sitting_landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.9, 0, 0.3)
        sitting_landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.95, 0, 0.1)
        sitting_landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.95, 0, 0.1)

        for _ in range(5):
            result = self.analyzer.analyze_posture(sitting_landmarks, 480, 640)

        # Remember state and probability
        sitting_probability = result["sitting_probability"]
        self.assertTrue(result["is_sitting"])

        # Now simulate 1 frame of standing person - shouldn't change detection yet
        standing_landmarks = [(0, 0, 0, 0.9)] * 33
        standing_landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.4, 0, 0.9)
        standing_landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.4, 0, 0.9)
        standing_landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.6, 0, 0.9)
        standing_landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.6, 0, 0.9)
        standing_landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.8, 0, 0.9)
        standing_landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.8, 0, 0.9)

        result = self.analyzer.analyze_posture(standing_landmarks, 480, 640)

        # After one standing frame detection should still show sitting
        # but probability should decrease somewhat
        self.assertTrue(result["is_sitting"])
        self.assertLess(result["sitting_probability"], sitting_probability)

        # Simulate more standing frames - now should change detection
        for _ in range(15):  # Need more frames to overcome sitting preference
            result = self.analyzer.analyze_posture(standing_landmarks, 480, 640)

        # Now we should detect standing
        self.assertFalse(result["is_sitting"])
        self.assertEqual(result["posture"], "standing")

    def test_update_thresholds(self):
        """Test updating detection thresholds."""
        # Save initial values
        initial_hip_threshold = self.analyzer.standing_hip_threshold
        initial_confidence_threshold = self.analyzer.confidence_threshold
        initial_smoothing_factor = self.analyzer.smoothing_factor
        initial_bias = self.analyzer.partial_visibility_bias

        # Update values
        new_hip_threshold = 0.6
        new_confidence_threshold = 0.7
        new_smoothing_factor = 0.5
        new_bias = 0.9

        self.analyzer.update_thresholds(
            standing_hip_threshold=new_hip_threshold,
            confidence_threshold=new_confidence_threshold,
            smoothing_factor=new_smoothing_factor,
            partial_visibility_bias=new_bias,
        )

        # Check if values were updated
        self.assertEqual(self.analyzer.standing_hip_threshold, new_hip_threshold)
        self.assertEqual(self.analyzer.confidence_threshold, new_confidence_threshold)
        self.assertEqual(self.analyzer.smoothing_factor, new_smoothing_factor)
        self.assertEqual(self.analyzer.partial_visibility_bias, new_bias)

        # Check if logger was called - but don't check how many times
        # (may be called during initialization and update)
        self.mock_logger.info.assert_called()

    def test_reset(self):
        """Test resetting analyzer state."""
        # First simulate several frames to set state
        landmarks = [(0, 0, 0, 0.9)] * 33

        for _ in range(3):
            self.analyzer.analyze_posture(landmarks, 480, 640)

        # Reset state
        self.analyzer.reset()

        # Check if state was reset
        self.assertIsNone(self.analyzer.is_sitting)
        self.assertEqual(self.analyzer.sitting_probability, 0.5)
        self.assertEqual(self.analyzer.history_buffer, [])
        self.assertEqual(self.analyzer.consecutive_frames, 0)

        # Check if logger was called
        self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()

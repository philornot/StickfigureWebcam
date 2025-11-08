#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests checking how system handles partial body visibility.
"""

import unittest
from unittest.mock import MagicMock

from src.pose.pose_detector import PoseDetector
from src.pose.posture_analyzer import PostureAnalyzer


class TestPartialVisibility(unittest.TestCase):
    """
    Tests verifying pose detection with limited body visibility,
    e.g., when only upper body is visible as typical in video conferences.
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
            partial_visibility_bias=0.9,  # High sitting preference
            logger=self.mock_logger,
        )

    def create_upper_body_only_landmarks(self):
        """
        Creates landmarks with high visibility only for upper body
        (head, shoulders, elbows), and low visibility for lower body (hips, legs).

        Returns:
            list: List of 33 landmarks in format (x, y, z, visibility)
        """
        # Create 33 empty points with low visibility
        landmarks = [(0, 0, 0, 0.1)] * 33

        # Head and upper body - high visibility
        landmarks[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.9)  # Nose
        landmarks[PoseDetector.LEFT_EYE] = (0.45, 0.08, 0, 0.9)  # Left eye
        landmarks[PoseDetector.RIGHT_EYE] = (0.55, 0.08, 0, 0.9)  # Right eye
        landmarks[PoseDetector.LEFT_EAR] = (0.4, 0.1, 0, 0.9)  # Left ear
        landmarks[PoseDetector.RIGHT_EAR] = (0.6, 0.1, 0, 0.9)  # Right ear

        # Shoulders - high visibility
        landmarks[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.9)
        landmarks[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.9)

        # Elbows - medium/high visibility
        landmarks[PoseDetector.LEFT_ELBOW] = (0.3, 0.3, 0, 0.8)
        landmarks[PoseDetector.RIGHT_ELBOW] = (0.7, 0.3, 0, 0.8)

        # Wrists - medium visibility (may be partially out of frame)
        landmarks[PoseDetector.LEFT_WRIST] = (0.25, 0.4, 0, 0.7)
        landmarks[PoseDetector.RIGHT_WRIST] = (0.75, 0.4, 0, 0.7)

        # Hips - low visibility (out of frame)
        landmarks[PoseDetector.LEFT_HIP] = (0.45, 0.5, 0, 0.3)
        landmarks[PoseDetector.RIGHT_HIP] = (0.55, 0.5, 0, 0.3)

        # Legs - very low visibility (out of frame)
        landmarks[PoseDetector.LEFT_KNEE] = (0.43, 0.7, 0, 0.1)
        landmarks[PoseDetector.RIGHT_KNEE] = (0.57, 0.7, 0, 0.1)
        landmarks[PoseDetector.LEFT_ANKLE] = (0.42, 0.9, 0, 0.1)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0.58, 0.9, 0, 0.1)

        return landmarks

    def test_upper_body_only_detection(self):
        """
        Test verifying if system correctly assumes sitting position,
        when only upper body is visible.
        """
        # Get upper body only landmarks
        landmarks = self.create_upper_body_only_landmarks()

        # Simulate multiple frames of same position to get stable detection
        for _ in range(10):  # Increase frame count for certainty
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Check if system correctly recognized sitting position
        self.assertTrue(
            result["is_sitting"],
            "System should assume sitting position when only upper body is visible",
        )
        self.assertEqual(result["posture"], "sitting")
        self.assertGreaterEqual(
            result["sitting_probability"],
            0.7,
            "Sitting probability should be high with limited visibility",
        )

        # Check if leg visibility analysis gives high sitting probability
        leg_score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(
            leg_score,
            0.8,
            "Leg visibility analysis should give high sitting probability",
        )

    def test_torso_only_detection(self):
        """
        Test verifying system behavior in extreme scenario,
        when very limited body part is visible (only torso).
        """
        # Create 33 empty points with low visibility
        landmarks = [(0, 0, 0, 0.1)] * 33

        # Only shoulders are well visible
        landmarks[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.8)
        landmarks[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.8)

        # Nose and head - medium visibility
        landmarks[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.7)

        # Elbows - low/medium visibility
        landmarks[PoseDetector.LEFT_ELBOW] = (0.3, 0.3, 0, 0.6)
        landmarks[PoseDetector.RIGHT_ELBOW] = (0.7, 0.3, 0, 0.6)

        # Simulate multiple frames of same position to get stable detection
        for _ in range(15):  # More frames for very limited visibility
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Check detection result
        self.assertTrue(
            result["is_sitting"],
            "System should prefer sitting position with very limited visibility",
        )
        self.assertEqual(result["posture"], "sitting")
        self.assertGreaterEqual(result["sitting_probability"], 0.6)

    def test_leg_visibility_analysis(self):
        """
        Detailed test of leg visibility analysis function, which is crucial
        for determining position with limited visibility.
        """
        # Test when no leg parts are visible
        landmarks = [(0, 0, 0, 0.9)] * 33  # All points with high visibility
        # Overwrite leg visibility to low
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.RIGHT_KNEE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.LEFT_ANKLE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0, 0, 0, 0.1)

        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(
            score, 0.9, "No leg visibility should give highest sitting probability"
        )

        # Test when only one knee is visible
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.7)  # Only left knee visible
        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(
            score,
            0.7,
            "One visible leg part should give high sitting probability",
        )

        # Test when all leg parts are visible
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.RIGHT_KNEE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.LEFT_ANKLE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0, 0, 0, 0.7)
        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertLessEqual(score, 0.2, "Full leg visibility should give low sitting probability")

    def test_visibility_type_detection(self):
        """
        Test checking body visibility type detection.
        """
        # Test for full body visibility
        full_body = [(0, 0, 0, 0.9)] * 33  # All points visible
        visibility_type = self.analyzer._analyze_visibility_type(full_body)
        self.assertEqual(visibility_type, "full_body", "Should detect full body visibility")

        # Test for upper body only visibility
        upper_body = self.create_upper_body_only_landmarks()
        visibility_type = self.analyzer._analyze_visibility_type(upper_body)
        self.assertEqual(visibility_type, "upper_body", "Should detect upper body only visibility")

        # Test for partial visibility
        partial_body = [(0, 0, 0, 0.1)] * 33  # Initially low visibility
        # Add only few visible upper body points
        partial_body[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.8)
        partial_body[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.8)
        partial_body[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.8)

        visibility_type = self.analyzer._analyze_visibility_type(partial_body)
        self.assertEqual(visibility_type, "partial_visibility", "Should detect partial visibility")

    def test_analyze_partial_visibility(self):
        """
        Test checking partial visibility analysis function.
        """
        landmarks = [(0, 0, 0, 0.9)] * 33  # Points not important for this function

        # With full visibility we should have neutral result
        score = self.analyzer._analyze_partial_visibility(landmarks, "full_body")
        self.assertAlmostEqual(
            score, 0.5, places=1, msg="Full visibility should give neutral result"
        )

        # With upper body only visibility we should have high sitting probability
        score = self.analyzer._analyze_partial_visibility(landmarks, "upper_body")
        self.assertGreaterEqual(score, 0.8, "Upper body should give high sitting probability")

        # With partial visibility we should also have high probability
        score = self.analyzer._analyze_partial_visibility(landmarks, "partial_visibility")
        self.assertGreaterEqual(
            score, 0.7, "Partial visibility should give high sitting probability"
        )

        # With unknown type we should have value from configuration
        score = self.analyzer._analyze_partial_visibility(landmarks, "unknown")
        self.assertEqual(
            score,
            self.analyzer.partial_visibility_bias,
            "Unknown type should use partial_visibility_bias from configuration",
        )


if __name__ == "__main__":
    unittest.main()

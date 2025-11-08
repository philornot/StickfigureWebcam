#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/pose/test_pose_detector.py
"""
Unit tests for pose detector (PoseDetector).
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from src.pose.pose_detector import PoseDetector


class TestPoseDetector(unittest.TestCase):
    """
    Tests for PoseDetector class that detects human pose in images.
    """

    def setUp(self):
        """Initialization before each test."""
        # Patch MediaPipe to avoid actual model initialization
        self.mp_pose_patch = patch("mediapipe.solutions.pose")
        self.mp_drawing_patch = patch("mediapipe.solutions.drawing_utils")
        self.mp_drawing_styles_patch = patch("mediapipe.solutions.drawing_styles")

        self.mock_mp_pose = self.mp_pose_patch.start()
        self.mock_mp_drawing = self.mp_drawing_patch.start()
        self.mock_mp_drawing_styles = self.mp_drawing_styles_patch.start()

        # Create mock for Pose object
        self.mock_pose = Mock()
        self.mock_mp_pose.Pose.return_value = self.mock_pose

        # Mock logger
        self.mock_logger = MagicMock()

        # Initialize pose detector with mocks
        self.detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            logger=self.mock_logger,
        )

    def tearDown(self):
        """Cleanup after each test."""
        self.mp_pose_patch.stop()
        self.mp_drawing_patch.stop()
        self.mp_drawing_styles_patch.stop()

    def test_initialization(self):
        """Test pose detector initialization."""
        # Check if initialization completed successfully
        self.assertEqual(self.detector.min_detection_confidence, 0.5)
        self.assertEqual(self.detector.min_tracking_confidence, 0.5)
        self.assertEqual(self.detector.model_complexity, 1)
        self.assertTrue(self.detector.smooth_landmarks)

        # Check if MediaPipe Pose was initialized with correct parameters
        self.mock_mp_pose.Pose.assert_called_once_with(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Check if logger was called
        self.mock_logger.info.assert_called_once()

    def test_detect_pose_no_landmarks(self):
        """Test pose detection when MediaPipe doesn't detect landmarks."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Configure MediaPipe process mock to return no detected landmarks
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        self.mock_pose.process.return_value = mock_results

        # Call pose detection
        success, pose_data = self.detector.detect_pose(test_image)

        # Check results
        self.assertFalse(success)
        self.assertFalse(pose_data["has_pose"])
        self.assertIsNone(pose_data["landmarks"])
        self.assertEqual(pose_data["frame_height"], 480)
        self.assertEqual(pose_data["frame_width"], 640)

    def test_detect_pose_with_landmarks(self):
        """Test pose detection when MediaPipe detects landmarks."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Configure MediaPipe process mock to return detected landmarks
        mock_results = MagicMock()

        # Create example landmarks (33 points for MediaPipe Pose)
        mock_landmarks = MagicMock()
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 0.9

        # Set 33 points (MediaPipe Pose uses 33 points)
        mock_landmarks.landmark = [mock_landmark] * 33
        mock_results.pose_landmarks = mock_landmarks

        # Also add world landmarks
        mock_world_landmarks = MagicMock()
        mock_world_landmarks.landmark = [mock_landmark] * 33
        mock_results.pose_world_landmarks = mock_world_landmarks

        self.mock_pose.process.return_value = mock_results

        # Call pose detection
        success, pose_data = self.detector.detect_pose(test_image)

        # Check results
        self.assertTrue(success)
        self.assertTrue(pose_data["has_pose"])
        self.assertIsNotNone(pose_data["landmarks"])
        self.assertEqual(len(pose_data["landmarks"]), 33)
        self.assertAlmostEqual(pose_data["detection_score"], 0.9)

    def test_draw_pose_on_image(self):
        """Test drawing pose on image."""
        # Create test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Create example landmarks
        landmarks = [(0.5, 0.5, 0.0, 0.9)] * 33

        # Call drawing method
        result_image = self.detector.draw_pose_on_image(
            test_image, landmarks, draw_connections=True
        )

        # Check if resulting image has correct dimensions
        self.assertEqual(result_image.shape, test_image.shape)

        # Hard to check exactly what was drawn without actual rendering,
        # but we can check if image was modified
        self.assertFalse(np.array_equal(result_image, test_image))

    def test_calculate_angle(self):
        """Test calculating angle between three points."""
        # Create example points
        # Points forming right angle (90 degrees)
        landmarks = [
            (0.0, 0.0, 0.0, 1.0),  # point 0
            (0.0, 1.0, 0.0, 1.0),  # point 1 (angle vertex)
            (1.0, 1.0, 0.0, 1.0),  # point 2
        ]

        # Calculate angle
        angle = self.detector.calculate_angle(landmarks, 0, 1, 2)

        # Check if angle is close to 90 degrees
        self.assertAlmostEqual(angle, 90.0, delta=1.0)

        # Now create points forming 45 degree angle
        landmarks = [
            (0.0, 0.0, 0.0, 1.0),  # point 0
            (0.0, 0.0, 0.0, 1.0),  # point 1 (angle vertex)
            (1.0, 1.0, 0.0, 1.0),  # point 2
        ]

        # Calculate angle
        angle = self.detector.calculate_angle(landmarks, 0, 1, 2)

        # Check if angle is close to 45 degrees
        self.assertAlmostEqual(angle, 45.0, delta=1.0)

    def test_get_landmark_position(self):
        """Test getting position of specific landmark."""
        # Create example points
        landmarks = [
            (0.1, 0.2, 0.0, 0.9),  # point 0
            (0.3, 0.4, 0.0, 0.8),  # point 1
            (0.5, 0.6, 0.0, 0.7),  # point 2
        ]

        # Get position of point 1
        position = self.detector.get_landmark_position(landmarks, 1, 640, 480)

        # Check if position is correct
        self.assertEqual(position[0], int(0.3 * 640))  # x
        self.assertEqual(position[1], int(0.4 * 480))  # y
        self.assertEqual(position[2], 0.0)  # z
        self.assertEqual(position[3], 0.8)  # visibility

    def test_get_detection_stats(self):
        """Test getting detection statistics."""
        # Update detection counters
        self.detector.frame_count = 100
        self.detector.detection_count = 80
        self.detector.last_detection_score = 0.85

        # Get statistics
        stats = self.detector.get_detection_stats()

        # Check statistics
        self.assertEqual(stats["total_frames"], 100)
        self.assertEqual(stats["detection_count"], 80)
        self.assertEqual(stats["detection_ratio"], 0.8)
        self.assertEqual(stats["last_detection_score"], 0.85)
        self.assertEqual(stats["model_complexity"], 1)

    def test_reset_stats(self):
        """Test resetting detection statistics."""
        # Update detection counters
        self.detector.frame_count = 100
        self.detector.detection_count = 80
        self.detector.last_detection_score = 0.85

        # Reset statistics
        self.detector.reset_stats()

        # Check if statistics were reset
        stats = self.detector.get_detection_stats()
        self.assertEqual(stats["total_frames"], 0)
        self.assertEqual(stats["detection_count"], 0)
        self.assertEqual(stats["detection_ratio"], 0)
        self.assertEqual(stats["last_detection_score"], 0.0)

    def test_close(self):
        """Test closing pose detector."""
        # Call close
        self.detector.close()

        # Check if pose.close() was called
        self.mock_pose.close.assert_called_once()

        # Check if logger was called
        self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for camera capture module."""

import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np

from src.camera.camera_capture import CameraCapture


class TestCameraCapture(unittest.TestCase):
    """Tests for CameraCapture class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()

        # Mock cv2.VideoCapture
        self.mock_cap = MagicMock()
        self.mock_cap.isOpened.return_value = True
        self.mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        self.mock_cap.get.side_effect = lambda prop: {3: 640, 4: 480, 5: 30}.get(int(prop), 0)

        self.cv2_patch = patch("cv2.VideoCapture", return_value=self.mock_cap)
        self.mock_cv2 = self.cv2_patch.start()

    def tearDown(self):
        """Clean up after tests."""
        self.cv2_patch.stop()

    def test_initialization(self):
        """Test camera initialization."""
        camera = CameraCapture(camera_id=0, width=640, height=480, fps=30, logger=self.mock_logger)

        self.assertEqual(camera.camera_id, 0)
        self.assertEqual(camera.width, 640)
        self.assertEqual(camera.height, 480)
        self.assertTrue(camera.is_open)

    def test_read_frame_success(self):
        """Test successful frame reading."""
        camera = CameraCapture(logger=self.mock_logger)

        success, frame = camera.read()

        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))

    def test_read_frame_failure(self):
        """Test frame reading failure."""
        self.mock_cap.read.return_value = (False, None)

        camera = CameraCapture(logger=self.mock_logger)
        success, frame = camera.read()

        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_flip_horizontal(self):
        """Test horizontal frame flipping."""
        camera = CameraCapture(logger=self.mock_logger)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 0] = [255, 0, 0]  # Red pixel at top-left

        flipped = camera.flip_horizontal(frame)

        # Pixel should now be at top-right
        self.assertTrue(np.array_equal(flipped[0, -1], [255, 0, 0]))

    def test_adjust_brightness_contrast(self):
        """Test brightness and contrast adjustment."""
        camera = CameraCapture(logger=self.mock_logger)

        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Increase brightness
        adjusted = camera.adjust_brightness_contrast(frame, brightness=0.2, contrast=1.0)
        self.assertGreater(np.mean(adjusted), np.mean(frame))

    def test_set_resolution(self):
        """Test changing camera resolution."""
        camera = CameraCapture(logger=self.mock_logger)

        result = camera.set_resolution(1280, 720)

        self.assertTrue(result)
        self.mock_cap.set.assert_called()

    def test_close(self):
        """Test closing camera."""
        camera = CameraCapture(logger=self.mock_logger)

        camera.close()

        self.assertFalse(camera.is_open)
        self.mock_cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

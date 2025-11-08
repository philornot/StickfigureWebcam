#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for virtual camera module."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.camera.virtual_camera import VirtualCamera, try_create_virtual_camera


class TestVirtualCamera(unittest.TestCase):
    """Tests for VirtualCamera class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()

        # Mock pyvirtualcam
        self.mock_pyvirtualcam = MagicMock()
        self.mock_camera = MagicMock()
        self.mock_camera.backend = "obs"
        self.mock_pyvirtualcam.Camera.return_value = self.mock_camera

        self.pyvirtualcam_patch = patch("pyvirtualcam.Camera", self.mock_camera)
        self.pyvirtualcam_patch.start()

    def tearDown(self):
        """Clean up after tests."""
        self.pyvirtualcam_patch.stop()

    @patch("pyvirtualcam.Camera")
    def test_initialization_success(self, mock_camera_class):
        """Test successful virtual camera initialization."""
        mock_cam = MagicMock()
        mock_cam.backend = "obs"
        mock_camera_class.return_value = mock_cam

        vcam = VirtualCamera(width=640, height=480, fps=30, logger=self.mock_logger)

        self.assertEqual(vcam.width, 640)
        self.assertEqual(vcam.height, 480)
        self.assertEqual(vcam.backend, "obs")

    @patch("pyvirtualcam.Camera")
    def test_initialization_failure(self, mock_camera_class):
        """Test virtual camera initialization failure."""
        mock_camera_class.side_effect = RuntimeError("OBS not running")

        with self.assertRaises(RuntimeError) as context:
            VirtualCamera(logger=self.mock_logger)

        self.assertIn("OBS", str(context.exception))

    @patch("pyvirtualcam.Camera")
    def test_send_frame(self, mock_camera_class):
        """Test sending frame to virtual camera."""
        mock_cam = MagicMock()
        mock_cam.backend = "obs"
        mock_camera_class.return_value = mock_cam

        vcam = VirtualCamera(logger=self.mock_logger)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        vcam.send(frame)

        mock_cam.send.assert_called_once()

    @patch("pyvirtualcam.Camera")
    def test_send_frame_wrong_size(self, mock_camera_class):
        """Test sending frame with wrong size (should resize)."""
        mock_cam = MagicMock()
        mock_cam.backend = "obs"
        mock_camera_class.return_value = mock_cam

        vcam = VirtualCamera(width=640, height=480, logger=self.mock_logger)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Should resize automatically
        vcam.send(frame)

        mock_cam.send.assert_called_once()

    @patch("pyvirtualcam.Camera")
    def test_close(self, mock_camera_class):
        """Test closing virtual camera."""
        mock_cam = MagicMock()
        mock_cam.backend = "obs"
        mock_camera_class.return_value = mock_cam

        vcam = VirtualCamera(logger=self.mock_logger)
        vcam.close()

        mock_cam.close.assert_called_once()
        self.assertIsNone(vcam.camera)

    @patch("pyvirtualcam.Camera")
    def test_context_manager(self, mock_camera_class):
        """Test using virtual camera as context manager."""
        mock_cam = MagicMock()
        mock_cam.backend = "obs"
        mock_camera_class.return_value = mock_cam

        with VirtualCamera(logger=self.mock_logger) as vcam:
            self.assertIsNotNone(vcam.camera)

        mock_cam.close.assert_called_once()

    @patch("src.camera.virtual_camera.VirtualCamera")
    def test_try_create_success(self, mock_vcam_class):
        """Test try_create_virtual_camera with success."""
        mock_vcam = MagicMock()
        mock_vcam_class.return_value = mock_vcam

        result = try_create_virtual_camera(logger=self.mock_logger)

        self.assertIsNotNone(result)

    @patch("src.camera.virtual_camera.VirtualCamera")
    def test_try_create_failure(self, mock_vcam_class):
        """Test try_create_virtual_camera with failure."""
        mock_vcam_class.side_effect = RuntimeError("Failed")

        result = try_create_virtual_camera(logger=self.mock_logger)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

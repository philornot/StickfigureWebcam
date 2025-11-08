#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for system requirements check module (system_check.py).
"""

import unittest
from unittest.mock import MagicMock, patch

from src.utils.system_check import SystemCheck, check_system_requirements


class TestSystemCheck(unittest.TestCase):
    """
    Tests for SystemCheck class that checks availability of required components.
    """

    def setUp(self):
        """Initialize before each test."""
        # Mock logger
        self.mock_logger = MagicMock()

        # Patch system detection to always return 'Windows'
        self.platform_system_patch = patch("platform.system", return_value="Windows")
        self.platform_system = self.platform_system_patch.start()

        # Patch pyvirtualcam import
        self.pyvirtualcam_patch = patch("src.utils.system_check.PYVIRTUALCAM_AVAILABLE", True)
        self.pyvirtualcam_available = self.pyvirtualcam_patch.start()

        # Patch mediapipe import
        self.mediapipe_patch = patch("src.utils.system_check.MEDIAPIPE_AVAILABLE", True)
        self.mediapipe_available = self.mediapipe_patch.start()

        # CHANGE: Completely new approach to mocking cv2.VideoCapture
        # Instead of trying to mock get() method, we mock camera properties as PropertyMock types
        self.mock_camera = MagicMock()
        self.mock_camera.isOpened.return_value = True
        self.mock_camera.read.return_value = (True, MagicMock())

        # Prepare constants for OpenCV property IDs
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5

        # Prepare manually mocked values returned by get()
        self.camera_properties = {
            CAP_PROP_FRAME_WIDTH: 640,
            CAP_PROP_FRAME_HEIGHT: 480,
            CAP_PROP_FPS: 30,
        }

        # Mock get() method
        def mock_get(prop_id):
            # Convert prop_id to int, because it might be passed as float
            prop_id = int(prop_id)
            return self.camera_properties.get(prop_id, 0)

        self.mock_camera.get = mock_get

        # Patch cv2.VideoCapture constructor
        self.cv2_videocapture_patch = patch("cv2.VideoCapture", return_value=self.mock_camera)
        self.mock_cv2_videocapture = self.cv2_videocapture_patch.start()

        # Initialize SystemCheck
        self.system_check = SystemCheck(logger=self.mock_logger)

    def tearDown(self):
        """Clean up after each test."""
        self.platform_system_patch.stop()
        self.pyvirtualcam_patch.stop()
        self.mediapipe_patch.stop()
        self.cv2_videocapture_patch.stop()

    def test_initialization(self):
        """Test SystemCheck object initialization."""
        # Check if initialization completed correctly
        self.assertEqual(self.system_check.system, "Windows")
        self.assertIsNotNone(self.system_check.results)
        self.assertIsNotNone(self.system_check.install_links)

        # Check if all required components are in results
        for component in ["camera", "virtual_camera", "mediapipe", "obs", "v4l2loopback"]:
            self.assertIn(component, self.system_check.results)

    def test_check_camera_success(self):
        """Test camera check when camera is available."""
        # Check camera
        result = self.system_check.check_camera(camera_id=0)

        # Check results
        self.assertTrue(result["status"])
        self.assertIn("Camera with ID: 0 works correctly", result["message"])
        self.assertEqual(result["details"]["camera_id"], 0)
        self.assertEqual(result["details"]["width"], 640)
        self.assertEqual(result["details"]["height"], 480)
        self.assertEqual(result["details"]["fps"], 30)

    def test_check_camera_failure(self):
        """Test camera check when camera is not available."""
        # Change camera mock to simulate unavailability
        self.mock_camera.isOpened.return_value = False

        # Check camera
        result = self.system_check.check_camera(camera_id=0)

        # Check results
        self.assertFalse(result["status"])
        self.assertIn("Cannot open camera", result["message"])

    @patch("pyvirtualcam.Camera")
    def test_check_virtual_camera_success(self, mock_pyvirtualcam_camera):
        """Test virtual camera check when it is available."""
        # Configure pyvirtualcam.Camera mock
        mock_camera = MagicMock()
        mock_camera.backend = "obs"
        mock_camera.width = 320
        mock_camera.height = 240
        mock_camera.fps = 20
        mock_pyvirtualcam_camera.return_value = mock_camera

        # Check virtual camera
        result = self.system_check.check_virtual_camera()

        # Check results
        self.assertTrue(result["status"])
        self.assertIn("Virtual camera works correctly", result["message"])
        self.assertEqual(result["details"]["backend"], "obs")
        self.assertEqual(result["details"]["width"], 320)
        self.assertEqual(result["details"]["height"], 240)
        self.assertEqual(result["details"]["fps"], 20)

    @patch("pyvirtualcam.Camera")
    def test_check_virtual_camera_failure(self, mock_pyvirtualcam_camera):
        """Test virtual camera check when it is not available."""
        # Configure pyvirtualcam.Camera mock to throw exception
        mock_pyvirtualcam_camera.side_effect = Exception("Virtual camera not available")

        # Check virtual camera
        result = self.system_check.check_virtual_camera()

        # Check results
        self.assertFalse(result["status"])
        self.assertIn("Error during virtual camera check", result["message"])
        self.assertEqual(result["details"]["error"], "Virtual camera not available")

    @patch("mediapipe.solutions.pose")
    def test_check_mediapipe_success(self, mock_mp_pose):
        """Test MediaPipe check when it is available."""
        # Configure mock for MediaPipe
        mock_pose = MagicMock()
        mock_mp_pose.Pose.return_value = mock_pose

        # Add mock for MediaPipe version
        with patch("mediapipe.__version__", "0.8.10"):
            # Check MediaPipe
            result = self.system_check.check_mediapipe()

            # Check results
            self.assertTrue(result["status"])
            self.assertIn("MediaPipe works correctly", result["message"])
            self.assertEqual(result["details"]["version"], "0.8.10")

    def test_check_mediapipe_not_installed(self):
        """Test MediaPipe check when it is not installed."""
        # Patch MEDIAPIPE_AVAILABLE to False
        with patch("src.utils.system_check.MEDIAPIPE_AVAILABLE", False):
            # Check MediaPipe
            result = self.system_check.check_mediapipe()

            # Check results
            self.assertFalse(result["status"])
            self.assertIn("MediaPipe is not installed", result["message"])
            self.assertEqual(result["details"]["install_command"], "pip install mediapipe")

    @patch("os.path.exists")
    def test_check_obs_installed(self, mock_exists):
        """Test OBS check when it is installed."""
        # Configure mock for os.path.exists
        mock_exists.return_value = True

        # Check OBS
        result = self.system_check.check_obs()

        # Check results
        self.assertTrue(result["status"])
        self.assertIn("OBS Studio is installed", result["message"])

    @patch("os.path.exists")
    def test_check_obs_not_installed(self, mock_exists):
        """Test OBS check when it is not installed."""
        # Configure mock for os.path.exists
        mock_exists.return_value = False

        # Check OBS
        result = self.system_check.check_obs()

        # Check results
        self.assertFalse(result["status"])
        self.assertIn("OBS Studio is not installed", result["message"])

    def test_get_missing_components_none_missing(self):
        """Test getting missing components when all are available."""
        # Set all components as available
        for component in self.system_check.results:
            self.system_check.results[component]["status"] = True

        # Get missing components
        missing = self.system_check.get_missing_components()

        # Check if list is empty
        self.assertEqual(len(missing), 0)

    def test_get_missing_components_some_missing(self):
        """Test getting missing components when some are unavailable."""
        # First reset all to True
        for component in self.system_check.results:
            self.system_check.results[component]["status"] = True

        # Now set some as unavailable
        self.system_check.results["virtual_camera"]["status"] = False
        self.system_check.results["obs"]["status"] = False

        # Get missing components
        missing = self.system_check.get_missing_components()

        # Check number of missing components
        self.assertEqual(len(missing), 2)

        # Check if appropriate components are on the list
        missing_components = [item["name"] for item in missing]
        self.assertIn("virtual_camera", missing_components)
        self.assertIn("obs", missing_components)

    def test_are_all_requirements_met_all_met(self):
        """Test checking if all requirements are met when they are."""
        # Set all components as available
        self.system_check.results["camera"]["status"] = True
        self.system_check.results["virtual_camera"]["status"] = True
        self.system_check.results["mediapipe"]["status"] = True
        self.system_check.results["obs"]["status"] = True

        # Check if all requirements are met
        result = self.system_check.are_all_requirements_met()

        # Check result
        self.assertTrue(result)

    def test_are_all_requirements_met_some_not_met(self):
        """Test checking if all requirements are met when some are not."""
        # Set some components as unavailable
        self.system_check.results["camera"]["status"] = True
        self.system_check.results["virtual_camera"]["status"] = False
        self.system_check.results["mediapipe"]["status"] = True
        self.system_check.results["obs"]["status"] = True

        # Check if all requirements are met
        result = self.system_check.are_all_requirements_met()

        # Check result
        self.assertFalse(result)

    def test_get_installation_instructions(self):
        """Test getting installation instructions for missing components."""
        # Set some components as unavailable
        self.system_check.results["camera"]["status"] = True
        self.system_check.results["virtual_camera"]["status"] = False
        self.system_check.results["mediapipe"]["status"] = False
        self.system_check.results["obs"]["status"] = True

        # Get installation instructions
        instructions = self.system_check.get_installation_instructions()

        # Check if instructions contain appropriate components
        self.assertIn("virtual_camera", instructions)
        self.assertIn("mediapipe", instructions)
        self.assertNotIn("camera", instructions)
        self.assertNotIn("obs", instructions)

        # Check if virtual_camera instructions contain OBS (for Windows)
        self.assertTrue(any("OBS" in instr for instr in instructions["virtual_camera"]))

    @patch("src.utils.system_check.SystemCheck")
    def test_check_system_requirements(self, mock_system_check_class):
        """Test check_system_requirements function."""
        # Configure mock for SystemCheck
        mock_system_check = MagicMock()
        mock_system_check.check_all.return_value = {"some": "results"}
        mock_system_check.are_all_requirements_met.return_value = True
        mock_system_check.get_missing_components.return_value = []
        mock_system_check.get_installation_instructions.return_value = {}

        mock_system_check_class.return_value = mock_system_check

        # Call function
        all_met, results = check_system_requirements(logger=self.mock_logger)

        # Check results
        self.assertTrue(all_met)
        self.assertEqual(results["all_met"], True)
        self.assertEqual(results["missing"], [])
        self.assertEqual(results["instructions"], {})

        # Check if SystemCheck methods were called
        mock_system_check.check_all.assert_called_once()
        mock_system_check.are_all_requirements_met.assert_called_once()
        mock_system_check.get_missing_components.assert_called_once()
        mock_system_check.get_installation_instructions.assert_called_once()


if __name__ == "__main__":
    unittest.main()

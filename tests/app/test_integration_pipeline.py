#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integration tests for video pipeline."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.app.config_manager import ConfigurationManager
from src.app.video_pipeline import VideoPipeline


class TestVideoPipelineIntegration(unittest.TestCase):
    """Integration tests for complete video processing pipeline."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()

        # Create realistic config
        self.config_manager = ConfigurationManager(logger=self.mock_logger)

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.face_mesh")
    @patch("mediapipe.solutions.hands")
    @patch("pyvirtualcam.Camera")
    def test_full_pipeline_with_detection(self, mock_vcam, mock_hands, mock_face_mesh, mock_cv2):
        """Test complete pipeline with face detection."""
        # Setup camera mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30}.get(int(x), 0)
        mock_cv2.return_value = mock_cap

        # Setup face mesh mock with detected face
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 0.9

        mock_face_landmarks = MagicMock()
        mock_face_landmarks.landmark = [mock_landmark] * 468  # Face mesh has 468 points

        mock_face_results = MagicMock()
        mock_face_results.multi_face_landmarks = [mock_face_landmarks]

        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.process.return_value = mock_face_results
        mock_face_mesh.FaceMesh.return_value = mock_face_mesh_instance

        # Setup hands mock with detected hands
        mock_hand_landmark = MagicMock()
        mock_hand_landmark.x = 0.3
        mock_hand_landmark.y = 0.6
        mock_hand_landmark.z = 0.0
        mock_hand_landmark.visibility = 0.8

        mock_hand_landmarks = MagicMock()
        mock_hand_landmarks.landmark = [mock_hand_landmark] * 21  # Hands have 21 points

        mock_hands_results = MagicMock()
        mock_hands_results.multi_hand_landmarks = [mock_hand_landmarks]

        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = mock_hands_results
        mock_hands.Hands.return_value = mock_hands_instance

        # Setup virtual camera mock
        mock_vcam_instance = MagicMock()
        mock_vcam.return_value = mock_vcam_instance

        # Create and initialize pipeline
        pipeline = VideoPipeline(
            self.config_manager.get_camera_config(),
            self.config_manager.get_processing_config(),
            self.mock_logger,
        )

        init_result = pipeline.initialize()
        self.assertTrue(init_result)

        # Process several frames
        for i in range(5):
            result = pipeline.process_frame()

            self.assertIsNotNone(result)
            self.assertIn("original_frame", result)
            self.assertIn("processed_frame", result)
            self.assertIn("face_data", result)

            # Verify face was detected
            self.assertTrue(result["face_data"]["has_face"])

            # Verify stick figure was rendered
            processed = result["processed_frame"]
            self.assertEqual(processed.shape, (480, 640, 3))

        pipeline.shutdown()

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.face_mesh")
    @patch("mediapipe.solutions.hands")
    def test_pipeline_without_detection(self, mock_hands, mock_face_mesh, mock_cv2):
        """Test pipeline when no face is detected."""
        # Setup camera mock
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30}.get(int(x), 0)
        mock_cv2.return_value = mock_cap

        # No face detected
        mock_face_results = MagicMock()
        mock_face_results.multi_face_landmarks = None

        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.process.return_value = mock_face_results
        mock_face_mesh.FaceMesh.return_value = mock_face_mesh_instance

        # No hands detected
        mock_hands_results = MagicMock()
        mock_hands_results.multi_hand_landmarks = None

        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = mock_hands_results
        mock_hands.Hands.return_value = mock_hands_instance

        # Create and initialize pipeline
        pipeline = VideoPipeline(
            self.config_manager.get_camera_config(),
            self.config_manager.get_processing_config(),
            self.mock_logger,
        )

        pipeline.initialize()

        # Process frame
        result = pipeline.process_frame()

        # Should still render default stick figure
        self.assertIsNotNone(result)
        self.assertFalse(result["face_data"]["has_face"])
        self.assertIsNotNone(result["processed_frame"])

        pipeline.shutdown()


if __name__ == "__main__":
    unittest.main()

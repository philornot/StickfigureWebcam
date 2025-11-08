#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for video pipeline."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.app.video_pipeline import VideoPipeline


class TestVideoPipeline(unittest.TestCase):
    """Tests for VideoPipeline class."""

    def setUp(self):
        """Initialize before each test."""
        self.mock_logger = MagicMock()

        self.camera_config = {"id": 0, "width": 640, "height": 480, "fps": 30}

        self.processing_config = {
            "line_thickness": 3,
            "bg_color": [255, 255, 255],
            "figure_color": [0, 0, 0],
        }

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.face_mesh")
    @patch("mediapipe.solutions.hands")
    def test_initialization(self, mock_hands, mock_face_mesh, mock_cv2):
        """Test pipeline initialization."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30}.get(int(x), 0)
        mock_cv2.return_value = mock_cap

        pipeline = VideoPipeline(self.camera_config, self.processing_config, self.mock_logger)

        result = pipeline.initialize()

        self.assertTrue(result)
        self.assertIsNotNone(pipeline.camera)

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.face_mesh")
    @patch("mediapipe.solutions.hands")
    def test_process_frame_success(self, mock_hands, mock_face_mesh, mock_cv2):
        """Test successful frame processing."""
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30}.get(int(x), 0)
        mock_cv2.return_value = mock_cap

        # Mock face mesh results
        mock_face_results = MagicMock()
        mock_face_results.multi_face_landmarks = None

        mock_face_mesh_instance = MagicMock()
        mock_face_mesh_instance.process.return_value = mock_face_results
        mock_face_mesh.FaceMesh.return_value = mock_face_mesh_instance

        # Mock hands results
        mock_hands_results = MagicMock()
        mock_hands_results.multi_hand_landmarks = None

        mock_hands_instance = MagicMock()
        mock_hands_instance.process.return_value = mock_hands_results
        mock_hands.Hands.return_value = mock_hands_instance

        pipeline = VideoPipeline(self.camera_config, self.processing_config, self.mock_logger)
        pipeline.initialize()

        result = pipeline.process_frame()

        self.assertIsNotNone(result)
        self.assertIn("original_frame", result)
        self.assertIn("processed_frame", result)
        self.assertIn("fps", result)

    @patch("cv2.VideoCapture")
    def test_camera_init_failure(self, mock_cv2):
        """Test pipeline behavior when camera fails to initialize."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap

        pipeline = VideoPipeline(self.camera_config, self.processing_config, self.mock_logger)

        result = pipeline.initialize()

        self.assertFalse(result)

    @patch("cv2.VideoCapture")
    @patch("mediapipe.solutions.face_mesh")
    @patch("mediapipe.solutions.hands")
    def test_analyze_expressions(self, mock_hands, mock_face_mesh, mock_cv2):
        """Test facial expression analysis."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda x: {3: 640, 4: 480, 5: 30}.get(int(x), 0)
        mock_cv2.return_value = mock_cap

        pipeline = VideoPipeline(self.camera_config, self.processing_config, self.mock_logger)

        # Create mock landmarks
        landmarks = [(0.5, 0.5, 0.0, 0.9) for _ in range(500)]

        expressions = pipeline._analyze_expressions(landmarks)

        self.assertIn("mouth_open", expressions)
        self.assertIn("smile", expressions)
        self.assertIn("left_eye_open", expressions)
        self.assertIn("right_eye_open", expressions)


if __name__ == "__main__":
    unittest.main()

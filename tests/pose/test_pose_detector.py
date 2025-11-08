#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/pose/test_pose_detector.py
"""
Testy jednostkowe dla detektora pozy (PoseDetector).
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from src.pose.pose_detector import PoseDetector


class TestPoseDetector(unittest.TestCase):
    """
    Testy dla klasy PoseDetector, która wykrywa pozę człowieka na obrazie.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Patchujemy MediaPipe, aby nie musiał rzeczywiście inicjalizować modeli
        self.mp_pose_patch = patch("mediapipe.solutions.pose")
        self.mp_drawing_patch = patch("mediapipe.solutions.drawing_utils")
        self.mp_drawing_styles_patch = patch("mediapipe.solutions.drawing_styles")

        self.mock_mp_pose = self.mp_pose_patch.start()
        self.mock_mp_drawing = self.mp_drawing_patch.start()
        self.mock_mp_drawing_styles = self.mp_drawing_styles_patch.start()

        # Tworzymy mocka dla obiektu Pose
        self.mock_pose = Mock()
        self.mock_mp_pose.Pose.return_value = self.mock_pose

        # Mock loggera
        self.mock_logger = MagicMock()

        # Inicjalizacja detektora pozy z mockami
        self.detector = PoseDetector(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
            smooth_landmarks=True,
            logger=self.mock_logger,
        )

    def tearDown(self):
        """Sprzątanie po każdym teście."""
        self.mp_pose_patch.stop()
        self.mp_drawing_patch.stop()
        self.mp_drawing_styles_patch.stop()

    def test_initialization(self):
        """Test inicjalizacji detektora pozy."""
        # Sprawdzamy czy inicjalizacja przebiegła poprawnie
        self.assertEqual(self.detector.min_detection_confidence, 0.5)
        self.assertEqual(self.detector.min_tracking_confidence, 0.5)
        self.assertEqual(self.detector.model_complexity, 1)
        self.assertTrue(self.detector.smooth_landmarks)

        # Sprawdzamy czy MediaPipe Pose został zainicjalizowany z poprawnymi parametrami
        self.mock_mp_pose.Pose.assert_called_once_with(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.info.assert_called_once()

    def test_detect_pose_no_landmarks(self):
        """Test detekcji pozy gdy MediaPipe nie wykryje punktów charakterystycznych."""
        # Tworzymy obraz testowy
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Konfigurujemy mocka procesu MediaPipe, aby zwracał brak wykrytych punktów
        mock_results = MagicMock()
        mock_results.pose_landmarks = None
        self.mock_pose.process.return_value = mock_results

        # Wywołujemy detekcję pozy
        success, pose_data = self.detector.detect_pose(test_image)

        # Sprawdzamy wyniki
        self.assertFalse(success)
        self.assertFalse(pose_data["has_pose"])
        self.assertIsNone(pose_data["landmarks"])
        self.assertEqual(pose_data["frame_height"], 480)
        self.assertEqual(pose_data["frame_width"], 640)

    def test_detect_pose_with_landmarks(self):
        """Test detekcji pozy gdy MediaPipe wykryje punkty charakterystyczne."""
        # Tworzymy obraz testowy
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Konfigurujemy mocka procesu MediaPipe, aby zwracał wykryte punkty
        mock_results = MagicMock()

        # Tworzymy przykładowe punkty charakterystyczne (33 punkty dla MediaPipe Pose)
        mock_landmarks = MagicMock()
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0
        mock_landmark.visibility = 0.9

        # Ustawiamy 33 punkty (MediaPipe Pose używa 33 punktów)
        mock_landmarks.landmark = [mock_landmark] * 33
        mock_results.pose_landmarks = mock_landmarks

        # Dodajemy również world landmarks
        mock_world_landmarks = MagicMock()
        mock_world_landmarks.landmark = [mock_landmark] * 33
        mock_results.pose_world_landmarks = mock_world_landmarks

        self.mock_pose.process.return_value = mock_results

        # Wywołujemy detekcję pozy
        success, pose_data = self.detector.detect_pose(test_image)

        # Sprawdzamy wyniki
        self.assertTrue(success)
        self.assertTrue(pose_data["has_pose"])
        self.assertIsNotNone(pose_data["landmarks"])
        self.assertEqual(len(pose_data["landmarks"]), 33)
        self.assertAlmostEqual(pose_data["detection_score"], 0.9)

    def test_draw_pose_on_image(self):
        """Test rysowania pozy na obrazie."""
        # Tworzymy obraz testowy
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Tworzymy przykładowe punkty charakterystyczne
        landmarks = [(0.5, 0.5, 0.0, 0.9)] * 33

        # Wywołujemy metodę rysowania
        result_image = self.detector.draw_pose_on_image(
            test_image, landmarks, draw_connections=True
        )

        # Sprawdzamy czy wynikowy obraz ma właściwe wymiary
        self.assertEqual(result_image.shape, test_image.shape)

        # Trudno sprawdzić dokładnie co zostało narysowane bez faktycznego renderowania,
        # ale możemy sprawdzić czy obraz został zmodyfikowany
        self.assertFalse(np.array_equal(result_image, test_image))

    def test_calculate_angle(self):
        """Test obliczania kąta między trzema punktami."""
        # Tworzymy przykładowe punkty
        # Punkty tworzące kąt prosty (90 stopni)
        landmarks = [
            (0.0, 0.0, 0.0, 1.0),  # punkt 0
            (0.0, 1.0, 0.0, 1.0),  # punkt 1 (wierzchołek kąta)
            (1.0, 1.0, 0.0, 1.0),  # punkt 2
        ]

        # Obliczamy kąt
        angle = self.detector.calculate_angle(landmarks, 0, 1, 2)

        # Sprawdzamy czy kąt jest bliski 90 stopni
        self.assertAlmostEqual(angle, 90.0, delta=1.0)

        # Teraz tworzymy punkty tworzące kąt 45 stopni
        landmarks = [
            (0.0, 0.0, 0.0, 1.0),  # punkt 0
            (0.0, 0.0, 0.0, 1.0),  # punkt 1 (wierzchołek kąta)
            (1.0, 1.0, 0.0, 1.0),  # punkt 2
        ]

        # Obliczamy kąt
        angle = self.detector.calculate_angle(landmarks, 0, 1, 2)

        # Sprawdzamy czy kąt jest bliski 45 stopni
        self.assertAlmostEqual(angle, 45.0, delta=1.0)

    def test_get_landmark_position(self):
        """Test pobierania pozycji konkretnego punktu charakterystycznego."""
        # Tworzymy przykładowe punkty
        landmarks = [
            (0.1, 0.2, 0.0, 0.9),  # punkt 0
            (0.3, 0.4, 0.0, 0.8),  # punkt 1
            (0.5, 0.6, 0.0, 0.7),  # punkt 2
        ]

        # Pobieramy pozycję punktu 1
        position = self.detector.get_landmark_position(landmarks, 1, 640, 480)

        # Sprawdzamy czy pozycja jest poprawna
        self.assertEqual(position[0], int(0.3 * 640))  # x
        self.assertEqual(position[1], int(0.4 * 480))  # y
        self.assertEqual(position[2], 0.0)  # z
        self.assertEqual(position[3], 0.8)  # visibility

    def test_get_detection_stats(self):
        """Test pobierania statystyk detekcji."""
        # Aktualizujemy liczniki detekcji
        self.detector.frame_count = 100
        self.detector.detection_count = 80
        self.detector.last_detection_score = 0.85

        # Pobieramy statystyki
        stats = self.detector.get_detection_stats()

        # Sprawdzamy statystyki
        self.assertEqual(stats["total_frames"], 100)
        self.assertEqual(stats["detection_count"], 80)
        self.assertEqual(stats["detection_ratio"], 0.8)
        self.assertEqual(stats["last_detection_score"], 0.85)
        self.assertEqual(stats["model_complexity"], 1)

    def test_reset_stats(self):
        """Test resetowania statystyk detekcji."""
        # Aktualizujemy liczniki detekcji
        self.detector.frame_count = 100
        self.detector.detection_count = 80
        self.detector.last_detection_score = 0.85

        # Resetujemy statystyki
        self.detector.reset_stats()

        # Sprawdzamy czy statystyki zostały zresetowane
        stats = self.detector.get_detection_stats()
        self.assertEqual(stats["total_frames"], 0)
        self.assertEqual(stats["detection_count"], 0)
        self.assertEqual(stats["detection_ratio"], 0)
        self.assertEqual(stats["last_detection_score"], 0.0)

    def test_close(self):
        """Test zamykania detektora pozy."""
        # Wywołujemy zamknięcie
        self.detector.close()

        # Sprawdzamy czy pose.close() został wywołany
        self.mock_pose.close.assert_called_once()

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()

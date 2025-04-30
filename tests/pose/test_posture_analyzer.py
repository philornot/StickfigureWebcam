#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/pose/test_posture_analyzer.py
"""
Testy jednostkowe dla analizatora postawy (PostureAnalyzer).
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from src.pose.posture_analyzer import PostureAnalyzer


class TestPostureAnalyzer(unittest.TestCase):
    """
    Testy dla klasy PostureAnalyzer, która analizuje pozę i określa czy użytkownik siedzi czy stoi.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Mock loggera
        self.mock_logger = MagicMock()

        # Tworzenie analizatora postawy z mockiem loggera
        self.analyzer = PostureAnalyzer(
            standing_hip_threshold=0.7,
            confidence_threshold=0.6,
            smoothing_factor=0.7,
            temporal_smoothing=3,
            logger=self.mock_logger
        )

    def test_analyze_posture_no_landmarks(self):
        """Test analizy postawy gdy brak punktów charakterystycznych."""
        result = self.analyzer.analyze_posture(None, 480, 640)

        # Sprawdzenie czy wynik zawiera oczekiwane klucze
        self.assertIsNone(result["is_sitting"])
        self.assertEqual(result["posture"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["visible_keypoints"], 0)

    def test_analyze_posture_too_few_landmarks(self):
        """Test analizy postawy gdy jest za mało punktów charakterystycznych."""
        # Tworzymy tylko 10 punktów zamiast wymaganych 33
        landmarks = [(0, 0, 0, 0.9)] * 10

        result = self.analyzer.analyze_posture(landmarks, 480, 640)

        self.assertIsNone(result["is_sitting"])
        self.assertEqual(result["posture"], "unknown")

    def test_analyze_posture_sitting(self):
        """Test analizy postawy w pozycji siedzącej."""
        # Tworzymy sztuczne dane punktów charakterystycznych dla postawy siedzącej
        # Format: (x, y, z, visibility)
        landmarks = [(0, 0, 0, 0.9)] * 33  # Najpierw inicjalizujemy wszystkie punkty

        # Teraz nadpisujemy konkretne punkty, które są ważne dla detekcji
        # Biodra umieszczamy nisko (wartość y > 0.7)
        landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.8, 0, 0.9)
        landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.8, 0, 0.9)

        # Ramiona umieszczamy wyżej
        landmarks[self.analyzer.LEFT_SHOULDER] = (0.35, 0.5, 0, 0.9)
        landmarks[self.analyzer.RIGHT_SHOULDER] = (0.65, 0.5, 0, 0.9)

        # Kolana umieszczamy nisko ale z niską widocznością (ukryte pod biurkiem)
        landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.9, 0, 0.3)
        landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.9, 0, 0.3)

        # Kostki są niewidoczne (ukryte pod biurkiem)
        landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.95, 0, 0.1)
        landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.95, 0, 0.1)

        # Symulujemy kilka klatek tej samej pozy aby uzyskać stabilną detekcję
        for _ in range(5):
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Po kilku klatkach powinniśmy mieć stabilną detekcję siedzenia
        self.assertTrue(result["is_sitting"])
        self.assertEqual(result["posture"], "sitting")
        self.assertGreater(result["sitting_probability"], 0.6)

    def test_analyze_posture_standing(self):
        """Test analizy postawy w pozycji stojącej."""
        # Tworzymy sztuczne dane punktów charakterystycznych dla postawy stojącej
        landmarks = [(0, 0, 0, 0.9)] * 33

        # Teraz nadpisujemy konkretne punkty, które są ważne dla detekcji
        # Biodra umieszczamy wysoko (wartość y < 0.7)
        landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.6, 0, 0.9)
        landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.6, 0, 0.9)

        # Ramiona umieszczamy jeszcze wyżej
        landmarks[self.analyzer.LEFT_SHOULDER] = (0.35, 0.3, 0, 0.9)
        landmarks[self.analyzer.RIGHT_SHOULDER] = (0.65, 0.3, 0, 0.9)

        # Kolana są widoczne
        landmarks[self.analyzer.LEFT_KNEE] = (0.4, 0.75, 0, 0.9)
        landmarks[self.analyzer.RIGHT_KNEE] = (0.6, 0.75, 0, 0.9)

        # Kostki są widoczne
        landmarks[self.analyzer.LEFT_ANKLE] = (0.4, 0.9, 0, 0.9)
        landmarks[self.analyzer.RIGHT_ANKLE] = (0.6, 0.9, 0, 0.9)

        # Symulujemy kilka klatek tej samej pozy aby uzyskać stabilną detekcję
        for _ in range(5):
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Po kilku klatkach powinniśmy mieć stabilną detekcję stania
        self.assertFalse(result["is_sitting"])
        self.assertEqual(result["posture"], "standing")
        self.assertLess(result["sitting_probability"], 0.4)

    def test_temporal_smoothing(self):
        """Test wygładzania czasowego detekcji."""
        # Najpierw symulujemy 5 klatek osoby siedzącej
        sitting_landmarks = [(0, 0, 0, 0.9)] * 33
        sitting_landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.8, 0, 0.9)
        sitting_landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.8, 0, 0.9)

        for _ in range(5):
            result = self.analyzer.analyze_posture(sitting_landmarks, 480, 640)

        # Zapamiętujemy stan i prawdopodobieństwo
        sitting_probability = result["sitting_probability"]
        self.assertTrue(result["is_sitting"])

        # Teraz symulujemy 1 klatkę osoby stojącej - nie powinno to jeszcze zmienić detekcji
        standing_landmarks = [(0, 0, 0, 0.9)] * 33
        standing_landmarks[self.analyzer.LEFT_HIP] = (0.4, 0.5, 0, 0.9)
        standing_landmarks[self.analyzer.RIGHT_HIP] = (0.6, 0.5, 0, 0.9)

        result = self.analyzer.analyze_posture(standing_landmarks, 480, 640)

        # Po jednej klatce stojącej detekcja wciąż powinna pokazywać siedzenie
        # ale probability powinno się trochę zmniejszyć
        self.assertTrue(result["is_sitting"])
        self.assertLess(result["sitting_probability"], sitting_probability)

        # Symulujemy jeszcze kilka klatek stojących - teraz powinno zmienić detekcję
        for _ in range(5):
            result = self.analyzer.analyze_posture(standing_landmarks, 480, 640)

        # Teraz powinniśmy wykryć stanie
        self.assertFalse(result["is_sitting"])
        self.assertEqual(result["posture"], "standing")

    def test_update_thresholds(self):
        """Test aktualizacji progów detekcji."""
        # Zapisujemy początkowe wartości
        initial_hip_threshold = self.analyzer.standing_hip_threshold
        initial_confidence_threshold = self.analyzer.confidence_threshold
        initial_smoothing_factor = self.analyzer.smoothing_factor

        # Aktualizujemy wartości
        new_hip_threshold = 0.6
        new_confidence_threshold = 0.7
        new_smoothing_factor = 0.5

        self.analyzer.update_thresholds(
            standing_hip_threshold=new_hip_threshold,
            confidence_threshold=new_confidence_threshold,
            smoothing_factor=new_smoothing_factor
        )

        # Sprawdzamy czy wartości zostały zaktualizowane
        self.assertEqual(self.analyzer.standing_hip_threshold, new_hip_threshold)
        self.assertEqual(self.analyzer.confidence_threshold, new_confidence_threshold)
        self.assertEqual(self.analyzer.smoothing_factor, new_smoothing_factor)

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.info.assert_called_once()

    def test_reset(self):
        """Test resetowania stanu analizatora."""
        # Najpierw symulujemy kilka klatek aby ustawić stan
        landmarks = [(0, 0, 0, 0.9)] * 33

        for _ in range(3):
            self.analyzer.analyze_posture(landmarks, 480, 640)

        # Resetujemy stan
        self.analyzer.reset()

        # Sprawdzamy czy stan został zresetowany
        self.assertIsNone(self.analyzer.is_sitting)
        self.assertEqual(self.analyzer.sitting_probability, 0.5)
        self.assertEqual(self.analyzer.history_buffer, [])
        self.assertEqual(self.analyzer.consecutive_frames, 0)

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.debug.assert_called()


if __name__ == "__main__":
    unittest.main()
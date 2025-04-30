#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testy sprawdzające jak system radzi sobie z częściową widocznością ciała.
"""

import unittest
from unittest.mock import MagicMock

from src.pose.pose_detector import PoseDetector
from src.pose.posture_analyzer import PostureAnalyzer


class TestPartialVisibility(unittest.TestCase):
    """
    Testy weryfikujące detekcję pozy przy ograniczonej widoczności ciała,
    np. gdy widoczna jest tylko górna część ciała typowa dla wideokonferencji.
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
            partial_visibility_bias=0.9,  # Wysoka preferencja siedzenia
            logger=self.mock_logger
        )

    def create_upper_body_only_landmarks(self):
        """
        Tworzy punkty charakterystyczne z wysoką widocznością tylko dla górnej części ciała
        (głowa, ramiona, łokcie), a niską widocznością dla dolnej części ciała (biodra, nogi).

        Returns:
            list: Lista 33 punktów charakterystycznych w formacie (x, y, z, visibility)
        """
        # Tworzymy 33 puste punkty z niską widocznością
        landmarks = [(0, 0, 0, 0.1)] * 33

        # Głowa i górna część ciała - wysoka widoczność
        landmarks[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.9)  # Nos
        landmarks[PoseDetector.LEFT_EYE] = (0.45, 0.08, 0, 0.9)  # Lewe oko
        landmarks[PoseDetector.RIGHT_EYE] = (0.55, 0.08, 0, 0.9)  # Prawe oko
        landmarks[PoseDetector.LEFT_EAR] = (0.4, 0.1, 0, 0.9)  # Lewe ucho
        landmarks[PoseDetector.RIGHT_EAR] = (0.6, 0.1, 0, 0.9)  # Prawe ucho

        # Ramiona - wysoka widoczność
        landmarks[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.9)
        landmarks[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.9)

        # Łokcie - średnia/wysoka widoczność
        landmarks[PoseDetector.LEFT_ELBOW] = (0.3, 0.3, 0, 0.8)
        landmarks[PoseDetector.RIGHT_ELBOW] = (0.7, 0.3, 0, 0.8)

        # Nadgarstki - średnia widoczność (mogą być częściowo poza kadrem)
        landmarks[PoseDetector.LEFT_WRIST] = (0.25, 0.4, 0, 0.7)
        landmarks[PoseDetector.RIGHT_WRIST] = (0.75, 0.4, 0, 0.7)

        # Biodra - niska widoczność (poza kadrem)
        landmarks[PoseDetector.LEFT_HIP] = (0.45, 0.5, 0, 0.3)
        landmarks[PoseDetector.RIGHT_HIP] = (0.55, 0.5, 0, 0.3)

        # Nogi - bardzo niska widoczność (poza kadrem)
        landmarks[PoseDetector.LEFT_KNEE] = (0.43, 0.7, 0, 0.1)
        landmarks[PoseDetector.RIGHT_KNEE] = (0.57, 0.7, 0, 0.1)
        landmarks[PoseDetector.LEFT_ANKLE] = (0.42, 0.9, 0, 0.1)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0.58, 0.9, 0, 0.1)

        return landmarks

    def test_upper_body_only_detection(self):
        """
        Test weryfikujący czy system poprawnie zakłada pozycję siedzącą,
        gdy widoczna jest tylko górna część ciała.
        """
        # Pobierz punkty charakterystyczne tylko górnej części ciała
        landmarks = self.create_upper_body_only_landmarks()

        # Symulujemy wiele klatek tej samej pozy aby uzyskać stabilną detekcję
        for _ in range(10):  # Zwiększamy liczbę klatek dla pewności
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Sprawdzamy czy system prawidłowo rozpoznał pozycję siedzącą
        self.assertTrue(result["is_sitting"],
                        "System powinien zakładać pozycję siedzącą przy widoczności tylko górnej części ciała")
        self.assertEqual(result["posture"], "sitting")
        self.assertGreaterEqual(result["sitting_probability"], 0.7,
                                "Prawdopodobieństwo siedzenia powinno być wysokie przy ograniczonej widoczności")

        # Sprawdzamy czy analiza oparta na widoczności nóg daje wysokie prawdopodobieństwo siedzenia
        leg_score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(leg_score, 0.8,
                                "Analiza widoczności nóg powinna dawać wysokie prawdopodobieństwo siedzenia")

    def test_torso_only_detection(self):
        """
        Test weryfikujący zachowanie systemu przy ekstremalnym scenariuszu,
        gdy widoczna jest bardzo ograniczona część ciała (tylko tors).
        """
        # Tworzymy 33 puste punkty z niską widocznością
        landmarks = [(0, 0, 0, 0.1)] * 33

        # Tylko ramiona są dobrze widoczne
        landmarks[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.8)
        landmarks[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.8)

        # Nos i głowa - średnia widoczność
        landmarks[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.7)

        # Łokcie - niska/średnia widoczność
        landmarks[PoseDetector.LEFT_ELBOW] = (0.3, 0.3, 0, 0.6)
        landmarks[PoseDetector.RIGHT_ELBOW] = (0.7, 0.3, 0, 0.6)

        # Symulujemy wiele klatek tej samej pozy aby uzyskać stabilną detekcję
        for _ in range(15):  # Więcej klatek dla bardzo ograniczonej widoczności
            result = self.analyzer.analyze_posture(landmarks, 480, 640)

        # Sprawdzamy wynik detekcji
        self.assertTrue(result["is_sitting"],
                        "System powinien preferować pozycję siedzącą przy bardzo ograniczonej widoczności")
        self.assertEqual(result["posture"], "sitting")
        self.assertGreaterEqual(result["sitting_probability"], 0.6)

    def test_leg_visibility_analysis(self):
        """
        Szczegółowy test funkcji analizy widoczności nóg, która jest kluczowa
        przy określaniu pozy przy ograniczonej widoczności.
        """
        # Test gdy żadna część nóg nie jest widoczna
        landmarks = [(0, 0, 0, 0.9)] * 33  # Wszystkie punkty z wysoką widocznością
        # Nadpisujemy widoczność nóg na niską
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.RIGHT_KNEE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.LEFT_ANKLE] = (0, 0, 0, 0.1)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0, 0, 0, 0.1)

        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(score, 0.9,
                                "Brak widoczności nóg powinien dawać najwyższe prawdopodobieństwo siedzenia")

        # Test gdy tylko jedno kolano jest widoczne
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.7)  # Tylko lewe kolano widoczne
        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertGreaterEqual(score, 0.7,
                                "Jedna widoczna część nóg powinna dawać wysokie prawdopodobieństwo siedzenia")

        # Test gdy wszystkie części nóg są widoczne
        landmarks[PoseDetector.LEFT_KNEE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.RIGHT_KNEE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.LEFT_ANKLE] = (0, 0, 0, 0.7)
        landmarks[PoseDetector.RIGHT_ANKLE] = (0, 0, 0, 0.7)
        score = self.analyzer._analyze_leg_visibility(landmarks)
        self.assertLessEqual(score, 0.2, "Pełna widoczność nóg powinna dawać niskie prawdopodobieństwo siedzenia")

    def test_visibility_type_detection(self):
        """
        Test sprawdzający wykrywanie typu widoczności ciała.
        """
        # Test dla pełnej widoczności ciała
        full_body = [(0, 0, 0, 0.9)] * 33  # Wszystkie punkty widoczne
        visibility_type = self.analyzer._analyze_visibility_type(full_body)
        self.assertEqual(visibility_type, "full_body", "Powinien wykryć pełną widoczność ciała")

        # Test dla widoczności tylko górnej części ciała
        upper_body = self.create_upper_body_only_landmarks()
        visibility_type = self.analyzer._analyze_visibility_type(upper_body)
        self.assertEqual(visibility_type, "upper_body", "Powinien wykryć widoczność tylko górnej części ciała")

        # Test dla częściowej widoczności
        partial_body = [(0, 0, 0, 0.1)] * 33  # Początkowo niska widoczność
        # Dodajemy tylko kilka widocznych punktów górnej części ciała
        partial_body[PoseDetector.NOSE] = (0.5, 0.1, 0, 0.8)
        partial_body[PoseDetector.LEFT_SHOULDER] = (0.4, 0.2, 0, 0.8)
        partial_body[PoseDetector.RIGHT_SHOULDER] = (0.6, 0.2, 0, 0.8)

        visibility_type = self.analyzer._analyze_visibility_type(partial_body)
        self.assertEqual(visibility_type, "partial_visibility", "Powinien wykryć częściową widoczność")

    def test_analyze_partial_visibility(self):
        """
        Test sprawdzający funkcję analizy częściowej widoczności.
        """
        landmarks = [(0, 0, 0, 0.9)] * 33  # Punkty nie są istotne dla tej funkcji

        # Przy pełnej widoczności powinniśmy mieć neutralny wynik
        score = self.analyzer._analyze_partial_visibility(landmarks, "full_body")
        self.assertAlmostEqual(score, 0.5, places=1, msg="Pełna widoczność powinna dawać neutralny wynik")

        # Przy widoczności tylko górnej części ciała powinniśmy mieć wysokie prawdopodobieństwo siedzenia
        score = self.analyzer._analyze_partial_visibility(landmarks, "upper_body")
        self.assertGreaterEqual(score, 0.8, "Górna część ciała powinna dawać wysokie prawdopodobieństwo siedzenia")

        # Przy częściowej widoczności również powinniśmy mieć wysokie prawdopodobieństwo
        score = self.analyzer._analyze_partial_visibility(landmarks, "partial_visibility")
        self.assertGreaterEqual(score, 0.7, "Częściowa widoczność powinna dawać wysokie prawdopodobieństwo siedzenia")

        # Przy nieznanym typie powinniśmy mieć wartość z konfiguracji
        score = self.analyzer._analyze_partial_visibility(landmarks, "unknown")
        self.assertEqual(score, self.analyzer.partial_visibility_bias,
                         "Nieznany typ powinien używać wartości partial_visibility_bias z konfiguracji")


if __name__ == "__main__":
    unittest.main()

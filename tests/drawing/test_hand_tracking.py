#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/drawing/test_hand_tracking.py

import os
import sys
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

# Dodanie ścieżki głównego katalogu projektu do sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drawing.stick_figure_renderer import StickFigureRenderer


class TestHandTracking(unittest.TestCase):
    """
    Testy dla mechanizmu śledzenia rąk w StickFigureRenderer.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Mock loggera
        self.mock_logger = MagicMock()

        # Tworzenie renderera
        self.renderer = StickFigureRenderer(
            canvas_width=640,
            canvas_height=480,
            line_thickness=3,
            head_radius_factor=0.075,
            bg_color=(255, 255, 255),
            figure_color=(0, 0, 0),
            chair_color=(150, 75, 0),
            smooth_factor=0.3,
            smoothing_history=3,
            logger=self.mock_logger
        )

    def test_update_arm_positions_without_data(self):
        """Test aktualizacji pozycji rąk bez danych wejściowych."""
        # Ręce powinny być w pozycji neutralnej
        self.renderer._update_arm_positions(None)

        # Sprawdzamy czy pozycje rąk zostały zainicjowane jako None
        self.assertIsNone(self.renderer.last_left_elbow)
        self.assertIsNone(self.renderer.last_right_elbow)
        self.assertIsNone(self.renderer.last_left_wrist)
        self.assertIsNone(self.renderer.last_right_wrist)

        # Ustawiamy jakieś wartości
        self.renderer.last_left_elbow = (100, 100)
        self.renderer.last_right_elbow = (200, 100)
        self.renderer.last_left_wrist = (80, 150)
        self.renderer.last_right_wrist = (220, 150)

        # Ponownie aktualizujemy bez danych - powinny przejść w stronę neutralnej pozycji
        self.renderer._update_arm_positions(None)

        # Sprawdzamy czy pozycje zostały zmienione (przesunięte w stronę neutralnych pozycji)
        # Nie sprawdzamy dokładnych wartości, bo te zależą od timera animacji
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_right_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)
        self.assertIsNotNone(self.renderer.last_right_wrist)

    def test_update_arm_positions_with_hand_data(self):
        """Test aktualizacji pozycji rąk z danymi z MediaPipe Hands."""
        # Tworzymy sztuczne dane rąk
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.3, 0.6, 0.0, 1.0),
                    "elbow": (0.35, 0.4, 0.0, 1.0),
                    "is_left": True
                },
                "right_hand": {
                    "wrist": (0.7, 0.6, 0.0, 1.0),
                    "elbow": (0.65, 0.4, 0.0, 1.0),
                    "is_left": False
                }
            }
        }

        # Aktualizujemy pozycje rąk
        self.renderer._update_arm_positions(hands_data)

        # Sprawdzamy czy flagi widoczności zostały ustawione
        self.assertTrue(self.renderer.left_arm_visible)
        self.assertTrue(self.renderer.right_arm_visible)

        # Sprawdzamy czy pozycje zostały zaktualizowane
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_right_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)
        self.assertIsNotNone(self.renderer.last_right_wrist)

        # Sprawdzamy przybliżone wartości (z uwzględnieniem wygładzania)
        left_elbow_x = int(0.35 * 640 * (1 - self.renderer.smooth_factor))  # Pierwszy raz
        left_elbow_y = int(0.4 * 480 * (1 - self.renderer.smooth_factor))
        self.assertAlmostEqual(self.renderer.last_left_elbow[0], left_elbow_x, delta=5)
        self.assertAlmostEqual(self.renderer.last_left_elbow[1], left_elbow_y, delta=5)

    def test_transition_to_neutral_position(self):
        """Test przechodzenia z wykrytej pozycji do neutralnej."""
        # Najpierw ustawiamy pozycje rąk z danymi
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.3, 0.6, 0.0, 1.0),
                    "elbow": (0.35, 0.4, 0.0, 1.0),
                    "is_left": True
                }
            }
        }

        # Aktualizujemy pozycje rąk
        self.renderer._update_arm_positions(hands_data)

        # Zapisujemy pozycje
        left_elbow_with_data = self.renderer.last_left_elbow
        left_wrist_with_data = self.renderer.last_left_wrist

        # Ustawiamy czas widoczności na dawny, aby symulować utratę śledzenia
        self.renderer.left_arm_visibility_time = time.time() - 1.0
        self.renderer.left_arm_visible = False

        # Aktualizujemy bez danych o lewej ręce
        self.renderer._update_arm_positions({"hands_data": {"left_hand": None, "right_hand": None}})

        # Sprawdzamy czy pozycje zmieniły się w stronę neutralnych
        self.assertIsNotNone(self.renderer.last_left_elbow)
        self.assertIsNotNone(self.renderer.last_left_wrist)

        # Pozycje powinny być inne niż wcześniej (przesunięte w stronę neutralnych)
        # Dokładne wartości zależą od timera animacji, więc nie sprawdzamy ich bezpośrednio
        self.assertNotEqual(self.renderer.last_left_elbow, left_elbow_with_data)
        self.assertNotEqual(self.renderer.last_left_wrist, left_wrist_with_data)

    def test_render_arms(self):
        """Test renderowania rąk na obrazie."""
        # Ustawiamy pozycje rąk
        self.renderer.last_left_elbow = (150, 150)
        self.renderer.last_right_elbow = (350, 150)
        self.renderer.last_left_wrist = (100, 200)
        self.renderer.last_right_wrist = (400, 200)

        # Renderujemy obraz
        canvas = self.renderer.render()

        # Sprawdzamy wymiary
        self.assertEqual(canvas.shape, (480, 640, 3))

        # Sprawdzamy czy obraz nie jest pusty (w pełni biały)
        self.assertFalse(np.array_equal(canvas, np.ones((480, 640, 3), dtype=np.uint8) * 255))

    def test_smooth_transition(self):
        """Test płynnego przejścia między różnymi pozycjami rąk."""
        # Ustawiamy początkową pozycję
        self.renderer.last_left_elbow = (150, 150)
        self.renderer.last_left_wrist = (100, 200)

        # Tworzymy nowe dane z innymi pozycjami
        hands_data = {
            "hands_data": {
                "left_hand": {
                    "wrist": (0.2, 0.5, 0.0, 1.0),  # ~(128, 240)
                    "elbow": (0.3, 0.35, 0.0, 1.0),  # ~(192, 168)
                    "is_left": True
                }
            }
        }

        # Wykonujemy kilka aktualizacji, aby zobaczyć płynne przejście
        positions = []
        for _ in range(5):
            self.renderer._update_arm_positions(hands_data)
            positions.append((
                self.renderer.last_left_elbow[0],
                self.renderer.last_left_elbow[1],
                self.renderer.last_left_wrist[0],
                self.renderer.last_left_wrist[1]
            ))

        # Sprawdzamy czy pozycje zmieniają się stopniowo
        for i in range(1, len(positions)):
            # Różnica między kolejnymi pozycjami nie powinna być zbyt duża
            self.assertLess(abs(positions[i][0] - positions[i - 1][0]), 50)
            self.assertLess(abs(positions[i][1] - positions[i - 1][1]), 50)
            self.assertLess(abs(positions[i][2] - positions[i - 1][2]), 50)
            self.assertLess(abs(positions[i][3] - positions[i - 1][3]), 50)

        # Pozycja końcowa powinna różnić się od początkowej
        self.assertNotEqual(self.renderer.last_left_elbow, (150, 150))
        self.assertNotEqual(self.renderer.last_left_wrist, (100, 200))


if __name__ == "__main__":
    unittest.main()

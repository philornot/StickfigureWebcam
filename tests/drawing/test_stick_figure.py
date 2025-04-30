#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/drawing/test_stick_figure.py
"""
Testy jednostkowe dla renderera stick figure (StickFigureRenderer).
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from src.drawing.stick_figure import StickFigureRenderer


class TestStickFigureRenderer(unittest.TestCase):
    """
    Testy dla klasy StickFigureRenderer, która rysuje stick figure na podstawie wykrytych punktów.
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

    def test_initialization(self):
        """Test inicjalizacji renderera stick figure."""
        # Sprawdzamy czy inicjalizacja przebiegła poprawnie
        self.assertEqual(self.renderer.canvas_width, 640)
        self.assertEqual(self.renderer.canvas_height, 480)
        self.assertEqual(self.renderer.line_thickness, 3)
        self.assertEqual(self.renderer.bg_color, (255, 255, 255))
        self.assertEqual(self.renderer.figure_color, (0, 0, 0))
        self.assertEqual(self.renderer.chair_color, (150, 75, 0))
        self.assertEqual(self.renderer.smooth_factor, 0.3)
        self.assertEqual(self.renderer.smoothing_history, 3)

        # Sprawdzamy obliczenie promienia głowy
        self.assertEqual(self.renderer.head_radius, int(0.075 * 480))

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.info.assert_called_once()

    def test_render_no_landmarks(self):
        """Test renderowania bez punktów charakterystycznych."""
        # Wywołujemy render bez punktów
        result = self.renderer.render(None, False, 0.0)

        # Sprawdzamy wymiary i kolor obrazu wynikowego
        self.assertEqual(result.shape, (480, 640, 3))
        # Sprawdzamy czy obraz jest biały (tło)
        self.assertTrue(np.all(result == 255))  # 255 to wartość dla białego koloru

        # Sprawdzamy czy logger został wywołany
        self.mock_logger.debug.assert_called_once()

    def create_mock_landmarks(self):
        """Tworzy sztuczne dane punktów charakterystycznych dla testów."""
        # Tworzymy 33 punkty z dobrą widocznością
        landmarks = [(0, 0, 0, 0.9)] * 33

        # Ustawiamy podstawowe punkty dla testu
        # Głowa
        landmarks[self.renderer.NOSE] = (0.5, 0.2, 0, 0.9)
        landmarks[self.renderer.LEFT_EYE] = (0.45, 0.18, 0, 0.9)
        landmarks[self.renderer.RIGHT_EYE] = (0.55, 0.18, 0, 0.9)

        # Ramiona
        landmarks[self.renderer.LEFT_SHOULDER] = (0.4, 0.3, 0, 0.9)
        landmarks[self.renderer.RIGHT_SHOULDER] = (0.6, 0.3, 0, 0.9)

        # Łokcie
        landmarks[self.renderer.LEFT_ELBOW] = (0.3, 0.4, 0, 0.9)
        landmarks[self.renderer.RIGHT_ELBOW] = (0.7, 0.4, 0, 0.9)

        # Nadgarstki
        landmarks[self.renderer.LEFT_WRIST] = (0.25, 0.5, 0, 0.9)
        landmarks[self.renderer.RIGHT_WRIST] = (0.75, 0.5, 0, 0.9)

        # Biodra
        landmarks[self.renderer.LEFT_HIP] = (0.45, 0.6, 0, 0.9)
        landmarks[self.renderer.RIGHT_HIP] = (0.55, 0.6, 0, 0.9)

        # Kolana
        landmarks[self.renderer.LEFT_KNEE] = (0.4, 0.8, 0, 0.9)
        landmarks[self.renderer.RIGHT_KNEE] = (0.6, 0.8, 0, 0.9)

        # Kostki
        landmarks[self.renderer.LEFT_ANKLE] = (0.4, 0.95, 0, 0.9)
        landmarks[self.renderer.RIGHT_ANKLE] = (0.6, 0.95, 0, 0.9)

        return landmarks

    def test_render_standing_figure(self):
        """Test renderowania stick figure w pozycji stojącej."""
        # Tworzymy sztuczne punkty charakterystyczne
        landmarks = self.create_mock_landmarks()

        # Renderujemy stick figure w pozycji stojącej
        result = self.renderer.render(landmarks, False, 0.8)

        # Sprawdzamy wymiary obrazu wynikowego
        self.assertEqual(result.shape, (480, 640, 3))

        # Sprawdzamy czy obraz nie jest całkowicie biały (coś zostało narysowane)
        self.assertFalse(np.all(result == 255))

        # Pewne punkty na obrazie powinny mieć kolor stick figure (czarny)
        # Na przykład w miejscu głowy
        nose_x, nose_y = int(landmarks[self.renderer.NOSE][0] * 640), int(landmarks[self.renderer.NOSE][1] * 480)
        # Sprawdzamy obszar wokół nosa - powinien mieć piksele koloru innego niż tło
        head_area = result[max(0, nose_y - 10):min(480, nose_y + 10), max(0, nose_x - 10):min(640, nose_x + 10)]
        self.assertFalse(np.all(head_area == 255))

    def test_render_sitting_figure(self):
        """Test renderowania stick figure w pozycji siedzącej."""
        # Tworzymy sztuczne punkty charakterystyczne
        landmarks = self.create_mock_landmarks()

        # Renderujemy stick figure w pozycji siedzącej
        result = self.renderer.render(landmarks, True, 0.8)

        # Sprawdzamy wymiary obrazu wynikowego
        self.assertEqual(result.shape, (480, 640, 3))

        # Sprawdzamy czy obraz nie jest całkowicie biały (coś zostało narysowane)
        self.assertFalse(np.all(result == 255))

        # W pozycji siedzącej powinno być narysowane krzesło
        # Sprawdzamy obszar pod biodrami - powinien mieć piksele koloru krzesła
        hip_x = int((landmarks[self.renderer.LEFT_HIP][0] + landmarks[self.renderer.RIGHT_HIP][0]) / 2 * 640)
        hip_y = int(landmarks[self.renderer.LEFT_HIP][1] * 480)
        chair_area = result[min(479, hip_y + 5):min(479, hip_y + 20), max(0, hip_x - 20):min(639, hip_x + 20)]

        # Sprawdzamy czy są piksele inne niż białe (tło) w obszarze krzesła
        self.assertFalse(np.all(chair_area == 255))

    def test_smooth_landmarks(self):
        """Test wygładzania punktów charakterystycznych."""
        # Tworzymy kilka zestawów punktów z małymi różnicami dla symulacji ruchu
        landmarks1 = self.create_mock_landmarks()
        landmarks2 = self.create_mock_landmarks()
        landmarks3 = self.create_mock_landmarks()

        # Dodajemy niewielkie przesunięcie do drugiego zestawu
        landmarks2[self.renderer.NOSE] = (0.51, 0.21, 0, 0.9)  # Przesunięcie nosa

        # Dodajemy większe przesunięcie do trzeciego zestawu
        landmarks3[self.renderer.NOSE] = (0.52, 0.22, 0, 0.9)  # Jeszcze większe przesunięcie

        # Renderujemy pierwszy zestaw punktów - inicjalizuje historię
        self.renderer.render(landmarks1, False, 0.8)

        # Renderujemy drugi zestaw - teraz wygładzanie powinno działać
        self.renderer.render(landmarks2, False, 0.8)

        # Sprawdzamy czy w historii są dwa zestawy punktów
        self.assertEqual(len(self.renderer.landmark_history), 2)

        # Renderujemy trzeci zestaw
        self.renderer.render(landmarks3, False, 0.8)

        # Sprawdzamy czy w historii są trzy zestawy punktów
        self.assertEqual(len(self.renderer.landmark_history), 3)

        # Wywołujemy bezpośrednio metodę wygładzania
        smoothed = self.renderer._smooth_landmarks(landmarks3)

        # Sprawdzamy czy wygładzone punkty różnią się od oryginalnych
        self.assertNotEqual(smoothed[self.renderer.NOSE], landmarks3[self.renderer.NOSE])

        # Wygładzone punkty powinny być pomiędzy wartościami w historii
        original_x = landmarks3[self.renderer.NOSE][0]
        smoothed_x = smoothed[self.renderer.NOSE][0]

        # Wygładzona wartość powinna być mniejsza niż ostatnia (ponieważ wcześniejsze wartości były mniejsze)
        self.assertLess(smoothed_x, original_x)

    def test_set_colors(self):
        """Test zmiany kolorów renderera."""
        # Początkowe kolory
        original_bg_color = self.renderer.bg_color
        original_figure_color = self.renderer.figure_color
        original_chair_color = self.renderer.chair_color

        # Nowe kolory
        new_bg_color = (240, 240, 240)
        new_figure_color = (50, 50, 50)
        new_chair_color = (100, 50, 0)

        # Ustawiamy nowe kolory
        self.renderer.set_colors(
            bg_color=new_bg_color,
            figure_color=new_figure_color,
            chair_color=new_chair_color
        )

        # Sprawdzamy czy kolory zostały zmienione
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, new_figure_color)
        self.assertEqual(self.renderer.chair_color, new_chair_color)

        # Sprawdzamy aktualizację częściową - tylko kolor figury
        newer_figure_color = (100, 100, 100)
        self.renderer.set_colors(figure_color=newer_figure_color)

        # Sprawdzamy czy tylko kolor figury został zmieniony
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, newer_figure_color)
        self.assertEqual(self.renderer.chair_color, new_chair_color)

    def test_set_line_thickness(self):
        """Test zmiany grubości linii."""
        # Początkowa grubość
        original_thickness = self.renderer.line_thickness

        # Nowa grubość
        new_thickness = 5

        # Ustawiamy nową grubość
        self.renderer.set_line_thickness(new_thickness)

        # Sprawdzamy czy grubość została zmieniona
        self.assertEqual(self.renderer.line_thickness, new_thickness)

        # Sprawdzamy czy zbyt mała wartość zostanie poprawiona
        self.renderer.set_line_thickness(0)

        # Grubość powinna być co najmniej 1
        self.assertEqual(self.renderer.line_thickness, 1)

    def test_set_smoothing(self):
        """Test zmiany parametrów wygładzania."""
        # Początkowe wartości
        original_smooth_factor = self.renderer.smooth_factor
        original_history_length = self.renderer.smoothing_history

        # Nowe wartości
        new_smooth_factor = 0.5
        new_history_length = 5

        # Ustawiamy nowe wartości
        self.renderer.set_smoothing(new_smooth_factor, new_history_length)

        # Sprawdzamy czy wartości zostały zmienione
        self.assertEqual(self.renderer.smooth_factor, new_smooth_factor)
        self.assertEqual(self.renderer.smoothing_history, new_history_length)

    def test_resize(self):
        """Test zmiany rozmiaru płótna."""
        # Początkowe wymiary
        original_width = self.renderer.canvas_width
        original_height = self.renderer.canvas_height
        original_head_radius = self.renderer.head_radius

        # Nowe wymiary
        new_width = 800
        new_height = 600

        # Zmieniamy rozmiar
        self.renderer.resize(new_width, new_height)

        # Sprawdzamy czy wymiary zostały zmienione
        self.assertEqual(self.renderer.canvas_width, new_width)
        self.assertEqual(self.renderer.canvas_height, new_height)

        # Promień głowy powinien zostać przeliczony
        expected_head_radius = int(self.renderer.head_radius_factor * new_height)
        self.assertEqual(self.renderer.head_radius, expected_head_radius)

        # Historia wygładzania powinna zostać zresetowana
        self.assertEqual(len(self.renderer.landmark_history), 0)

    def test_reset(self):
        """Test resetowania wewnętrznego stanu renderera."""
        # Najpierw dodajemy coś do historii wygładzania
        landmarks = self.create_mock_landmarks()
        self.renderer.render(landmarks, False, 0.8)
        self.renderer.render(landmarks, False, 0.8)

        # Sprawdzamy czy historia się wypełniła
        self.assertEqual(len(self.renderer.landmark_history), 2)

        # Resetujemy stan
        self.renderer.reset()

        # Sprawdzamy czy historia została wyczyszczona
        self.assertEqual(len(self.renderer.landmark_history), 0)


if __name__ == "__main__":
    unittest.main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/drawing/test_stick_figure.py
"""
Testy jednostkowe dla renderera stick figure (StickFigureRenderer).
"""

import unittest

import numpy as np

from src.drawing.pose_analyzer import PoseAnalyzer
from src.drawing.stick_figure_renderer import StickFigureRenderer


class TestStickFigureRenderer(unittest.TestCase):
    """
    Testy dla klasy StickFigureRenderer, która rysuje stick figure na podstawie wykrytych punktów.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        # Tworzenie renderera
        self.renderer = StickFigureRenderer(
            canvas_width=640,
            canvas_height=480,
            line_thickness=3,
            head_radius_factor=0.12,  # Zwiększony rozmiar głowy
            bg_color=(255, 255, 255),
            figure_color=(0, 0, 0),
            chair_color=(150, 75, 0),
            smooth_factor=0.3,
            smoothing_history=3
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
        self.assertEqual(self.renderer.head_radius, int(0.12 * 480))

        # Sprawdzamy czy mamy instancje analizatora pozy i renderera twarzy
        self.assertIsInstance(self.renderer.pose_analyzer, PoseAnalyzer)
        self.assertIsNotNone(self.renderer.face_renderer)

    def test_render_no_landmarks(self):
        """Test renderowania bez punktów charakterystycznych."""
        # Wywołujemy render bez punktów
        result = self.renderer.render(None)

        # Sprawdzamy wymiary i kolor obrazu wynikowego
        self.assertEqual(result.shape, (480, 640, 3))
        # Sprawdzamy czy obraz jest biały (tło)
        self.assertTrue(np.all(result == 255))  # 255 to wartość dla białego koloru

    def create_mock_landmarks(self):
        """Tworzy sztuczne dane punktów charakterystycznych dla testów."""
        # Tworzymy 15 punktów z dobrą widocznością
        landmarks = [(0, 0, 0, 0.9)] * 15

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

        return landmarks

    def test_render_upper_body(self):
        """Test renderowania stick figure z górną częścią ciała."""
        # Tworzymy sztuczne punkty charakterystyczne
        landmarks = self.create_mock_landmarks()

        # Przygotowujemy dane twarzy z informacją o rękach
        face_data = {
            "has_face": True,
            "landmarks": landmarks,
            "expressions": {
                "mouth_open": 0.2,
                "smile": 0.6,
                "left_eye_open": 1.0,
                "right_eye_open": 1.0
            },
            "hands_data": {
                "left_hand": {
                    "wrist": landmarks[self.renderer.LEFT_WRIST],
                    "elbow": landmarks[self.renderer.LEFT_ELBOW],
                    "is_left": True
                },
                "right_hand": {
                    "wrist": landmarks[self.renderer.RIGHT_WRIST],
                    "elbow": landmarks[self.renderer.RIGHT_ELBOW],
                    "is_left": False
                }
            }
        }

        # Renderujemy stick figure z górną częścią ciała
        result = self.renderer.render(face_data)

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

    def test_set_colors(self):
        """Test zmiany kolorów renderera."""
        # Początkowe kolory
        original_bg_color = self.renderer.bg_color
        original_figure_color = self.renderer.figure_color

        # Nowe kolory
        new_bg_color = (240, 240, 240)
        new_figure_color = (50, 50, 50)

        # Ustawiamy nowe kolory
        self.renderer.set_colors(
            bg_color=new_bg_color,
            figure_color=new_figure_color
        )

        # Sprawdzamy czy kolory zostały zmienione
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, new_figure_color)

        # Sprawdzamy aktualizację częściową - tylko kolor figury
        newer_figure_color = (100, 100, 100)
        self.renderer.set_colors(figure_color=newer_figure_color)

        # Sprawdzamy czy tylko kolor figury został zmieniony
        self.assertEqual(self.renderer.bg_color, new_bg_color)
        self.assertEqual(self.renderer.figure_color, newer_figure_color)

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

    def test_set_mood(self):
        """Test zmiany nastroju."""
        # Początkowy nastrój
        original_mood = self.renderer.mood

        # Nowy nastrój
        new_mood = "sad"

        # Ustawiamy nowy nastrój
        self.renderer.set_mood(new_mood)

        # Sprawdzamy czy nastrój został zmieniony
        self.assertEqual(self.renderer.mood, new_mood)

        # Sprawdzamy czy nieprawidłowy nastrój zostanie zignorowany
        self.renderer.set_mood("nieprawidłowy")

        # Nastrój nie powinien się zmienić
        self.assertEqual(self.renderer.mood, new_mood)

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
        face_data = {"landmarks": landmarks, "has_face": True}
        self.renderer.render(face_data)
        self.renderer.render(face_data)

        # Sprawdzamy czy historia się wypełniła
        self.assertTrue(hasattr(self.renderer, 'last_left_elbow'))
        self.assertTrue(hasattr(self.renderer, 'last_right_elbow'))

        # Resetujemy stan
        self.renderer.reset()

        # Sprawdzamy czy historia została wyczyszczona
        self.assertIsNone(self.renderer.last_left_elbow)
        self.assertIsNone(self.renderer.last_right_elbow)

        # Sprawdzamy czy nastrój wrócił do domyślnego
        self.assertEqual(self.renderer.mood, "happy")


class TestPoseAnalyzer(unittest.TestCase):
    """
    Testy dla klasy PoseAnalyzer, która analizuje pozę człowieka.
    """

    def setUp(self):
        """Inicjalizacja przed każdym testem."""
        self.analyzer = PoseAnalyzer()

    def test_initialization(self):
        """Test inicjalizacji analizatora pozy."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.sitting_threshold, 0.3)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# drawing/stick_figure_renderer.py

from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np

from .face_renderer import SimpleFaceRenderer


class StickFigureRenderer:
    """
    Uproszczona klasa do renderowania stick figure (patyczaka) skupiona na popierstu.
    Pozycja jest zawsze na środku ekranu, niezależnie od pozycji wykrytej przez kamerę.
    """

    def __init__(
        self,
        canvas_width: int = 640,
        canvas_height: int = 480,
        line_thickness: int = 3,
        head_radius_factor: float = 0.075,
        bg_color: Tuple[int, int, int] = (255, 255, 255),  # Białe tło
        figure_color: Tuple[int, int, int] = (0, 0, 0),  # Czarny patyczak
        chair_color: Tuple[int, int, int] = (150, 75, 0),  # Brązowe krzesło
        smooth_factor: float = 0.3,  # Współczynnik wygładzania ruchu
    ):
        """
        Inicjalizacja renderera stick figure.

        Args:
            canvas_width (int): Szerokość obszaru rysowania
            canvas_height (int): Wysokość obszaru rysowania
            line_thickness (int): Grubość linii stick figure
            head_radius_factor (float): Promień głowy jako ułamek wysokości
            bg_color (Tuple[int, int, int]): Kolor tła (BGR)
            figure_color (Tuple[int, int, int]): Kolor stick figure (BGR)
            chair_color (Tuple[int, int, int]): Kolor krzesła (BGR)
            smooth_factor (float): Współczynnik wygładzania ruchu (0.0-1.0)
        """
        # Parametry renderowania
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.line_thickness = max(2, line_thickness)
        self.head_radius_factor = head_radius_factor
        self.bg_color = bg_color
        self.figure_color = figure_color
        self.chair_color = chair_color
        self.smooth_factor = smooth_factor

        # Obliczanie promienia głowy
        self.head_radius = int(head_radius_factor * canvas_height)

        # Inicjalizacja renderera twarzy
        self.face_renderer = SimpleFaceRenderer(
            feature_color=figure_color,
            smooth_factor=smooth_factor
        )

        # Stan nastroju
        self.mood = "neutral"  # Domyślny neutralny nastrój

        # Pozycja zawsze na środku ekranu
        self.head_center = (canvas_width // 2, canvas_height // 3)

        # Czy pokazywać krzesło
        self.show_chair = False

        print(f"StickFigureRenderer zainicjalizowany ({canvas_width}x{canvas_height})")

    def render(
        self,
        face_data: Optional[Dict[str, Any]] = None,
        show_chair: bool = False
    ) -> np.ndarray:
        """
        Renderuje stick figure zawsze na środku ekranu.

        Args:
            face_data (Optional[Dict[str, Any]]): Dane twarzy (opcjonalne)
            show_chair (bool): Czy pokazać krzesło

        Returns:
            np.ndarray: Obraz z narysowanym stick figure
        """
        try:
            # Tworzenie pustego obrazu
            canvas = np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
            canvas[:] = self.bg_color

            # Zapisz opcję pokazywania krzesła
            self.show_chair = show_chair

            # Rysowanie stick figure (zawsze na środku)
            self._draw_figure(canvas)

            # Rysowanie twarzy
            self.face_renderer.draw_face(
                canvas,
                self.head_center,
                self.head_radius,
                self.mood,
                face_data
            )

            return canvas

        except Exception as e:
            print(f"Błąd podczas renderowania stick figure: {str(e)}")
            # W przypadku błędu zwracamy pusty biały obraz
            return np.ones((self.canvas_height, self.canvas_width, 3), dtype=np.uint8) * 255

    def _draw_figure(self, canvas: np.ndarray) -> None:
        """
        Rysuje stick figure względem pozycji głowy.

        Args:
            canvas (np.ndarray): Płótno do rysowania
        """
        try:
            # 1. Głowa - okrąg
            cv2.circle(canvas, self.head_center, self.head_radius, self.figure_color, self.line_thickness)

            # 2. Obliczenie pozycji innych części ciała względem głowy
            # Środek głowy
            head_x, head_y = self.head_center

            # Klatka piersiowa - poniżej głowy
            torso_top_y = head_y + self.head_radius + 10
            torso_length = int(self.head_radius * 2.5)
            torso_bottom_y = torso_top_y + torso_length

            # Ramiona - powyżej klatki piersiowej
            shoulder_y = torso_top_y + int(self.head_radius * 0.5)
            shoulder_width = int(self.head_radius * 3)
            left_shoulder_x = head_x - shoulder_width // 2
            right_shoulder_x = head_x + shoulder_width // 2

            # 3. Rysowanie tułowia (linia pionowa od głowy w dół)
            cv2.line(
                canvas,
                (head_x, torso_top_y),
                (head_x, torso_bottom_y),
                self.figure_color,
                self.line_thickness
            )

            # 4. Rysowanie linii barków
            cv2.line(
                canvas,
                (left_shoulder_x, shoulder_y),
                (right_shoulder_x, shoulder_y),
                self.figure_color,
                self.line_thickness
            )

            # 5. Rysowanie rąk
            # Łokcie - poniżej ramion
            elbow_offset_y = int(self.head_radius * 1.0)
            elbow_offset_x = int(self.head_radius * 0.7)

            # Lewy łokieć
            left_elbow_x = left_shoulder_x - elbow_offset_x
            left_elbow_y = shoulder_y + elbow_offset_y

            # Prawy łokieć
            right_elbow_x = right_shoulder_x + elbow_offset_x
            right_elbow_y = shoulder_y + elbow_offset_y

            # Nadgarstki - poniżej łokci
            wrist_offset_y = int(self.head_radius * 1.0)
            wrist_offset_x = int(self.head_radius * 0.5)

            # Lewy nadgarstek
            left_wrist_x = left_elbow_x - wrist_offset_x
            left_wrist_y = left_elbow_y + wrist_offset_y

            # Prawy nadgarstek
            right_wrist_x = right_elbow_x + wrist_offset_x
            right_wrist_y = right_elbow_y + wrist_offset_y

            # Rysowanie lewego ramienia i przedramienia
            cv2.line(
                canvas,
                (left_shoulder_x, shoulder_y),
                (left_elbow_x, left_elbow_y),
                self.figure_color,
                self.line_thickness
            )

            cv2.line(
                canvas,
                (left_elbow_x, left_elbow_y),
                (left_wrist_x, left_wrist_y),
                self.figure_color,
                self.line_thickness
            )

            # Rysowanie prawego ramienia i przedramienia
            cv2.line(
                canvas,
                (right_shoulder_x, shoulder_y),
                (right_elbow_x, right_elbow_y),
                self.figure_color,
                self.line_thickness
            )

            cv2.line(
                canvas,
                (right_elbow_x, right_elbow_y),
                (right_wrist_x, right_wrist_y),
                self.figure_color,
                self.line_thickness
            )

            # 6. Rysowanie krzesła (jeśli potrzeba)
            if self.show_chair:
                self._draw_simple_chair(canvas, head_x, torso_bottom_y)

        except Exception as e:
            print(f"Błąd podczas rysowania postaci: {str(e)}")

    def _draw_simple_chair(self, canvas: np.ndarray, head_x: int, torso_bottom_y: int) -> None:
        """
        Rysuje proste krzesło pod dolną częścią tułowia.

        Args:
            canvas (np.ndarray): Płótno do rysowania
            head_x (int): Współrzędna X środka głowy
            torso_bottom_y (int): Współrzędna Y dolnej części tułowia
        """
        try:
            # Parametry krzesła
            chair_width = self.head_radius * 3  # Szerokość siedziska
            seat_height = self.head_radius // 2  # Wysokość siedziska
            leg_height = self.head_radius * 1.5  # Długość nóg krzesła

            # Współrzędne siedziska
            seat_top_y = torso_bottom_y + 5  # Małe przesunięcie w dół od końca tułowia
            seat_bottom_y = seat_top_y + seat_height
            seat_left_x = head_x - chair_width // 2
            seat_right_x = head_x + chair_width // 2

            # Upewniamy się, że punkty są w granicach obrazu
            h, w, _ = canvas.shape
            seat_top_y = min(h - seat_height - 1, max(0, seat_top_y))
            seat_bottom_y = min(h - 1, seat_top_y + seat_height)
            seat_left_x = min(w - 1, max(0, seat_left_x))
            seat_right_x = min(w - 1, max(0, seat_right_x))

            # Rysujemy siedzisko
            cv2.rectangle(
                canvas,
                (seat_left_x, seat_top_y),
                (seat_right_x, seat_bottom_y),
                self.chair_color,
                -1  # wypełniony
            )

            # Obliczamy pozycje nóg krzesła
            leg_width = max(1, self.line_thickness - 1)

            # Lewa noga
            left_leg_x = seat_left_x + chair_width // 4

            # Prawa noga
            right_leg_x = seat_right_x - chair_width // 4

            # Sprawdzamy czy wszystkie punkty są w obrazie
            if (0 <= left_leg_x < w and
                0 <= right_leg_x < w and
                0 <= seat_bottom_y < h and
                0 <= seat_bottom_y + leg_height < h):

                # Rysujemy nogi
                cv2.line(
                    canvas,
                    (left_leg_x, seat_bottom_y),
                    (left_leg_x, seat_bottom_y + leg_height),
                    self.chair_color,
                    leg_width
                )

                cv2.line(
                    canvas,
                    (right_leg_x, seat_bottom_y),
                    (right_leg_x, seat_bottom_y + leg_height),
                    self.chair_color,
                    leg_width
                )

                # Oparcie - prosta pionowa linia
                backrest_height = self.head_radius * 1.5

                # Sprawdzamy czy oparcie jest w granicach obrazu
                if 0 <= seat_top_y - backrest_height < h:
                    cv2.line(
                        canvas,
                        (head_x, seat_top_y),
                        (head_x, seat_top_y - backrest_height),
                        self.chair_color,
                        leg_width
                    )

        except Exception as e:
            print(f"Błąd podczas rysowania krzesła: {str(e)}")

    def set_colors(
        self,
        bg_color: Optional[Tuple[int, int, int]] = None,
        figure_color: Optional[Tuple[int, int, int]] = None,
        chair_color: Optional[Tuple[int, int, int]] = None
    ) -> None:
        """
        Aktualizuje kolory używane do rysowania.

        Args:
            bg_color (Optional[Tuple[int, int, int]]): Nowy kolor tła (BGR)
            figure_color (Optional[Tuple[int, int, int]]): Nowy kolor stick figure (BGR)
            chair_color (Optional[Tuple[int, int, int]]): Nowy kolor krzesła (BGR)
        """
        if bg_color is not None:
            self.bg_color = bg_color

        if figure_color is not None:
            self.figure_color = figure_color
            # Aktualizacja koloru w rendererze twarzy
            self.face_renderer.feature_color = figure_color

        if chair_color is not None:
            self.chair_color = chair_color

    def set_mood(self, mood: str) -> None:
        """
        Ustawia nastrój stick figure, który wpływa na mimikę twarzy.

        Args:
            mood (str): Nastrój: "happy", "sad", "neutral", "surprised", "wink"
        """
        valid_moods = ["happy", "sad", "neutral", "surprised", "wink"]
        if mood in valid_moods:
            self.mood = mood
        else:
            print(f"Nieprawidłowy nastrój: {mood}. Dozwolone wartości: {valid_moods}")

    def set_line_thickness(self, thickness: int) -> None:
        """
        Aktualizuje grubość linii.

        Args:
            thickness (int): Nowa grubość linii
        """
        self.line_thickness = max(1, thickness)

    def resize(self, width: int, height: int) -> None:
        """
        Zmienia rozmiar płótna.

        Args:
            width (int): Nowa szerokość
            height (int): Nowa wysokość
        """
        self.canvas_width = width
        self.canvas_height = height

        # Aktualizacja promienia głowy
        self.head_radius = int(self.head_radius_factor * height)

        # Aktualizacja pozycji głowy na środku ekranu
        self.head_center = (width // 2, height // 3)

    def reset(self) -> None:
        """
        Resetuje wewnętrzny stan renderera.
        """
        self.mood = "neutral"  # Przywrócenie domyślnego nastroju
        self.face_renderer.reset()
